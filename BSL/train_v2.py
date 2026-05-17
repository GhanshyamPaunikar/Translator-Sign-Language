#!/usr/bin/env python3
"""
Sign -> Text v2. Transformer over improved pose+hand features.

Key improvements over train_recognizer.py:
  - Upper-body pose only (joints 0-24, skip legs/feet which are noise)
  - Hand features as bone vectors (relative to wrist) — scale/position invariant
  - Explicit velocity features (frame-to-frame delta) — the motion IS the sign
  - Transformer encoder instead of Bi-LSTM — better temporal attention
  - Defaults to words with >= 10 clips (24 classes, ~10 clips each)
  - Confusion matrix printed on test set

Usage:
    python3 train_v2.py                        # default: random split, >=10 clips
    python3 train_v2.py --split by_provider
    python3 train_v2.py --min-clips 8          # 46 classes
    python3 train_v2.py --save model.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT   = Path(__file__).parent
DATA   = ROOT / "bsl_dataset"
LM_DIR = DATA / "landmarks"
SPLITS = DATA / "splits.csv"

SEQ_LEN = 32

# Upper-body MediaPipe pose joints (skip 25-32: hips, knees, ankles, feet)
UPPER_JOINTS = list(range(25))          # 25 joints
POSE_DIM     = len(UPPER_JOINTS) * 4    # x,y,z,vis  -> 100
HAND_DIM     = 21 * 3                   # bone vectors x,y,z -> 63 per hand
VEL_DIM      = len(UPPER_JOINTS) * 2   # x,y velocity for pose -> 50
FEAT_DIM     = POSE_DIM + 2 * HAND_DIM + VEL_DIM   # 100+63+63+50 = 276


# ── feature engineering ──────────────────────────────────────────────────────

def _shoulder_norm(pose: np.ndarray):
    """Per-frame shoulder midpoint + width from (T,33,4) pose."""
    ls = pose[:, 11, :2]
    rs = pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    w = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    w = np.where(w < 1e-3, 1.0, w)
    return mid, w


def normalise_pose(pose: np.ndarray) -> np.ndarray:
    mid, w = _shoulder_norm(pose)
    out = pose[:, UPPER_JOINTS, :].copy()      # (T,25,4)
    out[:, :, 0] = (out[:, :, 0] - mid[:, 0:1]) / w
    out[:, :, 1] = (out[:, :, 1] - mid[:, 1:2]) / w
    return out


def hand_bone_vectors(hand: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Express each hand joint relative to the wrist (joint 0), then
    normalise by shoulder width so it's scale-invariant.

    hand: (T,21,3)  — already nan_to_num'd
    returns (T,21,3)
    """
    _, w = _shoulder_norm(pose)
    wrist = hand[:, 0:1, :]                    # (T,1,3) — wrist position
    bones = (hand - wrist) / w[:, :, None]     # relative, normalised
    return bones


def velocity(seq: np.ndarray) -> np.ndarray:
    """Frame-to-frame delta, padded with zeros at t=0.
    seq: (T, J, 2) — returns (T, J, 2)
    """
    v = np.zeros_like(seq)
    v[1:] = seq[1:] - seq[:-1]
    return v


def resample(seq: np.ndarray, n: int) -> np.ndarray:
    t = seq.shape[0]
    if t == n:
        return seq
    if t == 1:
        return np.broadcast_to(seq, (n,) + seq.shape[1:]).copy()
    idx = np.linspace(0, t - 1, n, dtype=np.float32)
    fl = np.floor(idx).astype(int)
    ce = np.minimum(fl + 1, t - 1)
    frac = (idx - fl).reshape((n,) + (1,) * (seq.ndim - 1))
    return (seq[fl] * (1 - frac) + seq[ce] * frac).astype(seq.dtype, copy=False)


def load_clip(npz_path: Path) -> np.ndarray | None:
    try:
        with np.load(npz_path) as z:
            pose = np.asarray(z["pose"],       dtype=np.float32)   # (T,33,4)
            lh   = np.asarray(z["left_hand"],  dtype=np.float32)   # (T,21,3)
            rh   = np.asarray(z["right_hand"], dtype=np.float32)   # (T,21,3)
    except Exception:
        return None
    if pose.shape[0] == 0:
        return None

    # drop clips where pose is mostly missing
    valid = ~np.isnan(pose[:, 0, 0])
    if valid.sum() < 4:
        return None
    if not valid.all():
        idx = np.where(valid, np.arange(len(pose)), -1)
        idx = np.maximum.accumulate(idx)
        idx = np.where(idx < 0, np.argmax(valid), idx)
        pose = pose[idx]
        lh   = lh[idx]
        rh   = rh[idx]

    lh = np.nan_to_num(lh, nan=0.0)
    rh = np.nan_to_num(rh, nan=0.0)

    pose_n = normalise_pose(pose)                          # (T,25,4)
    lh_n   = hand_bone_vectors(lh, pose)                   # (T,21,3)
    rh_n   = hand_bone_vectors(rh, pose)                   # (T,21,3)
    vel    = velocity(pose_n[:, :, :2])                    # (T,25,2)

    feats = np.concatenate([
        pose_n.reshape(len(pose), POSE_DIM),
        lh_n.reshape(len(pose), HAND_DIM),
        rh_n.reshape(len(pose), HAND_DIM),
        vel.reshape(len(pose), VEL_DIM),
    ], axis=1)                                             # (T, 276)

    return resample(feats, SEQ_LEN).astype(np.float32)    # (32, 276)


# ── augmentation ─────────────────────────────────────────────────────────────

def augment(feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T = feats.shape[0]

    # time crop
    if rng.random() < 0.6 and T > 8:
        span = rng.integers(int(T * 0.65), T + 1)
        start = rng.integers(0, T - span + 1)
        feats = resample(feats[start:start + span], T)
    else:
        feats = feats.copy()

    # coordinate jitter
    feats = feats + rng.normal(0, 0.01, size=feats.shape).astype(np.float32)

    # speed jitter: randomly stretch/compress by ±20%
    if rng.random() < 0.4:
        factor = rng.uniform(0.8, 1.2)
        new_len = max(4, int(T * factor))
        feats = resample(feats, new_len)
        feats = resample(feats, T)

    return feats


# ── dataset ───────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    def __init__(self, items, label_to_idx, train=False):
        self.label_to_idx = label_to_idx
        self.train = train
        self.rng = np.random.default_rng(42)
        self.cache: list = []
        for it in items:
            npz = LM_DIR / (Path(it["filename"]).stem + ".npz")
            feats = load_clip(npz) if npz.exists() else None
            self.cache.append(
                (feats, label_to_idx[it["word"]]) if feats is not None else None
            )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        item = self.cache[i]
        if item is None:
            return None
        feats, label = item
        if self.train:
            feats = augment(feats, self.rng)
        return feats, label


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return (
        torch.from_numpy(np.stack([b[0] for b in batch])),
        torch.tensor([b[1] for b in batch], dtype=torch.long),
    )


# ── model ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):                             # x: (B, T, d_model)
        return self.drop(x + self.pe[:, :x.size(1)])


class SignTransformer(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(FEAT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_classes),
        )

    def forward(self, x):                            # (B, T, FEAT_DIM)
        h = self.pos(self.proj(x))
        h = self.encoder(h)                          # (B, T, d_model)
        pooled = torch.cat([h.mean(1), h.amax(1)], dim=-1)   # mean + max pool
        return self.head(pooled)


# ── data loading ──────────────────────────────────────────────────────────────

def load_split(strategy, min_clips=10, words=None):
    rows = list(csv.DictReader(SPLITS.open()))
    col = f"split_{strategy}"
    if col not in rows[0]:
        raise SystemExit(f"split column not found: {col}")
    if words:
        keep = set(words)
    elif min_clips > 0:
        from collections import Counter
        counts = Counter(r["word"] for r in rows)
        keep = {w for w, c in counts.items() if c >= min_clips}
        print(f"words with >= {min_clips} clips: {len(keep)}  ({sum(counts[w] for w in keep)} clips)")
    else:
        keep = None
    if keep:
        rows = [r for r in rows if r["word"] in keep]
    train, val, test = [], [], []
    for r in rows:
        {"train": train, "val": val, "test": test}.get(r[col], []).append(r)
    labels = sorted({r["word"] for r in rows})
    return train, val, test, {w: i for i, w in enumerate(labels)}


# ── evaluation + confusion matrix ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, n_classes=None, return_preds=False):
    model.eval()
    correct = total = loss_sum = 0
    all_true, all_pred = [], []
    crit = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        if batch is None:
            continue
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        preds = logits.argmax(-1)
        correct += (preds == y).sum().item()
        total += y.numel()
        if return_preds:
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    if total == 0:
        return float("nan"), float("nan"), [], []
    acc = correct / total
    if return_preds:
        return loss_sum / total, acc, all_true, all_pred
    return loss_sum / total, acc


def print_confusion(true, pred, idx_to_label, top_n=15):
    from collections import defaultdict
    n = len(idx_to_label)
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(true, pred):
        mat[t][p] += 1

    print("\n── Confusion matrix (top misclassifications) ──")
    errors = []
    for t in range(n):
        for p in range(n):
            if t != p and mat[t][p] > 0:
                errors.append((mat[t][p], idx_to_label[t], idx_to_label[p]))
    errors.sort(reverse=True)
    print(f"{'count':>5}  {'true':<16}  {'predicted':<16}")
    print("-" * 40)
    for count, true_w, pred_w in errors[:top_n]:
        print(f"{count:>5}  {true_w:<16}  {pred_w:<16}")

    print("\n── Per-class accuracy ──")
    per_class = []
    for t in range(n):
        total_t = sum(mat[t])
        if total_t == 0:
            continue
        acc_t = mat[t][t] / total_t
        per_class.append((acc_t, idx_to_label[t], mat[t][t], total_t))
    per_class.sort()
    print(f"{'word':<16}  {'acc':>6}  correct/total")
    print("-" * 40)
    for acc_t, w, c, tot in per_class:
        bar = "█" * int(acc_t * 20)
        print(f"{w:<16}  {acc_t:>6.1%}  {c:>4}/{tot:<4}  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",     choices=["random", "by_provider"], default="random")
    ap.add_argument("--epochs",    type=int,   default=80)
    ap.add_argument("--batch",     type=int,   default=32)
    ap.add_argument("--lr",        type=float, default=5e-4)
    ap.add_argument("--d-model",   type=int,   default=128)
    ap.add_argument("--layers",    type=int,   default=3)
    ap.add_argument("--dropout",   type=float, default=0.3)
    ap.add_argument("--seed",      type=int,   default=0)
    ap.add_argument("--min-clips", type=int,   default=10,
                    help="only train words with >= N total clips (default 10 -> 24 words)")
    ap.add_argument("--words",     type=str,   default="",
                    help="comma-separated explicit word list")
    ap.add_argument("--save",      type=str,   default="",
                    help="save best-val checkpoint to this path")
    ap.add_argument("--no-confusion", action="store_true",
                    help="skip confusion matrix at the end")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  split=split_{args.split}  feat_dim={FEAT_DIM}")

    word_list = [w.strip() for w in args.words.split(",") if w.strip()] or None
    train_rows, val_rows, test_rows, label_to_idx = load_split(
        args.split, args.min_clips, word_list
    )
    idx_to_label = {i: w for w, i in label_to_idx.items()}
    n_classes = len(label_to_idx)
    print(f"classes={n_classes}  train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")
    print(f"random baseline: {1/n_classes:.1%}")

    print("loading landmarks...")
    train_ds  = ClipDataset(train_rows, label_to_idx, train=True)
    val_ds    = ClipDataset(val_rows,   label_to_idx, train=False)
    test_ds   = ClipDataset(test_rows,  label_to_idx, train=False)

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        miss = sum(1 for x in ds.cache if x is None)
        print(f"  {name}: {len(ds)-miss} usable, {miss} missing")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,                collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,                collate_fn=collate)
    train_eval   = DataLoader(ClipDataset(train_rows, label_to_idx, train=False),
                              batch_size=args.batch, collate_fn=collate)

    model = SignTransformer(n_classes, args.d_model, num_layers=args.layers,
                            dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # warmup for 10 epochs then cosine decay
    warmup = 10
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup),
        ],
        milestones=[warmup],
    )
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val  = -1.0
    best_test = -1.0
    best_state = None
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = n = 0
        for batch in train_loader:
            if batch is None:
                continue
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
        sched.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val  = val_acc
            _, best_test = evaluate(model, test_loader, device)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if args.save:
                torch.save({
                    "model_state": best_state,
                    "labels": [idx_to_label[i] for i in range(n_classes)],
                    "d_model": args.d_model,
                    "feat_dim": FEAT_DIM,
                    "seq_len": SEQ_LEN,
                }, args.save)

        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            _, train_acc = evaluate(model, train_eval, device)
            print(f"ep {ep:3d}  train={train_acc:.3f}  val={val_acc:.3f}  "
                  f"best_val={best_val:.3f}  test@best={best_test:.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    print(f"\nelapsed={(time.time()-t0)/60:.1f}m")
    print(f"final: best_val={best_val:.3f}  test_at_best_val={best_test:.3f}")

    # ── confusion matrix on test set (using best checkpoint) ─────────────────
    if not args.no_confusion and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        _, test_acc, true_labels, pred_labels = evaluate(
            model, test_loader, device, n_classes=n_classes, return_preds=True
        )
        print(f"\nTest accuracy (best checkpoint): {test_acc:.1%}")
        print_confusion(true_labels, pred_labels, idx_to_label)


if __name__ == "__main__":
    main()
