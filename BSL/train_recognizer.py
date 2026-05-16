#!/usr/bin/env python3
"""
Sign -> Text baseline. Bi-LSTM over MediaPipe pose+hand landmarks.

Reads bsl_dataset/landmarks/*.npz and bsl_dataset/splits.csv. Reports top-1
accuracy on val and test for the chosen split strategy.

Usage:
    python3 train_recognizer.py                      # split_random
    python3 train_recognizer.py --split by_provider  # held-out signers
    python3 train_recognizer.py --epochs 60 --lr 5e-4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
LM_DIR = DATA / "landmarks"
SPLITS = DATA / "splits.csv"

SEQ_LEN = 32           # resample every clip to this many frames
POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FEAT_DIM = POSE_DIM + 2 * HAND_DIM  # 132 + 63 + 63 = 258


def normalise_pose(pose: np.ndarray) -> np.ndarray:
    """Centre on midpoint of shoulders, scale by shoulder width.
    pose: (T, 33, 4). Returns (T, 33, 4) with x,y normalised; z and visibility
    untouched.
    """
    out = pose.copy()
    L_SHOULDER, R_SHOULDER = 11, 12
    # per-frame shoulder midpoint and width
    ls = pose[:, L_SHOULDER, :2]
    rs = pose[:, R_SHOULDER, :2]
    mid = (ls + rs) / 2.0  # (T, 2)
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)  # (T, 1)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (pose[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (pose[:, :, 1] - mid[:, 1:2]) / width
    return out


def normalise_hands(hands: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Same centring/scaling, applied to hand landmarks."""
    out = hands.copy()
    L_SHOULDER, R_SHOULDER = 11, 12
    ls = pose[:, L_SHOULDER, :2]
    rs = pose[:, R_SHOULDER, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (hands[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (hands[:, :, 1] - mid[:, 1:2]) / width
    return out


def resample(seq: np.ndarray, n: int) -> np.ndarray:
    """Linear-interpolate seq (T, ...) to length n along time."""
    t_src = seq.shape[0]
    if t_src == n:
        return seq
    if t_src == 1:
        return np.broadcast_to(seq, (n,) + seq.shape[1:]).copy()
    src_idx = np.linspace(0, t_src - 1, n, dtype=np.float32)
    floor = np.floor(src_idx).astype(int)
    ceil = np.minimum(floor + 1, t_src - 1)
    frac = (src_idx - floor).reshape((n,) + (1,) * (seq.ndim - 1))
    return (seq[floor] * (1 - frac) + seq[ceil] * frac).astype(seq.dtype, copy=False)


def load_clip(npz_path: Path) -> np.ndarray | None:
    """Return (SEQ_LEN, FEAT_DIM) float32 feature tensor for one clip."""
    try:
        with np.load(npz_path) as z:
            pose = np.asarray(z["pose"], dtype=np.float32)
            lh = np.asarray(z["left_hand"], dtype=np.float32)
            rh = np.asarray(z["right_hand"], dtype=np.float32)
    except Exception:
        return None
    if pose.shape[0] == 0:
        return None
    # Forward-fill NaN frames in pose (any frame where pose missing)
    # — pose is rarely missing; if it is, drop the whole clip.
    pose_valid = ~np.isnan(pose[:, 0, 0])
    if pose_valid.sum() < 4:
        return None
    if not pose_valid.all():
        idx = np.where(pose_valid, np.arange(len(pose)), -1)
        idx = np.maximum.accumulate(idx)
        idx = np.where(idx < 0, np.argmax(pose_valid), idx)
        pose = pose[idx]
    pose_n = normalise_pose(pose)
    lh_n = normalise_hands(np.nan_to_num(lh, nan=0.0), pose)
    rh_n = normalise_hands(np.nan_to_num(rh, nan=0.0), pose)
    # zero rows where hand was missing (post nan_to_num it's already 0)
    lh_n = np.where(np.isnan(lh)[..., :1], 0.0, lh_n)
    rh_n = np.where(np.isnan(rh)[..., :1], 0.0, rh_n)

    feats = np.concatenate(
        [pose_n.reshape(-1, POSE_DIM), lh_n.reshape(-1, HAND_DIM), rh_n.reshape(-1, HAND_DIM)],
        axis=1,
    )
    feats = resample(feats, SEQ_LEN)
    return feats.astype(np.float32)


def augment(feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-time augmentation. We deliberately do NOT mirror-flip — many BSL
    signs are one-handed or asymmetric, so flipping creates invalid examples.
    """
    out = feats
    T = out.shape[0]

    # Time crop (50%): take a contiguous [0.7, 1.0] window then resample back.
    if rng.random() < 0.5 and T > 8:
        span = rng.integers(int(T * 0.7), T + 1)
        start = rng.integers(0, T - span + 1)
        out = resample(out[start : start + span], T)
    else:
        out = out.copy()

    # Coordinate jitter
    out = out + rng.normal(0, 0.01, size=out.shape).astype(np.float32)
    return out


class ClipDataset(Dataset):
    def __init__(self, items: list[dict], label_to_idx: dict[str, int], train: bool = False):
        self.items = items
        self.label_to_idx = label_to_idx
        self.train = train
        self.rng = np.random.default_rng(0)
        self.cache: list[tuple[np.ndarray, int] | None] = []
        for it in items:
            npz = LM_DIR / (Path(it["filename"]).stem + ".npz")
            feats = load_clip(npz) if npz.exists() else None
            if feats is None:
                self.cache.append(None)
            else:
                self.cache.append((feats, label_to_idx[it["word"]]))

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
    xs = torch.from_numpy(np.stack([b[0] for b in batch]))
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 96, n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, hidden)
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=n_layers, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0, batch_first=True,
        )
        # Bi-LSTM out is 2*hidden; we pool with mean + max -> 4*hidden into the head.
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, n_classes),
        )

    def forward(self, x):  # x: (B, T, FEAT_DIM)
        h = torch.relu(self.proj(x))
        out, _ = self.lstm(h)
        pooled = torch.cat([out.mean(1), out.amax(1)], dim=-1)
        return self.head(pooled)


def load_split(
    strategy: str, min_clips: int = 0, words: list[str] | None = None
) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    rows = list(csv.DictReader(SPLITS.open()))
    col = f"split_{strategy}"
    if col not in rows[0]:
        raise SystemExit(f"split column not found: {col}")
    if words:
        keep = set(words)
        rows = [r for r in rows if r["word"] in keep]
        print(f"filtered to {len(keep)} words: {sorted(keep)}  ({len(rows)} clips)")
    elif min_clips > 0:
        from collections import Counter
        counts = Counter(r["word"] for r in rows)
        keep = {w for w, c in counts.items() if c >= min_clips}
        rows = [r for r in rows if r["word"] in keep]
        print(f"filtered to words with >= {min_clips} clips: "
              f"{len(keep)} words, {len(rows)} clips")
    train, val, test = [], [], []
    for r in rows:
        bucket = {"train": train, "val": val, "test": test}.get(r[col])
        if bucket is None:
            continue
        bucket.append(r)
    labels = sorted({r["word"] for r in rows})
    label_to_idx = {w: i for i, w in enumerate(labels)}
    return train, val, test, label_to_idx


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    if total == 0:
        return float("nan"), float("nan")
    return loss_sum / total, correct / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "by_provider"], default="random")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-clips", type=int, default=0,
                    help="only train on words with >= N total clips")
    ap.add_argument("--words", type=str, default="",
                    help="comma-separated explicit word list (overrides --min-clips)")
    ap.add_argument("--save", type=str, default="",
                    help="save best-val checkpoint to this path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  split=split_{args.split}")

    word_list = [w.strip() for w in args.words.split(",") if w.strip()] if args.words else None
    train_rows, val_rows, test_rows, label_to_idx = load_split(
        args.split, args.min_clips, word_list
    )
    print(f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}  "
          f"classes={len(label_to_idx)}")

    print("loading landmarks...")
    train_ds = ClipDataset(train_rows, label_to_idx, train=True)
    val_ds = ClipDataset(val_rows, label_to_idx, train=False)
    test_ds = ClipDataset(test_rows, label_to_idx, train=False)

    def report_missing(name, ds):
        n_miss = sum(1 for x in ds.cache if x is None)
        print(f"  {name}: {len(ds) - n_miss} usable, {n_miss} missing landmarks")

    report_missing("train", train_ds)
    report_missing("val", val_ds)
    report_missing("test", test_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch, collate_fn=collate)

    model = BiLSTMClassifier(n_classes=len(label_to_idx), hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = -1.0
    best_test = -1.0
    t0 = time.time()
    train_eval_loader = DataLoader(
        ClipDataset(train_rows, label_to_idx, train=False),
        batch_size=args.batch, collate_fn=collate,
    )
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        train_loss = loss_sum / max(n, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            _, best_test = evaluate(model, test_loader, device)
            if args.save:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "labels": [w for w, _ in sorted(label_to_idx.items(), key=lambda x: x[1])],
                        "hidden": args.hidden,
                        "feat_dim": FEAT_DIM,
                        "seq_len": SEQ_LEN,
                    },
                    args.save,
                )
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            _, train_acc = evaluate(model, train_eval_loader, device)
            print(
                f"ep {ep:3d}  train_loss={train_loss:.3f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  "
                f"best_val={best_val:.3f}  test@best_val={best_test:.3f}"
            )

    print(f"\nelapsed={(time.time()-t0)/60:.1f}m")
    print(f"final: best_val={best_val:.3f}  test_at_best_val={best_test:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
