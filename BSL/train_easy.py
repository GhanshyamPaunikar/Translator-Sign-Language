#!/usr/bin/env python3
"""
Train a small-vocab BSL recogniser the right way for tiny data:

  1. 5-fold cross-validation to get an honest accuracy estimate
     (val/test single-clip-per-class is too noisy to be informative).
  2. Train the final demo model on ALL clips of the curated vocabulary
     and save it to models/recognizer_easy.pt.

Default vocabulary is a hand-picked set of visually distinct BSL signs:
different body zones touched, different motion shapes, no confusable pairs.

Usage:
    python3 train_easy.py
    python3 train_easy.py --words hello,phone,toilet,home --epochs 80
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
LM_DIR = DATA / "landmarks"
META = DATA / "metadata.csv"

SEQ_LEN = 32
POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FEAT_DIM = POSE_DIM + 2 * HAND_DIM

DEFAULT_WORDS = [
    "hello", "phone", "toilet", "home", "family", "think", "see",
    "drink", "cold", "help", "child",
]


# --- preprocessing (same as train_recognizer.py) ---------------------------
def normalise_pose(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (pose[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (pose[:, :, 1] - mid[:, 1:2]) / width
    return out


def normalise_hands(hands: np.ndarray, pose: np.ndarray) -> np.ndarray:
    out = hands.copy()
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (hands[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (hands[:, :, 1] - mid[:, 1:2]) / width
    return out


def resample(seq: np.ndarray, n: int) -> np.ndarray:
    t = seq.shape[0]
    if t == n:
        return seq
    if t == 1:
        return np.broadcast_to(seq, (n,) + seq.shape[1:]).copy()
    src_idx = np.linspace(0, t - 1, n, dtype=np.float32)
    floor = np.floor(src_idx).astype(int)
    ceil = np.minimum(floor + 1, t - 1)
    frac = (src_idx - floor).reshape((n,) + (1,) * (seq.ndim - 1))
    return (seq[floor] * (1 - frac) + seq[ceil] * frac).astype(seq.dtype, copy=False)


def load_clip(npz_path: Path) -> np.ndarray | None:
    try:
        with np.load(npz_path) as z:
            pose = np.asarray(z["pose"], dtype=np.float32)
            lh = np.asarray(z["left_hand"], dtype=np.float32)
            rh = np.asarray(z["right_hand"], dtype=np.float32)
    except Exception:
        return None
    if pose.shape[0] == 0:
        return None
    pose_valid = ~np.isnan(pose[:, 0, 0])
    if pose_valid.sum() < 4:
        return None
    if not pose_valid.all():
        idx = np.where(pose_valid, np.arange(len(pose)), -1)
        idx = np.maximum.accumulate(idx)
        idx = np.where(idx < 0, np.argmax(pose_valid), idx)
        pose = pose[idx]
    pose_n = normalise_pose(pose)
    lh_mask = np.isnan(lh)[..., :1]
    rh_mask = np.isnan(rh)[..., :1]
    lh_n = normalise_hands(np.nan_to_num(lh, nan=0.0), pose)
    rh_n = normalise_hands(np.nan_to_num(rh, nan=0.0), pose)
    lh_n = np.where(lh_mask, 0.0, lh_n)
    rh_n = np.where(rh_mask, 0.0, rh_n)
    feats = np.concatenate(
        [pose_n.reshape(-1, POSE_DIM), lh_n.reshape(-1, HAND_DIM), rh_n.reshape(-1, HAND_DIM)],
        axis=1,
    ).astype(np.float32)
    return resample(feats, SEQ_LEN)


# --- augmentation ----------------------------------------------------------
def augment(feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = feats
    T = out.shape[0]
    # time crop, more aggressive (50-100%)
    if rng.random() < 0.7 and T > 8:
        span = rng.integers(int(T * 0.5), T + 1)
        start = rng.integers(0, T - span + 1)
        out = resample(out[start : start + span], T)
    else:
        out = out.copy()
    # frame dropout — zero ~10% random frames
    if rng.random() < 0.5:
        n_drop = max(1, int(T * 0.1))
        drop_idx = rng.choice(T, size=n_drop, replace=False)
        out[drop_idx] = 0.0
    # coordinate jitter
    out = out + rng.normal(0, 0.015, size=out.shape).astype(np.float32)
    return out


class ClipDataset(Dataset):
    def __init__(self, items: list[dict], label_to_idx: dict[str, int], train: bool):
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


# --- model ------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, hidden)
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=n_layers, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, n_classes),
        )

    def forward(self, x):
        h = torch.relu(self.proj(x))
        out, _ = self.lstm(h)
        pooled = torch.cat([out.mean(1), out.amax(1)], dim=-1)
        return self.head(pooled)


# --- training utilities ----------------------------------------------------
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one(train_rows, val_rows, label_to_idx, device, epochs, batch, lr, hidden, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = ClipDataset(train_rows, label_to_idx, train=True)
    val_ds = ClipDataset(val_rows, label_to_idx, train=False) if val_rows else None
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch, collate_fn=collate) if val_ds else None
    model = BiLSTMClassifier(n_classes=len(label_to_idx), hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val = -1.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for batch_ in train_loader:
            if batch_ is None:
                continue
            x, y = batch_
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if val_loader is not None:
            v = eval_acc(model, val_loader, device)
            if v > best_val:
                best_val = v
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def kfold_split(items_by_word: dict[str, list], k: int, seed: int = 0):
    """Yield (train_rows, val_rows) for k folds, stratified by word."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for word, items in items_by_word.items():
        idx = list(range(len(items)))
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(items[i])
    for f in range(k):
        val = folds[f]
        train = [r for j, fold in enumerate(folds) if j != f for r in fold]
        yield train, val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default=",".join(DEFAULT_WORDS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save", default=str(ROOT / "models" / "recognizer_easy.pt"))
    args = ap.parse_args()

    word_list = [w.strip() for w in args.words.split(",") if w.strip()]
    rows = [r for r in csv.DictReader(META.open()) if r["word"] in word_list]
    if not rows:
        print("no clips matched the chosen words.", file=sys.stderr)
        return 1
    by_word: dict[str, list] = {}
    for r in rows:
        by_word.setdefault(r["word"], []).append(r)
    label_to_idx = {w: i for i, w in enumerate(sorted(by_word))}
    print(f"vocab ({len(label_to_idx)}): {list(label_to_idx)}")
    print(f"clips per word: " + ", ".join(f"{w}={len(v)}" for w, v in sorted(by_word.items())))
    print(f"total clips: {len(rows)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    print(f"\n--- {args.folds}-fold cross-validation ---")
    fold_accs = []
    fold_top3 = []
    t0 = time.time()
    for f, (train_rows, val_rows) in enumerate(kfold_split(by_word, args.folds), 1):
        model, _ = train_one(
            train_rows, val_rows, label_to_idx, device,
            args.epochs, args.batch, args.lr, args.hidden, seed=f,
        )
        # final-pass acc + top-3
        val_ds = ClipDataset(val_rows, label_to_idx, train=False)
        loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)
        correct = top3 = total = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                top3_pred = logits.topk(3, dim=-1).indices
                top3 += (top3_pred == y.unsqueeze(-1)).any(-1).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        t3 = top3 / max(total, 1)
        fold_accs.append(acc)
        fold_top3.append(t3)
        print(f"  fold {f}: val={len(val_rows):3d}  top1={acc:.3f}  top3={t3:.3f}")
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_top3 = float(np.mean(fold_top3))
    print(f"\nCV top1: {mean_acc:.3f} ± {std_acc:.3f}   "
          f"top3: {mean_top3:.3f}   ({(time.time()-t0)/60:.1f}m)")

    print(f"\n--- final model: training on ALL {len(rows)} clips ---")
    final_model, _ = train_one(
        rows, [], label_to_idx, device,
        args.epochs, args.batch, args.lr, args.hidden, seed=0,
    )
    Path(args.save).parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "labels": [w for w, _ in sorted(label_to_idx.items(), key=lambda x: x[1])],
            "hidden": args.hidden,
            "n_layers": 1,
            "feat_dim": FEAT_DIM,
            "seq_len": SEQ_LEN,
            "cv_top1": mean_acc,
            "cv_top3": mean_top3,
            "vocab": sorted(by_word),
        },
        args.save,
    )
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
