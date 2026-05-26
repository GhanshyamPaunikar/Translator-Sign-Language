#!/usr/bin/env python3
"""Sign -> Text, improved trainer. Adds mixup, heavier augmentation, optional
upper-body-only features, and a per-class / confusion-matrix report.

Same dataset, same Bi-LSTM. Direct A/B against train_recognizer.py.

Usage:
    python3 train_improved.py --split by_provider --min-clips 8 --upper-body
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
LM_DIR = DATA / "landmarks"
SPLITS = DATA / "splits.csv"

SEQ_LEN = 32
N_POSE = 33
POSE_DIM_FULL = N_POSE * 4
HAND_DIM = 21 * 3
UPPER_BODY_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
POSE_DIM_UPPER = len(UPPER_BODY_IDX) * 4


def normalise_pose(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    ls = pose[:, 11, :2]
    rs = pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (pose[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (pose[:, :, 1] - mid[:, 1:2]) / width
    return out


def normalise_hands(hands: np.ndarray, pose: np.ndarray) -> np.ndarray:
    out = hands.copy()
    ls = pose[:, 11, :2]
    rs = pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (hands[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (hands[:, :, 1] - mid[:, 1:2]) / width
    return out


def resample(seq: np.ndarray, n: int) -> np.ndarray:
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


def load_clip(npz_path: Path, upper_body: bool) -> np.ndarray | None:
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
    lh_n = normalise_hands(np.nan_to_num(lh, nan=0.0), pose)
    rh_n = normalise_hands(np.nan_to_num(rh, nan=0.0), pose)
    lh_n = np.where(np.isnan(lh)[..., :1], 0.0, lh_n)
    rh_n = np.where(np.isnan(rh)[..., :1], 0.0, rh_n)

    if upper_body:
        pose_part = pose_n[:, UPPER_BODY_IDX, :].reshape(-1, POSE_DIM_UPPER)
    else:
        pose_part = pose_n.reshape(-1, POSE_DIM_FULL)
    feats = np.concatenate(
        [pose_part, lh_n.reshape(-1, HAND_DIM), rh_n.reshape(-1, HAND_DIM)], axis=1,
    )
    feats = resample(feats, SEQ_LEN)
    return feats.astype(np.float32)


def augment(feats: np.ndarray, pose_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Time speed-warp, small rotation, small scale, mild jitter, joint dropout."""
    out = feats.copy()
    T = out.shape[0]

    # Speed warp: resample a [0.7, 1.0]-length window back to T frames.
    if rng.random() < 0.7 and T > 8:
        span = int(rng.integers(int(T * 0.7), T + 1))
        start = int(rng.integers(0, T - span + 1))
        out = resample(out[start:start + span], T)

    # Small rotation around origin (shoulders are already at origin after norm).
    if rng.random() < 0.5:
        theta = float(rng.uniform(-0.18, 0.18))  # ~+/-10 deg
        c, s = np.cos(theta), np.sin(theta)
        out = _rotate_xy(out, pose_dim, c, s)

    # Small isotropic scale.
    if rng.random() < 0.5:
        scale = float(rng.uniform(0.9, 1.1))
        out = _scale_xy(out, pose_dim, scale)

    # Joint dropout: zero out ~10% of joints across whole clip.
    if rng.random() < 0.5:
        out = _joint_dropout(out, pose_dim, p=0.1, rng=rng)

    # Coordinate jitter.
    out = out + rng.normal(0, 0.01, size=out.shape).astype(np.float32)
    return out


def _split_streams(feats: np.ndarray, pose_dim: int):
    pose = feats[:, :pose_dim]
    lh = feats[:, pose_dim:pose_dim + HAND_DIM]
    rh = feats[:, pose_dim + HAND_DIM:]
    return pose, lh, rh


def _rotate_xy(feats: np.ndarray, pose_dim: int, c: float, s: float) -> np.ndarray:
    pose, lh, rh = _split_streams(feats, pose_dim)
    pose = pose.reshape(feats.shape[0], -1, 4).copy()
    x, y = pose[..., 0].copy(), pose[..., 1].copy()
    pose[..., 0] = c * x - s * y
    pose[..., 1] = s * x + c * y
    for h in (lh, rh):
        H = h.reshape(feats.shape[0], -1, 3)
        x, y = H[..., 0].copy(), H[..., 1].copy()
        H[..., 0] = c * x - s * y
        H[..., 1] = s * x + c * y
    return np.concatenate([pose.reshape(feats.shape[0], -1), lh, rh], axis=1)


def _scale_xy(feats: np.ndarray, pose_dim: int, scale: float) -> np.ndarray:
    out = feats.copy()
    pose = out[:, :pose_dim].reshape(out.shape[0], -1, 4)
    pose[..., 0] *= scale
    pose[..., 1] *= scale
    for off in (pose_dim, pose_dim + HAND_DIM):
        H = out[:, off:off + HAND_DIM].reshape(out.shape[0], -1, 3)
        H[..., 0] *= scale
        H[..., 1] *= scale
    return out


def _joint_dropout(feats: np.ndarray, pose_dim: int, p: float, rng) -> np.ndarray:
    out = feats.copy()
    pose = out[:, :pose_dim].reshape(out.shape[0], -1, 4)
    n_pose = pose.shape[1]
    mask = rng.random(n_pose) < p
    pose[:, mask, :] = 0.0
    for off in (pose_dim, pose_dim + HAND_DIM):
        H = out[:, off:off + HAND_DIM].reshape(out.shape[0], -1, 3)
        m = rng.random(H.shape[1]) < p
        H[:, m, :] = 0.0
    return out


class ClipDataset(Dataset):
    def __init__(self, items, label_to_idx, upper_body, train=False):
        self.items = items
        self.label_to_idx = label_to_idx
        self.upper_body = upper_body
        self.pose_dim = POSE_DIM_UPPER if upper_body else POSE_DIM_FULL
        self.train = train
        self.rng = np.random.default_rng(0)
        self.cache = []
        for it in items:
            npz = LM_DIR / (Path(it["filename"]).stem + ".npz")
            feats = load_clip(npz, upper_body) if npz.exists() else None
            self.cache.append(None if feats is None else (feats, label_to_idx[it["word"]]))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        item = self.cache[i]
        if item is None:
            return None
        feats, label = item
        if self.train:
            feats = augment(feats, self.pose_dim, self.rng)
        return feats, label


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs = torch.from_numpy(np.stack([b[0] for b in batch]))
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


class BiLSTMClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes, hidden=96, n_layers=2, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden)
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


def load_split(strategy, min_clips=0, words=None):
    rows = list(csv.DictReader(SPLITS.open()))
    col = f"split_{strategy}"
    if col not in rows[0]:
        raise SystemExit(f"split column not found: {col}")
    if words:
        keep = set(words)
        rows = [r for r in rows if r["word"] in keep]
    elif min_clips > 0:
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
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    preds_all, true_all = [], []
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        preds_all.append(pred.cpu().numpy())
        true_all.append(y.cpu().numpy())
    if total == 0:
        return float("nan"), float("nan"), np.array([]), np.array([])
    preds = np.concatenate(preds_all)
    trues = np.concatenate(true_all)
    return loss_sum / total, correct / total, preds, trues


def mixup_batch(x, y, n_classes, alpha, rng):
    if alpha <= 0:
        return x, nn.functional.one_hot(y, n_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_oh = nn.functional.one_hot(y, n_classes).float()
    y_mix = lam * y_oh + (1 - lam) * y_oh[perm]
    return x_mix, y_mix


def soft_ce(logits, target_dist, label_smoothing=0.1):
    n = logits.size(-1)
    log_p = nn.functional.log_softmax(logits, dim=-1)
    target = target_dist * (1 - label_smoothing) + label_smoothing / n
    return -(target * log_p).sum(-1).mean()


def bootstrap_ci(correct_mask, iters=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct_mask)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    arr = correct_mask.astype(np.float32)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        means.append(arr[idx].mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def write_report(out_path, labels, preds, trues, split_name):
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    per_class_acc = np.zeros(n_classes)
    per_class_n = cm.sum(axis=1)
    for i in range(n_classes):
        per_class_acc[i] = cm[i, i] / max(per_class_n[i], 1)

    # Macro F1
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    correct_mask = (preds == trues)
    lo, hi = bootstrap_ci(correct_mask)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# Report: {split_name}\n")
        f.write(f"n_samples = {len(trues)}\n")
        f.write(f"n_classes = {n_classes}\n")
        f.write(f"top1_acc  = {correct_mask.mean():.4f}\n")
        f.write(f"top1_95ci = [{lo:.4f}, {hi:.4f}]\n")
        f.write(f"macro_f1  = {macro_f1:.4f}\n\n")
        f.write("# Per-class (sorted worst to best)\n")
        order = np.argsort(per_class_acc)
        f.write(f"{'word':<20} {'acc':>6} {'n':>4}\n")
        for i in order:
            f.write(f"{labels[i]:<20} {per_class_acc[i]:>6.2f} {per_class_n[i]:>4}\n")
        f.write("\n# Top confusions (true -> pred, count)\n")
        confusions = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], labels[i], labels[j]))
        confusions.sort(reverse=True)
        for cnt, t, p in confusions[:25]:
            f.write(f"{t:>16}  ->  {p:<16}  x{cnt}\n")

    cm_path = out_path.with_suffix(".cm.csv")
    with cm_path.open("w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(labels[i] + "," + ",".join(str(int(v)) for v in row) + "\n")

    return {
        "top1": float(correct_mask.mean()),
        "top1_95ci": [lo, hi],
        "macro_f1": macro_f1,
        "report_path": str(out_path),
        "confusion_csv": str(cm_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "by_provider"], default="by_provider")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-clips", type=int, default=8)
    ap.add_argument("--words", type=str, default="")
    ap.add_argument("--upper-body", action="store_true", default=True)
    ap.add_argument("--no-upper-body", dest="upper_body", action="store_false")
    ap.add_argument("--mixup", type=float, default=0.2,
                    help="mixup alpha (0 = off)")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--report", type=str, default="reports/improved.txt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pose_dim = POSE_DIM_UPPER if args.upper_body else POSE_DIM_FULL
    feat_dim = pose_dim + 2 * HAND_DIM
    print(f"device={device}  split=split_{args.split}  upper_body={args.upper_body}  "
          f"feat_dim={feat_dim}  mixup={args.mixup}")

    word_list = [w.strip() for w in args.words.split(",") if w.strip()] or None
    train_rows, val_rows, test_rows, label_to_idx = load_split(
        args.split, args.min_clips, word_list
    )
    print(f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}  "
          f"classes={len(label_to_idx)}")

    print("loading landmarks...")
    train_ds = ClipDataset(train_rows, label_to_idx, args.upper_body, train=True)
    val_ds = ClipDataset(val_rows, label_to_idx, args.upper_body, train=False)
    test_ds = ClipDataset(test_rows, label_to_idx, args.upper_body, train=False)
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        n_miss = sum(1 for x in ds.cache if x is None)
        print(f"  {name}: {len(ds) - n_miss} usable, {n_miss} missing")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch, collate_fn=collate)
    train_eval_loader = DataLoader(
        ClipDataset(train_rows, label_to_idx, args.upper_body, train=False),
        batch_size=args.batch, collate_fn=collate,
    )

    n_classes = len(label_to_idx)
    model = BiLSTMClassifier(feat_dim, n_classes, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng_mix = np.random.default_rng(args.seed + 1)

    best_val = -1.0
    best_test = -1.0
    best_state = None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = n = 0
        for batch in train_loader:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            x_mix, y_dist = mixup_batch(x, y, n_classes, args.mixup, rng_mix)
            opt.zero_grad()
            logits = model(x_mix)
            loss = soft_ce(logits, y_dist, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        train_loss = loss_sum / max(n, 1)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            _, best_test, _, _ = evaluate(model, test_loader, device)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            _, train_acc, _, _ = evaluate(model, train_eval_loader, device)
            print(f"ep {ep:3d}  tr_loss={train_loss:.3f}  tr_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  "
                  f"best_val={best_val:.3f}  test@best={best_test:.3f}")

    print(f"\nelapsed={(time.time()-t0)/60:.1f}m")
    print(f"final: best_val={best_val:.3f}  test@best_val={best_test:.3f}")

    # Restore best, write reports.
    if best_state is not None:
        model.load_state_dict(best_state)
    labels = [w for w, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    _, _, val_preds, val_trues = evaluate(model, val_loader, device)
    _, _, test_preds, test_trues = evaluate(model, test_loader, device)
    val_stats = write_report(args.report.replace(".txt", ".val.txt"),
                             labels, val_preds, val_trues, "val")
    test_stats = write_report(args.report.replace(".txt", ".test.txt"),
                              labels, test_preds, test_trues, "test")
    print(f"\nval  top1={val_stats['top1']:.3f}  95%CI={val_stats['top1_95ci']}  "
          f"macro_f1={val_stats['macro_f1']:.3f}")
    print(f"test top1={test_stats['top1']:.3f}  95%CI={test_stats['top1_95ci']}  "
          f"macro_f1={test_stats['macro_f1']:.3f}")
    print(f"reports: {val_stats['report_path']}  {test_stats['report_path']}")

    if args.save and best_state is not None:
        torch.save(
            {
                "model_state": best_state,
                "labels": labels,
                "hidden": args.hidden,
                "feat_dim": feat_dim,
                "seq_len": SEQ_LEN,
                "upper_body": args.upper_body,
            },
            args.save,
        )
        print(f"saved checkpoint to {args.save}")

    summary = {
        "split": args.split,
        "min_clips": args.min_clips,
        "n_classes": n_classes,
        "best_val": best_val,
        "test_at_best_val": best_test,
        "val": val_stats,
        "test": test_stats,
        "config": {
            "upper_body": args.upper_body,
            "mixup": args.mixup,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
        },
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/improved_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
