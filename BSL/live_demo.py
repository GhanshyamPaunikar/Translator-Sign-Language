#!/usr/bin/env python3
"""
Live BSL recogniser — webcam in, top-K predictions overlaid.

Press SPACE to record a clip (~2.5 seconds), then the model classifies it.
Press 'q' to quit.  Press 'r' to reset between recordings.

Uses model_v2.pt (Transformer, 29.6% test accuracy on 24 words) by default.
Override with --model.

Tips for best results:
  - Frame yourself front-on, head + shoulders + both hands visible.
  - Good lighting, neutral background.
  - Match the timing of the training clips (most are 2-4 seconds).
  - Training data was studio-quality; webcam will score slightly lower.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT       = Path(__file__).parent
POSE_MODEL = ROOT / "models" / "pose_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"

SEQ_LEN        = 32
RECORD_SECONDS = 2.5
TOP_K          = 5

# v2 feature constants (must match train_v2.py)
UPPER_JOINTS = list(range(25))
POSE_DIM_V2  = len(UPPER_JOINTS) * 4   # 100
HAND_DIM_V2  = 21 * 3                  # 63
VEL_DIM_V2   = len(UPPER_JOINTS) * 2   # 50
FEAT_DIM_V2  = POSE_DIM_V2 + 2 * HAND_DIM_V2 + VEL_DIM_V2   # 276

# v1 feature constants (legacy)
FEAT_DIM_V1 = 33 * 4 + 2 * 21 * 3     # 258


# ── models ────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class SignTransformer(nn.Module):
    def __init__(self, n_classes, d_model=128, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(FEAT_DIM_V2, d_model), nn.LayerNorm(d_model), nn.ReLU(),
        )
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2), nn.Dropout(dropout), nn.Linear(d_model * 2, n_classes),
        )

    def forward(self, x):
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))


class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes, hidden=96, n_layers=2, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM_V1, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden), nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes),
        )

    def forward(self, x):
        h = torch.relu(self.proj(x))
        out, _ = self.lstm(h)
        return self.head(torch.cat([out.mean(1), out.amax(1)], dim=-1))


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    feat_dim = ckpt.get("feat_dim", FEAT_DIM_V1)
    if feat_dim == FEAT_DIM_V2:
        model = SignTransformer(len(labels), d_model=ckpt.get("d_model", 128))
        version = "v2 (Transformer)"
    else:
        model = BiLSTMClassifier(len(labels), hidden=ckpt.get("hidden", 96))
        version = "v1 (Bi-LSTM)"
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {version}  classes={len(labels)}  path={path}")
    return model, labels, feat_dim


# ── feature extraction ────────────────────────────────────────────────────────

def _shoulder_norm(pose):
    ls = pose[:, 11, :2]
    rs = pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    w = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    return mid, np.where(w < 1e-3, 1.0, w)


def resample(seq, n):
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


def build_features_v2(pose_frames, lh_frames, rh_frames) -> np.ndarray | None:
    pose = np.stack(pose_frames).astype(np.float32)
    lh   = np.nan_to_num(np.stack(lh_frames).astype(np.float32), nan=0.0)
    rh   = np.nan_to_num(np.stack(rh_frames).astype(np.float32), nan=0.0)

    valid = ~np.isnan(pose[:, 0, 0])
    if valid.sum() < 4:
        return None
    if not valid.all():
        idx = np.where(valid, np.arange(len(pose)), -1)
        idx = np.maximum.accumulate(idx)
        idx = np.where(idx < 0, np.argmax(valid), idx)
        pose = pose[idx]; lh = lh[idx]; rh = rh[idx]

    mid, w = _shoulder_norm(pose)
    pose_n = pose[:, UPPER_JOINTS, :].copy()
    pose_n[:, :, 0] = (pose_n[:, :, 0] - mid[:, 0:1]) / w
    pose_n[:, :, 1] = (pose_n[:, :, 1] - mid[:, 1:2]) / w

    def bones(hand):
        return (hand - hand[:, 0:1, :]) / w[:, :, None]

    lh_n = bones(lh)
    rh_n = bones(rh)

    vel = np.zeros_like(pose_n[:, :, :2])
    vel[1:] = pose_n[1:, :, :2] - pose_n[:-1, :, :2]

    T = len(pose)
    feats = np.concatenate([
        pose_n.reshape(T, POSE_DIM_V2),
        lh_n.reshape(T, HAND_DIM_V2),
        rh_n.reshape(T, HAND_DIM_V2),
        vel.reshape(T, VEL_DIM_V2),
    ], axis=1)
    return resample(feats, SEQ_LEN).astype(np.float32)


def build_features_v1(pose_frames, lh_frames, rh_frames) -> np.ndarray | None:
    pose = np.stack(pose_frames).astype(np.float32)
    lh   = np.stack(lh_frames).astype(np.float32)
    rh   = np.stack(rh_frames).astype(np.float32)

    valid = ~np.isnan(pose[:, 0, 0])
    if valid.sum() < 4:
        return None
    if not valid.all():
        idx = np.where(valid, np.arange(len(pose)), -1)
        idx = np.maximum.accumulate(idx)
        idx = np.where(idx < 0, np.argmax(valid), idx)
        pose = pose[idx]; lh = lh[idx]; rh = rh[idx]

    mid, w = _shoulder_norm(pose)
    pose_n = pose.copy()
    pose_n[:, :, 0] = (pose[:, :, 0] - mid[:, 0:1]) / w
    pose_n[:, :, 1] = (pose[:, :, 1] - mid[:, 1:2]) / w
    lh_n = np.nan_to_num(lh.copy(), nan=0.0)
    rh_n = np.nan_to_num(rh.copy(), nan=0.0)
    lh_n[:, :, 0] = (lh_n[:, :, 0] - mid[:, 0:1]) / w
    lh_n[:, :, 1] = (lh_n[:, :, 1] - mid[:, 1:2]) / w
    rh_n[:, :, 0] = (rh_n[:, :, 0] - mid[:, 0:1]) / w
    rh_n[:, :, 1] = (rh_n[:, :, 1] - mid[:, 1:2]) / w

    T = len(pose)
    feats = np.concatenate([
        pose_n.reshape(T, -1), lh_n.reshape(T, -1), rh_n.reshape(T, -1),
    ], axis=1)
    return resample(feats, SEQ_LEN).astype(np.float32)


# ── MediaPipe ─────────────────────────────────────────────────────────────────

def init_landmarkers():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
    )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
        )
    )
    return pose, hand, mp


def lm_to_array(lms, n_pts, with_vis):
    cols = 4 if with_vis else 3
    if not lms:
        return np.full((n_pts, cols), np.nan, dtype=np.float32)
    if with_vis:
        return np.array([[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in lms], dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)


def split_hands(hr):
    left  = np.full((21, 3), np.nan, dtype=np.float32)
    right = np.full((21, 3), np.nan, dtype=np.float32)
    if not hr or not hr.hand_landmarks:
        return left, right
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        arr = lm_to_array(lms, 21, with_vis=False)
        if hd[0].category_name == "Left":
            left = arr
        else:
            right = arr
    return left, right


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_skeleton(img, pose_lms, hand_res):
    h, w = img.shape[:2]
    if pose_lms:
        for p in pose_lms:
            cv2.circle(img, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)
    if hand_res and hand_res.hand_landmarks:
        for lms in hand_res.hand_landmarks:
            for p in lms:
                cv2.circle(img, (int(p.x * w), int(p.y * h)), 4, (0, 200, 255), -1)


def draw_ui(img, predictions, status, recording, progress):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.putText(img, status, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if recording:
        cv2.circle(img, (w - 28, 26), 11, (0, 0, 255), -1)
        bar_w = int(progress * (w - 20))
        cv2.rectangle(img, (10, 57), (10 + bar_w, 64), (0, 0, 255), -1)
        cv2.rectangle(img, (10, 57), (w - 10, 64), (200, 200, 200), 1)

    if predictions:
        ph = 32 + 28 * len(predictions)
        cv2.rectangle(img, (0, h - ph), (320, h), (20, 20, 20), -1)
        cv2.putText(img, "predictions:", (10, h - ph + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        for i, (word, prob) in enumerate(predictions):
            y = h - ph + 48 + i * 28
            cv2.rectangle(img, (110, y - 14), (110 + int(prob * 190), y - 3), (0, 210, 0), -1)
            cv2.putText(img, word,          (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(img, f"{prob:.2f}", (260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (180, 180, 180), 1)


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default=str(ROOT / "model_v2.pt"),
                    help="path to .pt checkpoint (v2 Transformer or v1 Bi-LSTM, auto-detected)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    model, labels, feat_dim = load_model(args.model)
    build_features = build_features_v2 if feat_dim == FEAT_DIM_V2 else build_features_v1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    pose_l, hand_l, mp_module = init_landmarkers()

    pose_frames: list = []
    lh_frames:   list = []
    rh_frames:   list = []
    recording    = False
    rec_start    = 0.0
    predictions: list[tuple[str, float]] = []
    status = "SPACE = record  |  r = reset  |  q = quit"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            pose_res = pose_l.detect(mp_img)
            hand_res = hand_l.detect(mp_img)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            draw_skeleton(frame, pose_lms, hand_res)

            progress = 0.0
            if recording:
                pose_frames.append(lm_to_array(pose_lms, 33, with_vis=True))
                lh, rh = split_hands(hand_res)
                lh_frames.append(lh)
                rh_frames.append(rh)

                elapsed  = time.time() - rec_start
                progress = min(1.0, elapsed / RECORD_SECONDS)

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    feats = build_features(pose_frames, lh_frames, rh_frames)
                    pose_frames, lh_frames, rh_frames = [], [], []
                    if feats is None:
                        status = "no pose detected — try again (SPACE)"
                        predictions = []
                    else:
                        with torch.no_grad():
                            logits = model(torch.from_numpy(feats).unsqueeze(0))
                            probs  = torch.softmax(logits, -1)[0].cpu().numpy()
                        top = np.argsort(-probs)[:TOP_K]
                        predictions = [(labels[i], float(probs[i])) for i in top]
                        status = f"  {predictions[0][0]}  ({predictions[0][1]:.0%})"

            draw_ui(frame, predictions, status, recording, progress)
            cv2.imshow("BSL recogniser — press SPACE", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                predictions = []
                status = "SPACE = record  |  r = reset  |  q = quit"
            elif key == ord(" ") and not recording:
                recording = True
                rec_start = time.time()
                pose_frames, lh_frames, rh_frames = [], [], []
                predictions = []
                status = "recording..."
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_l.close()
        hand_l.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
