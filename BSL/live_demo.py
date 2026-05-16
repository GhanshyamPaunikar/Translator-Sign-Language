#!/usr/bin/env python3
"""
Live BSL recogniser — webcam in, top-K predictions overlaid.

Press SPACE to record a clip (~2 seconds), then the model classifies it.
Press 'q' to quit. Press 'r' to reset between recordings.

Defaults to the 46-word model at models/recognizer_46w.pt. Override with
--model.

Caveats: the model was trained on signbsl.com studio clips (front-on, neutral
background, professional signers). Your webcam is a different distribution,
so accuracy will be worse than the offline test numbers. Frame yourself
front-on in good light, fit head + shoulders + hands in view, and try to
match the timing of the training clips (most are 2-4 seconds).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
POSE_MODEL = ROOT / "models" / "pose_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"

# Must match training-time constants
SEQ_LEN = 32
POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FEAT_DIM = POSE_DIM + 2 * HAND_DIM

RECORD_SECONDS = 2.5
TOP_K = 5


# ----- model (mirror of train_recognizer.BiLSTMClassifier) ------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 96, n_layers: int = 2, dropout: float = 0.5):
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


# ----- preprocessing (mirror of train_recognizer) ----------------------------
def normalise_pose(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    L_S, R_S = 11, 12
    ls = pose[:, L_S, :2]
    rs = pose[:, R_S, :2]
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


def build_features(pose_frames, lh_frames, rh_frames) -> np.ndarray | None:
    pose = np.stack(pose_frames).astype(np.float32)
    lh = np.stack(lh_frames).astype(np.float32)
    rh = np.stack(rh_frames).astype(np.float32)
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


# ----- MediaPipe ------------------------------------------------------------
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


def lm_to_array(lms, n_pts: int, with_vis: bool) -> np.ndarray:
    cols = 4 if with_vis else 3
    if not lms:
        return np.full((n_pts, cols), np.nan, dtype=np.float32)
    if with_vis:
        return np.array(
            [[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in lms],
            dtype=np.float32,
        )
    return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)


def split_hands(hr) -> tuple[np.ndarray, np.ndarray]:
    left = np.full((21, 3), np.nan, dtype=np.float32)
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


# ----- drawing helpers -------------------------------------------------------
def draw_pose(img, pose_lms):
    h, w = img.shape[:2]
    if not pose_lms:
        return
    for p in pose_lms:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)


def draw_hands(img, hand_result):
    if not hand_result or not hand_result.hand_landmarks:
        return
    h, w = img.shape[:2]
    for lms in hand_result.hand_landmarks:
        for p in lms:
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(img, (x, y), 3, (0, 200, 255), -1)


def draw_results(img, predictions, label_top: str, recording: bool, progress: float):
    h, w = img.shape[:2]
    # top banner
    cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(img, label_top, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # recording indicator
    if recording:
        cv2.circle(img, (w - 30, 25), 12, (0, 0, 255), -1)
        bar_w = int(progress * (w - 20))
        cv2.rectangle(img, (10, 56), (10 + bar_w, 64), (0, 0, 255), -1)
        cv2.rectangle(img, (10, 56), (w - 10, 64), (255, 255, 255), 1)

    # predictions panel
    if predictions:
        panel_h = 30 + 28 * len(predictions)
        cv2.rectangle(img, (0, h - panel_h), (320, h), (0, 0, 0), -1)
        cv2.putText(img, "predictions:", (12, h - panel_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for i, (word, prob) in enumerate(predictions):
            y = h - panel_h + 50 + i * 28
            bar = int(prob * 200)
            cv2.rectangle(img, (110, y - 14), (110 + bar, y - 2), (0, 220, 0), -1)
            cv2.putText(img, f"{word}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            cv2.putText(img, f"{prob:.2f}", (315 - 60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ----- main loop -------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "recognizer_46w.pt"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true",
                    help="mirror the camera (looks more natural for self-recording)")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    model = BiLSTMClassifier(
        n_classes=len(labels),
        hidden=ckpt["hidden"],
        n_layers=ckpt.get("n_layers", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded model: {args.model}  classes={len(labels)}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    pose_l, hand_l, mp_module = init_landmarkers()

    pose_frames: list = []
    lh_frames: list = []
    rh_frames: list = []
    recording = False
    rec_start = 0.0
    predictions: list[tuple[str, float]] = []
    label_top = "press SPACE to record  |  q: quit  |  r: reset"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            pose_res = pose_l.detect(mp_image)
            hand_res = hand_l.detect(mp_image)

            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
            draw_pose(frame, pose_lms)
            draw_hands(frame, hand_res)

            if recording:
                pose_frames.append(lm_to_array(pose_lms, 33, with_vis=True))
                lh, rh = split_hands(hand_res)
                lh_frames.append(lh)
                rh_frames.append(rh)
                elapsed = time.time() - rec_start
                progress = min(1.0, elapsed / RECORD_SECONDS)
                if elapsed >= RECORD_SECONDS:
                    recording = False
                    feats = build_features(pose_frames, lh_frames, rh_frames)
                    pose_frames, lh_frames, rh_frames = [], [], []
                    if feats is None:
                        label_top = "no pose detected — try again (SPACE)"
                        predictions = []
                    else:
                        with torch.no_grad():
                            x = torch.from_numpy(feats).unsqueeze(0)
                            logits = model(x)
                            probs = torch.softmax(logits, -1)[0].cpu().numpy()
                        idx = np.argsort(-probs)[:TOP_K]
                        predictions = [(labels[i], float(probs[i])) for i in idx]
                        label_top = f"top: {predictions[0][0]}  ({predictions[0][1]:.2f})"
            else:
                progress = 0.0

            draw_results(frame, predictions, label_top, recording, progress)
            cv2.imshow("BSL recogniser", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                predictions = []
                label_top = "press SPACE to record"
            elif key == ord(" ") and not recording:
                recording = True
                rec_start = time.time()
                pose_frames, lh_frames, rh_frames = [], [], []
                predictions = []
                label_top = "recording..."
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_l.close()
        hand_l.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
