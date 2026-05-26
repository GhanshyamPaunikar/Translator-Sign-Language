#!/usr/bin/env python3
"""Browser-facing BSL recogniser demo.

Serves static/index.html and accepts POST /predict with a JSON body
{"frames": [base64-jpeg, ...]}. Returns top-K word predictions.

Usage:
    python3 serve_demo.py --model models/recognizer_improved.pt --port 8000

Then open http://localhost:8000 in a browser. Allow camera access, press
"Record" and sign for ~2.5s. The page calls /predict and shows top-5.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
POSE_MODEL = ROOT / "models" / "pose_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"

SEQ_LEN = 32
N_POSE = 33
POSE_DIM_FULL = N_POSE * 4
HAND_DIM = 21 * 3
UPPER_BODY_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
POSE_DIM_UPPER = len(UPPER_BODY_IDX) * 4

TOP_K = 5


# ---------- model + feature pipeline (mirrors train_improved.py) -------------
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


def normalise_pose(pose):
    out = pose.copy()
    ls = pose[:, 11, :2]
    rs = pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (pose[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (pose[:, :, 1] - mid[:, 1:2]) / width
    return out


def normalise_hands(hands, pose):
    out = hands.copy()
    ls = pose[:, 11, :2]
    rs = pose[:, 12, :2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    width = np.where(width < 1e-3, 1.0, width)
    out[:, :, 0] = (hands[:, :, 0] - mid[:, 0:1]) / width
    out[:, :, 1] = (hands[:, :, 1] - mid[:, 1:2]) / width
    return out


def resample(seq, n):
    t = seq.shape[0]
    if t == n:
        return seq
    if t == 1:
        return np.broadcast_to(seq, (n,) + seq.shape[1:]).copy()
    src = np.linspace(0, t - 1, n, dtype=np.float32)
    floor = np.floor(src).astype(int)
    ceil = np.minimum(floor + 1, t - 1)
    frac = (src - floor).reshape((n,) + (1,) * (seq.ndim - 1))
    return (seq[floor] * (1 - frac) + seq[ceil] * frac).astype(seq.dtype, copy=False)


def build_features(pose_frames, lh_frames, rh_frames, upper_body):
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
    if upper_body:
        pose_part = pose_n[:, UPPER_BODY_IDX, :].reshape(-1, POSE_DIM_UPPER)
    else:
        pose_part = pose_n.reshape(-1, POSE_DIM_FULL)
    feats = np.concatenate(
        [pose_part, lh_n.reshape(-1, HAND_DIM), rh_n.reshape(-1, HAND_DIM)], axis=1,
    ).astype(np.float32)
    return resample(feats, SEQ_LEN)


def lm_to_array(lms, n_pts, with_vis):
    cols = 4 if with_vis else 3
    if not lms:
        return np.full((n_pts, cols), np.nan, dtype=np.float32)
    if with_vis:
        return np.array(
            [[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in lms],
            dtype=np.float32,
        )
    return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)


def split_hands(hr):
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


# ---------- landmarkers (created once at startup) ----------------------------
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


# ---------- HTTP handler -----------------------------------------------------
STATE = {}  # populated in main(): model, labels, pose_l, hand_l, mp, upper_body


def predict_from_frames(frames_b64):
    pose_l = STATE["pose_l"]
    hand_l = STATE["hand_l"]
    mp = STATE["mp"]
    upper_body = STATE["upper_body"]
    pose_frames, lh_frames, rh_frames = [], [], []
    for b64 in frames_b64:
        raw = base64.b64decode(b64.split(",", 1)[-1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_res = pose_l.detect(mp_image)
        hand_res = hand_l.detect(mp_image)
        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        pose_frames.append(lm_to_array(pose_lms, 33, with_vis=True))
        lh, rh = split_hands(hand_res)
        lh_frames.append(lh)
        rh_frames.append(rh)
    if not pose_frames:
        return {"error": "no frames"}
    feats = build_features(pose_frames, lh_frames, rh_frames, upper_body)
    if feats is None:
        return {"error": "no pose detected — try framing yourself front-on"}
    model = STATE["model"]
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0)
        probs = torch.softmax(model(x), -1)[0].cpu().numpy()
    labels = STATE["labels"]
    idx = np.argsort(-probs)[:TOP_K]
    return {
        "predictions": [{"word": labels[i], "prob": float(probs[i])} for i in idx],
        "n_frames": len(pose_frames),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

    def _send(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/vocab":
            self._send(200, "application/json",
                       json.dumps({"labels": STATE["labels"]}).encode())
            return
        if path == "/":
            path = "/index.html"
        f = STATIC / path.lstrip("/")
        if not f.exists() or not f.is_file():
            self._send(404, "text/plain", b"not found")
            return
        ct = {".html": "text/html", ".js": "application/javascript",
              ".css": "text/css"}.get(f.suffix, "application/octet-stream")
        self._send(200, ct, f.read_bytes())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            frames = data.get("frames", [])
            if not isinstance(frames, list) or not frames:
                self._send(400, "application/json", json.dumps({"error": "no frames"}).encode())
                return
            result = predict_from_frames(frames)
            self._send(200, "application/json", json.dumps(result).encode())
        except Exception as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "recognizer_improved.pt"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    feat_dim = ckpt.get("feat_dim", POSE_DIM_FULL + 2 * HAND_DIM)
    upper_body = ckpt.get("upper_body", False)
    model = BiLSTMClassifier(feat_dim, len(labels), hidden=ckpt.get("hidden", 96))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pose_l, hand_l, mp = init_landmarkers()
    STATE.update(model=model, labels=labels, pose_l=pose_l, hand_l=hand_l,
                 mp=mp, upper_body=upper_body)

    print(f"loaded {args.model}: {len(labels)} classes, feat_dim={feat_dim}, "
          f"upper_body={upper_body}")
    print(f"open http://{args.host}:{args.port} in your browser")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()
