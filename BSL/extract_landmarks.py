#!/usr/bin/env python3
"""
Extract MediaPipe pose + hand landmarks for every video in bsl_dataset/.
Caches one .npz per clip in bsl_dataset/landmarks/.

Uses PoseLandmarker + HandLandmarker separately (Tasks API). We skip face
mesh — for isolated-word sign recognition the manual features (hands +
upper-body pose) carry the signal. Add face later if needed.

Each .npz contains:
    pose       float32 (T, 33, 4)   x, y, z, visibility   (NaN if not detected)
    left_hand  float32 (T, 21, 3)   x, y, z              (NaN if not detected)
    right_hand float32 (T, 21, 3)   x, y, z              (NaN if not detected)
    fps        float32 scalar
    width      int32   scalar
    height     int32   scalar

Coordinates are in MediaPipe's normalised image space ([0, 1] for x/y).

Resumable: an existing .npz with > 0 frames is skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
LM_DIR = DATA / "landmarks"
POSE_MODEL = ROOT / "models" / "pose_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
LM_DIR.mkdir(exist_ok=True)

RESIZE_LONG = 480


def init_landmarkers():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = vision.PoseLandmarker.create_from_options(pose_opts)
    hand = vision.HandLandmarker.create_from_options(hand_opts)
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


def split_hands(hand_result) -> tuple[np.ndarray, np.ndarray]:
    """Return (left, right) each (21, 3); NaN if missing."""
    left = np.full((21, 3), np.nan, dtype=np.float32)
    right = np.full((21, 3), np.nan, dtype=np.float32)
    if not hand_result or not hand_result.hand_landmarks:
        return left, right
    for lms, hd in zip(hand_result.hand_landmarks, hand_result.handedness):
        # handedness is from camera POV; "Left" in result == signer's right hand
        # if the video is mirrored. We trust MediaPipe's label here — downstream
        # consumers can decide on convention.
        label = hd[0].category_name  # "Left" or "Right"
        arr = lm_to_array(lms, 21, with_vis=False)
        if label == "Left":
            left = arr
        else:
            right = arr
    return left, right


def process_video(path: Path, pose, hand, mp_module) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_frames, lh_frames, rh_frames = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        long_edge = max(frame.shape[:2])
        if long_edge > RESIZE_LONG:
            scale = RESIZE_LONG / long_edge
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)

        pose_res = pose.detect(mp_image)
        hand_res = hand.detect(mp_image)

        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        pose_frames.append(lm_to_array(pose_lms, 33, with_vis=True))
        left, right = split_hands(hand_res)
        lh_frames.append(left)
        rh_frames.append(right)
    cap.release()

    if not pose_frames:
        return None
    return {
        "pose": np.stack(pose_frames),
        "left_hand": np.stack(lh_frames),
        "right_hand": np.stack(rh_frames),
        "fps": np.float32(fps),
        "width": np.int32(w),
        "height": np.int32(h),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N (debug)")
    ap.add_argument("--reextract", action="store_true", help="redo even if .npz exists")
    args = ap.parse_args()

    for m in (POSE_MODEL, HAND_MODEL):
        if not m.exists():
            print(f"Model not found: {m}", file=sys.stderr)
            return 1

    videos = sorted(DATA.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"Found {len(videos)} videos. Cache dir: {LM_DIR}")

    pose, hand, mp_module = init_landmarkers()
    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    total_frames = 0

    try:
        for i, vid in enumerate(videos, 1):
            out = LM_DIR / (vid.stem + ".npz")
            if out.exists() and not args.reextract:
                try:
                    with np.load(out) as z:
                        if z["pose"].shape[0] > 0:
                            n_skipped += 1
                            continue
                except Exception:
                    pass

            ts = time.time()
            try:
                data = process_video(vid, pose, hand, mp_module)
            except Exception as e:
                print(f"  ! {vid.name}: {e}")
                n_failed += 1
                continue
            if data is None:
                print(f"  ! {vid.name}: no frames")
                n_failed += 1
                continue
            np.savez_compressed(out, **data)
            n_done += 1
            total_frames += data["pose"].shape[0]
            elapsed = time.time() - ts
            if i % 25 == 0 or i == len(videos):
                avg = (time.time() - t0) / max(n_done, 1)
                remaining = len(videos) - i
                print(
                    f"[{i}/{len(videos)}] {vid.name[:46]:46s} "
                    f"{data['pose'].shape[0]:3d}f {elapsed:4.1f}s  "
                    f"avg {avg:4.1f}s/clip  eta {remaining*avg/60:.1f}m"
                )
    finally:
        pose.close()
        hand.close()

    print()
    print(
        f"done={n_done}  skipped={n_skipped}  failed={n_failed}  "
        f"frames={total_frames}  elapsed={(time.time()-t0)/60:.1f}m"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
