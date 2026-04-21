"""MediaPipe Holistic landmark extraction for ASL videos.

The full 1662-dim per-frame vector is:
    pose:       33 landmarks × 4 (x, y, z, visibility)  = 132
    face:       468 landmarks × 3 (x, y, z)             = 1404
    left_hand:  21 landmarks × 3                        = 63
    right_hand: 21 landmarks × 3                        = 63
                                                 total = 1662
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

LANDMARK_DIM = 1662
POSE_DIM = 33 * 4
FACE_DIM = 468 * 3
HAND_DIM = 21 * 3

_mp_holistic = mp.solutions.holistic


def extract_frame_landmarks(results) -> np.ndarray:
    """Flatten a MediaPipe Holistic result into a 1662-dim vector."""
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(POSE_DIM)
    )
    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(FACE_DIM)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(HAND_DIM)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(HAND_DIM)
    )
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)


def extract_video_landmarks(
    video_path: str | Path,
    target_frames: int = 30,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Optional[np.ndarray]:
    """Process a video into a (target_frames, 1662) landmark array.

    Frames are uniformly sampled from the video's full length. Missing detections
    (e.g., hands out of frame) produce zero-filled slots.

    Returns None if the video can't be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, target_frames, dtype=int)
    sequence = []

    with _mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                sequence.append(np.zeros(LANDMARK_DIM, dtype=np.float32))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            sequence.append(extract_frame_landmarks(results))

    cap.release()
    return np.stack(sequence)


def normalize_landmarks(seq: np.ndarray) -> np.ndarray:
    """Optional: center pose on torso and scale by shoulder width.

    This dramatically improves signer-invariance. Only applies to pose/hands;
    face is left untouched since it's already normalized by MediaPipe.
    """
    seq = seq.copy()
    # Pose is first 132 dims, organized as 33 × (x,y,z,vis)
    pose = seq[:, :POSE_DIM].reshape(-1, 33, 4)
    # Landmark 11 = left shoulder, 12 = right shoulder, 0 = nose
    left_sh = pose[:, 11, :3]
    right_sh = pose[:, 12, :3]
    center = (left_sh + right_sh) / 2
    scale = np.linalg.norm(left_sh - right_sh, axis=-1, keepdims=True) + 1e-6

    # Normalize pose xyz
    pose[:, :, :3] = (pose[:, :, :3] - center[:, None, :]) / scale[:, None, :]
    seq[:, :POSE_DIM] = pose.reshape(-1, POSE_DIM)

    # Same for hands (both hands share the same torso frame)
    for offset in (POSE_DIM + FACE_DIM, POSE_DIM + FACE_DIM + HAND_DIM):
        hand = seq[:, offset : offset + HAND_DIM].reshape(-1, 21, 3)
        hand = (hand - center[:, None, :]) / scale[:, None, :]
        seq[:, offset : offset + HAND_DIM] = hand.reshape(-1, HAND_DIM)

    return seq
