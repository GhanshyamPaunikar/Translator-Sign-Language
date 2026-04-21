"""Real-time word-level ASL inference from webcam.

Captures 30 frames with MediaPipe Holistic in a sliding window, runs the
trained SignTransformer, and overlays top-3 predictions on the video feed.

Usage:
    python scripts/realtime_demo.py --model sign_transformer.pt

Controls
--------
  [q]     quit
  [space] force a prediction from the current buffer
  [c]     clear the frame buffer

Tips
----
- Sit at roughly arm's length from the webcam with even lighting.
- Sign naturally; the model uses a 30-frame window (~1-1.5s at 20-30 FPS).
"""
from __future__ import annotations

import argparse
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import SignTransformer
from landmarks import extract_frame_landmarks
import mediapipe as mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sign_transformer.pt")
    ap.add_argument("--window", type=int, default=30, help="Frames per prediction")
    ap.add_argument("--stride", type=int, default=5, help="Predict every N frames")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_gloss = SignTransformer.from_checkpoint(args.model, map_location=device)
    model.to(device).eval()
    print(f"Loaded model on {device} with {len(idx_to_gloss)} classes")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    buffer: deque = deque(maxlen=args.window)
    last_preds: list[tuple[str, float]] = []
    frame_count = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural signing UX
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Draw landmarks for user feedback
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            buffer.append(extract_frame_landmarks(results))
            frame_count += 1

            # Run inference every `stride` frames once the buffer is full
            if len(buffer) == args.window and frame_count % args.stride == 0:
                seq = np.stack(list(buffer))
                x = torch.from_numpy(seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                top3 = probs.argsort()[-3:][::-1]
                last_preds = [(idx_to_gloss[int(i)], float(probs[int(i)])) for i in top3]

            # Overlay UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 30 + 30 * max(len(last_preds), 1)), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, f"Buffer: {len(buffer)}/{args.window}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, (gloss, p) in enumerate(last_preds):
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(frame, f"{gloss}  {p*100:.1f}%",
                            (20, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("ASL Word Translator — press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                buffer.clear(); last_preds = []
            elif key == ord(" ") and len(buffer) == args.window:
                frame_count = 0  # force immediate prediction next loop

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
