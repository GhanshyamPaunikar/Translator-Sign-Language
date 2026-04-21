"""ASL Alphabet Recognition. Author: Ghanshyam Paunikar"""
import argparse
from collections import deque, Counter
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_hand(lm):
    c = lm - lm[0]
    s = np.linalg.norm(c, axis=1).max() + 1e-6
    return (c / s).flatten().astype(np.float32)

def extract_hand(results):
    if not results.multi_hand_landmarks:
        return None
    h = results.multi_hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in h.landmark])
    return normalize_hand(pts)

TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

def finger_states(pts):
    st = []
    for t, p in zip(TIPS, PIPS):
        st.append(np.linalg.norm(pts[t] - pts[0]) > np.linalg.norm(pts[p] - pts[0]) * 1.1)
    return st

def classify(feat):
    pts = feat.reshape(21, 3)
    th, ix, mi, rg, pk = finger_states(pts)
    n = sum([ix, mi, rg, pk])
    if ix and mi and rg and pk and not th: return "B", 0.85
    if all([th, ix, mi, rg, pk]): return "5 / open", 0.80
    if n == 0: return "A / S", 0.75
    if ix and not mi and not rg and not pk: return "D / 1", 0.80
    if ix and mi and not rg and not pk:
        return ("V / 2", 0.80) if np.linalg.norm(pts[8]-pts[12]) > 0.3 else ("U", 0.75)
    if ix and mi and rg and not pk: return "W / 3", 0.80
    if pk and not ix and not mi and not rg:
        return ("Y", 0.80) if th else ("I", 0.80)
    if th and ix and not mi and not rg and not pk: return "L", 0.80
    if mi and rg and pk and not ix:
        if np.linalg.norm(pts[4]-pts[8]) < 0.2: return "F", 0.75
    return "?", 0.3

def run():
    cap = cv2.VideoCapture(0)
    recent = deque(maxlen=10)
    print("Press Q in the window to quit.")
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feat = extract_hand(res)
            if res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=2))
            if feat is not None:
                letter, conf = classify(feat)
                recent.append(letter)
            else:
                letter, conf = "no hand", 0.0
            smooth = Counter(recent).most_common(1)[0][0] if recent else letter
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w,110), (0,0,0), -1)
            cv2.putText(frame, f"Letter: {smooth}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,0), 3)
            cv2.putText(frame, f"Confidence: {conf*100:.0f}%  Press Q to quit", (20,95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.imshow("ASL Alphabet - Ghanshyam Paunikar", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
