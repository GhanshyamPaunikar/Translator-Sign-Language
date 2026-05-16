#!/usr/bin/env python3
"""
Probe every clip in bsl_dataset/ and rewrite metadata.csv with:
  filename, word, variation, provider, source_url,
  duration_s, fps, width, height, n_frames, ok

Also fixes provider parsing (paths can be /videos/bsl/<provider>/...,
/videos/bsl/<provider>/mp4/..., /videos/<provider>/mp4/..., etc.).
"""
from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse

import cv2

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
META = DATA / "metadata.csv"

SKIP_SEGMENTS = {"videos", "bsl", "mp4", "img", "videos.mp4"}


def derive_provider(url: str) -> str:
    """First non-skip path segment after /videos/. Falls back to 'unknown'."""
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    # parts ~ ['videos', 'bsl', 'signstation', 'hello.mp4']
    #       or ['videos', 'wolver', 'mp4', '3636_Sign.mp4']
    for seg in parts:
        if seg.lower().endswith(".mp4"):
            break
        if seg in SKIP_SEGMENTS:
            continue
        return seg
    return "unknown"


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "n_frames": 0, "fps": 0.0, "width": 0, "height": 0, "duration_s": 0.0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = n / fps if fps > 0 else 0.0
    ok = n > 0 and w > 0 and h > 0
    return {
        "ok": ok,
        "n_frames": n,
        "fps": round(fps, 3),
        "width": w,
        "height": h,
        "duration_s": round(duration, 3),
    }


def main() -> int:
    rows_in = list(csv.DictReader(META.open()))
    print(f"Read {len(rows_in)} metadata rows")

    rows_out = []
    bad = []
    for i, r in enumerate(rows_in, 1):
        path = DATA / r["filename"]
        if not path.exists():
            print(f"  ! missing file: {r['filename']}")
            continue
        provider = derive_provider(r["source_url"])
        info = probe(path)
        out = {
            "filename": r["filename"],
            "word": r["word"],
            "variation": r["variation"],
            "provider": provider,
            "source_url": r["source_url"],
            **info,
        }
        rows_out.append(out)
        if not info["ok"]:
            bad.append(r["filename"])
        if i % 100 == 0:
            print(f"  probed {i}/{len(rows_in)}")

    fieldnames = [
        "filename", "word", "variation", "provider", "source_url",
        "duration_s", "fps", "width", "height", "n_frames", "ok",
    ]
    with META.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    n = len(rows_out)
    n_ok = sum(1 for r in rows_out if r["ok"])
    durations = [r["duration_s"] for r in rows_out if r["ok"]]
    print()
    print(f"Total: {n}  OK: {n_ok}  Broken: {n - n_ok}")
    if durations:
        durations.sort()
        print(
            f"Duration s — min {durations[0]:.2f}  median "
            f"{durations[len(durations)//2]:.2f}  max {durations[-1]:.2f}  "
            f"sum {sum(durations):.0f}"
        )
    if bad:
        print(f"Broken files ({len(bad)}):")
        for b in bad[:20]:
            print(f"  {b}")
    providers = {}
    for r in rows_out:
        providers[r["provider"]] = providers.get(r["provider"], 0) + 1
    print("Providers:")
    for p, c in sorted(providers.items(), key=lambda x: -x[1]):
        print(f"  {p:30s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
