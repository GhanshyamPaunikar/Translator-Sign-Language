#!/usr/bin/env python3
"""
Text -> Sign lookup. Given a sentence, return one BSL clip per recognised
word. Used as the "generation" half of the bidirectional system: a baseline
that concatenates dictionary clips. A real generative model can replace this
later.

Usage:
    python3 text_to_sign.py "hello please thank you"
    python3 text_to_sign.py --copy out/ "good morning my name"

Output without --copy: prints the chosen clip path per token, one per line.
With --copy: writes them as 0001_<word>.mp4, 0002_<word>.mp4, ... into the
target directory so they're easy to feed into ffmpeg/concat or a player.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
META = DATA / "metadata.csv"

# Providers we trust most (clean studio recordings, single signer, neutral bg).
PROVIDER_PRIORITY = [
    "signstation", "deafway", "signmonkey", "gpnhs", "bslfirst",
    "corpusngt", "ExeterDeafAcademyVoiceOff", "signon", "winkball",
    "nf", "youtube", "deafsigns", "wolver", "ict", "cs",
]


def load_index() -> dict[str, list[dict]]:
    by_word: dict[str, list[dict]] = defaultdict(list)
    for r in csv.DictReader(META.open()):
        by_word[r["word"].lower()].append(r)
    return by_word


def pick_best(rows: list[dict]) -> dict:
    """Pick the canonical clip: highest-priority provider, then variation 1."""
    rank = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}

    def key(r):
        return (rank.get(r["provider"], 999), int(r["variation"]))

    return min(rows, key=key)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def tokenise(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def lookup(tokens: list[str], by_word: dict[str, list[dict]]) -> list[tuple[str, dict | None]]:
    out: list[tuple[str, dict | None]] = []
    for tok in tokens:
        # Exact match first, then a few simple morphology fallbacks.
        candidates = [tok]
        if tok.endswith("s") and len(tok) > 2:
            candidates.append(tok[:-1])
        if tok.endswith("ing") and len(tok) > 4:
            candidates.append(tok[:-3])
        if tok.endswith("ed") and len(tok) > 3:
            candidates.append(tok[:-2])
        # Hyphenated multi-word forms (e.g. "thank-you")
        candidates.append(tok.replace(" ", "-"))
        chosen = None
        for c in candidates:
            if c in by_word:
                chosen = pick_best(by_word[c])
                break
        out.append((tok, chosen))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="English text to translate")
    ap.add_argument("--copy", metavar="DIR", help="copy chosen clips into DIR in order")
    args = ap.parse_args()

    by_word = load_index()
    # also expose hyphenated forms ("thank-you") as a single phrase token
    phrase_keys = [w for w in by_word if "-" in w]

    text = args.text.lower()
    # try to match longest hyphenated phrases first by replacing space-separated
    # forms with the hyphenated form when both are present in the dictionary
    for phrase in sorted(phrase_keys, key=len, reverse=True):
        spaced = phrase.replace("-", " ")
        text = re.sub(rf"\b{re.escape(spaced)}\b", phrase, text)
    tokens = tokenise(text)
    matches = lookup(tokens, by_word)

    out_dir = Path(args.copy) if args.copy else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    found = missing = 0
    for i, (tok, row) in enumerate(matches, 1):
        if row is None:
            print(f"  -- {tok}  (no sign in dictionary)")
            missing += 1
            continue
        src = DATA / row["filename"]
        print(f"  {tok:20s} -> {row['filename']}  [{row['provider']}]")
        found += 1
        if out_dir:
            dst = out_dir / f"{i:04d}_{tok}.mp4"
            shutil.copyfile(src, dst)

    print(f"\n{found} found, {missing} missing.")
    if out_dir:
        print(f"Copied to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
