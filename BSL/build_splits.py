#!/usr/bin/env python3
"""
Build train / val / test splits for the BSL dataset.

Two split strategies are produced:

  random       — random 80/10/10 over all clips (within-distribution baseline)
  by_provider  — held-out signers/providers go to val and test (tests
                 generalisation to unseen signers, the realistic scenario)

For both strategies, every word must appear in the train set; otherwise the
recogniser cannot learn it. Words that lack enough variations to satisfy this
fall back to "best effort" (all available variations go to train, none to
val/test for that word). This is logged.

Output: bsl_dataset/splits.csv with columns
    filename, word, provider, split_random, split_by_provider
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "bsl_dataset"
META = DATA / "metadata.csv"
OUT = DATA / "splits.csv"

SEED = 7
VAL_FRAC = 0.1
TEST_FRAC = 0.1
# Providers that go entirely into val/test in by_provider strategy. Picked from
# the long tail so the train set keeps the bulk of the data and most words.
HELDOUT_VAL_PROVIDERS = {"deafway", "winkball"}
HELDOUT_TEST_PROVIDERS = {"signon", "deafsigns"}


def main() -> int:
    rows = list(csv.DictReader(META.open()))
    random.seed(SEED)
    random.shuffle(rows)

    # --- random split (per-word stratified) ---
    by_word: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_word[r["word"]].append(r)

    split_random: dict[str, str] = {}
    for word, items in by_word.items():
        items = list(items)
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * TEST_FRAC))) if n >= 3 else 0
        n_val = max(1, int(round(n * VAL_FRAC))) if n >= 4 else 0
        # ensure at least 1 train sample per word
        if n - n_test - n_val < 1:
            n_test = max(0, n_test - 1)
            if n - n_test - n_val < 1:
                n_val = 0
        for i, r in enumerate(items):
            if i < n_test:
                split_random[r["filename"]] = "test"
            elif i < n_test + n_val:
                split_random[r["filename"]] = "val"
            else:
                split_random[r["filename"]] = "train"

    # --- by_provider split ---
    split_provider: dict[str, str] = {}
    words_in_train: set[str] = set()
    # First pass: assign by provider rule
    for r in rows:
        if r["provider"] in HELDOUT_TEST_PROVIDERS:
            split_provider[r["filename"]] = "test"
        elif r["provider"] in HELDOUT_VAL_PROVIDERS:
            split_provider[r["filename"]] = "val"
        else:
            split_provider[r["filename"]] = "train"
            words_in_train.add(r["word"])

    # Repair: any word that has no train sample steals one from val/test
    orphan_words = set(by_word) - words_in_train
    for word in orphan_words:
        # Promote one variation to train (prefer val over test so test stays clean)
        items = by_word[word]
        promoted = False
        for r in items:
            if split_provider[r["filename"]] == "val":
                split_provider[r["filename"]] = "train"
                promoted = True
                break
        if not promoted:
            for r in items:
                if split_provider[r["filename"]] == "test":
                    split_provider[r["filename"]] = "train"
                    break

    # --- write ---
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["filename", "word", "provider", "split_random", "split_by_provider"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "filename": r["filename"],
                    "word": r["word"],
                    "provider": r["provider"],
                    "split_random": split_random[r["filename"]],
                    "split_by_provider": split_provider[r["filename"]],
                }
            )

    # --- report ---
    def summary(split_map: dict[str, str], label: str):
        counts = defaultdict(int)
        words = defaultdict(set)
        for fn, sp in split_map.items():
            counts[sp] += 1
            for r in rows:
                if r["filename"] == fn:
                    words[sp].add(r["word"])
                    break
        print(f"\n{label}:")
        for sp in ("train", "val", "test"):
            print(f"  {sp:5s} clips={counts[sp]:4d}  words={len(words[sp]):3d}")
        if orphan_words and label.startswith("by_provider"):
            print(f"  promoted-to-train (no train coverage otherwise): {len(orphan_words)} words")

    summary(split_random, "random 80/10/10")
    summary(split_provider, "by_provider (val=deafway+winkball, test=signon+deafsigns)")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
