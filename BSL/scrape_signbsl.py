#!/usr/bin/env python3
"""
Download BSL sign videos from signbsl.com with proper word-based labels.

For each word in words.txt, fetches https://www.signbsl.com/sign/<word>,
extracts every <source src="..."> video (variations from different providers),
and saves them as <word>.mp4, <word>_var2.mp4, <word>_var3.mp4, ...

A metadata.csv is produced mapping filename -> word, variation, provider, source URL.
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).parent
WORDS_FILE = ROOT / "words.txt"
OUT_DIR = ROOT / "bsl_dataset"
META_FILE = OUT_DIR / "metadata.csv"
BASE = "https://www.signbsl.com/sign/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BSL-research-scraper)"}
DELAY = 0.8  # seconds between page fetches
TIMEOUT = 30

SOURCE_RE = re.compile(
    r'<source\s+src="(https?://media\.signbsl\.com/[^"]+\.mp4)"', re.IGNORECASE
)


def load_words() -> list[str]:
    words = []
    for line in WORDS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line.lower())
    return words


def provider_from_url(url: str) -> str:
    # e.g. https://media.signbsl.com/videos/bsl/signstation/hello.mp4 -> signstation
    parts = urlparse(url).path.strip("/").split("/")
    # videos / bsl / <provider> / [mp4/] file.mp4
    if len(parts) >= 3 and parts[0] == "videos":
        return parts[2]
    return "unknown"


def fetch_sources(session: requests.Session, word: str) -> list[str]:
    url = BASE + word
    r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    seen, ordered = set(), []
    for m in SOURCE_RE.findall(r.text):
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def download(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        with session.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dest)
        return True
    except Exception as e:
        print(f"    ! download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    words = load_words()
    print(f"Loaded {len(words)} words")

    session = requests.Session()
    rows: list[dict] = []
    no_videos: list[str] = []

    for i, word in enumerate(words, 1):
        print(f"[{i}/{len(words)}] {word}")
        try:
            sources = fetch_sources(session, word)
        except Exception as e:
            print(f"  ! page fetch failed: {e}")
            sources = []

        if not sources:
            print("  - no videos found")
            no_videos.append(word)
            time.sleep(DELAY)
            continue

        print(f"  + {len(sources)} variation(s)")
        for idx, src in enumerate(sources, 1):
            provider = provider_from_url(src)
            name = word if idx == 1 else f"{word}_var{idx}"
            filename = f"{name}__{provider}.mp4"
            dest = OUT_DIR / filename
            ok = download(session, src, dest)
            print(f"    {'ok' if ok else 'FAIL'}  {filename}")
            if ok:
                rows.append(
                    {
                        "filename": filename,
                        "word": word,
                        "variation": idx,
                        "provider": provider,
                        "source_url": src,
                    }
                )
            time.sleep(0.2)
        time.sleep(DELAY)

    with META_FILE.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["filename", "word", "variation", "provider", "source_url"]
        )
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"Done. {len(rows)} videos saved across {len({r['word'] for r in rows})} words.")
    print(f"Metadata: {META_FILE}")
    if no_videos:
        print(f"No videos for {len(no_videos)} words: {', '.join(no_videos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
