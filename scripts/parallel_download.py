#!/usr/bin/env python3
"""Parallel segmented downloader using byte-range requests.

Downloads a file using multiple concurrent connections to maximize throughput.
Supports resume from partial downloads.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://huggingface.co/datasets/confit/vctk-full/resolve/main/vctk.zip"
# After redirect, the actual CDN URL
FOLLOW_REDIRECTS = True
NUM_SEGMENTS = 8
CHUNK_SIZE = 1024 * 1024  # 1 MB read buffer


def get_final_url_and_size(url: str) -> tuple[str, int]:
    """Follow redirects and get Content-Length."""
    req = Request(url, method="HEAD")
    resp = urlopen(req)
    return resp.url, int(resp.headers["Content-Length"])


def download_segment(
    url: str,
    output_path: Path,
    start: int,
    end: int,
    segment_id: int,
    progress: dict,
):
    """Download a byte range to a segment file."""
    seg_path = output_path.parent / f".vctk_seg_{segment_id}"
    existing = seg_path.stat().st_size if seg_path.exists() else 0

    if existing >= (end - start + 1):
        progress[segment_id] = end - start + 1
        return

    current_start = start + existing
    req = Request(url)
    req.add_header("Range", f"bytes={current_start}-{end}")

    try:
        resp = urlopen(req)
        with open(seg_path, "ab") as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                progress[segment_id] = (
                    progress.get(segment_id, existing) + len(chunk)
                )
    except Exception as e:
        print(f"\nSegment {segment_id} error: {e}", file=sys.stderr)


def merge_segments(output_path: Path, num_segments: int):
    """Merge segment files into the final output."""
    with open(output_path, "wb") as out:
        for i in range(num_segments):
            seg_path = output_path.parent / f".vctk_seg_{i}"
            with open(seg_path, "rb") as seg:
                while True:
                    chunk = seg.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
            seg_path.unlink()


def main():
    output_path = Path("data/raw/vctk.zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Getting file info from {URL}...")
    final_url, total_size = get_final_url_and_size(URL)
    print(f"Final URL: {final_url[:80]}...")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Using {NUM_SEGMENTS} parallel connections")

    # Calculate segment ranges
    segment_size = total_size // NUM_SEGMENTS
    segments = []
    for i in range(NUM_SEGMENTS):
        start = i * segment_size
        end = (i + 1) * segment_size - 1 if i < NUM_SEGMENTS - 1 else total_size - 1
        segments.append((start, end))

    # Start threads
    progress: dict[int, int] = {}
    threads = []
    for i, (start, end) in enumerate(segments):
        t = threading.Thread(
            target=download_segment,
            args=(final_url, output_path, start, end, i, progress),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Monitor progress
    start_time = time.time()
    prev_downloaded = 0
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            downloaded = sum(progress.values())
            elapsed = time.time() - start_time
            speed = downloaded / elapsed if elapsed > 0 else 0
            delta = downloaded - prev_downloaded
            instant_speed = delta / 2 if delta > 0 else 0
            pct = downloaded / total_size * 100
            eta = (total_size - downloaded) / speed if speed > 0 else 0
            print(
                f"\r  {downloaded/(1024**2):.0f}/{total_size/(1024**2):.0f} MB "
                f"({pct:.1f}%) "
                f"avg={speed/(1024**2):.1f} MB/s "
                f"now={instant_speed/(1024**2):.1f} MB/s "
                f"ETA={eta/60:.0f}min",
                end="",
                flush=True,
            )
            prev_downloaded = downloaded
    except KeyboardInterrupt:
        print("\nInterrupted. Segments saved for resume.")
        sys.exit(1)

    print()

    # Verify all segments complete
    for i, (start, end) in enumerate(segments):
        seg_path = output_path.parent / f".vctk_seg_{i}"
        expected = end - start + 1
        actual = seg_path.stat().st_size
        if actual != expected:
            print(f"ERROR: Segment {i} incomplete ({actual}/{expected})")
            sys.exit(1)

    # Merge
    print("Merging segments...")
    merge_segments(output_path, NUM_SEGMENTS)
    print(f"Done: {output_path} ({output_path.stat().st_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main()
