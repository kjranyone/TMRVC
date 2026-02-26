#!/usr/bin/env python3
"""Download and extract VCTK Corpus 0.92 (CC BY 4.0) with resume support.

Source: University of Edinburgh, CSTR
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
URL: https://datashare.ed.ac.uk/handle/10283/3443
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests not found, installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
EXPECTED_SIZE = 11_000_000_000  # ~11GB


def download_with_resume(
    url: str, output_path: Path, chunk_size: int = 1024 * 1024
) -> bool:
    """Download with resume support. Returns True on success."""
    resume_pos = 0
    temp_path = output_path.with_suffix(".zip.partial")

    if temp_path.exists():
        resume_pos = temp_path.stat().st_size
        print(f"Resuming from {resume_pos / (1024 * 1024):.0f} MB")

    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)

        if resp.status_code == 416:
            print("Download already complete (server reports range not satisfiable)")
            if temp_path.exists():
                temp_path.rename(output_path)
            return True

        if resp.status_code == 206:
            total_size = resume_pos + int(resp.headers.get("Content-Length", 0))
        elif resp.status_code == 200:
            total_size = int(resp.headers.get("Content-Length", EXPECTED_SIZE))
            resume_pos = 0
        else:
            print(f"HTTP error: {resp.status_code}")
            return False

        print(f"Total size: {total_size / (1024 * 1024 * 1024):.1f} GB")

        mode = "ab" if resume_pos > 0 and resp.status_code == 206 else "wb"
        downloaded = resume_pos

        with open(temp_path, mode) as f:
            start_time = time.time()
            last_report = start_time

            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_report >= 1.0:
                    elapsed = now - start_time
                    if elapsed > 0 and downloaded > resume_pos:
                        speed = (downloaded - resume_pos) / elapsed / (1024 * 1024)
                        pct = downloaded / total_size * 100 if total_size > 0 else 0
                        remaining = (
                            (total_size - downloaded) / (speed * 1024 * 1024)
                            if speed > 0
                            else 0
                        )
                        sys.stdout.write(
                            f"\r  {downloaded / (1024 * 1024 * 1024):.2f}/{total_size / (1024 * 1024 * 1024):.2f} GB ({pct:.1f}%) - {speed:.1f} MB/s - ETA: {remaining / 60:.0f}m   "
                        )
                        sys.stdout.flush()
                        last_report = now

        print()

        if downloaded >= total_size * 0.99:
            temp_path.rename(output_path)
            return True
        else:
            print(f"Incomplete download: {downloaded} < {total_size}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"\nDownload error: {e}")
        print("Partial file saved. Run again to resume.")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted. Run again to resume.")
        return False


def download_vctk(output_dir: str | Path) -> Path:
    """Download and extract VCTK to output_dir/VCTK-Corpus-0.92/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vctk_dir = output_dir / "VCTK-Corpus-0.92"
    marker = vctk_dir / ".download_complete"

    if marker.exists():
        print(f"VCTK already downloaded at {vctk_dir}")
        return vctk_dir

    zip_path = output_dir / "VCTK-Corpus-0.92.zip"

    if not zip_path.exists() or zip_path.stat().st_size < EXPECTED_SIZE * 0.9:
        print(f"Downloading VCTK from {VCTK_URL}")
        print(f"  License: CC BY 4.0")
        print(f"  Destination: {zip_path}")

        while True:
            if download_with_resume(VCTK_URL, zip_path):
                break
            print("Retrying in 5 seconds...")
            time.sleep(5)

    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(output_dir))
    print("  Extraction complete.")

    marker.write_text("ok")

    return vctk_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download VCTK Corpus 0.92 with resume support"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to download and extract VCTK into.",
    )
    args = parser.parse_args()
    path = download_vctk(args.output_dir)
    print(f"VCTK ready at: {path}")
