#!/usr/bin/env python3
"""Download and extract VCTK Corpus 0.92 (CC BY 4.0).

Source: University of Edinburgh, CSTR
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
URL: https://datashare.ed.ac.uk/handle/10283/3443
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
EXPECTED_MD5 = None  # MD5 not published officially


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)")
        sys.stdout.flush()


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

    if not zip_path.exists():
        print(f"Downloading VCTK from {VCTK_URL}")
        print(f"  License: CC BY 4.0 (Creative Commons Attribution 4.0 International)")
        print(f"  Destination: {zip_path}")
        urlretrieve(VCTK_URL, str(zip_path), reporthook=_progress_hook)
        print("\n  Download complete.")
    else:
        print(f"ZIP already exists: {zip_path}")

    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(output_dir))
    print("  Extraction complete.")

    # Write marker
    marker.write_text("ok")

    # Optionally remove ZIP to save space
    # zip_path.unlink()

    return vctk_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VCTK Corpus 0.92")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to download and extract VCTK into.",
    )
    args = parser.parse_args()
    path = download_vctk(args.output_dir)
    print(f"VCTK ready at: {path}")
