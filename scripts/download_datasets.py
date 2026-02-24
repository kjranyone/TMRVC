#!/usr/bin/env python3
"""Download TTS training datasets.

Downloads and extracts:
- JSUT: Japanese single-speaker TTS corpus (~2 GB)
- LJSpeech: English single-speaker TTS corpus (~2.6 GB)
- JVNV: Japanese emotional speech corpus (~1.5 GB)

Usage::

    python scripts/download_datasets.py --dataset jsut --output-dir data/raw
    python scripts/download_datasets.py --dataset ljspeech --output-dir data/raw
    python scripts/download_datasets.py --dataset jvnv --output-dir data/raw
    python scripts/download_datasets.py --all --output-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DATASETS: dict[str, dict] = {
    "jsut": {
        "url": "https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip",
        "filename": "jsut_ver1.1.zip",
        "extract_to": "jsut",
        "description": "JSUT: Japanese single-speaker TTS corpus (~10h, 48kHz)",
        "size_mb": 2048,
    },
    "ljspeech": {
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "filename": "LJSpeech-1.1.tar.bz2",
        "extract_to": "ljspeech",
        "description": "LJSpeech: English single-speaker TTS corpus (~24h, 22.05kHz)",
        "size_mb": 2600,
    },
    "jvnv": {
        "url": "https://sites.google.com/site/shinaborulab/research/jvnv",
        "filename": None,  # Manual download required
        "extract_to": "jvnv",
        "description": "JVNV: Japanese emotional speech corpus (6 emotions, ~4h, 24kHz)",
        "size_mb": 1500,
        "manual": True,
    },
}


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("Downloading %s ...", desc or url)
    logger.info("  URL: %s", url)
    logger.info("  Destination: %s", dest)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TMRVC/0.1"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        logger.info(
                            "  Progress: %.1f%% (%d / %d MB)",
                            pct, downloaded // (1024 * 1024), total // (1024 * 1024),
                        )

        tmp.rename(dest)
        logger.info("Download complete: %s", dest)

    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {e}") from e


def _extract_zip(archive: Path, output_dir: Path) -> None:
    """Extract a ZIP archive."""
    logger.info("Extracting %s ...", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output_dir)
    logger.info("Extracted to %s", output_dir)


def _extract_tar(archive: Path, output_dir: Path) -> None:
    """Extract a tar archive (gz, bz2, xz)."""
    logger.info("Extracting %s ...", archive)
    with tarfile.open(archive) as tf:
        tf.extractall(output_dir, filter="data")
    logger.info("Extracted to %s", output_dir)


def download_jsut(output_dir: Path) -> Path:
    """Download and extract JSUT corpus.

    Returns:
        Path to extracted JSUT directory.
    """
    info = DATASETS["jsut"]
    archive = output_dir / info["filename"]
    extract_dir = output_dir / info["extract_to"]

    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("JSUT already extracted at %s", extract_dir)
        return extract_dir

    _download_file(info["url"], archive, info["description"])
    _extract_zip(archive, output_dir)

    # JSUT extracts to jsut_ver1.1/ — rename to jsut/
    extracted = output_dir / "jsut_ver1.1"
    if extracted.exists() and not extract_dir.exists():
        extracted.rename(extract_dir)
    elif extracted.exists():
        # Move contents
        for item in extracted.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))
        extracted.rmdir()

    return extract_dir


def download_ljspeech(output_dir: Path) -> Path:
    """Download and extract LJSpeech corpus.

    Returns:
        Path to extracted LJSpeech directory.
    """
    info = DATASETS["ljspeech"]
    archive = output_dir / info["filename"]
    extract_dir = output_dir / info["extract_to"]

    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("LJSpeech already extracted at %s", extract_dir)
        return extract_dir

    _download_file(info["url"], archive, info["description"])
    _extract_tar(archive, output_dir)

    # LJSpeech extracts to LJSpeech-1.1/ — rename
    extracted = output_dir / "LJSpeech-1.1"
    if extracted.exists() and not extract_dir.exists():
        extracted.rename(extract_dir)

    return extract_dir


def download_dataset(name: str, output_dir: Path) -> Path | None:
    """Download a dataset by name.

    Args:
        name: Dataset name.
        output_dir: Output directory.

    Returns:
        Path to extracted dataset, or None if manual download required.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[name]
    if info.get("manual"):
        logger.warning(
            "Dataset '%s' requires manual download. Visit: %s",
            name, info["url"],
        )
        logger.warning("  Download and extract to: %s", output_dir / info["extract_to"])
        return None

    if name == "jsut":
        return download_jsut(output_dir)
    elif name == "ljspeech":
        return download_ljspeech(output_dir)
    else:
        raise ValueError(f"No downloader for: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TTS training datasets.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            manual = " [MANUAL DOWNLOAD]" if info.get("manual") else ""
            print(f"  {name}: {info['description']} (~{info['size_mb']} MB){manual}")
        return

    if args.all:
        for name in DATASETS:
            try:
                download_dataset(name, args.output_dir)
            except Exception as e:
                logger.error("Failed to download %s: %s", name, e)
    elif args.dataset:
        download_dataset(args.dataset, args.output_dir)
    else:
        parser.error("Specify --dataset or --all")


if __name__ == "__main__":
    main()
