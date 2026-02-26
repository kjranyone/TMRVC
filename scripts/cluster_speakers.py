#!/usr/bin/env python3
"""Cluster flat audio files by speaker using ECAPA-TDNN + HDBSCAN.

Extracts 192-dim speaker embeddings from all audio files in a directory,
then clusters them with HDBSCAN to produce a speaker_map.json.

Usage::

    # Full pipeline (embed + cluster)
    python scripts/cluster_speakers.py --input data/raw/galge_voices --device xpu

    # Embedding only (resumable, saves every 1000 files)
    python scripts/cluster_speakers.py --input data/raw/galge_voices --step embed --device xpu

    # Cluster from existing embeddings
    python scripts/cluster_speakers.py --input data/raw/galge_voices --step cluster

    # Show cluster statistics
    python scripts/cluster_speakers.py --input data/raw/galge_voices --step report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
EMBED_FILENAME = "_speaker_embeds.npz"
MAP_FILENAME = "_speaker_map.json"


def extract_embeddings(
    root: Path,
    output_path: Path,
    device: str = "cpu",
    save_every: int = 1000,
) -> dict[str, np.ndarray]:
    """Extract 192-dim speaker embeddings from all audio files.

    Saves intermediate results every *save_every* files for resumability.
    """
    from tmrvc_data.speaker import SpeakerEncoder

    # Collect audio files
    audio_files = sorted(
        f for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    logger.info("Found %d audio files in %s", len(audio_files), root)

    # Load existing embeddings for resume
    existing: dict[str, np.ndarray] = {}
    if output_path.exists():
        data = np.load(str(output_path), allow_pickle=False)
        existing = {k: data[k] for k in data.files}
        logger.info("Resuming: loaded %d existing embeddings", len(existing))

    encoder = SpeakerEncoder(device=device)
    embeddings = dict(existing)
    processed = 0
    errors = 0
    t0 = time.time()

    for i, audio_path in enumerate(audio_files):
        name = audio_path.name
        if name in embeddings:
            continue

        try:
            emb = encoder.extract_from_file(str(audio_path))
            embeddings[name] = emb.numpy()
            processed += 1
        except Exception as e:
            logger.warning("Failed: %s: %s", name, e)
            errors += 1

        # Progress + intermediate save
        total_done = len(embeddings)
        if total_done % save_every == 0 and processed > 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining = len(audio_files) - (i + 1)
            eta_min = remaining / rate / 60 if rate > 0 else 0
            logger.info(
                "Progress: %d/%d done (%d new, %d errors), %.1f files/s, ETA %.0f min",
                total_done, len(audio_files), processed, errors, rate, eta_min,
            )
            np.savez(str(output_path), **embeddings)
            logger.info("  Saved checkpoint: %s (%d embeddings)", output_path, len(embeddings))

    # Final save
    np.savez(str(output_path), **embeddings)
    elapsed = time.time() - t0
    logger.info(
        "Embedding done: %d total (%d new, %d errors) in %.1f min",
        len(embeddings), processed, errors, elapsed / 60,
    )
    return embeddings


def cluster_embeddings(
    embeddings: dict[str, np.ndarray],
    method: str = "hdbscan",
    min_cluster_size: int = 20,
    min_samples: int = 5,
) -> dict[str, str]:
    """Cluster speaker embeddings with HDBSCAN.

    Returns mapping from filename to speaker label.
    Noise points (label -1) are mapped to ``"spk_noise"``.
    """
    import hdbscan

    names = sorted(embeddings.keys())
    matrix = np.stack([embeddings[n] for n in names])  # [N, 192]

    logger.info(
        "Clustering %d embeddings (method=%s, min_cluster_size=%d, min_samples=%d)",
        len(names), method, min_cluster_size, min_samples,
    )

    # L2-normalize before clustering — speaker embeddings are already L2-normed,
    # so euclidean distance on normalized vectors ∝ cosine distance.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    matrix = matrix / norms

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(matrix)

    # Build speaker map
    n_clusters = labels.max() + 1 if len(labels) > 0 else 0
    n_noise = int((labels == -1).sum())
    logger.info("Found %d clusters, %d noise points", n_clusters, n_noise)

    mapping: dict[str, str] = {}
    for name, label in zip(names, labels):
        if label == -1:
            mapping[name] = "spk_noise"
        else:
            mapping[name] = f"spk_{label + 1:04d}"

    return mapping


def save_speaker_map(
    mapping: dict[str, str],
    output_path: Path,
    method: str = "hdbscan",
) -> None:
    """Save speaker map as JSON."""
    speakers = set(v for v in mapping.values() if v != "spk_noise")
    n_noise = sum(1 for v in mapping.values() if v == "spk_noise")

    result = {
        "version": 1,
        "method": method,
        "n_speakers": len(speakers),
        "n_noise": n_noise,
        "mapping": mapping,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Saved speaker map: %s (%d speakers, %d noise)", output_path, len(speakers), n_noise)


def load_speaker_map(path: Path) -> dict[str, str]:
    """Load speaker map from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["mapping"]


def print_report(speaker_map: dict[str, str], embeddings: dict[str, np.ndarray] | None = None) -> None:
    """Print cluster statistics."""
    from collections import Counter

    counts = Counter(speaker_map.values())
    speakers = sorted(
        ((spk, cnt) for spk, cnt in counts.items() if spk != "spk_noise"),
        key=lambda x: -x[1],
    )
    noise_count = counts.get("spk_noise", 0)

    print(f"\n{'Speaker':<15} {'Files':>8}")
    print("-" * 25)
    for spk, cnt in speakers:
        print(f"{spk:<15} {cnt:>8}")
    print("-" * 25)
    print(f"{'Speakers':<15} {len(speakers):>8}")
    print(f"{'Noise':<15} {noise_count:>8}")
    print(f"{'Total':<15} {len(speaker_map):>8}")

    if speakers:
        file_counts = [cnt for _, cnt in speakers]
        print(f"\nFiles per speaker: min={min(file_counts)}, max={max(file_counts)}, "
              f"median={sorted(file_counts)[len(file_counts)//2]}, "
              f"mean={sum(file_counts)/len(file_counts):.0f}")


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Cluster audio files by speaker using ECAPA-TDNN + HDBSCAN.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input directory with audio files.")
    parser.add_argument(
        "--step",
        choices=["all", "embed", "cluster", "report"],
        default="all",
        help="Pipeline step to run (default: all).",
    )
    parser.add_argument("--device", default="cpu", help="Device for embedding extraction.")
    parser.add_argument("--min-cluster-size", type=int, default=20, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save embeddings every N files.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    embed_path = args.input / EMBED_FILENAME
    map_path = args.input / MAP_FILENAME

    # --- Embed ---
    if args.step in ("all", "embed"):
        extract_embeddings(
            args.input, embed_path,
            device=args.device,
            save_every=args.save_every,
        )

    # --- Cluster ---
    if args.step in ("all", "cluster"):
        if not embed_path.exists():
            logger.error("Embeddings not found: %s (run --step embed first)", embed_path)
            sys.exit(1)
        data = np.load(str(embed_path), allow_pickle=False)
        embeddings = {k: data[k] for k in data.files}

        mapping = cluster_embeddings(
            embeddings,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        save_speaker_map(mapping, map_path)

    # --- Report ---
    if args.step in ("all", "report"):
        if not map_path.exists():
            logger.error("Speaker map not found: %s (run --step cluster first)", map_path)
            sys.exit(1)
        speaker_map = load_speaker_map(map_path)
        embeddings = None
        if embed_path.exists():
            data = np.load(str(embed_path), allow_pickle=False)
            embeddings = {k: data[k] for k in data.files}
        print_report(speaker_map, embeddings)


if __name__ == "__main__":
    main()
