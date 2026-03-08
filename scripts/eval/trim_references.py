#!/usr/bin/env python3
"""Trim few-shot reference audio to frozen durations per evaluation-set-spec section 4.

Selects the highest-SNR continuous voiced span at each target duration
(3 s, 5 s, 10 s) and writes trimmed WAV files at 24 kHz.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFERENCE_LENGTHS_SEC: List[int] = [3, 5, 10]
TARGET_SR: int = 24_000

# VAD parameters (mirrors cleanup.py)
VAD_FRAME_SEC: float = 0.025
VAD_HOP_SEC: float = 0.010
ENERGY_FLOOR_DB: float = -40.0

FEW_SHOT_SUBSETS = {"few_shot_same_language", "few_shot_leakage_pairs"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    """Load full audio file as mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample via linear interpolation when sample rates differ.

    For production quality a proper resampler (e.g. libsamplerate) would be
    preferable, but numpy-only linear interpolation is acceptable here because
    the trimmed references are already high-quality voiced spans and we only
    need 24 kHz output.
    """
    if sr == target_sr:
        return audio
    ratio = target_sr / sr
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
    frac = indices - idx_floor
    return audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac


# ---------------------------------------------------------------------------
# Energy-based VAD
# ---------------------------------------------------------------------------

def _compute_frame_voiced(
    audio: np.ndarray,
    sr: int,
    frame_sec: float = VAD_FRAME_SEC,
    hop_sec: float = VAD_HOP_SEC,
    floor_db: float = ENERGY_FLOOR_DB,
) -> np.ndarray:
    """Return a boolean array indicating voiced frames."""
    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    if frame_len < 1 or hop_len < 1 or len(audio) < frame_len:
        return np.array([], dtype=bool)

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        energy_db = 10.0 * np.log10(np.mean(frame ** 2) + 1e-12)
        if energy_db > floor_db:
            voiced[i] = True
    return voiced


def _voiced_spans(voiced: np.ndarray, sr: int, hop_sec: float = VAD_HOP_SEC) -> List[Tuple[float, float]]:
    """Return list of (start_sec, end_sec) for continuous voiced regions."""
    if len(voiced) == 0:
        return []
    spans: List[Tuple[float, float]] = []
    hop_len = int(hop_sec * sr)
    in_span = False
    span_start = 0
    for i, v in enumerate(voiced):
        if v and not in_span:
            span_start = i
            in_span = True
        elif not v and in_span:
            start_sec = span_start * hop_len / sr
            end_sec = i * hop_len / sr
            spans.append((start_sec, end_sec))
            in_span = False
    if in_span:
        start_sec = span_start * hop_len / sr
        end_sec = len(voiced) * hop_len / sr
        spans.append((start_sec, end_sec))
    return spans


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------

def _estimate_noise_energy(audio: np.ndarray, voiced: np.ndarray, sr: int) -> float:
    """Estimate noise energy from non-voiced frames."""
    hop_len = int(VAD_HOP_SEC * sr)
    frame_len = int(VAD_FRAME_SEC * sr)
    noise_energy_sum = 0.0
    noise_count = 0
    for i, v in enumerate(voiced):
        if not v:
            start = i * hop_len
            frame = audio[start : start + frame_len]
            if len(frame) == frame_len:
                noise_energy_sum += np.mean(frame ** 2)
                noise_count += 1
    if noise_count == 0:
        return 1e-12
    return noise_energy_sum / noise_count


def _span_snr(audio: np.ndarray, sr: int, start_sec: float, end_sec: float, noise_energy: float) -> float:
    """Compute SNR (dB) for a given time span relative to estimated noise."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return -np.inf
    signal_energy = float(np.mean(segment ** 2))
    return 10.0 * np.log10(signal_energy / (noise_energy + 1e-12))


# ---------------------------------------------------------------------------
# Core trimming
# ---------------------------------------------------------------------------

def select_best_span(
    audio: np.ndarray,
    sr: int,
    target_duration_sec: int,
) -> Optional[Tuple[float, float, float]]:
    """Select the best voiced span for trimming.

    Returns (start_sec, end_sec, snr_db) or None if no qualifying span exists.
    """
    voiced = _compute_frame_voiced(audio, sr)
    if len(voiced) == 0:
        return None

    noise_energy = _estimate_noise_energy(audio, voiced, sr)
    spans = _voiced_spans(voiced, sr)

    candidates: List[Tuple[float, float, float]] = []
    for span_start, span_end in spans:
        span_dur = span_end - span_start
        if span_dur >= target_duration_sec:
            # Use the first target_duration_sec portion of this span
            trim_end = span_start + target_duration_sec
            snr = _span_snr(audio, sr, span_start, trim_end, noise_energy)
            candidates.append((span_start, trim_end, snr))

    if not candidates:
        return None

    # Sort by SNR descending, then by start time ascending (earliest) for ties
    candidates.sort(key=lambda c: (-c[2], c[0]))
    return candidates[0]


# ---------------------------------------------------------------------------
# Manifest processing
# ---------------------------------------------------------------------------

def process_manifest(
    manifest_path: Path,
    audio_dir: Path,
    output_dir: Path,
) -> Tuple[int, int]:
    """Process manifest and trim references.

    Returns (trimmed_count, skipped_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Filter to few-shot subsets
    few_shot_records = [r for r in records if r.get("subset") in FEW_SHOT_SUBSETS]
    logger.info("Found %d few-shot records out of %d total.", len(few_shot_records), len(records))

    # Collect unique (speaker_id, reference_audio_id, reference_length_sec) tuples
    seen: set[tuple[str, str, int]] = set()
    work_items: list[tuple[str, str, int]] = []
    for rec in few_shot_records:
        speaker_id = rec.get("speaker_id")
        ref_audio_id = rec.get("reference_audio_id")
        ref_len = rec.get("reference_length_sec")
        if speaker_id is None or ref_audio_id is None or ref_len is None:
            continue
        ref_len = int(ref_len)
        key = (speaker_id, ref_audio_id, ref_len)
        if key not in seen:
            seen.add(key)
            work_items.append(key)

    logger.info("Unique (speaker, ref_audio, ref_len) tuples: %d", len(work_items))

    trimmed = 0
    skipped = 0
    # Cache loaded audio per reference_audio_id
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for speaker_id, ref_audio_id, ref_len_sec in work_items:
        # Load source audio
        if ref_audio_id not in audio_cache:
            # Try common extensions
            source_path: Optional[Path] = None
            for ext in (".wav", ".flac", ".mp3", ".ogg", ""):
                candidate = audio_dir / f"{ref_audio_id}{ext}"
                if candidate.exists():
                    source_path = candidate
                    break
            if source_path is None:
                logger.warning(
                    "SKIP speaker=%s ref=%s len=%ds: source audio not found in %s",
                    speaker_id, ref_audio_id, ref_len_sec, audio_dir,
                )
                skipped += 1
                continue
            try:
                audio, sr = _load_audio_mono(source_path)
                audio_cache[ref_audio_id] = (audio, sr)
            except Exception as e:
                logger.warning(
                    "SKIP speaker=%s ref=%s len=%ds: failed to load audio: %s",
                    speaker_id, ref_audio_id, ref_len_sec, e,
                )
                skipped += 1
                continue

        audio, sr = audio_cache[ref_audio_id]

        result = select_best_span(audio, sr, ref_len_sec)
        if result is None:
            logger.warning(
                "SKIP speaker=%s ref=%s len=%ds: no qualifying voiced span of %ds found",
                speaker_id, ref_audio_id, ref_len_sec, ref_len_sec,
            )
            skipped += 1
            continue

        start_sec, end_sec, snr_db = result
        logger.info(
            "TRIM speaker=%s ref=%s len=%ds: span=[%.3f, %.3f]s SNR=%.1fdB",
            speaker_id, ref_audio_id, ref_len_sec, start_sec, end_sec, snr_db,
        )

        # Extract and resample
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment = audio[start_sample:end_sample]
        segment = _resample_if_needed(segment, sr, TARGET_SR)

        # Trim to exact sample count at target SR
        exact_samples = ref_len_sec * TARGET_SR
        if len(segment) > exact_samples:
            segment = segment[:exact_samples]
        elif len(segment) < exact_samples:
            # Pad with silence if slightly short due to rounding
            segment = np.pad(segment, (0, exact_samples - len(segment)))

        # Save
        out_path = output_dir / f"{speaker_id}_{ref_len_sec}s.wav"
        sf.write(str(out_path), segment, TARGET_SR, subtype="PCM_16")
        trimmed += 1

    return trimmed, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim few-shot reference audio per evaluation-set-spec section 4.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.jsonl",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing source reference audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for trimmed reference WAV output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        raise SystemExit(1)
    if not args.audio_dir.is_dir():
        logger.error("Audio directory not found: %s", args.audio_dir)
        raise SystemExit(1)

    trimmed, skipped = process_manifest(args.manifest, args.audio_dir, args.output_dir)

    print(f"\nSummary: {trimmed} trimmed, {skipped} skipped")
    if skipped > 0:
        print("WARNING: some speakers were excluded. See log for details.")


if __name__ == "__main__":
    main()
