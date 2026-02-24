"""Forced alignment wrapper for extracting phoneme-level durations.

Uses Montreal Forced Aligner (MFA) to align audio with phoneme transcriptions,
producing per-phoneme durations in frames (10ms hop).
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import HOP_LENGTH, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Seconds per frame
_FRAME_SEC = HOP_LENGTH / SAMPLE_RATE  # 0.01s = 10ms


@dataclass
class AlignmentResult:
    """Result of forced alignment for a single utterance."""

    phonemes: list[str]  # Phoneme sequence (from TextGrid)
    durations: np.ndarray  # [L] frame counts per phoneme
    start_times: np.ndarray  # [L] start time in seconds
    end_times: np.ndarray  # [L] end time in seconds


def _parse_textgrid(path: Path) -> list[tuple[float, float, str]]:
    """Parse a Praat TextGrid file (short format or long format).

    Returns list of (start_sec, end_sec, label) tuples for the first
    IntervalTier (typically "phones").
    """
    text = path.read_text(encoding="utf-8")

    intervals: list[tuple[float, float, str]] = []

    # Try short format first
    if '"IntervalTier"' in text or "IntervalTier" in text:
        # Find phone tier intervals
        # Short format: xmin\nxmax\ntext on separate lines
        lines = text.strip().split("\n")
        in_phone_tier = False
        i = 0
        while i < len(lines):
            line = lines[i].strip().strip('"')
            if line == "phones" or line == "phone":
                in_phone_tier = True
                # Skip xmin, xmax, n_intervals
                i += 1
                continue
            if in_phone_tier:
                # Try to parse triplets: xmin, xmax, text
                try:
                    xmin = float(lines[i].strip().strip('"'))
                    xmax = float(lines[i + 1].strip().strip('"'))
                    label = lines[i + 2].strip().strip('"')
                    intervals.append((xmin, xmax, label))
                    i += 3
                    continue
                except (ValueError, IndexError):
                    pass
            i += 1

    if not intervals:
        # Fallback: regex-based parsing for long format
        pattern = re.compile(
            r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"'
        )
        for match in pattern.finditer(text):
            xmin = float(match.group(1))
            xmax = float(match.group(2))
            label = match.group(3)
            intervals.append((xmin, xmax, label))

    return intervals


def alignment_to_durations(
    intervals: list[tuple[float, float, str]],
    total_frames: int | None = None,
) -> AlignmentResult:
    """Convert time-aligned intervals to frame-level durations.

    Args:
        intervals: List of (start_sec, end_sec, label) from TextGrid.
        total_frames: If provided, adjust last duration to match exactly.

    Returns:
        AlignmentResult with phonemes and frame durations.
    """
    phonemes: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    durations: list[int] = []

    for start, end, label in intervals:
        if not label or label == "":
            label = "<sil>"

        phonemes.append(label)
        starts.append(start)
        ends.append(end)

        # Convert time span to frame count
        dur = max(1, round((end - start) / _FRAME_SEC))
        durations.append(dur)

    dur_array = np.array(durations, dtype=np.int64)

    # Adjust to match total frames if provided
    if total_frames is not None and len(dur_array) > 0:
        diff = total_frames - dur_array.sum()
        if diff != 0:
            dur_array[-1] = max(1, dur_array[-1] + diff)

    return AlignmentResult(
        phonemes=phonemes,
        durations=dur_array,
        start_times=np.array(starts, dtype=np.float64),
        end_times=np.array(ends, dtype=np.float64),
    )


def run_mfa_align(
    audio_dir: Path,
    transcript_dir: Path,
    output_dir: Path,
    language: str = "japanese",
    dictionary: str | None = None,
    acoustic_model: str | None = None,
    num_jobs: int = 4,
) -> None:
    """Run Montreal Forced Aligner on a directory of audio files.

    Args:
        audio_dir: Directory containing .wav files.
        transcript_dir: Directory containing .lab or .txt transcript files.
        output_dir: Directory to write TextGrid outputs.
        language: MFA language preset (e.g., "japanese", "english").
        dictionary: Path to MFA dictionary (or pretrained name).
        acoustic_model: Path to MFA acoustic model (or pretrained name).
        num_jobs: Number of parallel MFA jobs.
    """
    cmd = [
        "mfa", "align",
        str(audio_dir),
        dictionary or language,
        acoustic_model or language,
        str(output_dir),
        "--num_jobs", str(num_jobs),
        "--clean",
        "--overwrite",
    ]

    logger.info("Running MFA: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error("MFA failed: %s", result.stderr)
        raise RuntimeError(f"MFA alignment failed: {result.stderr}")

    logger.info("MFA alignment complete: %s", output_dir)


def load_textgrid_durations(
    textgrid_path: Path,
    total_frames: int | None = None,
) -> AlignmentResult:
    """Load a TextGrid file and extract phoneme durations.

    Args:
        textgrid_path: Path to .TextGrid file from MFA.
        total_frames: If provided, adjust last phoneme duration to match.

    Returns:
        AlignmentResult with phonemes and frame-level durations.
    """
    intervals = _parse_textgrid(textgrid_path)
    if not intervals:
        raise ValueError(f"No intervals found in TextGrid: {textgrid_path}")
    return alignment_to_durations(intervals, total_frames)
