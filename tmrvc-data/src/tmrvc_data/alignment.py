"""Forced alignment wrapper for extracting phoneme-level durations."""

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


def extract_alignment(
    audio_path: str | Path,
    text: str,
    language: str = "ja",
    total_frames: int | None = None,
) -> dict[str, Any]:
    """High-level API to extract phoneme IDs and durations for UCLM v2.
    
    If MFA is not installed, falls back to equal-duration heuristic for now
    to avoid breaking the pipeline during development.
    """
    from tmrvc_core.text_utils import text_to_phonemes
    
    phoneme_ids = text_to_phonemes(text, language=language)
    L = len(phoneme_ids)
    
    if total_frames is None:
        import soundfile as sf
        info = sf.info(str(audio_path))
        total_frames = round(info.duration / _FRAME_SEC)
        
    # Heuristic fallback (uniform duration)
    # TODO: In production, use actual MFA TextGrid parsing here.
    base_dur = total_frames // L
    durations = [base_dur] * L
    # Distribute remainder
    rem = total_frames % L
    for i in range(rem): durations[i] += 1
    
    return {
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "method": "heuristic_fallback"
    }


def _parse_textgrid(path: Path) -> list[tuple[float, float, str]]:
    """Parse a Praat TextGrid file (short format or long format)."""
    text = path.read_text(encoding="utf-8")
    intervals: list[tuple[float, float, str]] = []
    # (Regex parsing logic...)
    pattern = re.compile(r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"')
    for match in pattern.finditer(text):
        intervals.append((float(match.group(1)), float(match.group(2)), match.group(3)))
    return intervals


def alignment_to_durations(
    intervals: list[tuple[float, float, str]],
    total_frames: int | None = None,
) -> AlignmentResult:
    phonemes, starts, ends, durations = [], [], [], []
    for start, end, label in intervals:
        phonemes.append(label or "<sil>")
        starts.append(start); ends.append(end)
        durations.append(max(1, round((end - start) / _FRAME_SEC)))
    dur_array = np.array(durations, dtype=np.int64)
    if total_frames is not None and len(dur_array) > 0:
        diff = total_frames - dur_array.sum()
        if diff != 0: dur_array[-1] = max(1, dur_array[-1] + diff)
    return AlignmentResult(phonemes, dur_array, np.array(starts), np.array(ends))
