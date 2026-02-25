"""Breath/pause event extraction and cache I/O for BPEH.

Events are stored per utterance as ``events.json`` in the feature cache::

    data/cache/{dataset}/train/{speaker}/{utterance}/events.json

Schema::

    {
      "events": [
        {"type": "breath", "start_frame": 12, "dur_ms": 340, "intensity": 0.71},
        {"type": "pause", "start_frame": 45, "dur_ms": 180}
      ]
    }

Event tensors (frame-level) for training:
- ``breath_onsets[T]``: binary, 1 at breath onset frames
- ``breath_durations[T]``: duration (ms) at onset frames, 0 elsewhere
- ``breath_intensity[T]``: intensity (0-1) at onset frames, 0 elsewhere
- ``pause_durations[T]``: duration (ms) at pause onset frames, 0 elsewhere
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import (
    BPEH_BREATH_THRESHOLD_DB,
    BPEH_MIN_BREATH_MS,
    BPEH_MIN_PAUSE_MS,
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# Frame duration in ms
FRAME_MS = HOP_LENGTH / SAMPLE_RATE * 1000.0  # 10 ms


def extract_events(
    mel: np.ndarray,
    f0: np.ndarray,
    breath_threshold_db: float = BPEH_BREATH_THRESHOLD_DB,
    min_pause_ms: float = BPEH_MIN_PAUSE_MS,
    min_breath_ms: float = BPEH_MIN_BREATH_MS,
) -> list[dict]:
    """Extract breath and pause events from mel spectrogram and F0.

    Breath detection heuristic:
    - Unvoiced frames (F0 == 0 or < 10 Hz)
    - Energy in mid-band (1-5 kHz) is above breath threshold
    - Duration meets minimum breath length

    Pause detection:
    - Unvoiced frames with energy below breath threshold
    - Duration meets minimum pause length

    Args:
        mel: ``[80, T]`` log-mel spectrogram.
        f0: ``[1, T]`` or ``[T]`` F0 in Hz.
        breath_threshold_db: Energy threshold (dB) for breath vs pause.
        min_pause_ms: Minimum pause duration to label.
        min_breath_ms: Minimum breath duration to label.

    Returns:
        List of event dicts with keys: type, start_frame, dur_ms, intensity.
    """
    if f0.ndim == 2:
        f0 = f0.squeeze(0)
    T = mel.shape[-1]

    # Frame energy (mean mel across frequency bins)
    frame_energy = mel.mean(axis=0)  # [T]

    # Mid-band energy (bins ~13-40 correspond to ~1-5 kHz for 80 mel bins)
    mid_band_energy = mel[13:40, :].mean(axis=0)  # [T]

    # Unvoiced mask
    unvoiced = f0 < 10.0  # [T]

    min_pause_frames = max(1, int(min_pause_ms / FRAME_MS))
    min_breath_frames = max(1, int(min_breath_ms / FRAME_MS))

    events: list[dict] = []

    # Find contiguous unvoiced regions
    regions = _find_contiguous_regions(unvoiced)

    for start, end in regions:
        dur_frames = end - start
        dur_ms = round(dur_frames * FRAME_MS, 1)
        region_energy = float(frame_energy[start:end].mean())
        region_mid_energy = float(mid_band_energy[start:end].mean())

        # Classify: breath has notable mid-band energy, pause is near-silent
        if region_mid_energy > breath_threshold_db and dur_frames >= min_breath_frames:
            # Breath: intensity is relative energy level
            intensity = float(np.clip(
                (region_mid_energy - breath_threshold_db) / 20.0, 0.0, 1.0,
            ))
            events.append({
                "type": "breath",
                "start_frame": int(start),
                "dur_ms": dur_ms,
                "intensity": round(intensity, 3),
            })
        elif dur_frames >= min_pause_frames:
            events.append({
                "type": "pause",
                "start_frame": int(start),
                "dur_ms": dur_ms,
            })

    return events


def _find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find start/end of contiguous True regions in a boolean array."""
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_region:
            start = i
            in_region = True
        elif not v and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(mask)))
    return regions


def events_to_tensors(
    events: list[dict],
    n_frames: int,
) -> dict[str, torch.Tensor]:
    """Convert event list to frame-level tensors.

    Returns:
        Dict with keys: breath_onsets, breath_durations, breath_intensity, pause_durations.
        All tensors have shape ``[T]``.
    """
    breath_onsets = torch.zeros(n_frames)
    breath_durations = torch.zeros(n_frames)
    breath_intensity = torch.zeros(n_frames)
    pause_durations = torch.zeros(n_frames)

    for evt in events:
        frame = evt["start_frame"]
        if frame >= n_frames:
            continue
        if evt["type"] == "breath":
            breath_onsets[frame] = 1.0
            breath_durations[frame] = evt["dur_ms"]
            breath_intensity[frame] = evt.get("intensity", 0.5)
        elif evt["type"] == "pause":
            pause_durations[frame] = evt["dur_ms"]

    return {
        "breath_onsets": breath_onsets,
        "breath_durations": breath_durations,
        "breath_intensity": breath_intensity,
        "pause_durations": pause_durations,
    }


def save_events(events: list[dict], utt_dir: Path) -> Path:
    """Save events to events.json in the utterance cache directory."""
    path = utt_dir / "events.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=2)
    return path


def load_events(utt_dir: Path) -> list[dict]:
    """Load events from events.json. Returns empty list if not found."""
    path = utt_dir / "events.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])
