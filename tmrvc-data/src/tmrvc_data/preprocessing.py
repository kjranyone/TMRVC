"""Audio preprocessing: resample, loudness normalisation, silence trimming, segmentation."""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from tmrvc_core.constants import (
    LOUDNESS_TARGET_LUFS,
    SAMPLE_RATE,
    SEGMENT_MAX_SEC,
    SEGMENT_MIN_SEC,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def load_and_resample(
    path: str | object,
    target_sr: int = SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample to *target_sr*.

    Uses soundfile for reading to avoid torchcodec dependency.

    Returns:
        ``(waveform, target_sr)`` where ``waveform`` is ``[1, T]`` float32.
    """
    data, sr = sf.read(str(path), dtype="float32")
    # soundfile returns [T] for mono, [T, C] for multi-channel
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    else:
        waveform = waveform.T  # [C, T]
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform, target_sr


# ---------------------------------------------------------------------------
# Loudness normalisation
# ---------------------------------------------------------------------------


def normalize_loudness(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = LOUDNESS_TARGET_LUFS,
) -> torch.Tensor:
    """Normalise integrated loudness to *target_lufs* (ITU-R BS.1770-4).

    Args:
        waveform: ``[1, T]`` float32.

    Returns:
        Loudness-normalised ``[1, T]`` tensor.
    """
    audio_np = waveform.squeeze(0).numpy()
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio_np)

    if np.isinf(current_lufs):
        logger.warning("Silent audio detected, skipping loudness normalisation")
        return waveform

    normalised = pyln.normalize.loudness(audio_np, current_lufs, target_lufs)
    return torch.from_numpy(normalised).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# Silence trimming (energy-based, no external VAD dependency)
# ---------------------------------------------------------------------------


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -40.0,
    frame_ms: float = 10.0,
    min_speech_ms: float = 200.0,
) -> torch.Tensor:
    """Trim leading/trailing silence using a simple energy-based VAD.

    This avoids a hard dependency on Silero VAD at import time.  For
    production preprocessing the CLI can optionally use Silero via
    ``--vad silero``.

    Args:
        waveform: ``[1, T]`` float32.
        threshold_db: Energy threshold relative to full scale.
        frame_ms: Analysis frame length in milliseconds.
        min_speech_ms: Minimum speech region to keep.

    Returns:
        Trimmed ``[1, T']`` tensor (or the original if entirely below threshold).
    """
    frame_len = int(sample_rate * frame_ms / 1000.0)
    audio = waveform.squeeze(0)
    n_frames = audio.shape[0] // frame_len

    if n_frames == 0:
        return waveform

    # RMS energy per frame
    frames = audio[: n_frames * frame_len].view(n_frames, frame_len)
    rms = frames.pow(2).mean(dim=1).sqrt()
    rms_db = 20.0 * torch.log10(rms.clamp(min=1e-10))

    voiced = rms_db > threshold_db
    indices = torch.where(voiced)[0]

    if len(indices) == 0:
        return waveform

    start_frame = indices[0].item()
    end_frame = indices[-1].item() + 1

    # Enforce minimum speech duration
    min_frames = int(min_speech_ms / frame_ms)
    if (end_frame - start_frame) < min_frames:
        return waveform

    start_sample = start_frame * frame_len
    end_sample = min(end_frame * frame_len, audio.shape[0])
    return audio[start_sample:end_sample].unsqueeze(0)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_utterance(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    min_sec: float = SEGMENT_MIN_SEC,
    max_sec: float = SEGMENT_MAX_SEC,
) -> Iterator[torch.Tensor]:
    """Split a waveform into segments of *min_sec* to *max_sec* duration.

    Segments shorter than *min_sec* at the tail are discarded.

    Yields:
        ``[1, T_seg]`` tensors.
    """
    total_samples = waveform.shape[-1]
    min_samples = int(min_sec * sample_rate)
    max_samples = int(max_sec * sample_rate)

    if total_samples <= max_samples:
        if total_samples >= min_samples:
            yield waveform
        return

    offset = 0
    while offset < total_samples:
        remaining = total_samples - offset
        seg_len = min(max_samples, remaining)
        if seg_len < min_samples:
            break
        seg = waveform[..., offset : offset + seg_len]
        yield seg
        offset += seg_len


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_audio(
    path: str | object,
    target_sr: int = SAMPLE_RATE,
    target_lufs: float = LOUDNESS_TARGET_LUFS,
    trim: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load, resample, normalise loudness, and optionally trim silence.

    Returns:
        ``(waveform, sample_rate)`` where ``waveform`` is ``[1, T]``.
    """
    waveform, sr = load_and_resample(path, target_sr)
    waveform = normalize_loudness(waveform, sr, target_lufs)
    if trim:
        waveform = trim_silence(waveform, sr)
    return waveform, sr
