"""Stage 1: Cleanup - VAD, clipping detection, corruption detection.

Removes obviously unusable audio before expensive inference stages.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..models import CurationRecord, Provenance, RecordStatus
from ..providers import ProviderOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIPPING_THRESHOLD = 0.99  # sample values near +/-1.0
MAX_CLIPPING_RATIO = 0.05  # reject if > 5% of samples clip
MIN_USABLE_DURATION_SEC = 0.3
ENERGY_FLOOR_DB = -60.0  # frames below this are silence
VAD_FRAME_SEC = 0.025  # 25 ms VAD frames
VAD_HOP_SEC = 0.010  # 10 ms hop


def _load_audio_mono(path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
    """Load audio segment as mono float32."""
    info = sf.info(path)
    sr = info.samplerate
    start_frame = int(start * sr)
    n_frames = int((end - start) * sr)
    audio, sr = sf.read(path, start=start_frame, frames=n_frames, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def detect_corruption(audio: np.ndarray) -> Dict[str, Any]:
    """Check for NaN, Inf, or abnormal RMS."""
    has_nan = bool(np.any(np.isnan(audio)))
    has_inf = bool(np.any(np.isinf(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
    # All-zero or near-zero audio is effectively corrupted/silent
    is_silent = rms < 1e-7
    corrupted = has_nan or has_inf or is_silent
    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "rms": round(rms, 8),
        "is_silent": is_silent,
        "corrupted": corrupted,
    }


def detect_clipping(audio: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> Dict[str, Any]:
    """Detect clipped samples (values near +/-1.0)."""
    if len(audio) == 0:
        return {"clipping_ratio": 0.0, "clipped_samples": 0}
    clipped = np.sum(np.abs(audio) >= threshold)
    ratio = float(clipped) / len(audio)
    return {
        "clipping_ratio": round(ratio, 6),
        "clipped_samples": int(clipped),
    }


def energy_vad(audio: np.ndarray, sr: int,
               frame_sec: float = VAD_FRAME_SEC,
               hop_sec: float = VAD_HOP_SEC,
               floor_db: float = ENERGY_FLOOR_DB) -> Dict[str, Any]:
    """Simple energy-based VAD. Returns usable (speech) duration estimate.

    Computes short-time energy in dB, marks frames above `floor_db` as speech.
    """
    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    if frame_len < 1 or hop_len < 1 or len(audio) < frame_len:
        return {
            "usable_duration_sec": 0.0,
            "speech_ratio": 0.0,
            "n_speech_frames": 0,
            "n_total_frames": 0,
        }

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    speech_count = 0

    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start: start + frame_len]
        energy = np.mean(frame ** 2)
        energy_db = 10.0 * np.log10(energy + 1e-12)
        if energy_db > floor_db:
            speech_count += 1

    total_dur = len(audio) / sr
    speech_ratio = speech_count / n_frames if n_frames > 0 else 0.0
    usable_duration = speech_ratio * total_dur

    return {
        "usable_duration_sec": round(usable_duration, 4),
        "speech_ratio": round(speech_ratio, 4),
        "n_speech_frames": speech_count,
        "n_total_frames": n_frames,
    }


def run_cleanup(record: CurationRecord) -> Optional[CurationRecord]:
    """Process a single record through Stage 1: Cleanup.

    Updates record attributes with cleanup results and rejects if
    audio is corrupted, excessively clipped, or has no usable speech.

    Returns:
        Updated CurationRecord, or None if the record should be removed.
    """
    try:
        audio, sr = _load_audio_mono(
            record.source_path,
            record.segment_start_sec,
            record.segment_end_sec,
        )
    except Exception as e:
        logger.warning("Cleanup: cannot load %s: %s", record.record_id, e)
        record.status = RecordStatus.REJECTED
        record.rejection_reasons.append("audio_load_failed")
        record.providers["cleanup"] = Provenance(
            stage="cleanup",
            provider="builtin_cleanup",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata={"error": str(e)},
        )
        return record

    # --- Corruption ---
    corruption = detect_corruption(audio)
    record.attributes["corruption"] = corruption

    if corruption["corrupted"]:
        record.status = RecordStatus.REJECTED
        record.rejection_reasons.append("audio_corrupted")
        record.attributes["corrupted"] = True
        record.providers["cleanup"] = Provenance(
            stage="cleanup",
            provider="builtin_cleanup",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata=corruption,
        )
        return record

    # --- Clipping ---
    clipping = detect_clipping(audio)
    record.attributes["clipping_ratio"] = clipping["clipping_ratio"]
    record.attributes["clipped_samples"] = clipping["clipped_samples"]

    if clipping["clipping_ratio"] > MAX_CLIPPING_RATIO:
        record.review_reasons.append("excessive_clipping")

    # --- VAD / usable duration ---
    vad = energy_vad(audio, sr)
    record.attributes["usable_duration_sec"] = vad["usable_duration_sec"]
    record.attributes["speech_ratio"] = vad["speech_ratio"]
    record.attributes["vad_n_speech_frames"] = vad["n_speech_frames"]
    record.attributes["vad_n_total_frames"] = vad["n_total_frames"]

    if vad["usable_duration_sec"] < MIN_USABLE_DURATION_SEC:
        record.status = RecordStatus.REJECTED
        record.rejection_reasons.append("no_usable_speech")
        record.providers["cleanup"] = Provenance(
            stage="cleanup",
            provider="builtin_cleanup",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata={"reason": "usable_duration_below_threshold", **vad},
        )
        return record

    # --- RMS / SNR rough estimate ---
    rms_db = 20.0 * np.log10(corruption["rms"] + 1e-12)
    record.attributes["rms_db"] = round(float(rms_db), 2)

    # Update status to annotating (passed cleanup)
    if record.status == RecordStatus.INGESTED:
        record.status = RecordStatus.ANNOTATING

    record.providers["cleanup"] = Provenance(
        stage="cleanup",
        provider="builtin_cleanup",
        version="1.0.0",
        timestamp=time.time(),
        confidence=1.0,
        metadata={
            "clipping_ratio": clipping["clipping_ratio"],
            "usable_duration_sec": vad["usable_duration_sec"],
            "speech_ratio": vad["speech_ratio"],
            "rms_db": record.attributes["rms_db"],
        },
    )

    return record
