"""Stage 4: Transcript Recovery - ASR-based transcript generation.

Creates text-side supervision for pointer TTS by running ASR on audio
segments and recording transcripts, word timestamps, language, and
confidence scores.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..models import CurationRecord, Provenance, RecordStatus
from ..providers import ProviderRegistry

logger = logging.getLogger(__name__)


def _load_audio_mono(path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
    """Load audio segment as mono float32."""
    info = sf.info(path)
    sr = info.samplerate
    start_frame = int(start * sr)
    n_frames = int((end - start) * sr)
    audio, sr = sf.read(path, start=start_frame, frames=n_frames,
                        dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _run_faster_whisper(
    audio: np.ndarray, sr: int
) -> Optional[Dict[str, Any]]:
    """Attempt transcription with faster-whisper if available.

    Returns dict with transcript, language, confidence, word_timestamps,
    or None if not available.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    try:
        # Use small model by default for throughput
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments_gen, info = model.transcribe(
            audio,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )

        segments_list = list(segments_gen)
        if not segments_list:
            return {
                "transcript": "",
                "language": info.language,
                "confidence": 0.0,
                "word_timestamps": [],
                "asr_segments": [],
            }

        full_text = " ".join(seg.text.strip() for seg in segments_list)

        # Compute confidence - handle both older (avg_log_prob) and newer API
        log_probs = []
        for seg in segments_list:
            if hasattr(seg, "avg_log_prob"):
                log_probs.append(seg.avg_log_prob)
            elif hasattr(seg, "avg_logprob"):
                log_probs.append(seg.avg_logprob)
        if log_probs:
            avg_confidence = float(np.mean(log_probs))
            confidence = float(np.clip(np.exp(avg_confidence), 0.0, 1.0))
        else:
            confidence = 0.5  # moderate default when log probs unavailable

        word_timestamps = []
        for seg in segments_list:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    word_timestamps.append({
                        "word": w.word,
                        "start": round(w.start, 4),
                        "end": round(w.end, 4),
                        "probability": round(getattr(w, "probability", 0.0), 4),
                    })

        asr_segments = []
        for seg in segments_list:
            entry = {
                "text": seg.text.strip(),
                "start": round(seg.start, 4),
                "end": round(seg.end, 4),
            }
            if hasattr(seg, "avg_log_prob"):
                entry["avg_log_prob"] = round(seg.avg_log_prob, 4)
            elif hasattr(seg, "avg_logprob"):
                entry["avg_log_prob"] = round(seg.avg_logprob, 4)
            asr_segments.append(entry)

        return {
            "transcript": full_text,
            "language": info.language,
            "confidence": round(confidence, 4),
            "word_timestamps": word_timestamps,
            "asr_segments": asr_segments,
        }
    except Exception as e:
        logger.warning("faster-whisper transcription failed: %s", e)
        return None


def _run_builtin_placeholder(
    audio: np.ndarray, sr: int
) -> Dict[str, Any]:
    """Placeholder when no ASR provider is available.

    Returns empty transcript with zero confidence so downstream stages
    know that ASR was attempted but no result was produced.
    """
    return {
        "transcript": "",
        "language": None,
        "confidence": 0.0,
        "word_timestamps": [],
        "asr_segments": [],
        "asr_provider": "none_available",
    }


def run_transcript_recovery(
    record: CurationRecord,
    registry: Optional[ProviderRegistry] = None,
) -> Optional[CurationRecord]:
    """Process a single record through Stage 4: Transcript Recovery.

    Tries ASR providers in order:
    1. External provider from registry (if available)
    2. faster-whisper (if installed)
    3. Placeholder with zero confidence

    Args:
        record: The curation record to process.
        registry: Optional provider registry for external ASR models.

    Returns:
        Updated CurationRecord.
    """
    if record.status == RecordStatus.REJECTED:
        return record

    # --- Try external ASR provider ---
    if registry is not None:
        provider = registry.get_primary("asr")
        if provider is not None and provider.is_available():
            try:
                output = provider.process(record)
                for key, value in output.fields.items():
                    if key == "attributes" and isinstance(value, dict):
                        record.attributes.update(value)
                    elif hasattr(record, key):
                        setattr(record, key, value)
                if output.provenance:
                    record.providers["transcript_recovery"] = output.provenance
                else:
                    record.providers["transcript_recovery"] = Provenance(
                        stage="transcript_recovery",
                        provider=provider.name,
                        version=provider.version,
                        timestamp=time.time(),
                        confidence=output.confidence,
                    )
                return record
            except Exception as e:
                logger.warning(
                    "ASR provider %s failed for %s: %s",
                    provider.name, record.record_id, e,
                )

    # --- Load audio ---
    try:
        audio, sr = _load_audio_mono(
            record.source_path,
            record.segment_start_sec,
            record.segment_end_sec,
        )
    except Exception as e:
        logger.warning("Transcript recovery: cannot load %s: %s", record.record_id, e)
        record.transcript = ""
        record.transcript_confidence = 0.0
        record.providers["transcript_recovery"] = Provenance(
            stage="transcript_recovery",
            provider="builtin",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata={"error": str(e)},
        )
        return record

    # --- Try faster-whisper ---
    result = _run_faster_whisper(audio, sr)
    if result is not None:
        provider_name = "faster_whisper"
    else:
        # --- Fallback placeholder ---
        result = _run_builtin_placeholder(audio, sr)
        provider_name = "none_available"

    # Apply results to record
    record.transcript = result["transcript"]
    record.transcript_confidence = result["confidence"]

    if result.get("language"):
        record.language = result["language"]

    record.attributes["word_timestamps"] = result.get("word_timestamps", [])
    record.attributes["asr_segments"] = result.get("asr_segments", [])
    record.attributes["asr_provider_used"] = provider_name

    # Store raw ASR output for potential multi-ASR refinement in Stage 5
    asr_outputs = record.attributes.get("asr_outputs", [])
    asr_outputs.append({
        "provider": provider_name,
        "text": result["transcript"],
        "confidence": result["confidence"],
        "language": result.get("language"),
        "word_timestamps": result.get("word_timestamps", []),
    })
    record.attributes["asr_outputs"] = asr_outputs

    record.providers["transcript_recovery"] = Provenance(
        stage="transcript_recovery",
        provider=provider_name,
        version="1.0.0",
        timestamp=time.time(),
        confidence=result["confidence"],
        metadata={
            "language": result.get("language"),
            "n_words": len(result.get("word_timestamps", [])),
            "n_segments": len(result.get("asr_segments", [])),
        },
    )

    return record
