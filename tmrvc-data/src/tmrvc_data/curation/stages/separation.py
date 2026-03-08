"""Stage 2: Separation / Enhancement - Source separation for mixed audio.

For mixed long-form audio, runs separation providers to improve annotation
quality. Separation outputs are annotation aids, NOT automatic waveform
teachers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from ..models import CurationRecord, Provenance, RecordStatus
from ..providers import (
    BaseProvider,
    ProviderOutput,
    ProviderRegistry,
    SeparationProvider,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristics for deciding whether separation is needed
# ---------------------------------------------------------------------------
MULTI_SPEAKER_ENERGY_VARIANCE_THRESHOLD = 0.15  # high variance hints at mixed audio
CLEAN_SPEECH_RATIO_THRESHOLD = 0.85  # if speech ratio is high, likely clean


def _estimate_spectral_flatness(audio: np.ndarray, sr: int) -> float:
    """Estimate spectral flatness as a rough noise/mixture indicator.

    Values closer to 1.0 suggest noise-like (flat) spectrum; closer to 0.0
    suggest tonal/clean speech.
    """
    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return 0.5

    # Use a single window for a quick estimate
    windowed = audio[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum = spectrum[1:]  # drop DC
    spectrum = np.maximum(spectrum, 1e-12)

    geo_mean = np.exp(np.mean(np.log(spectrum)))
    arith_mean = np.mean(spectrum)
    flatness = geo_mean / (arith_mean + 1e-12)
    return float(np.clip(flatness, 0.0, 1.0))


def _is_likely_clean(record: CurationRecord) -> bool:
    """Heuristic: skip separation if audio appears to be clean single-speaker."""
    attrs = record.attributes
    speech_ratio = attrs.get("speech_ratio", 0.0)
    if speech_ratio >= CLEAN_SPEECH_RATIO_THRESHOLD:
        return True
    return False


def run_separation(
    record: CurationRecord,
    registry: Optional[ProviderRegistry] = None,
) -> Optional[CurationRecord]:
    """Process a single record through Stage 2: Separation / Enhancement.

    If a SeparationProvider is available in the registry, delegates to it.
    Otherwise, performs a lightweight spectral analysis to annotate the
    record with mixture indicators without modifying the waveform.

    Args:
        record: The curation record to process.
        registry: Optional provider registry for external separation models.

    Returns:
        Updated CurationRecord.
    """
    # Skip records that have already been rejected
    if record.status == RecordStatus.REJECTED:
        return record

    # Skip if audio is likely clean single-speaker
    if _is_likely_clean(record):
        record.attributes["separation_skipped"] = True
        record.attributes["separation_skip_reason"] = "clean_single_speaker"
        record.attributes["separation_confidence"] = 1.0
        record.attributes["artifact_notes"] = []
        record.providers["separation"] = Provenance(
            stage="separation",
            provider="builtin_heuristic",
            version="1.0.0",
            timestamp=time.time(),
            confidence=1.0,
            metadata={"skipped": True, "reason": "clean_single_speaker"},
        )
        return record

    # --- Try external provider ---
    if registry is not None:
        provider = registry.get_primary("separation")
        if provider is not None and provider.is_available():
            try:
                output = provider.process(record)
                # Merge provider output fields into record
                for key, value in output.fields.items():
                    if key == "attributes" and isinstance(value, dict):
                        record.attributes.update(value)
                    else:
                        setattr(record, key, value)
                record.attributes["separation_confidence"] = output.confidence
                if output.warnings:
                    record.attributes["artifact_notes"] = output.warnings
                if output.provenance:
                    record.providers["separation"] = output.provenance
                else:
                    record.providers["separation"] = Provenance(
                        stage="separation",
                        provider=provider.name,
                        version=provider.version,
                        timestamp=time.time(),
                        confidence=output.confidence,
                    )
                return record
            except Exception as e:
                logger.warning(
                    "Separation provider %s failed for %s: %s",
                    provider.name, record.record_id, e,
                )
                # Fall through to builtin analysis

    # --- Builtin spectral analysis (no waveform modification) ---
    try:
        info = sf.info(record.source_path)
        sr = info.samplerate
        start_frame = int(record.segment_start_sec * sr)
        n_frames = int((record.segment_end_sec - record.segment_start_sec) * sr)
        audio, _ = sf.read(
            record.source_path, start=start_frame, frames=n_frames,
            dtype="float32", always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        logger.warning("Separation: cannot load %s: %s", record.record_id, e)
        record.attributes["separation_confidence"] = 0.0
        record.attributes["artifact_notes"] = [f"load_error: {e}"]
        record.providers["separation"] = Provenance(
            stage="separation",
            provider="builtin_spectral",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata={"error": str(e)},
        )
        return record

    flatness = _estimate_spectral_flatness(audio, sr)
    record.attributes["spectral_flatness"] = round(flatness, 4)

    # Estimate energy variance across short windows as a mixture indicator
    win_samples = int(0.5 * sr)  # 500 ms windows
    if len(audio) > win_samples:
        n_wins = len(audio) // win_samples
        energies = []
        for i in range(n_wins):
            chunk = audio[i * win_samples: (i + 1) * win_samples]
            energies.append(float(np.mean(chunk ** 2)))
        energy_var = float(np.std(energies) / (np.mean(energies) + 1e-12))
    else:
        energy_var = 0.0

    record.attributes["energy_variance"] = round(energy_var, 4)

    # Heuristic confidence: low flatness + low variance = likely clean
    confidence = max(0.0, min(1.0, 1.0 - flatness * 0.5 - energy_var * 0.5))
    record.attributes["separation_confidence"] = round(confidence, 4)

    artifact_notes = []
    if flatness > 0.5:
        artifact_notes.append("high_spectral_flatness_possible_noise")
    if energy_var > MULTI_SPEAKER_ENERGY_VARIANCE_THRESHOLD:
        artifact_notes.append("high_energy_variance_possible_mixture")
    record.attributes["artifact_notes"] = artifact_notes

    # Mark as annotation aid only
    record.attributes["separation_skipped"] = True
    record.attributes["separation_skip_reason"] = "no_provider_builtin_analysis_only"

    record.providers["separation"] = Provenance(
        stage="separation",
        provider="builtin_spectral",
        version="1.0.0",
        timestamp=time.time(),
        confidence=confidence,
        metadata={
            "spectral_flatness": record.attributes["spectral_flatness"],
            "energy_variance": record.attributes["energy_variance"],
            "method": "annotation_aid_only",
        },
    )

    return record
