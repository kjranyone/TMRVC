"""ASR provider adapters (Worker 08).

Mainline: Qwen3-ASR-1.7B
Throughput fallback: faster-whisper (already in ``..providers.FasterWhisperASR``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import CurationRecord, Provenance
from . import ASRProvider, ProviderOutput, ProviderUnavailableError

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """A single ASR output segment with word-level detail."""

    text: str
    start_sec: float
    end_sec: float
    confidence: float = 0.0
    language: Optional[str] = None
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)


class Qwen3ASRProvider(ASRProvider):
    """ASR provider wrapping Qwen3-ASR-1.7B.

    This is the mainline ASR provider for the curation pipeline.
    Outputs: transcript, word timestamps, confidence per segment,
    detected language.

    The actual model loading is deferred -- this class defines the
    interface contract and confidence normalization logic.  In stub
    mode (model not loaded), ``process()`` raises ``NotImplementedError``.
    """

    name = "qwen3_asr"
    version = "1.7b-v1"

    # Provider metadata for registry pinning
    artifact_id: str = "Qwen/Qwen3-ASR-1.7B"
    runtime_backend: str = "transformers"
    calibration_version: str = "uncalibrated"
    supported_languages: List[str] = field(default_factory=lambda: [
        "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar",
    ])  # type: ignore[assignment]

    def __init__(
        self,
        *,
        artifact_id: str = "Qwen/Qwen3-ASR-1.7B",
        runtime_backend: str = "transformers",
        calibration_version: str = "uncalibrated",
    ) -> None:
        self.artifact_id = artifact_id
        self.runtime_backend = runtime_backend
        self.calibration_version = calibration_version
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        """Check if transformers and the model weights are accessible."""
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        """Run ASR on the audio referenced by *record*.

        Returns a ``ProviderOutput`` with fields:
        - transcript: str
        - transcript_confidence: float  (normalized [0, 1])
        - language: str
        - attributes.word_timestamps: list of {word, start, end, probability}
        - attributes.asr_segments: list of segment dicts

        Raises:
            NotImplementedError: Model is not loaded (stub mode).
            ProviderUnavailableError: Required packages not installed.
        """
        if not self.is_available():
            raise ProviderUnavailableError(
                "transformers is required for Qwen3-ASR-1.7B"
            )

        # --- Stub: real implementation would load model and run inference ---
        raise NotImplementedError(
            "Qwen3ASRProvider.process() requires model initialization. "
            "Provide a loaded model via the curation pipeline runtime."
        )

    # ------------------------------------------------------------------
    # Confidence normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_confidence(raw_log_prob: float) -> float:
        """Convert raw log-probability to [0, 1] confidence.

        Qwen3-ASR emits average log-probabilities per segment.  We map
        them through ``exp()`` and clamp to [0, 1].
        """
        import math

        return max(0.0, min(1.0, math.exp(raw_log_prob)))

    # ------------------------------------------------------------------
    # Output construction helpers (used by real runtime)
    # ------------------------------------------------------------------

    def _build_output(
        self,
        segments: List[ASRSegment],
        detected_language: Optional[str],
    ) -> ProviderOutput:
        """Assemble normalized ``ProviderOutput`` from decoded segments."""
        if not segments:
            return ProviderOutput(
                fields={
                    "transcript": "",
                    "transcript_confidence": 0.0,
                    "language": detected_language,
                },
                confidence=0.0,
                warnings=["No segments produced by ASR"],
                provenance=self.make_provenance(
                    confidence=0.0,
                    metadata={
                        "artifact_id": self.artifact_id,
                        "runtime_backend": self.runtime_backend,
                        "calibration_version": self.calibration_version,
                    },
                ),
            )

        full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        # Aggregate confidence: weighted average by segment duration
        total_dur = sum(max(seg.end_sec - seg.start_sec, 0.001) for seg in segments)
        weighted_conf = sum(
            seg.confidence * max(seg.end_sec - seg.start_sec, 0.001)
            for seg in segments
        ) / (total_dur + 1e-12)
        confidence = round(max(0.0, min(1.0, weighted_conf)), 4)

        word_timestamps: List[Dict[str, Any]] = []
        asr_segments: List[Dict[str, Any]] = []
        for seg in segments:
            asr_segments.append({
                "text": seg.text.strip(),
                "start": round(seg.start_sec, 4),
                "end": round(seg.end_sec, 4),
                "confidence": round(seg.confidence, 4),
                "language": seg.language or detected_language,
            })
            word_timestamps.extend(seg.word_timestamps)

        return ProviderOutput(
            fields={
                "transcript": full_text,
                "transcript_confidence": confidence,
                "language": detected_language,
                "attributes": {
                    "word_timestamps": word_timestamps,
                    "asr_segments": asr_segments,
                    "asr_provider_used": self.name,
                },
            },
            confidence=confidence,
            provenance=self.make_provenance(
                confidence=confidence,
                metadata={
                    "artifact_id": self.artifact_id,
                    "runtime_backend": self.runtime_backend,
                    "calibration_version": self.calibration_version,
                    "n_segments": len(segments),
                    "n_words": len(word_timestamps),
                    "detected_language": detected_language,
                },
            ),
        )
