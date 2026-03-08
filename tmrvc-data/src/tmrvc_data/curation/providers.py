"""Provider abstraction layer for AI Curation System.

Each curation stage (ASR, diarization, separation, etc.) can have multiple
provider implementations. This module defines the base interface and registry.
"""
from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .models import CurationRecord, Provenance

logger = logging.getLogger(__name__)


class ProviderUnavailableError(Exception):
    """Raised when a provider's dependencies are not installed."""


@dataclass
class ProviderOutput:
    """Normalized output from a provider."""
    fields: Dict[str, Any]  # Fields to merge into CurationRecord
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None


class BaseProvider(abc.ABC):
    """Base class for all curation providers."""

    name: str = "base"
    version: str = "0.0.0"
    stage: str = "unknown"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if provider dependencies are installed."""
        ...

    @abc.abstractmethod
    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        """Process a single record and return normalized output."""
        ...

    def make_provenance(self, confidence: float | None = None,
                        metadata: dict | None = None) -> Provenance:
        return Provenance(
            stage=self.stage,
            provider=self.name,
            version=self.version,
            timestamp=time.time(),
            confidence=confidence,
            metadata=metadata or {},
        )


class ASRProvider(BaseProvider):
    """Base class for ASR providers."""
    stage = "asr"


class DiarizationProvider(BaseProvider):
    """Base class for diarization providers."""
    stage = "diarization"


class SeparationProvider(BaseProvider):
    """Base class for separation/enhancement providers."""
    stage = "separation"


class SpeakerClusteringProvider(BaseProvider):
    """Base class for cross-file speaker clustering."""
    stage = "speaker_clustering"


class EventExtractionProvider(BaseProvider):
    """Base class for prosody/event extraction."""
    stage = "event_extraction"


class TranscriptRefinementProvider(BaseProvider):
    """Base class for transcript refinement."""
    stage = "transcript_refinement"


class VoiceStateEstimationProvider(BaseProvider):
    """Base class for 8-D voice state estimation."""
    stage = "voice_state_estimation"


class ProviderRegistry:
    """Registry for provider implementations per stage."""

    def __init__(self) -> None:
        self._providers: Dict[str, List[BaseProvider]] = {}

    def register(self, provider: BaseProvider) -> None:
        stage = provider.stage
        if stage not in self._providers:
            self._providers[stage] = []
        self._providers[stage].append(provider)
        logger.info("Registered provider: %s/%s v%s",
                     stage, provider.name, provider.version)

    def get_providers(self, stage: str) -> List[BaseProvider]:
        return self._providers.get(stage, [])

    def get_available(self, stage: str) -> List[BaseProvider]:
        return [p for p in self.get_providers(stage) if p.is_available()]

    def get_primary(self, stage: str) -> Optional[BaseProvider]:
        available = self.get_available(stage)
        return available[0] if available else None

    def get_fallback(self, stage: str) -> Optional[BaseProvider]:
        available = self.get_available(stage)
        return available[1] if len(available) > 1 else None


# -- Concrete stub providers ------------------------------------------------


class FasterWhisperASR(ASRProvider):
    """ASR provider using faster-whisper."""
    name = "faster_whisper"
    version = "1.0.0"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        if not self.is_available():
            raise ProviderUnavailableError("faster-whisper not installed")
        raise NotImplementedError(
            "FasterWhisperASR.process() requires model initialization. "
            "Use within a configured curation pipeline."
        )


class PyannoteDializer(DiarizationProvider):
    """Diarization provider using pyannote.audio."""
    name = "pyannote"
    version = "1.0.0"

    def is_available(self) -> bool:
        try:
            import pyannote.audio  # noqa: F401
            return True
        except ImportError:
            return False

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        if not self.is_available():
            raise ProviderUnavailableError("pyannote.audio not installed")
        raise NotImplementedError(
            "PyannoteDializer.process() requires pipeline initialization."
        )


class TranscriptRefiner(TranscriptRefinementProvider):
    """Multi-ASR transcript refinement engine.

    Compares outputs from multiple ASR providers, computes agreement,
    and produces a refined transcript with disagreement metrics.
    """
    name = "multi_asr_refiner"
    version = "1.0.0"

    def is_available(self) -> bool:
        return True  # Pure logic, no external deps

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        """Refine transcript using multiple ASR outputs.

        Expects kwargs:
            asr_outputs: list of dicts with 'text' and 'confidence' keys
        """
        asr_outputs = kwargs.get("asr_outputs", [])
        if not asr_outputs:
            return ProviderOutput(
                fields={},
                confidence=0.0,
                warnings=["No ASR outputs provided for refinement"],
                provenance=self.make_provenance(confidence=0.0),
            )

        # Simple majority vote / highest confidence selection
        best = max(asr_outputs, key=lambda x: x.get("confidence", 0.0))

        # Compute agreement ratio
        texts = [o.get("text", "") for o in asr_outputs]
        agreement = sum(1 for t in texts if t == best["text"]) / len(texts)

        return ProviderOutput(
            fields={
                "transcript": best["text"],
                "transcript_confidence": best.get("confidence", 0.0),
                "attributes": {
                    "refinement_agreement": round(agreement, 4),
                    "refinement_n_sources": len(asr_outputs),
                },
            },
            confidence=agreement,
            provenance=self.make_provenance(
                confidence=agreement,
                metadata={"n_sources": len(asr_outputs), "method": "majority_vote"},
            ),
        )


def create_default_registry() -> ProviderRegistry:
    """Create a registry with all known providers."""
    registry = ProviderRegistry()
    registry.register(FasterWhisperASR())
    registry.register(PyannoteDializer())
    registry.register(TranscriptRefiner())
    return registry
