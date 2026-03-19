"""Diarization provider adapters (Worker 08).

Mainline: pyannote/speaker-diarization-community-1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import CurationRecord, Provenance
from . import DiarizationProvider, ProviderOutput, ProviderUnavailableError

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A single diarized speaker segment."""

    speaker_id: str
    start_sec: float
    end_sec: float
    confidence: float = 0.0


class PyAnnoteDiarizationProvider(DiarizationProvider):
    """Diarization provider wrapping pyannote/speaker-diarization-community-1.

    Outputs: speaker segments with speaker_ids, timestamps, overlap
    regions, and estimated number of speakers.

    Stub mode: ``process()`` raises ``NotImplementedError`` when the
    pipeline is not loaded.
    """

    name = "pyannote_community"
    version = "1.0.0"

    artifact_id: str = "pyannote/speaker-diarization-community-1"
    runtime_backend: str = "pyannote.audio"
    calibration_version: str = "uncalibrated"

    def __init__(
        self,
        *,
        artifact_id: str = "pyannote/speaker-diarization-community-1",
        runtime_backend: str = "pyannote.audio",
        calibration_version: str = "uncalibrated",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> None:
        self.artifact_id = artifact_id
        self.runtime_backend = runtime_backend
        self.calibration_version = calibration_version
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

    def is_available(self) -> bool:
        try:
            import pyannote.audio  # noqa: F401
            return True
        except ImportError:
            return False

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        """Run diarization on the audio referenced by *record*.

        Returns a ``ProviderOutput`` with fields:
        - speaker_cluster: str  (primary speaker)
        - diarization_confidence: float  [0, 1]
        - attributes.speaker_turns: list of turn dicts
        - attributes.overlap_flags: list of overlap dicts
        - attributes.n_speakers_detected: int

        Raises:
            NotImplementedError: Pipeline is not loaded (stub mode).
        """
        if not self.is_available():
            raise ProviderUnavailableError(
                "pyannote.audio is required for PyAnnoteDiarizationProvider"
            )

        raise NotImplementedError(
            "PyAnnoteDiarizationProvider.process() requires pipeline "
            "initialization. Use within a configured curation pipeline."
        )

    # ------------------------------------------------------------------
    # Output construction helpers (used by real runtime)
    # ------------------------------------------------------------------

    def _build_output(
        self,
        segments: List[SpeakerSegment],
        *,
        n_speakers: int = 0,
        overlap_regions: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderOutput:
        """Assemble normalized ``ProviderOutput`` from diarization result."""
        if not segments:
            return ProviderOutput(
                fields={
                    "speaker_cluster": "spk_000",
                    "diarization_confidence": 0.0,
                    "attributes": {
                        "speaker_turns": [],
                        "overlap_flags": [],
                        "n_speakers_detected": 0,
                    },
                },
                confidence=0.0,
                warnings=["No speaker segments produced by diarization"],
                provenance=self.make_provenance(
                    confidence=0.0,
                    metadata={
                        "artifact_id": self.artifact_id,
                        "runtime_backend": self.runtime_backend,
                        "calibration_version": self.calibration_version,
                    },
                ),
            )

        # Build speaker turns
        speaker_turns: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            speaker_turns.append({
                "turn_index": i,
                "speaker": seg.speaker_id,
                "start_sec": round(seg.start_sec, 4),
                "end_sec": round(seg.end_sec, 4),
                "duration_sec": round(seg.end_sec - seg.start_sec, 4),
                "confidence": round(seg.confidence, 4),
            })

        # Primary speaker: most total duration
        speaker_durations: Dict[str, float] = {}
        for seg in segments:
            dur = seg.end_sec - seg.start_sec
            speaker_durations[seg.speaker_id] = (
                speaker_durations.get(seg.speaker_id, 0.0) + dur
            )
        primary_speaker = max(speaker_durations, key=speaker_durations.get)  # type: ignore[arg-type]

        # Aggregate confidence
        total_dur = sum(max(seg.end_sec - seg.start_sec, 0.001) for seg in segments)
        weighted_conf = sum(
            seg.confidence * max(seg.end_sec - seg.start_sec, 0.001)
            for seg in segments
        ) / (total_dur + 1e-12)
        confidence = round(max(0.0, min(1.0, weighted_conf)), 4)

        unique_speakers = set(seg.speaker_id for seg in segments)
        if n_speakers == 0:
            n_speakers = len(unique_speakers)

        return ProviderOutput(
            fields={
                "speaker_cluster": primary_speaker,
                "diarization_confidence": confidence,
                "attributes": {
                    "speaker_turns": speaker_turns,
                    "overlap_flags": overlap_regions or [],
                    "n_speakers_detected": n_speakers,
                    "diarization_provider_used": self.name,
                },
            },
            confidence=confidence,
            provenance=self.make_provenance(
                confidence=confidence,
                metadata={
                    "artifact_id": self.artifact_id,
                    "runtime_backend": self.runtime_backend,
                    "calibration_version": self.calibration_version,
                    "n_speakers": n_speakers,
                    "n_segments": len(segments),
                },
            ),
        )
