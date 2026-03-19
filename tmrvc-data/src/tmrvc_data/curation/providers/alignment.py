"""Forced alignment provider adapters (Worker 08).

Mainline: Qwen3-ForcedAligner-0.6B
Fallback: none (unsupported languages emit explicit absence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import CurationRecord, Provenance
from . import BaseProvider, ProviderOutput, ProviderUnavailableError

logger = logging.getLogger(__name__)

# Canonical frame convention for alignment output
CANONICAL_HOP_LENGTH = 240
CANONICAL_SAMPLE_RATE = 24000
CANONICAL_FRAME_SHIFT_SEC = CANONICAL_HOP_LENGTH / CANONICAL_SAMPLE_RATE  # 0.01s


@dataclass
class PhonemeAlignment:
    """A single phoneme with its time boundaries."""

    phoneme: str
    start_sec: float
    end_sec: float
    start_frame: int  # in canonical frame convention
    end_frame: int
    confidence: float = 0.0


class AlignmentProvider(BaseProvider):
    """Base class for forced alignment providers."""

    stage = "alignment"


class Qwen3AlignerProvider(AlignmentProvider):
    """Forced alignment provider wrapping Qwen3-ForcedAligner-0.6B.

    Outputs phoneme-level timestamps projected to the canonical frame
    convention (hop_length=240, sample_rate=24000).

    Stub mode: ``process()`` raises ``NotImplementedError`` when the
    model is not loaded.
    """

    name = "qwen3_aligner"
    version = "0.6b-v1"

    artifact_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    runtime_backend: str = "transformers"
    calibration_version: str = "uncalibrated"

    def __init__(
        self,
        *,
        artifact_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        runtime_backend: str = "transformers",
        calibration_version: str = "uncalibrated",
    ) -> None:
        self.artifact_id = artifact_id
        self.runtime_backend = runtime_backend
        self.calibration_version = calibration_version
        self._model = None

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        """Run forced alignment on the audio + transcript in *record*.

        Expects ``record.transcript`` to be set (from ASR stage).

        Returns a ``ProviderOutput`` with fields:
        - attributes.phoneme_alignments: list of phoneme dicts
        - attributes.alignment_frame_shift_sec: float (canonical)
        - attributes.alignment_hop_length: int (canonical)
        - attributes.alignment_sample_rate: int (canonical)

        Raises:
            NotImplementedError: Model is not loaded (stub mode).
        """
        if not self.is_available():
            raise ProviderUnavailableError(
                "transformers is required for Qwen3-ForcedAligner-0.6B"
            )

        raise NotImplementedError(
            "Qwen3AlignerProvider.process() requires model initialization. "
            "Provide a loaded model via the curation pipeline runtime."
        )

    # ------------------------------------------------------------------
    # Frame projection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def seconds_to_canonical_frame(time_sec: float) -> int:
        """Project a timestamp to the canonical frame index.

        Uses hop_length=240 at sample_rate=24000 (10 ms per frame).
        """
        return int(round(time_sec / CANONICAL_FRAME_SHIFT_SEC))

    @staticmethod
    def canonical_frame_to_seconds(frame_idx: int) -> float:
        """Convert a canonical frame index back to seconds."""
        return frame_idx * CANONICAL_FRAME_SHIFT_SEC

    # ------------------------------------------------------------------
    # Output construction helpers
    # ------------------------------------------------------------------

    def _build_output(
        self,
        phonemes: List[PhonemeAlignment],
        *,
        language: Optional[str] = None,
    ) -> ProviderOutput:
        """Assemble normalized ``ProviderOutput`` from alignment result."""
        if not phonemes:
            return ProviderOutput(
                fields={
                    "attributes": {
                        "phoneme_alignments": [],
                        "alignment_frame_shift_sec": CANONICAL_FRAME_SHIFT_SEC,
                        "alignment_hop_length": CANONICAL_HOP_LENGTH,
                        "alignment_sample_rate": CANONICAL_SAMPLE_RATE,
                        "alignment_provider_used": self.name,
                    },
                },
                confidence=0.0,
                warnings=["No phoneme alignments produced"],
                provenance=self.make_provenance(
                    confidence=0.0,
                    metadata={
                        "artifact_id": self.artifact_id,
                        "runtime_backend": self.runtime_backend,
                        "calibration_version": self.calibration_version,
                        "language": language,
                    },
                ),
            )

        phoneme_dicts: List[Dict[str, Any]] = []
        for p in phonemes:
            phoneme_dicts.append({
                "phoneme": p.phoneme,
                "start_sec": round(p.start_sec, 4),
                "end_sec": round(p.end_sec, 4),
                "start_frame": p.start_frame,
                "end_frame": p.end_frame,
                "confidence": round(p.confidence, 4),
            })

        avg_conf = sum(p.confidence for p in phonemes) / len(phonemes)
        confidence = round(max(0.0, min(1.0, avg_conf)), 4)

        return ProviderOutput(
            fields={
                "attributes": {
                    "phoneme_alignments": phoneme_dicts,
                    "alignment_frame_shift_sec": CANONICAL_FRAME_SHIFT_SEC,
                    "alignment_hop_length": CANONICAL_HOP_LENGTH,
                    "alignment_sample_rate": CANONICAL_SAMPLE_RATE,
                    "alignment_provider_used": self.name,
                    "n_phonemes": len(phonemes),
                },
            },
            confidence=confidence,
            provenance=self.make_provenance(
                confidence=confidence,
                metadata={
                    "artifact_id": self.artifact_id,
                    "runtime_backend": self.runtime_backend,
                    "calibration_version": self.calibration_version,
                    "language": language,
                    "n_phonemes": len(phonemes),
                },
            ),
        )


def project_word_timestamps_acoustic(
    word_timestamps: List[Dict[str, Any]],
    phoneme_ids: List[int],
    word_to_phoneme_map: List[List[int]],
    num_samples: int,
    energy_flux: Optional[torch.Tensor] = None,
) -> BootstrapAlignment:
    """Refine coarse word timestamps into sharp phoneme boundaries (Worker 10).
    
    Uses Energy Delta / Spectral Flux to 'snap' word boundaries to acoustic edges.
    """
    import torch
    import numpy as np
    
    total_frames = int(np.ceil(num_samples / CANONICAL_HOP_LENGTH))
    phoneme_indices = torch.zeros(total_frames, dtype=torch.long)
    
    # 1. Map words to frame spans
    for word_idx, word in enumerate(word_timestamps):
        start_f = int(round(word["start"] / CANONICAL_FRAME_SHIFT_SEC))
        end_f = int(round(word["end"] / CANONICAL_FRAME_SHIFT_SEC))
        
        # 2. Boundary Refinement: Search for acoustic edge near word boundaries
        if energy_flux is not None:
            search_win = 5 # +/- 50ms
            
            # Snap start boundary to local flux peak
            s_min = max(0, start_f - search_win)
            s_max = min(total_frames - 1, start_f + search_win)
            if s_max > s_min:
                start_f = s_min + int(torch.argmax(energy_flux[s_min:s_max]))
                
            # Snap end boundary to local flux peak
            e_min = max(0, end_f - search_win)
            e_max = min(total_frames - 1, end_f + search_win)
            if e_max > e_min:
                end_f = e_min + int(torch.argmax(energy_flux[e_min:e_max]))

        # 3. Distribute phonemes uniformly within refined word span
        # (Fine-grained phoneme alignment is handled by internal MAS later)
        p_indices = word_to_phoneme_map[word_idx]
        if not p_indices: continue
        
        span_len = end_f - start_f
        for i, p_idx in enumerate(p_indices):
            p_start = start_f + int(i * span_len / len(p_indices))
            p_end = start_f + int((i + 1) * span_len / len(p_indices))
            phoneme_indices[p_start:p_end] = p_idx

    return BootstrapAlignment(phoneme_indices=phoneme_indices.tolist())


@dataclass
class BootstrapAlignment:
    """Container for projected bootstrap alignment data."""
    phoneme_indices: List[int]
    
    def validate(self) -> List[str]:
        if not self.phoneme_indices:
            return ["Empty alignment indices"]
        return []
        
    def to_dict(self) -> Dict[str, Any]:
        return {"phoneme_indices": self.phoneme_indices}
