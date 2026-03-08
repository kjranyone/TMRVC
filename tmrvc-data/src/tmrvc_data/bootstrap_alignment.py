"""Bootstrap alignment projection (Worker 03).

Deterministically projects ASR word/token timestamps onto canonical
phoneme-index space and validates the result.

Frame convention (frozen):
    sample_rate = 24000
    hop_length = 240
    start_frame is inclusive
    end_frame is exclusive
    T = ceil(num_samples / 240)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Frozen frame convention
SAMPLE_RATE = 24000
HOP_LENGTH = 240


@dataclass
class AlignmentSpan:
    """A single aligned span mapping a phoneme index to frame range."""

    text_unit_index: int
    start_frame: int  # inclusive
    end_frame: int  # exclusive
    confidence: float = 1.0
    projection_source: str = "bootstrap"

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class BootstrapAlignment:
    """Canonical bootstrap alignment artifact.

    Contains the projected alignment from ASR timestamps into canonical
    phoneme-index space, with provenance metadata.
    """

    spans: list[AlignmentSpan] = field(default_factory=list)
    num_text_units: int = 0
    num_frames: int = 0
    provenance: str = "bootstrap_projection"
    projection_method: str = "uniform_within_word"

    def validate(self) -> list[str]:
        """Validate alignment integrity. Returns list of errors."""
        errors: list[str] = []
        if not self.spans:
            errors.append("alignment has no spans")
            return errors

        # Check monotonicity
        for i in range(1, len(self.spans)):
            prev = self.spans[i - 1]
            curr = self.spans[i]
            if curr.text_unit_index <= prev.text_unit_index:
                errors.append(
                    f"non-monotonic text_unit_index at span {i}: "
                    f"{prev.text_unit_index} -> {curr.text_unit_index}"
                )
            if curr.start_frame < prev.end_frame:
                errors.append(
                    f"overlapping frames at span {i}: "
                    f"prev.end={prev.end_frame}, curr.start={curr.start_frame}"
                )

        # Check no gaps in text_unit_index coverage
        covered_indices = {s.text_unit_index for s in self.spans}
        expected = set(range(self.num_text_units))
        missing = expected - covered_indices
        if missing:
            errors.append(f"missing text_unit_indices: {sorted(missing)}")

        extra = covered_indices - expected
        if extra:
            errors.append(f"extra text_unit_indices beyond num_text_units: {sorted(extra)}")

        # Check frame bounds
        for span in self.spans:
            if span.start_frame < 0:
                errors.append(f"negative start_frame at index {span.text_unit_index}")
            if span.end_frame > self.num_frames:
                errors.append(
                    f"end_frame {span.end_frame} exceeds num_frames {self.num_frames} "
                    f"at index {span.text_unit_index}"
                )
            if span.start_frame >= span.end_frame:
                errors.append(
                    f"zero-length span at index {span.text_unit_index}: "
                    f"[{span.start_frame}, {span.end_frame})"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "num_text_units": self.num_text_units,
            "num_frames": self.num_frames,
            "provenance": self.provenance,
            "projection_method": self.projection_method,
            "spans": [
                {
                    "text_unit_index": s.text_unit_index,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "confidence": s.confidence,
                    "projection_source": s.projection_source,
                }
                for s in self.spans
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapAlignment:
        """Deserialize from dict."""
        spans = [AlignmentSpan(**s) for s in data.get("spans", [])]
        return cls(
            spans=spans,
            num_text_units=data.get("num_text_units", len(spans)),
            num_frames=data.get("num_frames", 0),
            provenance=data.get("provenance", "unknown"),
            projection_method=data.get("projection_method", "unknown"),
        )


def samples_to_frames(num_samples: int) -> int:
    """Convert sample count to frame count using frozen convention."""
    return math.ceil(num_samples / HOP_LENGTH)


def seconds_to_frame(seconds: float) -> int:
    """Convert timestamp in seconds to frame index."""
    sample = int(seconds * SAMPLE_RATE)
    return sample // HOP_LENGTH


def project_word_timestamps(
    word_timestamps: list[dict[str, Any]],
    phoneme_ids: list[int],
    word_to_phoneme_map: list[tuple[int, int]],
    num_samples: int,
    confidence: float = 0.5,
) -> BootstrapAlignment:
    """Project ASR word-level timestamps onto canonical phoneme indices.

    Args:
        word_timestamps: List of {"word": str, "start": float, "end": float}
            from ASR output.
        phoneme_ids: Canonical phoneme ID sequence [L].
        word_to_phoneme_map: List of (start_phoneme_idx, end_phoneme_idx)
            mapping each word to its phoneme span (exclusive end).
        num_samples: Total audio length in samples.
        confidence: Default confidence for projected spans.

    Returns:
        BootstrapAlignment with validated spans.
    """
    num_frames = samples_to_frames(num_samples)
    num_text_units = len(phoneme_ids)
    spans: list[AlignmentSpan] = []

    for word_idx, (word_ts, (ph_start, ph_end)) in enumerate(
        zip(word_timestamps, word_to_phoneme_map)
    ):
        word_start_frame = seconds_to_frame(word_ts["start"])
        word_end_frame = seconds_to_frame(word_ts["end"])
        # Clamp to valid range
        word_start_frame = max(0, min(word_start_frame, num_frames))
        word_end_frame = max(word_start_frame, min(word_end_frame, num_frames))

        n_phones = ph_end - ph_start
        if n_phones <= 0:
            continue

        word_duration = word_end_frame - word_start_frame
        if word_duration <= 0:
            # Assign at least 1 frame per phone
            word_end_frame = min(word_start_frame + n_phones, num_frames)
            word_duration = word_end_frame - word_start_frame

        # Uniform distribution within word (baseline projection)
        for i in range(n_phones):
            ph_idx = ph_start + i
            if ph_idx >= num_text_units:
                break
            sf = word_start_frame + (i * word_duration) // n_phones
            ef = word_start_frame + ((i + 1) * word_duration) // n_phones
            # Ensure at least 1 frame
            if ef <= sf:
                ef = sf + 1
            ef = min(ef, num_frames)
            spans.append(
                AlignmentSpan(
                    text_unit_index=ph_idx,
                    start_frame=sf,
                    end_frame=ef,
                    confidence=confidence,
                    projection_source="word_timestamp_uniform",
                )
            )

    alignment = BootstrapAlignment(
        spans=spans,
        num_text_units=num_text_units,
        num_frames=num_frames,
        provenance="asr_word_timestamp",
        projection_method="uniform_within_word",
    )
    return alignment
