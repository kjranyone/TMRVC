"""Dataset supervision report (Worker 03).

Defines the canonical DatasetReport dataclass and generation utilities
per the Worker 03 Dataset Report Specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tmrvc_data.code_switch import code_switch_ratio


@dataclass
class DatasetReport:
    """Machine-readable supervision and normalization report per dataset.

    All fields from Worker 03 Dataset Report Specification are required.
    """

    # Identity
    dataset_name: str = ""
    split: str = "train"

    # Basic counts
    num_utterances: int = 0

    # Text supervision
    text_supervision_coverage: float = 0.0
    canonical_text_unit_coverage: float = 0.0
    legacy_duration_coverage: float = 0.0

    # Phone normalization quality
    unknown_phone_ratio: float = 0.0
    direct_hit_ratio: float = 0.0
    alias_hit_ratio: float = 0.0
    active_phone_inventory: list[str] = field(default_factory=list)
    unmapped_phone_counts: dict[str, int] = field(default_factory=dict)
    top_unmapped_examples: list[tuple[str, int]] = field(default_factory=list)
    per_language_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Expressive readiness
    dialogue_context_coverage: float = 0.0
    same_text_multi_context_coverage: float = 0.0
    code_switch_coverage: float = 0.0
    cross_lingual_prompt_coverage: float = 0.0
    g2p_fallback_coverage: float = 0.0

    # Voice state supervision
    voice_state_supervision_coverage: float = 0.0
    voice_state_observed_ratio: float = 0.0
    voice_state_confidence_summary: dict[str, float] = field(default_factory=dict)

    # Few-shot prompt pairing (Worker 03)
    prompt_pairing_coverage: float = 0.0
    prompt_leakage_risk_count: int = 0

    # Suprasegmental and alignment coverage (Worker 03)
    suprasegmental_coverage: float = 0.0
    bootstrap_alignment_coverage: float = 0.0

    # Curation asset consumption (Worker 03)
    curation_record_coverage: float = 0.0

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.dataset_name:
            errors.append("dataset_name is empty")
        if self.num_utterances < 0:
            errors.append("num_utterances is negative")
        for name in (
            "text_supervision_coverage",
            "canonical_text_unit_coverage",
            "legacy_duration_coverage",
            "unknown_phone_ratio",
            "direct_hit_ratio",
            "alias_hit_ratio",
            "dialogue_context_coverage",
            "voice_state_supervision_coverage",
            "prompt_pairing_coverage",
            "suprasegmental_coverage",
            "bootstrap_alignment_coverage",
            "curation_record_coverage",
        ):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} out of [0,1]: {val}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for JSON export."""
        import dataclasses
        return dataclasses.asdict(self)

    def compute_code_switch_coverage(
        self,
        texts: list[str],
        primary_language: str = "ja",
    ) -> float:
        """Compute and update code_switch_coverage from a list of utterance texts.

        Uses Unicode script detection to identify language switches within
        each text. Updates self.code_switch_coverage in place.

        Args:
            texts: List of utterance text strings from the dataset.
            primary_language: Expected primary language of the dataset.

        Returns:
            The computed code_switch_coverage ratio.
        """
        self.code_switch_coverage = code_switch_ratio(texts, primary_language)
        return self.code_switch_coverage


# All fields required by the Worker 03 spec
REQUIRED_REPORT_FIELDS = frozenset({
    "dataset_name",
    "split",
    "num_utterances",
    "text_supervision_coverage",
    "canonical_text_unit_coverage",
    "legacy_duration_coverage",
    "unknown_phone_ratio",
    "direct_hit_ratio",
    "alias_hit_ratio",
    "active_phone_inventory",
    "unmapped_phone_counts",
    "top_unmapped_examples",
    "per_language_stats",
    "dialogue_context_coverage",
    "same_text_multi_context_coverage",
    "code_switch_coverage",
    "cross_lingual_prompt_coverage",
    "g2p_fallback_coverage",
    "voice_state_supervision_coverage",
    "voice_state_observed_ratio",
    "voice_state_confidence_summary",
    "prompt_pairing_coverage",
    "prompt_leakage_risk_count",
    "suprasegmental_coverage",
    "bootstrap_alignment_coverage",
    "curation_record_coverage",
})
