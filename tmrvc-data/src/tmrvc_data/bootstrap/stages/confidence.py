"""Stage 11: Confidence scoring and supervision tier classification.

Aggregates per-field confidence scores and classifies each utterance
into a supervision tier (A/B/C/D).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from tmrvc_data.bootstrap.contracts import BootstrapConfig, BootstrapStage, BootstrapUtterance
from tmrvc_data.bootstrap.supervision import SupervisionTierClassifier

logger = logging.getLogger(__name__)


class ConfidenceStage:
    """Aggregate confidence scores and assign supervision tiers."""

    def __init__(self, config: BootstrapConfig) -> None:
        self.config = config
        self.tier_classifier = SupervisionTierClassifier()

    def process(self, utterances: List[BootstrapUtterance]) -> List[BootstrapUtterance]:
        """Score confidence and assign tiers for all utterances."""
        active = [u for u in utterances if not u.is_rejected]

        for utt in active:
            self._score(utt)
            utt.stage_completed = BootstrapStage.CONFIDENCE_SCORING

        # Log tier distribution
        tier_counts: dict[str, int] = {}
        for utt in active:
            tier_counts[utt.supervision_tier] = tier_counts.get(utt.supervision_tier, 0) + 1
        logger.info("Tier distribution: %s", tier_counts)

        return utterances

    def _score(self, utt: BootstrapUtterance) -> None:
        """Score a single utterance."""
        has_semantic = bool(
            utt.acting_annotations.get("scene_summary")
            or utt.acting_annotations.get("emotion_description")
        )

        utt.supervision_tier = self.tier_classifier.classify(
            transcript_confidence=utt.transcript_confidence,
            diarization_confidence=utt.diarization_confidence,
            physical_observed_mask=utt.physical_observed_mask,
            physical_confidence=utt.physical_confidence,
            has_semantic_annotations=has_semantic,
        )

        # Composite quality score
        phys_coverage = 0.0
        if utt.physical_observed_mask is not None:
            phys_coverage = float(np.mean(utt.physical_observed_mask))

        utt.quality_score = (
            0.3 * utt.transcript_confidence
            + 0.3 * utt.diarization_confidence
            + 0.2 * phys_coverage
            + 0.2 * (1.0 if has_semantic else 0.0)
        )
