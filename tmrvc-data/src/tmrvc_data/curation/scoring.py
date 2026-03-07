"""Quality scoring, rejection, review, and promotion for curated records.

Implements Worker 09: turns raw provider outputs into actionable decisions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import CurationRecord, RecordStatus, PromotionBucket, LegalityStatus

logger = logging.getLogger(__name__)


@dataclass
class BucketThresholds:
    """Promotion thresholds for a single bucket."""

    transcript_confidence: float = 0.90
    cross_asr_agreement: float = 0.85
    diarization_confidence: float = 0.80
    max_overlap_ratio: float = 0.10
    quality_score: float = 0.85
    allowed_legality: Tuple[str, ...] = ("owned", "licensed")


# Default thresholds from plan/ai_curation_system.md
DEFAULT_THRESHOLDS: Dict[str, BucketThresholds] = {
    "tts_mainline": BucketThresholds(
        transcript_confidence=0.90,
        cross_asr_agreement=0.85,
        diarization_confidence=0.80,
        max_overlap_ratio=0.10,
        quality_score=0.85,
        allowed_legality=("owned", "licensed"),
    ),
    "vc_prior": BucketThresholds(
        transcript_confidence=0.60,
        cross_asr_agreement=0.0,
        diarization_confidence=0.0,
        max_overlap_ratio=0.30,
        quality_score=0.70,
        allowed_legality=("owned", "licensed", "research-restricted"),
    ),
    "expressive_prior": BucketThresholds(
        transcript_confidence=0.50,
        cross_asr_agreement=0.0,
        diarization_confidence=0.0,
        max_overlap_ratio=0.20,
        quality_score=0.75,
        allowed_legality=("owned", "licensed", "research-restricted"),
    ),
    "holdout_eval": BucketThresholds(
        transcript_confidence=0.90,
        cross_asr_agreement=0.85,
        diarization_confidence=0.80,
        max_overlap_ratio=0.05,
        quality_score=0.90,
        allowed_legality=("owned", "licensed"),
    ),
}


@dataclass
class ScoringConfig:
    """Configuration for the quality scoring engine."""

    # Score weights
    w_transcript: float = 0.30
    w_asr_agreement: float = 0.15
    w_diarization: float = 0.15
    w_snr: float = 0.15
    w_duration_sanity: float = 0.10
    w_event_completeness: float = 0.10
    w_language_consistency: float = 0.05

    # Bucket thresholds
    thresholds: Dict[str, BucketThresholds] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )

    # Holdout set
    holdout_record_ids: Set[str] = field(default_factory=set)


class QualityScoringEngine:
    """Computes quality scores and makes promotion/rejection decisions."""

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        self.config = config or ScoringConfig()

    def compute_score(self, record: CurationRecord) -> float:
        """Compute composite quality score for a record."""
        cfg = self.config
        attrs = record.attributes

        t_conf = record.transcript_confidence or 0.0
        asr_agree = attrs.get("refinement_agreement", t_conf)
        diar_conf = record.diarization_confidence or 0.0
        snr = attrs.get("snr_db", 0.0)
        snr_norm = min(snr / 40.0, 1.0) if snr > 0 else 0.0

        # Duration sanity: 0.3-30s is ideal
        dur = record.duration_sec
        dur_score = (
            1.0
            if 0.3 <= dur <= 30.0
            else max(0.0, 1.0 - abs(dur - 15.0) / 30.0)
        )

        # Event completeness
        has_events = (
            1.0
            if attrs.get("pause_events") or attrs.get("breath_events")
            else 0.0
        )

        # Language consistency
        lang_ok = 1.0 if record.language else 0.0

        score = (
            cfg.w_transcript * t_conf
            + cfg.w_asr_agreement * asr_agree
            + cfg.w_diarization * diar_conf
            + cfg.w_snr * snr_norm
            + cfg.w_duration_sanity * dur_score
            + cfg.w_event_completeness * has_events
            + cfg.w_language_consistency * lang_ok
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def check_hard_reject(self, record: CurationRecord) -> List[str]:
        """Return list of hard-reject reasons (empty = no rejection)."""
        reasons: List[str] = []
        if not record.transcript and record.transcript != "":
            reasons.append("transcript_empty")
        if record.duration_sec <= 0:
            reasons.append("invalid_duration")
        attrs = record.attributes
        overlap = attrs.get("overlap_ratio", 0.0)
        if overlap > 0.5:
            reasons.append("severe_overlap")
        if attrs.get("corrupted", False):
            reasons.append("audio_corrupted")
        sep_damage = attrs.get("separation_damage", 0.0)
        if sep_damage > 0.5:
            reasons.append("separation_damage_high")
        return reasons

    def check_review(self, record: CurationRecord) -> List[str]:
        """Return list of review reasons (empty = no review needed)."""
        reasons: List[str] = []
        t_conf = record.transcript_confidence or 0.0
        if 0.4 <= t_conf < 0.6:
            reasons.append("marginal_transcript_confidence")
        attrs = record.attributes
        agree = attrs.get("refinement_agreement", 1.0)
        if agree < 0.7:
            reasons.append("provider_disagreement")
        diar = record.diarization_confidence or 0.0
        if 0.3 <= diar < 0.6:
            reasons.append("uncertain_speaker_cluster")
        return reasons

    def determine_bucket(
        self, record: CurationRecord
    ) -> Optional[PromotionBucket]:
        """Determine which promotion bucket a record qualifies for."""
        legality = record.source_legality
        score = record.quality_score
        t_conf = record.transcript_confidence or 0.0

        # Anti-contamination: holdout cannot leak into train
        if record.record_id in self.config.holdout_record_ids:
            return PromotionBucket.HOLDOUT_EVAL

        # Try buckets from strictest to loosest
        for bucket_name, bucket_enum in [
            ("tts_mainline", PromotionBucket.TTS_MAINLINE),
            ("expressive_prior", PromotionBucket.EXPRESSIVE_PRIOR),
            ("vc_prior", PromotionBucket.VC_PRIOR),
        ]:
            thresh = self.config.thresholds.get(bucket_name)
            if thresh is None:
                continue
            if legality not in thresh.allowed_legality:
                continue
            if t_conf < thresh.transcript_confidence:
                continue
            if score < thresh.quality_score:
                continue
            return bucket_enum

        return None

    def score_and_decide(self, record: CurationRecord) -> CurationRecord:
        """Score a record and set its status and promotion bucket."""
        # Compute quality score
        record.quality_score = self.compute_score(record)

        # Check hard reject
        reject_reasons = self.check_hard_reject(record)
        if reject_reasons:
            record.status = RecordStatus.REJECTED
            record.rejection_reasons = reject_reasons
            record.promotion_bucket = PromotionBucket.NONE
            return record

        # Check review
        review_reasons = self.check_review(record)
        if review_reasons:
            record.status = RecordStatus.REVIEW
            record.review_reasons = review_reasons
            record.promotion_bucket = PromotionBucket.NONE
            return record

        # Determine bucket
        bucket = self.determine_bucket(record)
        if bucket is not None:
            record.status = RecordStatus.PROMOTED
            record.promotion_bucket = bucket
        else:
            record.status = RecordStatus.REVIEW
            record.review_reasons = ["no_qualifying_bucket"]

        return record

    def generate_report(self, records: List[CurationRecord]) -> Dict[str, Any]:
        """Generate summary report of scoring decisions."""
        status_counts: Dict[str, int] = {}
        bucket_counts: Dict[str, int] = {}
        rejection_reasons: Dict[str, int] = {}
        review_reasons: Dict[str, int] = {}
        scores: List[float] = []

        for r in records:
            status_counts[r.status.value] = (
                status_counts.get(r.status.value, 0) + 1
            )
            bucket_counts[r.promotion_bucket.value] = (
                bucket_counts.get(r.promotion_bucket.value, 0) + 1
            )
            scores.append(r.quality_score)
            for reason in r.rejection_reasons:
                rejection_reasons[reason] = (
                    rejection_reasons.get(reason, 0) + 1
                )
            for reason in r.review_reasons:
                review_reasons[reason] = review_reasons.get(reason, 0) + 1

        return {
            "total": len(records),
            "status_distribution": status_counts,
            "bucket_distribution": bucket_counts,
            "top_rejection_reasons": dict(
                sorted(rejection_reasons.items(), key=lambda x: -x[1])[:10]
            ),
            "top_review_reasons": dict(
                sorted(review_reasons.items(), key=lambda x: -x[1])[:10]
            ),
            "score_mean": (
                round(sum(scores) / len(scores), 4) if scores else 0.0
            ),
            "score_min": round(min(scores), 4) if scores else 0.0,
            "score_max": round(max(scores), 4) if scores else 0.0,
        }
