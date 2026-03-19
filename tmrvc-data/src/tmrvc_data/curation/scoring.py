"""Quality scoring, rejection, review, and promotion for curated records.

Implements Worker 09: turns raw provider outputs into actionable decisions.

Features:
- Composite quality score with configurable weights
- Hard-reject, review, and promotion decision policies
- Promotion buckets (tts_mainline, vc_prior, expressive_prior, holdout_eval)
- Anti-contamination: holdout samples never leak into training buckets
- Same-text clustering to prevent data leakage
- Human approval policy (auto_promote, auditor_review, double_approval)
- Voice-state coverage requirements per bucket
- Calibration-aware thresholds per provider
- Score histogram and rejection-reason reporting
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import CurationRecord, RecordStatus, PromotionBucket, LegalityStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucket thresholds
# ---------------------------------------------------------------------------


@dataclass
class BucketThresholds:
    """Promotion thresholds for a single bucket."""

    transcript_confidence: float = 0.90
    cross_asr_agreement: float = 0.85
    diarization_confidence: float = 0.80
    max_overlap_ratio: float = 0.10
    quality_score: float = 0.85
    allowed_legality: Tuple[str, ...] = ("owned", "licensed")


# Default thresholds from docs/design/curation-contract.md
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


# ---------------------------------------------------------------------------
# Score weights configuration
# ---------------------------------------------------------------------------


@dataclass
class ScoreWeights:
    """Configurable weights for quality_score computation.

    Each weight corresponds to one scoring dimension.  Weights are
    normalised internally so they need not sum to 1.0.
    """

    transcript_confidence: float = 0.25
    asr_agreement: float = 0.10
    diarization: float = 0.15
    audio_quality: float = 0.10       # SNR + artifact rate
    duration_sanity: float = 0.10
    event_completeness: float = 0.10
    language_consistency: float = 0.05
    voice_state_coverage: float = 0.10
    voice_state_confidence: float = 0.05
    speaker_similarity: float = 0.0   # optional, disabled by default


# ---------------------------------------------------------------------------
# Calibration-aware thresholds
# ---------------------------------------------------------------------------


@dataclass
class ProviderCalibration:
    """Per-provider calibration entry.

    When a provider's ``calibration_version`` is ``"uncalibrated"``, its
    outputs must not auto-promote into ``tts_mainline``.
    """

    provider_id: str
    calibration_version: str = "uncalibrated"
    confidence_offset: float = 0.0   # additive adjustment
    confidence_scale: float = 1.0    # multiplicative adjustment
    trusted: bool = False            # False => cannot auto-promote to mainline


# ---------------------------------------------------------------------------
# Human approval policy
# ---------------------------------------------------------------------------


class ApprovalPolicy:
    """Determines human-approval requirements for promotion decisions.

    Policies:
    - ``auto_promote``:    quality_score > high_threshold
    - ``auditor_review``:  quality_score between low and high threshold
    - ``double_approval``: always required for holdout_eval bucket

    Override justifications are logged via the ``approval_log`` list.
    """

    def __init__(
        self,
        auto_threshold: float = 0.90,
        review_threshold: float = 0.70,
    ) -> None:
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold
        self.approval_log: List[Dict[str, Any]] = []

    def required_approval(
        self,
        record: CurationRecord,
        bucket: Optional[PromotionBucket],
    ) -> str:
        """Return the approval level needed: auto_promote | auditor_review | double_approval."""
        if bucket == PromotionBucket.HOLDOUT_EVAL:
            return "double_approval"
        score = record.quality_score
        if score >= self.auto_threshold:
            return "auto_promote"
        if score >= self.review_threshold:
            return "auditor_review"
        return "auditor_review"

    def log_override(
        self,
        record_id: str,
        actor_id: str,
        approval_level: str,
        rationale: str,
    ) -> None:
        """Record an explicit human override with justification."""
        self.approval_log.append({
            "record_id": record_id,
            "actor_id": actor_id,
            "approval_level": approval_level,
            "rationale": rationale,
        })


# ---------------------------------------------------------------------------
# Voice-state promotion policy
# ---------------------------------------------------------------------------


@dataclass
class VoiceStateCoveragePolicy:
    """Coverage requirements for voice_state dimensions per bucket.

    Attributes:
        min_observed_dims: minimum number of observed dimensions (out of 8)
        min_mean_confidence: minimum mean confidence across observed dims
        required_for_buckets: set of bucket names that enforce this policy
        allow_absent_with_metadata: if True, records with explicit
            ``voice_state_absent=True`` attribute may pass with a penalty
    """

    min_observed_dims: int = 4
    min_mean_confidence: float = 0.3
    required_for_buckets: Tuple[str, ...] = ("tts_mainline",)
    allow_absent_with_metadata: bool = True

    def check(
        self,
        record: CurationRecord,
        bucket_name: str,
    ) -> Tuple[bool, str]:
        """Return (passes, reason) for the voice_state policy.

        Returns ``(True, "")`` if the policy passes or the bucket does
        not require voice_state coverage.
        """
        if bucket_name not in self.required_for_buckets:
            return True, ""

        attrs = record.attributes

        # Explicit absence metadata is allowed with a flag
        if attrs.get("voice_state_absent", False) and self.allow_absent_with_metadata:
            return True, ""

        # Check observed mask.  If the record has no voice_state data at
        # all (mask is empty / missing) we treat it the same as explicit
        # absence when allow_absent_with_metadata is True, so that
        # records produced before voice_state was introduced still pass.
        mask = attrs.get("voice_state_observed_mask", [])
        if not mask:
            if self.allow_absent_with_metadata:
                return True, ""
            return False, "voice_state_not_available"

        observed_count = sum(1 for m in mask if m)
        if observed_count < self.min_observed_dims:
            return (
                False,
                f"voice_state_low_coverage:{observed_count}/{self.min_observed_dims}",
            )

        # Check confidence
        conf_list = attrs.get("voice_state_confidence", [])
        if conf_list:
            # Handle both scalar and list confidence
            if isinstance(conf_list, (int, float)):
                mean_conf = float(conf_list)
            else:
                observed_confs = [
                    float(c)
                    for c, m in zip(conf_list, mask)
                    if m
                ]
                mean_conf = (
                    sum(observed_confs) / len(observed_confs)
                    if observed_confs
                    else 0.0
                )
            if mean_conf < self.min_mean_confidence:
                return (
                    False,
                    f"voice_state_low_confidence:{mean_conf:.3f}<{self.min_mean_confidence}",
                )

        return True, ""


# ---------------------------------------------------------------------------
# Same-text clustering (anti-contamination)
# ---------------------------------------------------------------------------


def _text_fingerprint(text: str) -> str:
    """Normalise and hash transcript text for deduplication clustering."""
    normalised = text.strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def cluster_same_text(
    records: List[CurationRecord],
) -> Dict[str, List[str]]:
    """Group records by normalised transcript text.

    Returns a mapping from text fingerprint to list of record_ids.
    Only clusters with more than one member are included.
    """
    clusters: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r.transcript:
            fp = _text_fingerprint(r.transcript)
            clusters[fp].append(r.record_id)
    return {k: v for k, v in clusters.items() if len(v) > 1}


def enforce_same_text_holdout_isolation(
    records: List[CurationRecord],
    holdout_ids: Set[str],
) -> Set[str]:
    """Return record_ids that must be excluded from training buckets.

    If any member of a same-text cluster is in the holdout set, *all*
    members of that cluster are tainted and must not appear in training.
    """
    clusters = cluster_same_text(records)
    tainted: Set[str] = set()
    for _fp, member_ids in clusters.items():
        if any(rid in holdout_ids for rid in member_ids):
            tainted.update(member_ids)
    return tainted


# ---------------------------------------------------------------------------
# Scoring config (enhanced)
# ---------------------------------------------------------------------------


@dataclass
class ScoringConfig:
    """Configuration for the quality scoring engine."""

    # Score weights (legacy flat fields kept for backward compat)
    w_transcript: float = 0.25
    w_asr_agreement: float = 0.10
    w_diarization: float = 0.15
    w_snr: float = 0.10
    w_duration_sanity: float = 0.10
    w_event_completeness: float = 0.10
    w_language_consistency: float = 0.05
    w_voice_state_density: float = 0.10
    w_voice_state_confidence: float = 0.05

    # Structured weights (preferred over flat fields)
    score_weights: Optional[ScoreWeights] = None

    # Bucket thresholds
    thresholds: Dict[str, BucketThresholds] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )

    # Holdout set
    holdout_record_ids: Set[str] = field(default_factory=set)

    # Calibration registry: provider_id -> ProviderCalibration
    provider_calibrations: Dict[str, ProviderCalibration] = field(
        default_factory=dict,
    )

    # Human approval policy thresholds
    auto_promote_threshold: float = 0.90
    auditor_review_threshold: float = 0.70

    # Voice-state policy
    voice_state_policy: Optional[VoiceStateCoveragePolicy] = None

    # Provider-supported language gate
    supported_languages: Optional[Dict[str, Set[str]]] = None

    def get_weights(self) -> ScoreWeights:
        """Return the structured ScoreWeights, falling back to flat fields."""
        if self.score_weights is not None:
            return self.score_weights
        return ScoreWeights(
            transcript_confidence=self.w_transcript,
            asr_agreement=self.w_asr_agreement,
            diarization=self.w_diarization,
            audio_quality=self.w_snr,
            duration_sanity=self.w_duration_sanity,
            event_completeness=self.w_event_completeness,
            language_consistency=self.w_language_consistency,
            voice_state_coverage=self.w_voice_state_density,
            voice_state_confidence=self.w_voice_state_confidence,
        )


# ---------------------------------------------------------------------------
# Quality Scoring Engine
# ---------------------------------------------------------------------------


class QualityScoringEngine:
    """Computes quality scores and makes promotion/rejection decisions.

    Enhanced with:
    - Configurable score weights via ``ScoreWeights``
    - Calibration-aware provider confidence adjustment
    - Voice-state coverage policy
    - Human approval policy
    - Same-text anti-contamination
    - Score histogram and rejection-reason reporting
    """

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        self.config = config or ScoringConfig()
        self._approval_policy = ApprovalPolicy(
            auto_threshold=self.config.auto_promote_threshold,
            review_threshold=self.config.auditor_review_threshold,
        )
        self._vs_policy = (
            self.config.voice_state_policy or VoiceStateCoveragePolicy()
        )

    @property
    def approval_policy(self) -> ApprovalPolicy:
        return self._approval_policy

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        provider_id: Optional[str],
    ) -> Tuple[float, bool]:
        """Apply calibration adjustment to a provider confidence score.

        Returns ``(adjusted_confidence, is_calibrated)``.
        """
        if provider_id is None:
            return raw_confidence, False
        cal = self.config.provider_calibrations.get(provider_id)
        if cal is None:
            return raw_confidence, False
        adjusted = (raw_confidence + cal.confidence_offset) * cal.confidence_scale
        adjusted = min(max(adjusted, 0.0), 1.0)
        is_calibrated = cal.calibration_version != "uncalibrated"
        return adjusted, is_calibrated

    def _is_provider_calibrated(self, provider_id: str) -> bool:
        """Return True if the provider has been calibrated."""
        cal = self.config.provider_calibrations.get(provider_id)
        if cal is None:
            return False
        return cal.calibration_version != "uncalibrated"

    def _is_provider_trusted(self, provider_id: str) -> bool:
        """Return True if the provider is trusted for auto-promotion."""
        cal = self.config.provider_calibrations.get(provider_id)
        if cal is None:
            return False
        return cal.trusted

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_score(self, record: CurationRecord) -> float:
        """Compute composite quality score for a record."""
        cfg = self.config
        w = cfg.get_weights()
        attrs = record.attributes

        t_conf = record.transcript_confidence or 0.0
        asr_agree = attrs.get("refinement_agreement", t_conf)
        diar_conf = record.diarization_confidence or 0.0
        snr = attrs.get("snr_db", 0.0)
        snr_norm = min(snr / 40.0, 1.0) if snr > 0 else 0.0

        # Audio quality: combine SNR and artifact rate
        artifact_rate = attrs.get("artifact_rate", 0.0)
        audio_quality = snr_norm * (1.0 - min(artifact_rate, 1.0))

        # Speaker similarity (optional)
        speaker_sim = attrs.get("speaker_similarity", 0.0)

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

        # Voice state supervision quality.
        # When voice_state data is completely absent (no density key and
        # no observed mask), default to 1.0 so that records produced before
        # voice_state was introduced are not penalised.
        _has_vs_data = (
            "voice_state_density" in attrs
            or "voice_state_observed_mask" in attrs
        )
        vs_density = attrs.get("voice_state_density", 0.0 if _has_vs_data else 1.0)
        vs_conf = attrs.get("voice_state_confidence", 0.0 if _has_vs_data else 1.0)
        # Handle list-type voice_state_confidence (per-dimension)
        if isinstance(vs_conf, (list, tuple)):
            vs_conf = sum(vs_conf) / len(vs_conf) if vs_conf else 0.0

        score = (
            w.transcript_confidence * t_conf
            + w.asr_agreement * asr_agree
            + w.diarization * diar_conf
            + w.audio_quality * audio_quality
            + w.duration_sanity * dur_score
            + w.event_completeness * has_events
            + w.language_consistency * lang_ok
            + w.voice_state_coverage * vs_density
            + w.voice_state_confidence * vs_conf
            + w.speaker_similarity * speaker_sim
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def compute_score_components(
        self, record: CurationRecord
    ) -> Dict[str, float]:
        """Return individual score components for debugging / reporting."""
        w = self.config.get_weights()
        attrs = record.attributes

        t_conf = record.transcript_confidence or 0.0
        asr_agree = attrs.get("refinement_agreement", t_conf)
        diar_conf = record.diarization_confidence or 0.0
        snr = attrs.get("snr_db", 0.0)
        snr_norm = min(snr / 40.0, 1.0) if snr > 0 else 0.0
        artifact_rate = attrs.get("artifact_rate", 0.0)
        audio_quality = snr_norm * (1.0 - min(artifact_rate, 1.0))
        speaker_sim = attrs.get("speaker_similarity", 0.0)
        dur = record.duration_sec
        dur_score = (
            1.0
            if 0.3 <= dur <= 30.0
            else max(0.0, 1.0 - abs(dur - 15.0) / 30.0)
        )
        has_events = (
            1.0
            if attrs.get("pause_events") or attrs.get("breath_events")
            else 0.0
        )
        lang_ok = 1.0 if record.language else 0.0
        _has_vs_data = (
            "voice_state_density" in attrs
            or "voice_state_observed_mask" in attrs
        )
        vs_density = attrs.get("voice_state_density", 0.0 if _has_vs_data else 1.0)
        vs_conf = attrs.get("voice_state_confidence", 0.0 if _has_vs_data else 1.0)
        if isinstance(vs_conf, (list, tuple)):
            vs_conf = sum(vs_conf) / len(vs_conf) if vs_conf else 0.0

        return {
            "transcript_confidence": round(w.transcript_confidence * t_conf, 4),
            "asr_agreement": round(w.asr_agreement * asr_agree, 4),
            "diarization": round(w.diarization * diar_conf, 4),
            "audio_quality": round(w.audio_quality * audio_quality, 4),
            "duration_sanity": round(w.duration_sanity * dur_score, 4),
            "event_completeness": round(w.event_completeness * has_events, 4),
            "language_consistency": round(w.language_consistency * lang_ok, 4),
            "voice_state_coverage": round(w.voice_state_coverage * vs_density, 4),
            "voice_state_confidence": round(w.voice_state_confidence * vs_conf, 4),
            "speaker_similarity": round(w.speaker_similarity * speaker_sim, 4),
        }

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

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
        # Clipping detection
        if attrs.get("clipping_detected", False):
            reasons.append("audio_clipping")
        # Wrong language
        if attrs.get("language_mismatch", False):
            reasons.append("wrong_language")
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
        # Partial event extraction failure
        if attrs.get("event_extraction_partial", False):
            reasons.append("partial_event_extraction")
        # Uncalibrated provider gate: must review, not auto-promote
        uncal_providers = attrs.get("uncalibrated_providers", [])
        if uncal_providers:
            reasons.append("uncalibrated_provider_output")
        return reasons

    def _check_language_gate(
        self, record: CurationRecord, bucket_name: str
    ) -> Optional[str]:
        """Check provider-supported language gate.

        Returns a review reason string if the language is unsupported,
        or None if OK.
        """
        supported = self.config.supported_languages
        if supported is None:
            return None
        lang = record.language
        if lang is None:
            return "language_unknown"
        bucket_langs = supported.get(bucket_name)
        if bucket_langs is None:
            return None  # no gate defined for this bucket
        if lang not in bucket_langs:
            return f"language_unsupported_for_{bucket_name}"
        return None

    def _check_separation_waveform_gate(
        self, record: CurationRecord, bucket_name: str
    ) -> Optional[str]:
        """Enforce separation-derived waveform policy.

        Per plan: tts_mainline does not auto-adopt separation-derived
        waveform teachers in the initial mainline.
        """
        attrs = record.attributes
        if bucket_name == "tts_mainline":
            if attrs.get("uses_separated_waveform", False):
                return "separated_waveform_blocked_for_mainline"
        return None

    def determine_bucket(
        self, record: CurationRecord
    ) -> Optional[PromotionBucket]:
        """Determine which promotion bucket a record qualifies for."""
        legality = record.source_legality
        score = record.quality_score
        t_conf = record.transcript_confidence or 0.0
        attrs = record.attributes

        # Anti-contamination: holdout cannot leak into train
        if record.record_id in self.config.holdout_record_ids:
            return PromotionBucket.HOLDOUT_EVAL

        # Same-text tainted records cannot go to training (checked externally)
        if attrs.get("_same_text_tainted", False):
            return None

        # Expressive prior requires prosody extraction
        has_prosody = bool(
            attrs.get("pause_events")
            or attrs.get("breath_events")
            or attrs.get("voice_state_observed_mask")
        )

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

            # Additional overlap check
            overlap = attrs.get("overlap_ratio", 0.0)
            if overlap > thresh.max_overlap_ratio:
                continue

            # Diarization check
            diar = record.diarization_confidence or 0.0
            if diar < thresh.diarization_confidence:
                continue

            # Cross-ASR agreement check
            agree = attrs.get("refinement_agreement", t_conf)
            if agree < thresh.cross_asr_agreement:
                continue

            # Expressive prior requires prosody data
            if bucket_name == "expressive_prior" and not has_prosody:
                continue

            # Separation waveform gate
            sep_gate = self._check_separation_waveform_gate(
                record, bucket_name
            )
            if sep_gate is not None:
                continue

            # Language gate
            lang_gate = self._check_language_gate(record, bucket_name)
            if lang_gate is not None:
                continue

            # Voice-state coverage policy
            vs_ok, _vs_reason = self._vs_policy.check(record, bucket_name)
            if not vs_ok:
                continue

            # Calibration gate: uncalibrated providers cannot auto-promote
            # to tts_mainline
            if bucket_name == "tts_mainline":
                uncal = attrs.get("uncalibrated_providers", [])
                if uncal:
                    continue

            return bucket_enum

        return None

    def score_and_decide(self, record: CurationRecord) -> CurationRecord:
        """Score a record and set its status and promotion bucket.

        Also annotates the record with ``score_components`` and
        ``approval_level`` in its attributes.
        """
        # Compute quality score
        record.quality_score = self.compute_score(record)

        # Store score components for reporting
        record.attributes["score_components"] = self.compute_score_components(
            record
        )

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
            # Determine approval level
            approval = self._approval_policy.required_approval(record, bucket)
            record.attributes["approval_level"] = approval

            if approval == "auto_promote":
                record.status = RecordStatus.PROMOTED
                record.promotion_bucket = bucket
            elif approval == "double_approval":
                # Holdout requires double approval -- mark for review
                record.status = RecordStatus.REVIEW
                record.review_reasons = ["requires_double_approval"]
                record.promotion_bucket = bucket
            else:
                # auditor_review: promote but flag for audit
                record.status = RecordStatus.PROMOTED
                record.promotion_bucket = bucket
                record.review_reasons = ["auditor_review_recommended"]
        else:
            record.status = RecordStatus.REVIEW
            record.review_reasons = ["no_qualifying_bucket"]

        return record

    # ------------------------------------------------------------------
    # Batch scoring with anti-contamination
    # ------------------------------------------------------------------

    def score_batch(
        self,
        records: List[CurationRecord],
    ) -> List[CurationRecord]:
        """Score a batch of records with same-text anti-contamination.

        Records whose transcript text appears in the same cluster as a
        holdout sample are excluded from training buckets.
        """
        holdout_ids = self.config.holdout_record_ids

        # Enforce same-text holdout isolation
        if holdout_ids:
            tainted = enforce_same_text_holdout_isolation(
                records, holdout_ids
            )
            for r in records:
                if r.record_id in tainted and r.record_id not in holdout_ids:
                    r.attributes["_same_text_tainted"] = True

        results = []
        for r in records:
            results.append(self.score_and_decide(r))
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

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

    def generate_score_histogram(
        self,
        records: List[CurationRecord],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Generate a score histogram with configurable bins.

        Returns bin edges and counts, plus per-bucket breakdowns.
        """
        scores = [r.quality_score for r in records]
        if not scores:
            return {"bins": [], "counts": [], "per_bucket": {}}

        bin_width = 1.0 / n_bins
        bins = [round(i * bin_width, 4) for i in range(n_bins + 1)]
        counts = [0] * n_bins
        per_bucket: Dict[str, List[int]] = {}

        for r in records:
            idx = min(int(r.quality_score / bin_width), n_bins - 1)
            counts[idx] += 1

            bname = r.promotion_bucket.value
            if bname not in per_bucket:
                per_bucket[bname] = [0] * n_bins
            per_bucket[bname][idx] += 1

        return {
            "bins": bins,
            "counts": counts,
            "per_bucket": per_bucket,
        }

    def generate_rejection_breakdown(
        self,
        records: List[CurationRecord],
    ) -> Dict[str, int]:
        """Return a sorted breakdown of rejection reasons."""
        counts: Dict[str, int] = {}
        for r in records:
            for reason in r.rejection_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def generate_bucket_summary(
        self,
        records: List[CurationRecord],
    ) -> Dict[str, Dict[str, Any]]:
        """Return per-bucket statistics: count, mean score, score range."""
        buckets: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            buckets[r.promotion_bucket.value].append(r.quality_score)

        summary: Dict[str, Dict[str, Any]] = {}
        for bname, bscores in buckets.items():
            summary[bname] = {
                "count": len(bscores),
                "mean_score": round(sum(bscores) / len(bscores), 4),
                "min_score": round(min(bscores), 4),
                "max_score": round(max(bscores), 4),
            }
        return summary

    def generate_full_report(
        self,
        records: List[CurationRecord],
        n_histogram_bins: int = 10,
    ) -> Dict[str, Any]:
        """Generate a comprehensive report combining all sub-reports."""
        return {
            "summary": self.generate_report(records),
            "score_histogram": self.generate_score_histogram(
                records, n_bins=n_histogram_bins
            ),
            "rejection_breakdown": self.generate_rejection_breakdown(records),
            "bucket_summary": self.generate_bucket_summary(records),
        }
