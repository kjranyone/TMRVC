"""Curation validation and acceptance testing.

Implements Worker 11: proves the curation system improves data quality
rather than only producing impressive metadata.

Classes:
- ValidationConfig: thresholds for curation validation
- CurationValidator: validation checks (promotion, legality, holdout, provenance)
- ProviderAcceptanceThresholds: per-provider acceptance criteria
- StageBenchmark: evaluates each curation stage independently on known-good samples
- SampleAuditor: human review workflow with dual-review and audit trail
- DownstreamComparison: compares naive vs curated data quality metrics
- VoiceStateValidator: validates voice_state pseudo-label utility
- ProviderRecalibrator: recalibrates provider confidences on held-out samples
- HumanWorkflowValidator: verifies audit trail, role separation, optimistic locking
- SplitIntegrityValidator: holdout leak detection and deterministic split checks
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .models import CurationRecord, RecordStatus, PromotionBucket, Provenance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------


@dataclass
class ValidationConfig:
    """Thresholds for curation validation."""

    min_promote_ratio: float = 0.10
    max_false_promote_rate: float = 0.05
    max_false_reject_rate: float = 0.10
    min_asr_spot_accuracy: float = 0.90
    min_speaker_nmi: float = 0.80
    max_holdout_leak_ratio: float = 0.0


# ---------------------------------------------------------------------------
# CurationValidator
# ---------------------------------------------------------------------------


class CurationValidator:
    """Validates curation system quality and policy compliance."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate_promotion_distribution(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Check promote/review/reject distribution."""
        total = len(records)
        if total == 0:
            return {"pass": False, "reason": "no_records", "total": 0}

        counts: Dict[str, int] = {}
        for r in records:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1

        promoted = counts.get("promoted", 0)
        promote_ratio = promoted / total

        return {
            "pass": promote_ratio >= self.config.min_promote_ratio,
            "total": total,
            "distribution": counts,
            "promote_ratio": round(promote_ratio, 4),
            "threshold": self.config.min_promote_ratio,
        }

    def validate_legality_gating(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Verify no unknown-rights sources in mainline export."""
        violations = []
        for r in records:
            if r.status != RecordStatus.PROMOTED:
                continue
            if r.promotion_bucket in (
                PromotionBucket.TTS_MAINLINE,
                PromotionBucket.HOLDOUT_EVAL,
            ):
                if r.source_legality not in ("owned", "licensed"):
                    violations.append({
                        "record_id": r.record_id,
                        "bucket": r.promotion_bucket.value,
                        "legality": r.source_legality,
                    })

        return {
            "pass": len(violations) == 0,
            "violations": violations,
            "total_checked": sum(
                1 for r in records if r.status == RecordStatus.PROMOTED
            ),
        }

    def validate_holdout_integrity(
        self, records: List[CurationRecord],
        holdout_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Verify holdout records don't leak into training buckets."""
        if holdout_ids is None:
            holdout_ids = {
                r.record_id for r in records
                if r.promotion_bucket == PromotionBucket.HOLDOUT_EVAL
            }

        leaks = []
        train_buckets = {
            PromotionBucket.TTS_MAINLINE,
            PromotionBucket.VC_PRIOR,
            PromotionBucket.EXPRESSIVE_PRIOR,
        }
        for r in records:
            if r.record_id in holdout_ids and r.promotion_bucket in train_buckets:
                leaks.append(r.record_id)

        return {
            "pass": len(leaks) == 0,
            "leaks": leaks,
            "holdout_size": len(holdout_ids),
        }

    def validate_provenance_completeness(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Check that promoted records have complete provenance."""
        promoted = [r for r in records if r.status == RecordStatus.PROMOTED]
        incomplete = []
        for r in promoted:
            if not r.providers:
                incomplete.append(r.record_id)

        return {
            "pass": len(incomplete) == 0,
            "total_promoted": len(promoted),
            "incomplete": incomplete[:20],  # Limit output
        }

    def run_all(self, records: List[CurationRecord]) -> Dict[str, Any]:
        """Run all validation checks and return combined report."""
        results = {
            "promotion_distribution": self.validate_promotion_distribution(records),
            "legality_gating": self.validate_legality_gating(records),
            "holdout_integrity": self.validate_holdout_integrity(records),
            "provenance_completeness": self.validate_provenance_completeness(records),
        }

        all_pass = all(v.get("pass", False) for v in results.values())
        results["overall"] = {"pass": all_pass}

        return results

    def save_report(self, report: Dict[str, Any], output_path: Path | str) -> None:
        """Save validation report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Validation report saved to %s", path)


# ---------------------------------------------------------------------------
# Provider Acceptance Thresholds (Worker 11 task 4)
# ---------------------------------------------------------------------------


@dataclass
class ProviderAcceptanceThresholds:
    """Per-provider acceptance criteria.

    Each provider family has its own acceptance thresholds.
    Pooled thresholds across different provider families are forbidden
    until calibration parity is demonstrated (per plan).

    Attributes:
        provider_id: pinned provider identifier
        calibration_version: the calibration version these thresholds apply to
        asr_wer_threshold: maximum acceptable word error rate (ASR only)
        asr_confidence_calibration_ece: max expected calibration error (ASR)
        diarization_der_threshold: max diarization error rate
        diarization_speaker_count_accuracy: min speaker count accuracy
        voice_state_coverage_threshold: min fraction of observed dimensions
        voice_state_calibration_mae: max mean absolute error vs reference
        alignment_timing_tolerance_sec: max boundary timing deviation
    """

    provider_id: str = ""
    calibration_version: str = "uncalibrated"

    # ASR thresholds
    asr_wer_threshold: float = 0.15
    asr_confidence_calibration_ece: float = 0.10

    # Diarization thresholds
    diarization_der_threshold: float = 0.20
    diarization_speaker_count_accuracy: float = 0.85

    # Voice state thresholds
    voice_state_coverage_threshold: float = 0.50
    voice_state_calibration_mae: float = 0.20

    # Alignment thresholds
    alignment_timing_tolerance_sec: float = 0.05

    def check_asr(
        self, wer: float, calibration_ece: float
    ) -> Dict[str, Any]:
        """Check ASR provider against acceptance criteria."""
        wer_pass = wer <= self.asr_wer_threshold
        ece_pass = calibration_ece <= self.asr_confidence_calibration_ece
        return {
            "pass": wer_pass and ece_pass,
            "provider_id": self.provider_id,
            "calibration_version": self.calibration_version,
            "wer": {"value": round(wer, 4), "threshold": self.asr_wer_threshold, "pass": wer_pass},
            "confidence_calibration_ece": {
                "value": round(calibration_ece, 4),
                "threshold": self.asr_confidence_calibration_ece,
                "pass": ece_pass,
            },
        }

    def check_diarization(
        self, der: float, speaker_count_accuracy: float
    ) -> Dict[str, Any]:
        """Check diarization provider against acceptance criteria."""
        der_pass = der <= self.diarization_der_threshold
        sc_pass = speaker_count_accuracy >= self.diarization_speaker_count_accuracy
        return {
            "pass": der_pass and sc_pass,
            "provider_id": self.provider_id,
            "calibration_version": self.calibration_version,
            "der": {
                "value": round(der, 4),
                "threshold": self.diarization_der_threshold,
                "pass": der_pass,
            },
            "speaker_count_accuracy": {
                "value": round(speaker_count_accuracy, 4),
                "threshold": self.diarization_speaker_count_accuracy,
                "pass": sc_pass,
            },
        }

    def check_voice_state(
        self, coverage: float, calibration_mae: float
    ) -> Dict[str, Any]:
        """Check voice_state provider against acceptance criteria."""
        cov_pass = coverage >= self.voice_state_coverage_threshold
        cal_pass = calibration_mae <= self.voice_state_calibration_mae
        return {
            "pass": cov_pass and cal_pass,
            "provider_id": self.provider_id,
            "calibration_version": self.calibration_version,
            "coverage": {
                "value": round(coverage, 4),
                "threshold": self.voice_state_coverage_threshold,
                "pass": cov_pass,
            },
            "calibration_mae": {
                "value": round(calibration_mae, 4),
                "threshold": self.voice_state_calibration_mae,
                "pass": cal_pass,
            },
        }


# ---------------------------------------------------------------------------
# Stage Benchmark (Worker 11 task 1)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSample:
    """A known-good test sample for stage benchmarking."""

    sample_id: str
    record: CurationRecord
    reference_outputs: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"transcript": "hello world", "speaker_count": 2, ...}


@dataclass
class StageAuditResult:
    """Result of auditing a single stage on benchmark samples."""

    stage_name: str
    provider_id: str
    n_samples: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    format_violations: List[str] = field(default_factory=list)


class StageBenchmark:
    """Evaluates each curation stage independently on known-good samples.

    Runs a provider on benchmark samples and measures accuracy against
    reference outputs.  Also validates stage output format compliance.
    """

    def __init__(
        self,
        acceptance_thresholds: Optional[Dict[str, ProviderAcceptanceThresholds]] = None,
    ) -> None:
        self.acceptance_thresholds = acceptance_thresholds or {}

    def audit_stage(
        self,
        stage_name: str,
        provider_id: str,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        match_key: str = "transcript",
    ) -> StageAuditResult:
        """Run a per-provider audit on known-good test samples.

        Args:
            stage_name: name of the curation stage (e.g. "asr", "diarization")
            provider_id: identifier of the provider being audited
            predictions: list of predicted output dicts
            references: list of reference (ground truth) output dicts
            match_key: the key to compare for correctness

        Returns:
            StageAuditResult with accuracy and error details.
        """
        n = min(len(predictions), len(references))
        correct = 0
        errors: List[Dict[str, Any]] = []

        for i in range(n):
            pred_val = predictions[i].get(match_key)
            ref_val = references[i].get(match_key)
            if pred_val == ref_val:
                correct += 1
            else:
                errors.append({
                    "index": i,
                    "predicted": pred_val,
                    "reference": ref_val,
                })

        accuracy = correct / n if n > 0 else 0.0

        return StageAuditResult(
            stage_name=stage_name,
            provider_id=provider_id,
            n_samples=n,
            n_correct=correct,
            accuracy=round(accuracy, 4),
            errors=errors[:20],  # cap error detail output
        )

    def validate_output_format(
        self,
        stage_name: str,
        outputs: List[Dict[str, Any]],
        required_keys: List[str],
    ) -> List[str]:
        """Validate that stage outputs contain required keys.

        Returns list of format violation descriptions.
        """
        violations: List[str] = []
        for i, out in enumerate(outputs):
            missing = [k for k in required_keys if k not in out]
            if missing:
                violations.append(
                    f"Sample {i}: missing keys {missing} for stage '{stage_name}'"
                )
        return violations

    def run_benchmark(
        self,
        stage_name: str,
        provider_id: str,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        match_key: str = "transcript",
        required_keys: Optional[List[str]] = None,
    ) -> StageAuditResult:
        """Full benchmark: audit + format validation."""
        result = self.audit_stage(
            stage_name, provider_id, predictions, references, match_key
        )
        if required_keys:
            result.format_violations = self.validate_output_format(
                stage_name, predictions, required_keys
            )
        return result


# ---------------------------------------------------------------------------
# Sample Auditor (Worker 11 task 2)
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """A single audit decision on a sample."""

    record_id: str
    auditor_id: str
    decision: str  # "correct" | "incorrect" | "uncertain"
    rationale: str = ""
    timestamp: float = 0.0
    bucket: str = ""


class SampleAuditor:
    """Human review workflow for curation quality.

    Tracks promoted/review/rejected sample quality with dual-review
    support for holdout_eval samples.
    """

    def __init__(self) -> None:
        self._audit_log: List[AuditEntry] = []

    @property
    def audit_log(self) -> List[AuditEntry]:
        return list(self._audit_log)

    def submit_audit(
        self,
        record_id: str,
        auditor_id: str,
        decision: str,
        rationale: str = "",
        bucket: str = "",
    ) -> AuditEntry:
        """Submit a single audit decision."""
        if decision not in ("correct", "incorrect", "uncertain"):
            raise ValueError(
                f"Invalid decision '{decision}'; "
                "must be 'correct', 'incorrect', or 'uncertain'"
            )
        entry = AuditEntry(
            record_id=record_id,
            auditor_id=auditor_id,
            decision=decision,
            rationale=rationale,
            timestamp=time.time(),
            bucket=bucket,
        )
        self._audit_log.append(entry)
        return entry

    def get_audits_for_record(self, record_id: str) -> List[AuditEntry]:
        """Return all audit entries for a given record."""
        return [e for e in self._audit_log if e.record_id == record_id]

    def check_dual_review(self, record_id: str) -> Dict[str, Any]:
        """Check whether a holdout_eval record has dual review.

        Returns pass status and agreement information.
        """
        audits = self.get_audits_for_record(record_id)
        unique_auditors = {a.auditor_id for a in audits}
        has_dual = len(unique_auditors) >= 2

        if len(audits) < 2:
            return {
                "pass": False,
                "record_id": record_id,
                "reason": "insufficient_reviews",
                "n_reviews": len(audits),
            }

        # Check agreement
        decisions = [a.decision for a in audits]
        agree = len(set(decisions)) == 1

        return {
            "pass": has_dual,
            "record_id": record_id,
            "n_reviews": len(audits),
            "n_unique_auditors": len(unique_auditors),
            "agreement": agree,
            "decisions": decisions,
        }

    def generate_audit_report(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Generate audit report across all reviewed samples.

        Computes false-promote rate, false-reject rate, and
        per-bucket quality metrics.
        """
        bucket_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0, "incorrect": 0, "uncertain": 0}
        )

        for entry in self._audit_log:
            bucket = entry.bucket or "unknown"
            bucket_stats[bucket]["total"] += 1
            bucket_stats[bucket][entry.decision] += 1

        # Compute false-promote rate (incorrect decisions on promoted samples)
        promoted_audits = [
            e for e in self._audit_log
            if e.bucket in ("tts_mainline", "vc_prior", "expressive_prior", "holdout_eval")
        ]
        n_promoted_audited = len(promoted_audits)
        n_false_promotes = sum(
            1 for e in promoted_audits if e.decision == "incorrect"
        )
        false_promote_rate = (
            n_false_promotes / n_promoted_audited
            if n_promoted_audited > 0
            else 0.0
        )

        # Compute false-reject rate (incorrect decisions on rejected samples)
        rejected_audits = [
            e for e in self._audit_log if e.bucket == "rejected"
        ]
        n_rejected_audited = len(rejected_audits)
        n_false_rejects = sum(
            1 for e in rejected_audits if e.decision == "incorrect"
        )
        false_reject_rate = (
            n_false_rejects / n_rejected_audited
            if n_rejected_audited > 0
            else 0.0
        )

        return {
            "total_audited": len(self._audit_log),
            "per_bucket": dict(bucket_stats),
            "false_promote_rate": round(false_promote_rate, 4),
            "false_reject_rate": round(false_reject_rate, 4),
            "n_promoted_audited": n_promoted_audited,
            "n_rejected_audited": n_rejected_audited,
        }


# ---------------------------------------------------------------------------
# Downstream Comparison (Worker 11 task 3)
# ---------------------------------------------------------------------------


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset subset (naive or curated)."""

    label: str  # "naive" or "curated"
    n_samples: int = 0
    mean_transcript_confidence: float = 0.0
    mean_quality_score: float = 0.0
    mean_diarization_confidence: float = 0.0
    mean_snr_db: float = 0.0
    coverage_voice_state: float = 0.0
    # Downstream training stability proxies
    mean_duration_sec: float = 0.0
    reject_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n_samples": self.n_samples,
            "mean_transcript_confidence": round(self.mean_transcript_confidence, 4),
            "mean_quality_score": round(self.mean_quality_score, 4),
            "mean_diarization_confidence": round(self.mean_diarization_confidence, 4),
            "mean_snr_db": round(self.mean_snr_db, 4),
            "coverage_voice_state": round(self.coverage_voice_state, 4),
            "mean_duration_sec": round(self.mean_duration_sec, 4),
            "reject_rate": round(self.reject_rate, 4),
        }


class DownstreamComparison:
    """Compare naive (uncurated) vs curated data quality metrics.

    Implements Layer 3 from the validation plan: proves curation
    provides measurable quality uplift over naive ingestion.
    """

    @staticmethod
    def compute_metrics(
        records: List[CurationRecord],
        label: str = "unknown",
    ) -> QualityMetrics:
        """Compute aggregate quality metrics for a set of records."""
        n = len(records)
        if n == 0:
            return QualityMetrics(label=label)

        t_confs = [r.transcript_confidence or 0.0 for r in records]
        q_scores = [r.quality_score for r in records]
        d_confs = [r.diarization_confidence or 0.0 for r in records]
        snrs = [r.attributes.get("snr_db", 0.0) for r in records]
        durations = [r.duration_sec for r in records]

        # Voice state coverage: fraction of records with voice_state data
        vs_count = sum(
            1 for r in records
            if r.attributes.get("voice_state_observed_mask")
        )

        rejected = sum(
            1 for r in records if r.status == RecordStatus.REJECTED
        )

        return QualityMetrics(
            label=label,
            n_samples=n,
            mean_transcript_confidence=sum(t_confs) / n,
            mean_quality_score=sum(q_scores) / n,
            mean_diarization_confidence=sum(d_confs) / n,
            mean_snr_db=sum(snrs) / n,
            coverage_voice_state=vs_count / n,
            mean_duration_sec=sum(durations) / n,
            reject_rate=rejected / n,
        )

    @staticmethod
    def compare(
        naive: QualityMetrics,
        curated: QualityMetrics,
    ) -> Dict[str, Any]:
        """Compare two sets of quality metrics and compute uplift.

        Returns a dict with per-metric uplift values (curated - naive).
        Positive values indicate improvement from curation.
        """
        uplift: Dict[str, float] = {}
        for metric in (
            "mean_transcript_confidence",
            "mean_quality_score",
            "mean_diarization_confidence",
            "mean_snr_db",
            "coverage_voice_state",
        ):
            v_naive = getattr(naive, metric, 0.0)
            v_curated = getattr(curated, metric, 0.0)
            uplift[metric] = round(v_curated - v_naive, 4)

        # For reject_rate, lower is better, so we invert
        uplift["reject_rate_reduction"] = round(
            naive.reject_rate - curated.reject_rate, 4
        )

        all_positive = all(v >= 0 for v in uplift.values())

        return {
            "pass": all_positive,
            "naive": naive.to_dict(),
            "curated": curated.to_dict(),
            "uplift": uplift,
        }


# ---------------------------------------------------------------------------
# Voice State Pseudo-label Validation (Worker 11 task 5)
# ---------------------------------------------------------------------------


class VoiceStateValidator:
    """Validate voice_state pseudo-label utility for promoted subsets.

    Checks:
    1. Coverage per dimension (fraction of records with observed mask)
    2. Calibration against known reference values
    3. Controllability uplift (placeholder for downstream measurement)
    """

    def __init__(
        self,
        min_coverage_per_dim: float = 0.50,
        max_calibration_mae: float = 0.20,
        n_dimensions: int = 12,
    ) -> None:
        self.min_coverage_per_dim = min_coverage_per_dim
        self.max_calibration_mae = max_calibration_mae
        self.n_dimensions = n_dimensions

    def check_coverage(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Check voice_state coverage per dimension across records.

        Returns per-dimension coverage fractions and overall pass status.
        """
        n = len(records)
        if n == 0:
            return {"pass": False, "reason": "no_records", "per_dim": []}

        dim_counts = [0] * self.n_dimensions
        for r in records:
            mask = r.attributes.get("voice_state_observed_mask", [])
            for d in range(min(len(mask), self.n_dimensions)):
                if mask[d]:
                    dim_counts[d] += 1

        per_dim = [
            {"dim": d, "coverage": round(dim_counts[d] / n, 4)}
            for d in range(self.n_dimensions)
        ]

        all_pass = all(
            dim_counts[d] / n >= self.min_coverage_per_dim
            for d in range(self.n_dimensions)
        )

        return {
            "pass": all_pass,
            "n_records": n,
            "per_dim": per_dim,
            "threshold": self.min_coverage_per_dim,
        }

    def check_calibration(
        self,
        predicted_states: List[List[float]],
        reference_states: List[List[float]],
    ) -> Dict[str, Any]:
        """Check calibration of voice_state predictions against references.

        Computes per-dimension mean absolute error.

        Args:
            predicted_states: list of predicted 12-D vectors
            reference_states: list of reference 12-D vectors
        """
        n = min(len(predicted_states), len(reference_states))
        if n == 0:
            return {
                "pass": False,
                "reason": "no_samples",
                "per_dim_mae": [],
                "overall_mae": 0.0,
            }

        dim_errors: List[List[float]] = [[] for _ in range(self.n_dimensions)]
        for i in range(n):
            pred = predicted_states[i]
            ref = reference_states[i]
            for d in range(min(len(pred), len(ref), self.n_dimensions)):
                dim_errors[d].append(abs(pred[d] - ref[d]))

        per_dim_mae = []
        for d in range(self.n_dimensions):
            if dim_errors[d]:
                mae = sum(dim_errors[d]) / len(dim_errors[d])
            else:
                mae = 0.0
            per_dim_mae.append(round(mae, 4))

        overall_mae = (
            sum(per_dim_mae) / len(per_dim_mae) if per_dim_mae else 0.0
        )
        passes = overall_mae <= self.max_calibration_mae

        return {
            "pass": passes,
            "n_samples": n,
            "per_dim_mae": per_dim_mae,
            "overall_mae": round(overall_mae, 4),
            "threshold": self.max_calibration_mae,
        }

    def check_controllability_uplift(
        self,
        baseline_metric: float,
        with_labels_metric: float,
    ) -> Dict[str, Any]:
        """Check whether voice_state labels improve controllability.

        A simple comparison: with_labels_metric should be >= baseline_metric.

        Args:
            baseline_metric: performance metric without voice_state labels
            with_labels_metric: performance metric with voice_state labels
        """
        uplift = with_labels_metric - baseline_metric
        return {
            "pass": uplift >= 0.0,
            "baseline": round(baseline_metric, 4),
            "with_labels": round(with_labels_metric, 4),
            "uplift": round(uplift, 4),
        }


# ---------------------------------------------------------------------------
# Provider Recalibrator (Worker 11 task 6)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationRecord:
    """Result of calibrating a provider on held-out samples."""

    provider_id: str
    calibration_version: str
    n_samples: int
    confidence_offset: float
    confidence_scale: float
    ece: float  # expected calibration error
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "calibration_version": self.calibration_version,
            "n_samples": self.n_samples,
            "confidence_offset": round(self.confidence_offset, 6),
            "confidence_scale": round(self.confidence_scale, 6),
            "ece": round(self.ece, 6),
            "timestamp": self.timestamp,
        }


class ProviderRecalibrator:
    """Recalibrate provider confidences on held-out samples.

    Produces CalibrationRecord entries consumed by Worker 09 scoring
    via the ProviderCalibration dataclass in scoring.py.

    Calibration approach:
    - Compare provider confidence to empirical accuracy
    - Compute additive offset and multiplicative scale to align them
    - Emit a versioned calibration record
    """

    def __init__(self, version_prefix: str = "cal") -> None:
        self.version_prefix = version_prefix
        self._calibration_records: List[CalibrationRecord] = []

    @property
    def calibration_records(self) -> List[CalibrationRecord]:
        return list(self._calibration_records)

    def calibrate(
        self,
        provider_id: str,
        confidences: List[float],
        correct: List[bool],
        n_bins: int = 10,
    ) -> CalibrationRecord:
        """Run calibration on held-out samples.

        Args:
            provider_id: the provider being calibrated
            confidences: provider confidence scores [0, 1]
            correct: whether each prediction was correct
            n_bins: number of bins for ECE computation

        Returns:
            CalibrationRecord with offset, scale, and ECE.
        """
        n = min(len(confidences), len(correct))
        if n == 0:
            rec = CalibrationRecord(
                provider_id=provider_id,
                calibration_version=f"{self.version_prefix}_empty",
                n_samples=0,
                confidence_offset=0.0,
                confidence_scale=1.0,
                ece=0.0,
                timestamp=time.time(),
            )
            self._calibration_records.append(rec)
            return rec

        # Compute ECE (expected calibration error)
        bin_width = 1.0 / n_bins
        ece = 0.0
        mean_conf = sum(confidences[:n]) / n
        mean_acc = sum(1 for c in correct[:n] if c) / n

        for b in range(n_bins):
            lo = b * bin_width
            hi = (b + 1) * bin_width
            bin_confs = [
                (confidences[i], correct[i])
                for i in range(n)
                if lo <= confidences[i] < hi
            ]
            if not bin_confs:
                continue
            bin_mean_conf = sum(c for c, _ in bin_confs) / len(bin_confs)
            bin_acc = sum(1 for _, c in bin_confs if c) / len(bin_confs)
            ece += (len(bin_confs) / n) * abs(bin_mean_conf - bin_acc)

        # Simple affine recalibration: offset = mean_acc - mean_conf
        offset = mean_acc - mean_conf
        # Scale: if mean_conf is near zero, keep scale at 1
        scale = 1.0
        if mean_conf > 0.01:
            scale = mean_acc / mean_conf

        version = f"{self.version_prefix}_{provider_id}_{n}_{int(time.time())}"

        rec = CalibrationRecord(
            provider_id=provider_id,
            calibration_version=version,
            n_samples=n,
            confidence_offset=offset,
            confidence_scale=scale,
            ece=ece,
            timestamp=time.time(),
        )
        self._calibration_records.append(rec)
        return rec

    def get_latest_calibration(
        self, provider_id: str
    ) -> Optional[CalibrationRecord]:
        """Return the most recent calibration for a provider."""
        matching = [
            r for r in self._calibration_records
            if r.provider_id == provider_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda r: r.timestamp)


# ---------------------------------------------------------------------------
# Human Workflow Integrity (Worker 11 task 7)
# ---------------------------------------------------------------------------


class HumanWorkflowValidator:
    """Verify audit trail completeness and role separation.

    Checks:
    1. Every promoted/exported record has actor, timestamp, and rationale
    2. Producer != auditor (role separation)
    3. Optimistic locking prevents concurrent edits (via metadata_version)
    """

    def validate_audit_trail(
        self,
        audit_entries: List[Dict[str, Any]],
        promoted_record_ids: Set[str],
    ) -> Dict[str, Any]:
        """Check that every promoted record has audit trail entries.

        Args:
            audit_entries: list of audit log dicts with record_id, actor_id,
                           action_type, timestamp, rationale
            promoted_record_ids: set of record IDs that were promoted
        """
        audited_ids = {e["record_id"] for e in audit_entries}
        missing = promoted_record_ids - audited_ids

        # Check completeness of each entry
        incomplete_entries: List[Dict[str, Any]] = []
        for entry in audit_entries:
            missing_fields = []
            for required in ("record_id", "actor_id", "timestamp"):
                if not entry.get(required):
                    missing_fields.append(required)
            if missing_fields:
                incomplete_entries.append({
                    "record_id": entry.get("record_id", "unknown"),
                    "missing_fields": missing_fields,
                })

        return {
            "pass": len(missing) == 0 and len(incomplete_entries) == 0,
            "promoted_without_audit": sorted(missing)[:20],
            "incomplete_entries": incomplete_entries[:20],
            "total_audit_entries": len(audit_entries),
            "total_promoted": len(promoted_record_ids),
        }

    def validate_role_separation(
        self,
        audit_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check that producer and auditor are different people.

        Groups entries by record_id and checks that no single actor
        both produced (action_type="promote") and audited
        (action_type="audit") the same record.
        """
        per_record: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for entry in audit_entries:
            record_id = entry.get("record_id", "")
            actor = entry.get("actor_id", "")
            action = entry.get("action_type", "")
            per_record[record_id][action].add(actor)

        violations: List[Dict[str, Any]] = []
        for record_id, actions in per_record.items():
            producers = actions.get("promote", set()) | actions.get("score", set())
            auditors = actions.get("audit", set()) | actions.get("review", set())
            overlap = producers & auditors
            if overlap:
                violations.append({
                    "record_id": record_id,
                    "overlapping_actors": sorted(overlap),
                })

        return {
            "pass": len(violations) == 0,
            "violations": violations[:20],
            "total_records_checked": len(per_record),
        }

    def validate_optimistic_locking(
        self,
        records: List[CurationRecord],
    ) -> Dict[str, Any]:
        """Check that all promoted records have metadata_version > 1.

        Records that have been through the pipeline should have been
        updated at least once (version incremented from 1).
        """
        promoted = [r for r in records if r.status == RecordStatus.PROMOTED]
        unversioned = [
            r.record_id for r in promoted if r.metadata_version <= 1
        ]

        return {
            "pass": len(unversioned) == 0,
            "total_promoted": len(promoted),
            "unversioned_records": unversioned[:20],
        }


# ---------------------------------------------------------------------------
# Split Integrity (Worker 11 task 8)
# ---------------------------------------------------------------------------


class SplitIntegrityValidator:
    """Verify holdout/training split integrity.

    Checks:
    1. Holdout samples never appear in training buckets
    2. Split assignments are deterministic (based on audio_hash)
    3. Same-text contamination is detected
    """

    def __init__(self, holdout_fraction: float = 0.05) -> None:
        self.holdout_fraction = holdout_fraction

    def validate_no_holdout_leakage(
        self,
        records: List[CurationRecord],
        holdout_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Verify that holdout records never appear in training buckets.

        Delegates to CurationValidator.validate_holdout_integrity but
        adds same-text contamination checking.
        """
        if holdout_ids is None:
            holdout_ids = {
                r.record_id for r in records
                if r.promotion_bucket == PromotionBucket.HOLDOUT_EVAL
            }

        train_buckets = {
            PromotionBucket.TTS_MAINLINE,
            PromotionBucket.VC_PRIOR,
            PromotionBucket.EXPRESSIVE_PRIOR,
        }

        direct_leaks = []
        for r in records:
            if r.record_id in holdout_ids and r.promotion_bucket in train_buckets:
                direct_leaks.append(r.record_id)

        # Same-text contamination
        text_to_ids: Dict[str, List[str]] = defaultdict(list)
        for r in records:
            if r.transcript:
                fp = hashlib.sha256(
                    r.transcript.strip().lower().encode("utf-8")
                ).hexdigest()[:16]
                text_to_ids[fp].append(r.record_id)

        text_leaks: List[Dict[str, Any]] = []
        for fp, ids in text_to_ids.items():
            has_holdout = any(rid in holdout_ids for rid in ids)
            has_train = any(
                any(
                    r.promotion_bucket in train_buckets
                    for r in records
                    if r.record_id == rid
                )
                for rid in ids
            )
            if has_holdout and has_train:
                text_leaks.append({
                    "text_fingerprint": fp,
                    "record_ids": ids,
                })

        return {
            "pass": len(direct_leaks) == 0 and len(text_leaks) == 0,
            "direct_leaks": direct_leaks,
            "text_contamination_leaks": text_leaks[:20],
            "holdout_size": len(holdout_ids),
        }

    def validate_deterministic_splits(
        self,
        records: List[CurationRecord],
    ) -> Dict[str, Any]:
        """Verify that split assignments are deterministic.

        Uses audio_hash to compute expected split assignment and
        checks that actual assignment matches.
        """
        mismatches: List[Dict[str, Any]] = []

        for r in records:
            if r.promotion_bucket == PromotionBucket.NONE:
                continue

            # Deterministic split: hash the audio_hash to decide holdout
            hash_val = int(hashlib.sha256(
                r.audio_hash.encode("utf-8")
            ).hexdigest()[:8], 16)
            expected_holdout = (hash_val % 100) < (self.holdout_fraction * 100)

            actual_holdout = (
                r.promotion_bucket == PromotionBucket.HOLDOUT_EVAL
            )

            if expected_holdout and not actual_holdout:
                mismatches.append({
                    "record_id": r.record_id,
                    "expected": "holdout_eval",
                    "actual": r.promotion_bucket.value,
                })

        return {
            "pass": len(mismatches) == 0,
            "mismatches": mismatches[:20],
            "total_checked": sum(
                1 for r in records
                if r.promotion_bucket != PromotionBucket.NONE
            ),
        }


# ---------------------------------------------------------------------------
# Comprehensive Validation Runner (Worker 11 integration)
# ---------------------------------------------------------------------------


class ComprehensiveValidator:
    """Run all Worker 11 validation layers and produce a unified report.

    Combines:
    - CurationValidator checks
    - StageBenchmark results (when provided)
    - SampleAuditor report (when provided)
    - DownstreamComparison (when provided)
    - VoiceStateValidator checks
    - HumanWorkflowValidator checks
    - SplitIntegrityValidator checks
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
    ) -> None:
        self.curation = CurationValidator(config)
        self.human_workflow = HumanWorkflowValidator()
        self.split_integrity = SplitIntegrityValidator()
        self.voice_state = VoiceStateValidator()

    def run_all(
        self,
        records: List[CurationRecord],
        audit_entries: Optional[List[Dict[str, Any]]] = None,
        holdout_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Run all validation checks and return a unified report."""
        results: Dict[str, Any] = {}

        # Layer 1 & 2: Core curation checks
        results["promotion_distribution"] = (
            self.curation.validate_promotion_distribution(records)
        )
        results["legality_gating"] = (
            self.curation.validate_legality_gating(records)
        )
        results["holdout_integrity"] = (
            self.curation.validate_holdout_integrity(records, holdout_ids)
        )
        results["provenance_completeness"] = (
            self.curation.validate_provenance_completeness(records)
        )

        # Layer 4: Human workflow
        if audit_entries is not None:
            promoted_ids = {
                r.record_id for r in records
                if r.status == RecordStatus.PROMOTED
            }
            results["audit_trail"] = (
                self.human_workflow.validate_audit_trail(
                    audit_entries, promoted_ids
                )
            )
            results["role_separation"] = (
                self.human_workflow.validate_role_separation(audit_entries)
            )

        results["optimistic_locking"] = (
            self.human_workflow.validate_optimistic_locking(records)
        )

        # Split integrity
        results["holdout_leakage"] = (
            self.split_integrity.validate_no_holdout_leakage(
                records, holdout_ids
            )
        )

        # Voice state coverage
        promoted = [r for r in records if r.status == RecordStatus.PROMOTED]
        if promoted:
            results["voice_state_coverage"] = (
                self.voice_state.check_coverage(promoted)
            )

        # Overall pass
        check_results = [
            v for k, v in results.items()
            if isinstance(v, dict) and "pass" in v
        ]
        all_pass = all(v.get("pass", False) for v in check_results)
        results["overall"] = {
            "pass": all_pass,
            "n_checks": len(check_results),
            "n_passed": sum(1 for v in check_results if v.get("pass", False)),
        }

        return results

    def save_report(
        self, report: Dict[str, Any], output_path: Path | str
    ) -> None:
        """Save comprehensive validation report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Comprehensive validation report saved to %s", path)
