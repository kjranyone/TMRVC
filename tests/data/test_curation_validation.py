"""Tests for Worker 11: Curation Validation and Acceptance.

Covers all validation layers:
- StageBenchmark: per-provider audit on known-good samples
- SampleAuditor: human review workflow with dual-review
- DownstreamComparison: naive vs curated quality uplift
- ProviderAcceptanceThresholds: per-provider acceptance criteria
- VoiceStateValidator: pseudo-label coverage and calibration
- ProviderRecalibrator: confidence recalibration on held-out data
- HumanWorkflowValidator: audit trail, role separation, locking
- SplitIntegrityValidator: holdout leakage and deterministic splits
- ComprehensiveValidator: unified validation runner
"""

from __future__ import annotations

import json
import time

import pytest

from tmrvc_data.curation.models import (
    CurationRecord,
    RecordStatus,
    PromotionBucket,
    Provenance,
)
from tmrvc_data.curation.validation import (
    CurationValidator,
    ValidationConfig,
    ComprehensiveValidator,
    DownstreamComparison,
    HumanWorkflowValidator,
    ProviderAcceptanceThresholds,
    ProviderRecalibrator,
    QualityMetrics,
    SampleAuditor,
    SplitIntegrityValidator,
    StageBenchmark,
    VoiceStateValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    record_id: str = "r1",
    status: RecordStatus = RecordStatus.PROMOTED,
    bucket: PromotionBucket = PromotionBucket.TTS_MAINLINE,
    source_legality: str = "owned",
    transcript: str = "hello world",
    transcript_confidence: float = 0.95,
    quality_score: float = 0.9,
    metadata_version: int = 2,
    audio_hash: str = "abc123",
    attributes: dict | None = None,
    providers: dict | None = None,
) -> CurationRecord:
    r = CurationRecord(
        record_id=record_id,
        source_path="/tmp/audio.wav",
        audio_hash=audio_hash,
        transcript=transcript,
        transcript_confidence=transcript_confidence,
        quality_score=quality_score,
        status=status,
        promotion_bucket=bucket,
        source_legality=source_legality,
        metadata_version=metadata_version,
        duration_sec=5.0,
        language="en",
    )
    if attributes:
        r.attributes = attributes
    if providers is not None:
        r.providers = providers
    else:
        r.providers = {
            "asr": Provenance(
                stage="asr", provider="whisper", version="1.0",
                timestamp=1000.0, confidence=0.9,
            )
        }
    return r


def _make_records(n: int = 10, **kwargs) -> list[CurationRecord]:
    return [_make_record(record_id=f"r{i}", audio_hash=f"hash_{i}", **kwargs) for i in range(n)]


# ===========================================================================
# Test: ProviderAcceptanceThresholds
# ===========================================================================


class TestProviderAcceptanceThresholds:
    def test_asr_passes(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="whisper_v3")
        result = thresholds.check_asr(wer=0.10, calibration_ece=0.05)
        assert result["pass"] is True
        assert result["provider_id"] == "whisper_v3"

    def test_asr_fails_high_wer(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="whisper_v3")
        result = thresholds.check_asr(wer=0.25, calibration_ece=0.05)
        assert result["pass"] is False
        assert result["wer"]["pass"] is False

    def test_asr_fails_high_ece(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="whisper_v3")
        result = thresholds.check_asr(wer=0.10, calibration_ece=0.20)
        assert result["pass"] is False
        assert result["confidence_calibration_ece"]["pass"] is False

    def test_diarization_passes(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="pyannote")
        result = thresholds.check_diarization(der=0.15, speaker_count_accuracy=0.90)
        assert result["pass"] is True

    def test_diarization_fails_high_der(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="pyannote")
        result = thresholds.check_diarization(der=0.30, speaker_count_accuracy=0.90)
        assert result["pass"] is False
        assert result["der"]["pass"] is False

    def test_diarization_fails_low_speaker_accuracy(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="pyannote")
        result = thresholds.check_diarization(der=0.10, speaker_count_accuracy=0.70)
        assert result["pass"] is False

    def test_voice_state_passes(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="vs_estimator")
        result = thresholds.check_voice_state(coverage=0.80, calibration_mae=0.10)
        assert result["pass"] is True

    def test_voice_state_fails_low_coverage(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="vs_estimator")
        result = thresholds.check_voice_state(coverage=0.30, calibration_mae=0.10)
        assert result["pass"] is False

    def test_voice_state_fails_high_mae(self):
        thresholds = ProviderAcceptanceThresholds(provider_id="vs_estimator")
        result = thresholds.check_voice_state(coverage=0.80, calibration_mae=0.30)
        assert result["pass"] is False

    def test_calibration_version_tracked(self):
        thresholds = ProviderAcceptanceThresholds(
            provider_id="test", calibration_version="v2.1"
        )
        result = thresholds.check_asr(wer=0.10, calibration_ece=0.05)
        assert result["calibration_version"] == "v2.1"


# ===========================================================================
# Test: StageBenchmark
# ===========================================================================


class TestStageBenchmark:
    def test_audit_perfect_accuracy(self):
        benchmark = StageBenchmark()
        preds = [{"transcript": "hello"}, {"transcript": "world"}]
        refs = [{"transcript": "hello"}, {"transcript": "world"}]
        result = benchmark.audit_stage("asr", "whisper", preds, refs)
        assert result.accuracy == 1.0
        assert result.n_correct == 2
        assert result.n_samples == 2
        assert len(result.errors) == 0

    def test_audit_partial_accuracy(self):
        benchmark = StageBenchmark()
        preds = [{"transcript": "hello"}, {"transcript": "wrld"}]
        refs = [{"transcript": "hello"}, {"transcript": "world"}]
        result = benchmark.audit_stage("asr", "whisper", preds, refs)
        assert result.accuracy == 0.5
        assert result.n_correct == 1
        assert len(result.errors) == 1

    def test_audit_empty_samples(self):
        benchmark = StageBenchmark()
        result = benchmark.audit_stage("asr", "whisper", [], [])
        assert result.accuracy == 0.0
        assert result.n_samples == 0

    def test_validate_output_format_passes(self):
        benchmark = StageBenchmark()
        outputs = [{"transcript": "hi", "confidence": 0.9}]
        violations = benchmark.validate_output_format(
            "asr", outputs, ["transcript", "confidence"]
        )
        assert len(violations) == 0

    def test_validate_output_format_missing_keys(self):
        benchmark = StageBenchmark()
        outputs = [{"transcript": "hi"}]
        violations = benchmark.validate_output_format(
            "asr", outputs, ["transcript", "confidence"]
        )
        assert len(violations) == 1
        assert "confidence" in violations[0]

    def test_run_benchmark_combined(self):
        benchmark = StageBenchmark()
        preds = [{"transcript": "hello", "confidence": 0.9}]
        refs = [{"transcript": "hello"}]
        result = benchmark.run_benchmark(
            "asr", "whisper", preds, refs,
            required_keys=["transcript", "confidence"],
        )
        assert result.accuracy == 1.0
        assert len(result.format_violations) == 0

    def test_run_benchmark_with_format_violations(self):
        benchmark = StageBenchmark()
        preds = [{"transcript": "hello"}]
        refs = [{"transcript": "hello"}]
        result = benchmark.run_benchmark(
            "asr", "whisper", preds, refs,
            required_keys=["transcript", "confidence"],
        )
        assert result.accuracy == 1.0
        assert len(result.format_violations) == 1

    def test_custom_match_key(self):
        benchmark = StageBenchmark()
        preds = [{"speaker_count": 2}]
        refs = [{"speaker_count": 2}]
        result = benchmark.audit_stage(
            "diarization", "pyannote", preds, refs,
            match_key="speaker_count",
        )
        assert result.accuracy == 1.0


# ===========================================================================
# Test: SampleAuditor
# ===========================================================================


class TestSampleAuditor:
    def test_submit_audit(self):
        auditor = SampleAuditor()
        entry = auditor.submit_audit("r1", "auditor_A", "correct", "looks good")
        assert entry.record_id == "r1"
        assert entry.decision == "correct"
        assert len(auditor.audit_log) == 1

    def test_invalid_decision_raises(self):
        auditor = SampleAuditor()
        with pytest.raises(ValueError, match="Invalid decision"):
            auditor.submit_audit("r1", "auditor_A", "bad_value")

    def test_get_audits_for_record(self):
        auditor = SampleAuditor()
        auditor.submit_audit("r1", "A", "correct")
        auditor.submit_audit("r2", "A", "incorrect")
        auditor.submit_audit("r1", "B", "correct")
        audits = auditor.get_audits_for_record("r1")
        assert len(audits) == 2

    def test_dual_review_passes(self):
        auditor = SampleAuditor()
        auditor.submit_audit("r1", "auditor_A", "correct")
        auditor.submit_audit("r1", "auditor_B", "correct")
        result = auditor.check_dual_review("r1")
        assert result["pass"] is True
        assert result["agreement"] is True
        assert result["n_unique_auditors"] == 2

    def test_dual_review_fails_single_reviewer(self):
        auditor = SampleAuditor()
        auditor.submit_audit("r1", "auditor_A", "correct")
        result = auditor.check_dual_review("r1")
        assert result["pass"] is False
        assert result["reason"] == "insufficient_reviews"

    def test_dual_review_same_auditor_fails(self):
        auditor = SampleAuditor()
        auditor.submit_audit("r1", "auditor_A", "correct")
        auditor.submit_audit("r1", "auditor_A", "correct")
        result = auditor.check_dual_review("r1")
        assert result["pass"] is False  # same auditor, not dual

    def test_dual_review_disagreement(self):
        auditor = SampleAuditor()
        auditor.submit_audit("r1", "auditor_A", "correct")
        auditor.submit_audit("r1", "auditor_B", "incorrect")
        result = auditor.check_dual_review("r1")
        assert result["pass"] is True
        assert result["agreement"] is False

    def test_generate_audit_report(self):
        auditor = SampleAuditor()
        auditor.submit_audit("r1", "A", "correct", bucket="tts_mainline")
        auditor.submit_audit("r2", "A", "incorrect", bucket="tts_mainline")
        auditor.submit_audit("r3", "B", "correct", bucket="rejected")
        report = auditor.generate_audit_report([])
        assert report["total_audited"] == 3
        assert report["false_promote_rate"] == 0.5  # 1 of 2 promoted audits incorrect
        assert report["false_reject_rate"] == 0.0

    def test_generate_audit_report_empty(self):
        auditor = SampleAuditor()
        report = auditor.generate_audit_report([])
        assert report["total_audited"] == 0
        assert report["false_promote_rate"] == 0.0


# ===========================================================================
# Test: DownstreamComparison
# ===========================================================================


class TestDownstreamComparison:
    def test_compute_metrics(self):
        records = _make_records(5, quality_score=0.85)
        metrics = DownstreamComparison.compute_metrics(records, label="curated")
        assert metrics.label == "curated"
        assert metrics.n_samples == 5
        assert metrics.mean_quality_score == pytest.approx(0.85)

    def test_compute_metrics_empty(self):
        metrics = DownstreamComparison.compute_metrics([], label="empty")
        assert metrics.n_samples == 0

    def test_compare_shows_uplift(self):
        naive = QualityMetrics(
            label="naive", n_samples=100,
            mean_transcript_confidence=0.70,
            mean_quality_score=0.60,
            mean_diarization_confidence=0.50,
            mean_snr_db=15.0,
            coverage_voice_state=0.30,
            reject_rate=0.40,
        )
        curated = QualityMetrics(
            label="curated", n_samples=80,
            mean_transcript_confidence=0.90,
            mean_quality_score=0.85,
            mean_diarization_confidence=0.80,
            mean_snr_db=25.0,
            coverage_voice_state=0.70,
            reject_rate=0.10,
        )
        result = DownstreamComparison.compare(naive, curated)
        assert result["pass"] is True
        assert result["uplift"]["mean_quality_score"] > 0
        assert result["uplift"]["reject_rate_reduction"] > 0

    def test_compare_fails_when_no_uplift(self):
        naive = QualityMetrics(
            label="naive", n_samples=100,
            mean_quality_score=0.90,
            reject_rate=0.05,
        )
        curated = QualityMetrics(
            label="curated", n_samples=100,
            mean_quality_score=0.80,  # worse
            reject_rate=0.10,
        )
        result = DownstreamComparison.compare(naive, curated)
        assert result["pass"] is False

    def test_compare_with_voice_state_coverage(self):
        records_naive = _make_records(5, attributes={})
        records_curated = _make_records(
            5,
            attributes={
                "voice_state_observed_mask": [True] * 8,
            },
        )
        naive = DownstreamComparison.compute_metrics(records_naive, "naive")
        curated = DownstreamComparison.compute_metrics(records_curated, "curated")
        result = DownstreamComparison.compare(naive, curated)
        assert result["uplift"]["coverage_voice_state"] > 0


# ===========================================================================
# Test: VoiceStateValidator
# ===========================================================================


class TestVoiceStateValidator:
    def test_coverage_passes_all_observed(self):
        validator = VoiceStateValidator(min_coverage_per_dim=0.50)
        records = _make_records(
            5,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        result = validator.check_coverage(records)
        assert result["pass"] is True
        assert result["n_records"] == 5

    def test_coverage_fails_none_observed(self):
        validator = VoiceStateValidator(min_coverage_per_dim=0.50)
        records = _make_records(
            5,
            attributes={"voice_state_observed_mask": [False] * 8},
        )
        result = validator.check_coverage(records)
        assert result["pass"] is False

    def test_coverage_empty_records(self):
        validator = VoiceStateValidator()
        result = validator.check_coverage([])
        assert result["pass"] is False

    def test_coverage_partial(self):
        validator = VoiceStateValidator(min_coverage_per_dim=0.50)
        # 3 records with all observed, 2 with none
        records = (
            _make_records(3, attributes={"voice_state_observed_mask": [True] * 8})
            + _make_records(2, attributes={"voice_state_observed_mask": [False] * 8})
        )
        # Deduplicate record IDs
        for i, r in enumerate(records):
            r.record_id = f"r_{i}"
        result = validator.check_coverage(records)
        # 3/5 = 0.6 >= 0.5
        assert result["pass"] is True

    def test_calibration_passes(self):
        validator = VoiceStateValidator(max_calibration_mae=0.20)
        predicted = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
        reference = [[0.5, 0.6, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]]
        result = validator.check_calibration(predicted, reference)
        assert result["pass"] is True
        assert result["overall_mae"] <= 0.20

    def test_calibration_fails_high_mae(self):
        validator = VoiceStateValidator(max_calibration_mae=0.05)
        predicted = [[0.1] * 8]
        reference = [[0.9] * 8]
        result = validator.check_calibration(predicted, reference)
        assert result["pass"] is False
        assert result["overall_mae"] > 0.05

    def test_calibration_empty(self):
        validator = VoiceStateValidator()
        result = validator.check_calibration([], [])
        assert result["pass"] is False

    def test_controllability_uplift_passes(self):
        validator = VoiceStateValidator()
        result = validator.check_controllability_uplift(
            baseline_metric=0.60, with_labels_metric=0.75
        )
        assert result["pass"] is True
        assert result["uplift"] == pytest.approx(0.15)

    def test_controllability_uplift_fails(self):
        validator = VoiceStateValidator()
        result = validator.check_controllability_uplift(
            baseline_metric=0.80, with_labels_metric=0.70
        )
        assert result["pass"] is False
        assert result["uplift"] < 0


# ===========================================================================
# Test: ProviderRecalibrator
# ===========================================================================


class TestProviderRecalibrator:
    def test_calibrate_well_calibrated(self):
        recal = ProviderRecalibrator(version_prefix="test")
        # Provider says 0.8 confidence, and is correct 80% of the time
        confidences = [0.8] * 10
        correct = [True] * 8 + [False] * 2
        rec = recal.calibrate("whisper", confidences, correct)
        assert rec.provider_id == "whisper"
        assert rec.n_samples == 10
        assert rec.ece < 0.05  # well-calibrated
        assert "test_whisper_" in rec.calibration_version

    def test_calibrate_overconfident(self):
        recal = ProviderRecalibrator()
        # Provider says 0.95 confidence but only correct 50% of the time
        confidences = [0.95] * 20
        correct = [True] * 10 + [False] * 10
        rec = recal.calibrate("overconfident_asr", confidences, correct)
        assert rec.confidence_offset < 0  # needs downward correction
        assert rec.ece > 0.1

    def test_calibrate_underconfident(self):
        recal = ProviderRecalibrator()
        # Provider says 0.3 confidence but correct 90% of the time
        confidences = [0.3] * 10
        correct = [True] * 9 + [False] * 1
        rec = recal.calibrate("underconf_asr", confidences, correct)
        assert rec.confidence_offset > 0  # needs upward correction

    def test_calibrate_empty(self):
        recal = ProviderRecalibrator()
        rec = recal.calibrate("empty", [], [])
        assert rec.n_samples == 0
        assert "empty" in rec.calibration_version

    def test_get_latest_calibration(self):
        recal = ProviderRecalibrator()
        recal.calibrate("asr_a", [0.8], [True])
        recal.calibrate("asr_b", [0.7], [False])
        recal.calibrate("asr_a", [0.9], [True])

        latest = recal.get_latest_calibration("asr_a")
        assert latest is not None
        assert latest.n_samples == 1
        # Should be the second calibration for asr_a
        assert latest.timestamp >= recal.calibration_records[0].timestamp

    def test_get_latest_calibration_missing(self):
        recal = ProviderRecalibrator()
        assert recal.get_latest_calibration("nonexistent") is None

    def test_calibration_record_to_dict(self):
        recal = ProviderRecalibrator()
        rec = recal.calibrate("test", [0.5, 0.6], [True, False])
        d = rec.to_dict()
        assert d["provider_id"] == "test"
        assert "calibration_version" in d
        assert "ece" in d


# ===========================================================================
# Test: HumanWorkflowValidator
# ===========================================================================


class TestHumanWorkflowValidator:
    def test_audit_trail_complete(self):
        validator = HumanWorkflowValidator()
        entries = [
            {"record_id": "r0", "actor_id": "user1", "action_type": "promote", "timestamp": "2024-01-01"},
            {"record_id": "r1", "actor_id": "user2", "action_type": "promote", "timestamp": "2024-01-01"},
        ]
        result = validator.validate_audit_trail(entries, {"r0", "r1"})
        assert result["pass"] is True

    def test_audit_trail_missing_records(self):
        validator = HumanWorkflowValidator()
        entries = [
            {"record_id": "r0", "actor_id": "user1", "action_type": "promote", "timestamp": "2024-01-01"},
        ]
        result = validator.validate_audit_trail(entries, {"r0", "r1"})
        assert result["pass"] is False
        assert "r1" in result["promoted_without_audit"]

    def test_audit_trail_incomplete_entry(self):
        validator = HumanWorkflowValidator()
        entries = [
            {"record_id": "r0", "actor_id": "", "action_type": "promote", "timestamp": "2024-01-01"},
        ]
        result = validator.validate_audit_trail(entries, {"r0"})
        assert result["pass"] is False
        assert len(result["incomplete_entries"]) == 1

    def test_role_separation_passes(self):
        validator = HumanWorkflowValidator()
        entries = [
            {"record_id": "r0", "actor_id": "user1", "action_type": "promote"},
            {"record_id": "r0", "actor_id": "user2", "action_type": "audit"},
        ]
        result = validator.validate_role_separation(entries)
        assert result["pass"] is True

    def test_role_separation_violation(self):
        validator = HumanWorkflowValidator()
        entries = [
            {"record_id": "r0", "actor_id": "user1", "action_type": "promote"},
            {"record_id": "r0", "actor_id": "user1", "action_type": "audit"},
        ]
        result = validator.validate_role_separation(entries)
        assert result["pass"] is False
        assert len(result["violations"]) == 1
        assert "user1" in result["violations"][0]["overlapping_actors"]

    def test_role_separation_different_records_ok(self):
        validator = HumanWorkflowValidator()
        entries = [
            {"record_id": "r0", "actor_id": "user1", "action_type": "promote"},
            {"record_id": "r1", "actor_id": "user1", "action_type": "audit"},
        ]
        result = validator.validate_role_separation(entries)
        assert result["pass"] is True

    def test_optimistic_locking_passes(self):
        validator = HumanWorkflowValidator()
        records = _make_records(5, metadata_version=3)
        result = validator.validate_optimistic_locking(records)
        assert result["pass"] is True

    def test_optimistic_locking_fails_unversioned(self):
        validator = HumanWorkflowValidator()
        records = _make_records(5, metadata_version=1)
        result = validator.validate_optimistic_locking(records)
        assert result["pass"] is False
        assert len(result["unversioned_records"]) == 5


# ===========================================================================
# Test: SplitIntegrityValidator
# ===========================================================================


class TestSplitIntegrityValidator:
    def test_no_holdout_leakage(self):
        validator = SplitIntegrityValidator()
        records = _make_records(5)
        result = validator.validate_no_holdout_leakage(records, holdout_ids=set())
        assert result["pass"] is True

    def test_holdout_direct_leak_detected(self):
        validator = SplitIntegrityValidator()
        records = _make_records(5)
        # r0 is in holdout but assigned to tts_mainline
        result = validator.validate_no_holdout_leakage(records, holdout_ids={"r0"})
        assert result["pass"] is False
        assert "r0" in result["direct_leaks"]

    def test_holdout_text_contamination(self):
        validator = SplitIntegrityValidator()
        # Two records with same transcript, one holdout, one training
        holdout_rec = _make_record(
            record_id="h1",
            bucket=PromotionBucket.HOLDOUT_EVAL,
            transcript="identical text",
        )
        train_rec = _make_record(
            record_id="t1",
            bucket=PromotionBucket.TTS_MAINLINE,
            transcript="identical text",
        )
        result = validator.validate_no_holdout_leakage(
            [holdout_rec, train_rec],
            holdout_ids={"h1"},
        )
        assert result["pass"] is False
        assert len(result["text_contamination_leaks"]) > 0

    def test_no_text_contamination_different_texts(self):
        validator = SplitIntegrityValidator()
        holdout_rec = _make_record(
            record_id="h1",
            bucket=PromotionBucket.HOLDOUT_EVAL,
            transcript="unique holdout text",
        )
        train_rec = _make_record(
            record_id="t1",
            bucket=PromotionBucket.TTS_MAINLINE,
            transcript="different training text",
        )
        result = validator.validate_no_holdout_leakage(
            [holdout_rec, train_rec],
            holdout_ids={"h1"},
        )
        assert result["pass"] is True

    def test_deterministic_splits(self):
        validator = SplitIntegrityValidator(holdout_fraction=0.0)
        records = _make_records(5)
        result = validator.validate_deterministic_splits(records)
        # With 0% holdout fraction, no records should be expected in holdout
        assert result["pass"] is True

    def test_deterministic_splits_with_holdout(self):
        validator = SplitIntegrityValidator(holdout_fraction=1.0)
        # All records expected to be holdout but are in tts_mainline
        records = _make_records(3)
        result = validator.validate_deterministic_splits(records)
        # All should be mismatched
        assert result["pass"] is False
        assert len(result["mismatches"]) == 3


# ===========================================================================
# Test: ComprehensiveValidator
# ===========================================================================


class TestComprehensiveValidator:
    def test_run_all_passes(self):
        validator = ComprehensiveValidator()
        records = _make_records(
            10, metadata_version=3,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        report = validator.run_all(records)
        assert report["overall"]["pass"] is True
        assert "promotion_distribution" in report
        assert "legality_gating" in report
        assert "holdout_integrity" in report

    def test_run_all_with_audit_entries(self):
        validator = ComprehensiveValidator()
        records = _make_records(
            3, metadata_version=3,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        audit_entries = [
            {
                "record_id": f"r{i}",
                "actor_id": "system",
                "action_type": "promote",
                "timestamp": "2024-01-01",
            }
            for i in range(3)
        ]
        report = validator.run_all(records, audit_entries=audit_entries)
        assert "audit_trail" in report
        assert report["audit_trail"]["pass"] is True

    def test_run_all_fails_legality(self):
        validator = ComprehensiveValidator()
        records = _make_records(5, metadata_version=3)
        records[0].source_legality = "unknown"
        report = validator.run_all(records)
        assert report["overall"]["pass"] is False
        assert report["legality_gating"]["pass"] is False

    def test_run_all_with_holdout_ids(self):
        validator = ComprehensiveValidator()
        records = _make_records(
            5, metadata_version=3,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        report = validator.run_all(records, holdout_ids={"r0"})
        # r0 is in tts_mainline but should be holdout
        assert report["holdout_integrity"]["pass"] is False

    def test_save_report(self, tmp_path):
        validator = ComprehensiveValidator()
        records = _make_records(
            5, metadata_version=3,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        report = validator.run_all(records)
        path = tmp_path / "comprehensive_report.json"
        validator.save_report(report, path)
        loaded = json.loads(path.read_text())
        assert loaded["overall"]["pass"] is True

    def test_overall_counts(self):
        validator = ComprehensiveValidator()
        records = _make_records(
            5, metadata_version=3,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        report = validator.run_all(records)
        assert report["overall"]["n_checks"] > 0
        assert report["overall"]["n_passed"] == report["overall"]["n_checks"]

    def test_voice_state_coverage_included(self):
        validator = ComprehensiveValidator()
        records = _make_records(
            5,
            metadata_version=3,
            attributes={"voice_state_observed_mask": [True] * 8},
        )
        report = validator.run_all(records)
        assert "voice_state_coverage" in report


# ===========================================================================
# Test: Legacy CurationValidator (preserved behavior)
# ===========================================================================


class TestLegacyValidation:
    def test_run_all_preserved(self):
        validator = CurationValidator()
        records = _make_records(10)
        report = validator.run_all(records)
        assert report["overall"]["pass"] is True

    def test_save_report(self, tmp_path):
        validator = CurationValidator()
        records = _make_records(10)
        report = validator.run_all(records)
        path = tmp_path / "report.json"
        validator.save_report(report, path)
        loaded = json.loads(path.read_text())
        assert loaded["overall"]["pass"] is True
