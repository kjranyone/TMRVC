"""Tests for Worker 09: Quality Scoring, Rejection, and Promotion.

Covers:
- Promotion buckets with explicit thresholds
- Score weights and quality_score computation
- Anti-contamination (holdout isolation, same-text clustering)
- Human approval policy
- Voice-state promotion policy
- Calibration-aware thresholds
- Reporting (histogram, rejection breakdown, bucket summary)
"""
from __future__ import annotations

import pytest
from copy import deepcopy

from tmrvc_data.curation.models import (
    CurationRecord,
    PromotionBucket,
    RecordStatus,
)
from tmrvc_data.curation.scoring import (
    ApprovalPolicy,
    BucketThresholds,
    DEFAULT_THRESHOLDS,
    ProviderCalibration,
    QualityScoringEngine,
    ScoreWeights,
    ScoringConfig,
    VoiceStateCoveragePolicy,
    cluster_same_text,
    enforce_same_text_holdout_isolation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> CurationRecord:
    """Build a high-quality record suitable for tts_mainline promotion."""
    defaults = dict(
        record_id="r1",
        source_path="/audio/test.wav",
        audio_hash="abc123",
        transcript="hello world",
        transcript_confidence=0.95,
        diarization_confidence=0.90,
        duration_sec=5.0,
        source_legality="owned",
        language="ja",
        quality_score=0.0,
        attributes={
            "snr_db": 30.0,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "voice_state_density": 0.8,
            "voice_state_confidence": 0.7,
            "voice_state_observed_mask": [True] * 8,
        },
    )
    defaults.update(overrides)
    attrs = defaults.pop("attributes", {})
    r = CurationRecord(**defaults)
    r.attributes = attrs
    return r


# ---------------------------------------------------------------------------
# 1. Promotion Buckets
# ---------------------------------------------------------------------------


class TestPromotionBuckets:
    """Bucket-specific threshold enforcement."""

    def test_tts_mainline_thresholds(self):
        t = DEFAULT_THRESHOLDS["tts_mainline"]
        assert t.transcript_confidence == 0.90
        assert t.cross_asr_agreement == 0.85
        assert t.quality_score == 0.85

    def test_vc_prior_thresholds(self):
        t = DEFAULT_THRESHOLDS["vc_prior"]
        assert t.transcript_confidence == 0.60
        assert t.quality_score == 0.70

    def test_expressive_prior_thresholds(self):
        t = DEFAULT_THRESHOLDS["expressive_prior"]
        assert t.transcript_confidence == 0.50
        assert t.quality_score == 0.75

    def test_holdout_eval_thresholds(self):
        t = DEFAULT_THRESHOLDS["holdout_eval"]
        assert t.transcript_confidence == 0.90
        assert t.quality_score == 0.90

    def test_promote_to_tts_mainline(self):
        engine = QualityScoringEngine()
        record = _make_record()
        result = engine.score_and_decide(record)
        assert result.promotion_bucket == PromotionBucket.TTS_MAINLINE

    def test_promote_to_vc_prior(self):
        """Record with moderate transcript confidence goes to vc_prior."""
        engine = QualityScoringEngine()
        record = _make_record(
            transcript_confidence=0.65,
            diarization_confidence=0.3,
            attributes={
                "snr_db": 25.0,
                "refinement_agreement": 0.65,
                "voice_state_density": 0.5,
                "voice_state_confidence": 0.4,
            },
        )
        result = engine.score_and_decide(record)
        # Should not qualify for tts_mainline (low agreement) but
        # may qualify for vc_prior or expressive_prior depending on score
        assert result.promotion_bucket in (
            PromotionBucket.VC_PRIOR,
            PromotionBucket.EXPRESSIVE_PRIOR,
            PromotionBucket.NONE,
        )

    def test_promote_to_expressive_prior_requires_prosody(self):
        """Expressive prior requires prosody data."""
        engine = QualityScoringEngine()
        record = _make_record(
            transcript_confidence=0.55,
            diarization_confidence=0.3,
            attributes={
                "snr_db": 25.0,
                "refinement_agreement": 0.55,
                "voice_state_density": 0.6,
                "voice_state_confidence": 0.5,
                # No pause_events, breath_events, or voice_state_observed_mask
            },
        )
        result = engine.score_and_decide(record)
        # Without prosody data, should not get expressive_prior
        assert result.promotion_bucket != PromotionBucket.EXPRESSIVE_PRIOR

    def test_holdout_forced_by_config(self):
        config = ScoringConfig(holdout_record_ids={"r1"})
        engine = QualityScoringEngine(config)
        record = _make_record()
        record.quality_score = 0.95
        bucket = engine.determine_bucket(record)
        assert bucket == PromotionBucket.HOLDOUT_EVAL

    def test_unknown_legality_blocks_all_buckets(self):
        engine = QualityScoringEngine()
        record = _make_record(source_legality="unknown")
        record.quality_score = 0.95
        bucket = engine.determine_bucket(record)
        assert bucket is None


# ---------------------------------------------------------------------------
# 2. Score Weights and Thresholds
# ---------------------------------------------------------------------------


class TestScoreWeights:
    def test_default_weights_sum_to_one(self):
        w = ScoreWeights()
        total = (
            w.transcript_confidence
            + w.asr_agreement
            + w.diarization
            + w.audio_quality
            + w.duration_sanity
            + w.event_completeness
            + w.language_consistency
            + w.voice_state_coverage
            + w.voice_state_confidence
            + w.speaker_similarity
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_change_score(self):
        w1 = ScoreWeights(transcript_confidence=0.90, asr_agreement=0.0,
                          diarization=0.0, audio_quality=0.0,
                          duration_sanity=0.0, event_completeness=0.0,
                          language_consistency=0.0, voice_state_coverage=0.0,
                          voice_state_confidence=0.0, speaker_similarity=0.10)
        cfg = ScoringConfig(score_weights=w1)
        engine = QualityScoringEngine(cfg)
        record = _make_record()
        score = engine.compute_score(record)
        # Dominated by transcript_confidence (0.95 * 0.90 = 0.855)
        assert score > 0.80

    def test_audio_quality_includes_artifact_rate(self):
        engine = QualityScoringEngine()
        clean = _make_record(attributes={
            "snr_db": 30.0,
            "artifact_rate": 0.0,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "voice_state_density": 0.8,
            "voice_state_confidence": 0.7,
            "voice_state_observed_mask": [True] * 8,
        })
        noisy = _make_record(attributes={
            "snr_db": 30.0,
            "artifact_rate": 0.8,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "voice_state_density": 0.8,
            "voice_state_confidence": 0.7,
            "voice_state_observed_mask": [True] * 8,
        })
        assert engine.compute_score(clean) > engine.compute_score(noisy)

    def test_speaker_similarity_weight(self):
        w = ScoreWeights(
            speaker_similarity=0.50,
            transcript_confidence=0.50,
            asr_agreement=0.0,
            diarization=0.0,
            audio_quality=0.0,
            duration_sanity=0.0,
            event_completeness=0.0,
            language_consistency=0.0,
            voice_state_coverage=0.0,
            voice_state_confidence=0.0,
        )
        cfg = ScoringConfig(score_weights=w)
        engine = QualityScoringEngine(cfg)

        high_sim = _make_record(attributes={
            "speaker_similarity": 0.95,
            "refinement_agreement": 0.95,
        })
        low_sim = _make_record(attributes={
            "speaker_similarity": 0.10,
            "refinement_agreement": 0.95,
        })
        assert engine.compute_score(high_sim) > engine.compute_score(low_sim)

    def test_score_components_available(self):
        engine = QualityScoringEngine()
        record = _make_record()
        components = engine.compute_score_components(record)
        assert "transcript_confidence" in components
        assert "audio_quality" in components
        assert "voice_state_coverage" in components
        assert "speaker_similarity" in components


# ---------------------------------------------------------------------------
# 3. Anti-contamination Rules
# ---------------------------------------------------------------------------


class TestAntiContamination:
    def test_holdout_never_in_training(self):
        config = ScoringConfig(holdout_record_ids={"r1"})
        engine = QualityScoringEngine(config)
        record = _make_record()
        record.quality_score = 0.99
        bucket = engine.determine_bucket(record)
        assert bucket == PromotionBucket.HOLDOUT_EVAL

    def test_same_text_clustering(self):
        records = [
            _make_record(record_id="a", transcript="Hello world"),
            _make_record(record_id="b", transcript="hello world"),
            _make_record(record_id="c", transcript="different text"),
        ]
        clusters = cluster_same_text(records)
        # "Hello world" and "hello world" should cluster (case-insensitive)
        assert any(
            set(ids) == {"a", "b"} for ids in clusters.values()
        )
        # "c" should not be in any cluster
        for ids in clusters.values():
            assert "c" not in ids

    def test_same_text_holdout_isolation(self):
        records = [
            _make_record(record_id="h1", transcript="shared text"),
            _make_record(record_id="t1", transcript="shared text"),
            _make_record(record_id="t2", transcript="unique text"),
        ]
        holdout_ids = {"h1"}
        tainted = enforce_same_text_holdout_isolation(records, holdout_ids)
        assert "h1" in tainted
        assert "t1" in tainted  # same text as holdout
        assert "t2" not in tainted

    def test_score_batch_applies_same_text_taint(self):
        config = ScoringConfig(holdout_record_ids={"h1"})
        engine = QualityScoringEngine(config)
        records = [
            _make_record(record_id="h1", transcript="shared text"),
            _make_record(record_id="t1", transcript="shared text"),
            _make_record(record_id="t2", transcript="unique text"),
        ]
        results = engine.score_batch(records)
        # h1 goes to holdout
        r_h1 = next(r for r in results if r.record_id == "h1")
        assert r_h1.promotion_bucket == PromotionBucket.HOLDOUT_EVAL or \
            r_h1.review_reasons  # holdout requires double_approval

        # t1 should NOT be in training buckets (tainted)
        r_t1 = next(r for r in results if r.record_id == "t1")
        assert r_t1.promotion_bucket != PromotionBucket.TTS_MAINLINE or \
            r_t1.status == RecordStatus.REVIEW


# ---------------------------------------------------------------------------
# 4. Human Approval Policy
# ---------------------------------------------------------------------------


class TestApprovalPolicy:
    def test_auto_promote_high_score(self):
        policy = ApprovalPolicy(auto_threshold=0.90)
        record = _make_record()
        record.quality_score = 0.95
        assert policy.required_approval(record, PromotionBucket.TTS_MAINLINE) == "auto_promote"

    def test_auditor_review_medium_score(self):
        policy = ApprovalPolicy(auto_threshold=0.90, review_threshold=0.70)
        record = _make_record()
        record.quality_score = 0.80
        assert policy.required_approval(record, PromotionBucket.TTS_MAINLINE) == "auditor_review"

    def test_double_approval_holdout(self):
        policy = ApprovalPolicy()
        record = _make_record()
        record.quality_score = 0.99
        assert policy.required_approval(record, PromotionBucket.HOLDOUT_EVAL) == "double_approval"

    def test_override_logging(self):
        policy = ApprovalPolicy()
        policy.log_override("r1", "admin_001", "double_approval", "Manual QA passed")
        assert len(policy.approval_log) == 1
        assert policy.approval_log[0]["actor_id"] == "admin_001"
        assert policy.approval_log[0]["rationale"] == "Manual QA passed"

    def test_score_and_decide_sets_approval_level(self):
        engine = QualityScoringEngine()
        record = _make_record()
        result = engine.score_and_decide(record)
        assert "approval_level" in result.attributes

    def test_holdout_requires_review_status(self):
        """Holdout bucket records should get REVIEW status (double_approval needed)."""
        config = ScoringConfig(holdout_record_ids={"r1"})
        engine = QualityScoringEngine(config)
        record = _make_record()
        result = engine.score_and_decide(record)
        # Should be in holdout bucket but require review for double_approval
        assert result.promotion_bucket == PromotionBucket.HOLDOUT_EVAL
        assert result.status == RecordStatus.REVIEW
        assert "requires_double_approval" in result.review_reasons


# ---------------------------------------------------------------------------
# 5. Voice-state Promotion Policy
# ---------------------------------------------------------------------------


class TestVoiceStatePolicy:
    def test_tts_mainline_requires_voice_state(self):
        policy = VoiceStateCoveragePolicy(
            min_observed_dims=4,
            min_mean_confidence=0.3,
            required_for_buckets=("tts_mainline",),
            allow_absent_with_metadata=False,
        )
        record = _make_record(attributes={})  # no voice_state
        ok, reason = policy.check(record, "tts_mainline")
        assert not ok
        assert "voice_state_not_available" in reason

    def test_vc_prior_does_not_require_voice_state(self):
        policy = VoiceStateCoveragePolicy(
            required_for_buckets=("tts_mainline",),
        )
        record = _make_record(attributes={})
        ok, reason = policy.check(record, "vc_prior")
        assert ok

    def test_sufficient_coverage_passes(self):
        policy = VoiceStateCoveragePolicy(min_observed_dims=4)
        record = _make_record(attributes={
            "voice_state_observed_mask": [True, True, True, True, False, False, False, False],
            "voice_state_confidence": [0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0],
        })
        ok, reason = policy.check(record, "tts_mainline")
        assert ok

    def test_insufficient_coverage_fails(self):
        policy = VoiceStateCoveragePolicy(min_observed_dims=4)
        record = _make_record(attributes={
            "voice_state_observed_mask": [True, True, False, False, False, False, False, False],
            "voice_state_confidence": [0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        ok, reason = policy.check(record, "tts_mainline")
        assert not ok
        assert "voice_state_low_coverage" in reason

    def test_low_confidence_fails(self):
        policy = VoiceStateCoveragePolicy(
            min_observed_dims=4,
            min_mean_confidence=0.5,
        )
        record = _make_record(attributes={
            "voice_state_observed_mask": [True] * 8,
            "voice_state_confidence": [0.1] * 8,
        })
        ok, reason = policy.check(record, "tts_mainline")
        assert not ok
        assert "voice_state_low_confidence" in reason

    def test_absent_with_metadata_allowed(self):
        policy = VoiceStateCoveragePolicy(allow_absent_with_metadata=True)
        record = _make_record(attributes={"voice_state_absent": True})
        ok, _reason = policy.check(record, "tts_mainline")
        assert ok

    def test_absent_without_metadata_flag_fails(self):
        policy = VoiceStateCoveragePolicy(allow_absent_with_metadata=False)
        record = _make_record(attributes={"voice_state_absent": True})
        ok, reason = policy.check(record, "tts_mainline")
        assert not ok


# ---------------------------------------------------------------------------
# 6. Calibration-aware Thresholds
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_calibrate_confidence_adjusts_score(self):
        config = ScoringConfig(
            provider_calibrations={
                "whisper_v3": ProviderCalibration(
                    provider_id="whisper_v3",
                    calibration_version="v1.0",
                    confidence_offset=-0.05,
                    confidence_scale=1.0,
                    trusted=True,
                ),
            }
        )
        engine = QualityScoringEngine(config)
        adj, is_cal = engine._calibrate_confidence(0.90, "whisper_v3")
        assert abs(adj - 0.85) < 1e-6
        assert is_cal is True

    def test_unknown_provider_returns_raw(self):
        engine = QualityScoringEngine()
        adj, is_cal = engine._calibrate_confidence(0.90, "unknown_provider")
        assert adj == 0.90
        assert is_cal is False

    def test_uncalibrated_provider_blocks_mainline(self):
        """Uncalibrated provider output must not auto-promote to tts_mainline."""
        engine = QualityScoringEngine()
        record = _make_record(attributes={
            "snr_db": 30.0,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "voice_state_density": 0.8,
            "voice_state_confidence": 0.7,
            "voice_state_observed_mask": [True] * 8,
            "uncalibrated_providers": ["some_provider"],
        })
        record.quality_score = 0.95
        bucket = engine.determine_bucket(record)
        assert bucket != PromotionBucket.TTS_MAINLINE

    def test_uncalibrated_triggers_review(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={
            "snr_db": 30.0,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "uncalibrated_providers": ["some_provider"],
        })
        reasons = engine.check_review(record)
        assert "uncalibrated_provider_output" in reasons

    def test_calibrated_provider_is_trusted(self):
        config = ScoringConfig(
            provider_calibrations={
                "whisper_v3": ProviderCalibration(
                    provider_id="whisper_v3",
                    calibration_version="v1.0",
                    trusted=True,
                ),
            }
        )
        engine = QualityScoringEngine(config)
        assert engine._is_provider_trusted("whisper_v3") is True
        assert engine._is_provider_calibrated("whisper_v3") is True

    def test_per_provider_calibration_versions(self):
        config = ScoringConfig(
            provider_calibrations={
                "provA": ProviderCalibration(
                    provider_id="provA",
                    calibration_version="v2.0",
                    confidence_offset=0.0,
                    confidence_scale=0.95,
                    trusted=True,
                ),
                "provB": ProviderCalibration(
                    provider_id="provB",
                    calibration_version="uncalibrated",
                    trusted=False,
                ),
            }
        )
        engine = QualityScoringEngine(config)
        assert engine._is_provider_calibrated("provA") is True
        assert engine._is_provider_calibrated("provB") is False


# ---------------------------------------------------------------------------
# 7. Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    def _make_scored_records(self, n=20):
        engine = QualityScoringEngine()
        records = []
        for i in range(n):
            r = _make_record(record_id=f"r{i}")
            records.append(engine.score_and_decide(r))
        return records, engine

    def test_report_basic_fields(self):
        records, engine = self._make_scored_records()
        report = engine.generate_report(records)
        assert report["total"] == 20
        assert "status_distribution" in report
        assert "bucket_distribution" in report
        assert "score_mean" in report
        assert "score_min" in report
        assert "score_max" in report

    def test_score_histogram(self):
        records, engine = self._make_scored_records()
        hist = engine.generate_score_histogram(records, n_bins=5)
        assert len(hist["bins"]) == 6  # 5 bins + 1 right edge
        assert len(hist["counts"]) == 5
        assert sum(hist["counts"]) == 20
        assert "per_bucket" in hist

    def test_score_histogram_empty(self):
        engine = QualityScoringEngine()
        hist = engine.generate_score_histogram([], n_bins=5)
        assert hist["bins"] == []
        assert hist["counts"] == []

    def test_rejection_breakdown(self):
        engine = QualityScoringEngine()
        records = [
            engine.score_and_decide(_make_record(record_id="a", transcript=None)),
            engine.score_and_decide(_make_record(record_id="b", duration_sec=0)),
            engine.score_and_decide(_make_record(record_id="c", transcript=None)),
        ]
        breakdown = engine.generate_rejection_breakdown(records)
        assert breakdown.get("transcript_empty", 0) == 2
        assert breakdown.get("invalid_duration", 0) == 1

    def test_bucket_summary(self):
        records, engine = self._make_scored_records(10)
        summary = engine.generate_bucket_summary(records)
        # At least one bucket should have entries
        assert any(v["count"] > 0 for v in summary.values())
        for v in summary.values():
            assert "mean_score" in v
            assert "min_score" in v
            assert "max_score" in v

    def test_full_report(self):
        records, engine = self._make_scored_records(10)
        full = engine.generate_full_report(records)
        assert "summary" in full
        assert "score_histogram" in full
        assert "rejection_breakdown" in full
        assert "bucket_summary" in full


# ---------------------------------------------------------------------------
# Hard reject and review extended tests
# ---------------------------------------------------------------------------


class TestHardRejectExtended:
    def test_clipping_detected(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={"clipping_detected": True})
        reasons = engine.check_hard_reject(record)
        assert "audio_clipping" in reasons

    def test_wrong_language(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={"language_mismatch": True})
        reasons = engine.check_hard_reject(record)
        assert "wrong_language" in reasons

    def test_severe_overlap(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={"overlap_ratio": 0.8})
        reasons = engine.check_hard_reject(record)
        assert "severe_overlap" in reasons

    def test_separation_damage(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={"separation_damage": 0.7})
        reasons = engine.check_hard_reject(record)
        assert "separation_damage_high" in reasons


class TestReviewExtended:
    def test_partial_event_extraction(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={"event_extraction_partial": True})
        reasons = engine.check_review(record)
        assert "partial_event_extraction" in reasons


# ---------------------------------------------------------------------------
# Separation waveform gate
# ---------------------------------------------------------------------------


class TestSeparationWaveformGate:
    def test_separated_waveform_blocked_for_mainline(self):
        engine = QualityScoringEngine()
        record = _make_record(attributes={
            "snr_db": 30.0,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "voice_state_density": 0.8,
            "voice_state_confidence": 0.7,
            "voice_state_observed_mask": [True] * 8,
            "uses_separated_waveform": True,
        })
        record.quality_score = 0.95
        bucket = engine.determine_bucket(record)
        assert bucket != PromotionBucket.TTS_MAINLINE

    def test_separated_waveform_ok_for_vc_prior(self):
        engine = QualityScoringEngine()
        record = _make_record(
            transcript_confidence=0.65,
            diarization_confidence=0.0,
            attributes={
                "snr_db": 25.0,
                "refinement_agreement": 0.65,
                "voice_state_density": 0.5,
                "voice_state_confidence": 0.4,
                "uses_separated_waveform": True,
            },
        )
        record.quality_score = 0.80
        bucket = engine.determine_bucket(record)
        # vc_prior does not block separated waveforms
        # bucket may be vc_prior or None depending on other gates
        assert bucket != PromotionBucket.TTS_MAINLINE


# ---------------------------------------------------------------------------
# Language gate
# ---------------------------------------------------------------------------


class TestLanguageGate:
    def test_unsupported_language_blocks_bucket(self):
        config = ScoringConfig(
            supported_languages={"tts_mainline": {"ja", "en"}},
        )
        engine = QualityScoringEngine(config)
        record = _make_record(language="zh")
        record.quality_score = 0.95
        record.attributes.update({
            "snr_db": 30.0,
            "refinement_agreement": 0.95,
            "pause_events": [{"type": "pause"}],
            "voice_state_density": 0.8,
            "voice_state_confidence": 0.7,
            "voice_state_observed_mask": [True] * 8,
        })
        bucket = engine.determine_bucket(record)
        assert bucket != PromotionBucket.TTS_MAINLINE

    def test_supported_language_passes(self):
        config = ScoringConfig(
            supported_languages={"tts_mainline": {"ja", "en"}},
        )
        engine = QualityScoringEngine(config)
        record = _make_record(language="ja")
        record.quality_score = 0.95
        bucket = engine.determine_bucket(record)
        assert bucket == PromotionBucket.TTS_MAINLINE


# ---------------------------------------------------------------------------
# Backward compatibility with existing tests
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Ensure existing test_curation.py TestScoring tests still pass."""

    def _make_record(self, **overrides):
        defaults = dict(
            record_id="r1", source_path="p", audio_hash="h",
            transcript="hello", transcript_confidence=0.95,
            diarization_confidence=0.9, duration_sec=5.0,
            source_legality="owned", language="ja",
            quality_score=0.0,
            attributes={
                "snr_db": 30.0, "refinement_agreement": 0.95,
                "pause_events": [{"type": "pause"}],
            },
        )
        defaults.update(overrides)
        attrs = defaults.pop("attributes", {})
        r = CurationRecord(**defaults)
        r.attributes = attrs
        return r

    def test_compute_score_high_quality(self):
        engine = QualityScoringEngine()
        record = self._make_record()
        score = engine.compute_score(record)
        assert 0.5 < score <= 1.0

    def test_hard_reject_empty_transcript(self):
        engine = QualityScoringEngine()
        record = self._make_record(transcript=None)
        reasons = engine.check_hard_reject(record)
        assert "transcript_empty" in reasons

    def test_review_marginal_confidence(self):
        engine = QualityScoringEngine()
        record = self._make_record(transcript_confidence=0.5)
        reasons = engine.check_review(record)
        assert "marginal_transcript_confidence" in reasons

    def test_holdout_anti_contamination(self):
        config = ScoringConfig(holdout_record_ids={"r1"})
        engine = QualityScoringEngine(config)
        record = self._make_record()
        bucket = engine.determine_bucket(record)
        assert bucket == PromotionBucket.HOLDOUT_EVAL

    def test_default_thresholds_present(self):
        assert "tts_mainline" in DEFAULT_THRESHOLDS
        assert "vc_prior" in DEFAULT_THRESHOLDS
        assert "expressive_prior" in DEFAULT_THRESHOLDS
        assert "holdout_eval" in DEFAULT_THRESHOLDS
