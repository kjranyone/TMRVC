"""v4 bootstrap quality gate tests.

Tests for the raw-audio bootstrap pipeline, supervision tier classification,
and quality gate evaluation.
"""

import numpy as np
import pytest

from tmrvc_data.bootstrap.supervision import (
    SupervisionTierClassifier,
    TierThresholds,
    QualityGateReport,
)
from tmrvc_data.bootstrap.quality_gates import (
    QualityGateConfig,
    evaluate_bootstrap_quality,
)
from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
    BootstrapResult,
)
from tmrvc_data.bootstrap.pipeline import BootstrapPipeline


class TestSupervisionTierClassifier:
    """Test supervision tier classification."""

    def setup_method(self):
        self.classifier = SupervisionTierClassifier()

    def test_tier_a_all_high_confidence(self):
        mask = np.ones((100, 12), dtype=bool)
        conf = np.ones((100, 12)) * 0.9
        tier = self.classifier.classify(
            transcript_confidence=0.95,
            diarization_confidence=0.90,
            physical_observed_mask=mask,
            physical_confidence=conf,
            has_semantic_annotations=True,
        )
        assert tier == "tier_a"

    def test_tier_b_partial_physical(self):
        mask = np.ones((100, 12), dtype=bool)
        mask[:, 8:] = False  # new v4 dims not observed
        conf = np.ones((100, 12)) * 0.7
        tier = self.classifier.classify(
            transcript_confidence=0.85,
            diarization_confidence=0.75,
            physical_observed_mask=mask,
            physical_confidence=conf,
            has_semantic_annotations=False,
        )
        assert tier == "tier_b"

    def test_tier_c_sparse_physical(self):
        mask = np.zeros((100, 12), dtype=bool)
        mask[:, :4] = True  # only pitch, energy, etc
        conf = np.ones((100, 12)) * 0.5
        tier = self.classifier.classify(
            transcript_confidence=0.60,
            diarization_confidence=0.50,
            physical_observed_mask=mask,
            physical_confidence=conf,
            has_semantic_annotations=False,
        )
        assert tier == "tier_c"

    def test_tier_d_low_confidence(self):
        tier = self.classifier.classify(
            transcript_confidence=0.2,
            diarization_confidence=0.1,
            physical_observed_mask=None,
            physical_confidence=None,
            has_semantic_annotations=False,
        )
        assert tier == "tier_d"

    def test_tier_weights(self):
        weights_a = self.classifier.compute_tier_weights("tier_a")
        weights_d = self.classifier.compute_tier_weights("tier_d")

        assert weights_a["physical_loss"] == 1.0
        assert weights_d["physical_loss"] == 0.0
        assert weights_a["codec_loss"] >= weights_d["codec_loss"]


class TestQualityGates:
    """Test bootstrap quality gate evaluation."""

    def test_passing_report(self):
        report = QualityGateReport(
            corpus_id="test",
            diarization_purity=0.90,
            speaker_cluster_consistency=0.85,
            overlap_rejection_precision=0.90,
            transcript_wer_proxy=0.10,
            physical_label_coverage=0.75,
            physical_confidence_calibration_error=0.08,
            languages_detected=["ja", "en"],
            tier_distribution={"tier_a": 30, "tier_b": 40, "tier_c": 20, "tier_d": 10},
        )
        result = evaluate_bootstrap_quality(report)
        assert result.gates_passed is True
        assert len(result.failed_gates) == 0

    def test_failing_diarization(self):
        report = QualityGateReport(
            corpus_id="test",
            diarization_purity=0.50,  # below threshold
            speaker_cluster_consistency=0.85,
            overlap_rejection_precision=0.90,
            transcript_wer_proxy=0.10,
            physical_label_coverage=0.75,
            physical_confidence_calibration_error=0.08,
            languages_detected=["ja"],
            tier_distribution={"tier_a": 30, "tier_b": 40, "tier_c": 20, "tier_d": 10},
        )
        result = evaluate_bootstrap_quality(report)
        assert result.gates_passed is False
        assert any("diarization_purity" in g for g in result.failed_gates)

    def test_failing_tier_distribution(self):
        report = QualityGateReport(
            corpus_id="test",
            diarization_purity=0.90,
            speaker_cluster_consistency=0.85,
            overlap_rejection_precision=0.90,
            transcript_wer_proxy=0.10,
            physical_label_coverage=0.75,
            physical_confidence_calibration_error=0.08,
            languages_detected=["ja"],
            tier_distribution={"tier_a": 1, "tier_b": 5, "tier_c": 30, "tier_d": 64},
        )
        result = evaluate_bootstrap_quality(report)
        assert result.gates_passed is False

    def test_custom_config(self):
        config = QualityGateConfig(
            min_diarization_purity=0.50,  # relaxed
        )
        report = QualityGateReport(
            corpus_id="test",
            diarization_purity=0.60,
            speaker_cluster_consistency=0.85,
            overlap_rejection_precision=0.90,
            transcript_wer_proxy=0.10,
            physical_label_coverage=0.75,
            physical_confidence_calibration_error=0.08,
            languages_detected=["ja"],
            tier_distribution={"tier_a": 30, "tier_b": 40, "tier_c": 20, "tier_d": 10},
        )
        result = evaluate_bootstrap_quality(report, config)
        assert result.gates_passed is True


class TestBootstrapStages:
    """Test bootstrap pipeline stage definitions."""

    def test_stage_count(self):
        assert len(BootstrapStage) == 13

    def test_stage_order(self):
        assert BootstrapStage.INGEST == 0
        assert BootstrapStage.CACHE_EXPORT == 12

    def test_all_stages_sequential(self):
        for i, stage in enumerate(BootstrapStage):
            assert stage.value == i


class TestBootstrapPipeline:
    """Test bootstrap pipeline instantiation."""

    def test_default_config(self):
        pipeline = BootstrapPipeline()
        assert pipeline.config.physical_dim == 12

    def test_custom_config(self):
        config = BootstrapConfig(num_workers=8, physical_dim=12)
        pipeline = BootstrapPipeline(config)
        assert pipeline.config.num_workers == 8

    def test_supported_formats(self):
        assert ".wav" in BootstrapPipeline.SUPPORTED_FORMATS
        assert ".flac" in BootstrapPipeline.SUPPORTED_FORMATS
        assert ".mp3" in BootstrapPipeline.SUPPORTED_FORMATS
