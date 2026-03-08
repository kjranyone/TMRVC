"""Tests for tmrvc_train.eval_metrics module."""

from __future__ import annotations

import pytest
import torch

from tmrvc_train.eval_metrics import (
    acting_alignment_score,
    cfg_responsiveness_score,
    external_baseline_delta,
    f0_correlation,
    few_shot_speaker_score,
    prosody_transfer_leakage_score,
    speaker_embedding_cosine_similarity,
    suprasegmental_integrity_score,
    timbre_prosody_disentanglement_score,
    utmos_proxy,
    voice_state_calibration_error,
    voice_state_label_utility_score,
)


class TestSECS:
    def test_identical_embeddings(self):
        embed = torch.randn(192)
        score = speaker_embedding_cosine_similarity(embed, embed)
        assert abs(score - 1.0) < 1e-5

    def test_opposite_embeddings(self):
        embed = torch.randn(192)
        score = speaker_embedding_cosine_similarity(embed, -embed)
        assert abs(score - (-1.0)) < 1e-5

    def test_batch_mode(self):
        embed_a = torch.randn(4, 192)
        embed_b = torch.randn(4, 192)
        score = speaker_embedding_cosine_similarity(embed_a, embed_b)
        assert -1.0 <= score <= 1.0

    def test_1d_input(self):
        a = torch.randn(192)
        b = torch.randn(192)
        score = speaker_embedding_cosine_similarity(a, b)
        assert -1.0 <= score <= 1.0


class TestF0Correlation:
    def test_identical_f0(self):
        f0 = torch.tensor([200.0, 210.0, 220.0, 230.0, 240.0])
        corr = f0_correlation(f0, f0)
        assert abs(corr - 1.0) < 1e-5

    def test_opposite_f0(self):
        f0_a = torch.tensor([200.0, 210.0, 220.0, 230.0, 240.0])
        f0_b = torch.tensor([240.0, 230.0, 220.0, 210.0, 200.0])
        corr = f0_correlation(f0_a, f0_b)
        assert abs(corr - (-1.0)) < 1e-5

    def test_unvoiced_frames_excluded(self):
        f0_a = torch.tensor([200.0, 0.0, 220.0, 0.0, 240.0])
        f0_b = torch.tensor([200.0, 0.0, 220.0, 0.0, 240.0])
        corr = f0_correlation(f0_a, f0_b)
        assert abs(corr - 1.0) < 1e-5

    def test_all_unvoiced_returns_zero(self):
        f0_a = torch.zeros(10)
        f0_b = torch.zeros(10)
        corr = f0_correlation(f0_a, f0_b)
        assert corr == 0.0

    def test_2d_input(self):
        f0_a = torch.tensor([[200.0, 210.0, 220.0]])
        f0_b = torch.tensor([[200.0, 210.0, 220.0]])
        corr = f0_correlation(f0_a, f0_b)
        assert abs(corr - 1.0) < 1e-5

    def test_custom_voiced_threshold(self):
        f0_a = torch.tensor([5.0, 200.0, 210.0])
        f0_b = torch.tensor([5.0, 200.0, 210.0])
        # With default threshold=10, frame 0 is unvoiced
        corr = f0_correlation(f0_a, f0_b, voiced_threshold=10.0)
        assert abs(corr - 1.0) < 1e-5


class TestUTMOSProxy:
    def test_identical_mels(self):
        mel = torch.randn(80, 100)
        score = utmos_proxy(mel, mel)
        assert abs(score - 5.0) < 1e-4

    def test_different_mels(self):
        mel_a = torch.randn(80, 100)
        mel_b = torch.randn(80, 100)
        score = utmos_proxy(mel_a, mel_b)
        assert 0.0 <= score <= 5.0

    def test_3d_batch_input(self):
        mel_a = torch.randn(2, 80, 100)
        mel_b = torch.randn(2, 80, 100)
        score = utmos_proxy(mel_a, mel_b)
        assert 0.0 <= score <= 5.0

    def test_mismatched_lengths_aligned(self):
        mel_a = torch.randn(80, 100)
        mel_b = torch.randn(80, 120)
        score = utmos_proxy(mel_a, mel_b)
        assert 0.0 <= score <= 5.0


# ---------------------------------------------------------------------------
# Extended metrics (Worker 06)
# ---------------------------------------------------------------------------


class TestActingAlignmentScore:
    def test_identical_embeddings(self):
        embed = torch.randn(256)
        assert abs(acting_alignment_score(embed, embed) - 1.0) < 1e-5

    def test_orthogonal_embeddings(self):
        a = torch.zeros(4)
        a[0] = 1.0
        b = torch.zeros(4)
        b[1] = 1.0
        assert abs(acting_alignment_score(a, b)) < 1e-5

    def test_batch_mode(self):
        ctx = torch.randn(4, 128)
        pros = torch.randn(4, 128)
        score = acting_alignment_score(ctx, pros)
        assert -1.0 <= score <= 1.0


class TestCFGResponsivenessScore:
    def test_monotonic_increase(self):
        scales = [1.0, 2.0, 3.0, 4.0]
        variances = [0.1, 0.2, 0.3, 0.4]
        score = cfg_responsiveness_score(variances, scales)
        assert abs(score - 1.0) < 1e-5

    def test_no_response(self):
        scales = [1.0, 2.0, 3.0, 4.0]
        variances = [0.2, 0.2, 0.2, 0.2]
        score = cfg_responsiveness_score(variances, scales)
        assert abs(score) < 1e-5

    def test_single_point_returns_zero(self):
        assert cfg_responsiveness_score([0.1], [1.0]) == 0.0

    def test_inverse_response(self):
        scales = [1.0, 2.0, 3.0, 4.0]
        variances = [0.4, 0.3, 0.2, 0.1]
        score = cfg_responsiveness_score(variances, scales)
        assert abs(score - (-1.0)) < 1e-5


class TestTimbreProsodyDisentanglement:
    def test_identical_features_zero_variance(self):
        feat = torch.randn(64)
        score = timbre_prosody_disentanglement_score([feat, feat, feat])
        assert score < 1e-6

    def test_different_features_positive_variance(self):
        feats = [torch.randn(64) for _ in range(5)]
        score = timbre_prosody_disentanglement_score(feats)
        assert score > 0.0

    def test_single_context_returns_zero(self):
        assert timbre_prosody_disentanglement_score([torch.randn(32)]) == 0.0


class TestProsodyTransferLeakage:
    def test_identical_f0_high_leakage(self):
        f0 = torch.tensor([200.0, 210.0, 220.0, 230.0, 240.0])
        score = prosody_transfer_leakage_score(f0, f0)
        assert abs(score - 1.0) < 1e-5

    def test_uncorrelated_low_leakage(self):
        ref = torch.tensor([200.0, 210.0, 220.0, 230.0, 240.0])
        gen = torch.tensor([240.0, 200.0, 240.0, 200.0, 240.0])
        score = prosody_transfer_leakage_score(ref, gen)
        assert score < 0.5


class TestFewShotSpeakerScore:
    def test_perfect_scores(self):
        score = few_shot_speaker_score(1.0, 1.0)
        assert abs(score - 1.0) < 1e-5

    def test_weighted_combination(self):
        score = few_shot_speaker_score(0.8, 0.6, sim_weight=0.5)
        assert abs(score - 0.7) < 1e-5

    def test_zero_scores(self):
        assert few_shot_speaker_score(0.0, 0.0) == 0.0


class TestVoiceStateLabelUtility:
    def test_positive_uplift(self):
        score = voice_state_label_utility_score(0.8, 0.6)
        expected = (0.8 - 0.6) / 0.6
        assert abs(score - expected) < 1e-5

    def test_no_uplift(self):
        score = voice_state_label_utility_score(0.5, 0.5)
        assert abs(score) < 1e-5

    def test_negative_uplift(self):
        score = voice_state_label_utility_score(0.4, 0.6)
        assert score < 0.0

    def test_zero_baseline_returns_zero(self):
        assert voice_state_label_utility_score(0.5, 0.0) == 0.0


class TestVoiceStateCalibrationError:
    def test_perfectly_calibrated(self):
        conf = torch.tensor([0.9, 0.9, 0.9, 0.9])
        err = torch.tensor([0.1, 0.1, 0.1, 0.1])  # 1 - err = 0.9 = conf
        ece = voice_state_calibration_error(conf, err)
        assert ece < 0.05

    def test_miscalibrated(self):
        conf = torch.tensor([0.9, 0.9, 0.9, 0.9])
        err = torch.tensor([0.9, 0.9, 0.9, 0.9])  # 1 - err = 0.1 ≠ 0.9
        ece = voice_state_calibration_error(conf, err)
        assert ece > 0.5

    def test_empty_returns_zero(self):
        assert voice_state_calibration_error(torch.tensor([]), torch.tensor([])) == 0.0


class TestSuprasegmentalIntegrity:
    def test_all_present(self):
        accent = torch.randn(50)
        tone = torch.randn(50)
        boundary = torch.randn(50)
        score = suprasegmental_integrity_score(accent, tone, boundary)
        assert abs(score - 1.0) < 1e-5

    def test_all_missing(self):
        score = suprasegmental_integrity_score(None, None, None)
        assert score == 0.0

    def test_partial_features(self):
        accent = torch.randn(50)
        score = suprasegmental_integrity_score(accent, None, None)
        assert abs(score - 1.0 / 3.0) < 1e-5

    def test_zero_features_rejected(self):
        zeros = torch.zeros(50)
        score = suprasegmental_integrity_score(zeros, torch.randn(50), None)
        assert abs(score - 1.0 / 3.0) < 1e-5


class TestExternalBaselineDelta:
    def test_positive_delta(self):
        assert external_baseline_delta(0.9, 0.8) == pytest.approx(0.1)

    def test_negative_delta(self):
        assert external_baseline_delta(0.7, 0.8) == pytest.approx(-0.1)

    def test_zero_delta(self):
        assert external_baseline_delta(0.5, 0.5) == 0.0
