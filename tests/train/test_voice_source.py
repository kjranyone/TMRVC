"""Tests for voice source parameter range constraints, stats tracking, and blending."""

import json

import numpy as np
import torch
import pytest

from tmrvc_core.constants import (
    N_ACOUSTIC_PARAMS,
    N_IR_PARAMS,
    N_MELS,
    N_VOICE_SOURCE_PARAMS,
    VOICE_SOURCE_PARAM_NAMES,
)
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.voice_source_stats import VoiceSourceStatsTracker, compute_group_preset


class TestVoiceSourceParamRanges:
    """Verify that voice source parameters (indices 24-31) obey their activation ranges."""

    @pytest.fixture
    def model(self):
        return IREstimator()

    def _get_acoustic_params(self, model: IREstimator, n_samples: int = 50) -> torch.Tensor:
        """Run model on random inputs and return acoustic_params [n_samples, 32]."""
        model.eval()
        results = []
        with torch.no_grad():
            for _ in range(n_samples):
                mel = torch.randn(1, N_MELS, 10)
                params, _ = model(mel)
                results.append(params)
        return torch.cat(results, dim=0)  # [n_samples, 32]

    def test_output_dim(self, model):
        mel = torch.randn(1, N_MELS, 10)
        params, _ = model(mel)
        assert params.shape == (1, N_ACOUSTIC_PARAMS)

    def test_n_voice_source_params(self):
        assert N_VOICE_SOURCE_PARAMS == 8
        assert N_ACOUSTIC_PARAMS == N_IR_PARAMS + N_VOICE_SOURCE_PARAMS

    def test_breathiness_range(self, model):
        """breathiness_low/high [24:26] should be in [0, 1] (sigmoid)."""
        params = self._get_acoustic_params(model)
        breathiness = params[:, 24:26]
        assert (breathiness >= 0.0).all()
        assert (breathiness <= 1.0).all()

    def test_tension_range(self, model):
        """tension_low/high [26:28] should be in [-1, 1] (tanh)."""
        params = self._get_acoustic_params(model)
        tension = params[:, 26:28]
        assert (tension >= -1.0).all()
        assert (tension <= 1.0).all()

    def test_jitter_range(self, model):
        """jitter [28] should be in [0, 0.1] (sigmoid * 0.1)."""
        params = self._get_acoustic_params(model)
        jitter = params[:, 28]
        assert (jitter >= 0.0).all()
        assert (jitter <= 0.1).all()

    def test_shimmer_range(self, model):
        """shimmer [29] should be in [0, 0.1] (sigmoid * 0.1)."""
        params = self._get_acoustic_params(model)
        shimmer = params[:, 29]
        assert (shimmer >= 0.0).all()
        assert (shimmer <= 0.1).all()

    def test_formant_shift_range(self, model):
        """formant_shift [30] should be in [-1, 1] (tanh)."""
        params = self._get_acoustic_params(model)
        formant_shift = params[:, 30]
        assert (formant_shift >= -1.0).all()
        assert (formant_shift <= 1.0).all()

    def test_roughness_range(self, model):
        """roughness [31] should be in [0, 1] (sigmoid)."""
        params = self._get_acoustic_params(model)
        roughness = params[:, 31]
        assert (roughness >= 0.0).all()
        assert (roughness <= 1.0).all()

    def test_ir_params_ranges_preserved(self, model):
        """Existing IR params (0-23) should still obey their original ranges."""
        params = self._get_acoustic_params(model)
        # RT60 [0:8]: [0.05, 3.0]
        rt60 = params[:, :8]
        assert (rt60 >= 0.05).all()
        assert (rt60 <= 3.0).all()
        # DRR [8:16]: [-10, 30]
        drr = params[:, 8:16]
        assert (drr >= -10.0).all()
        assert (drr <= 30.0).all()
        # Tilt [16:24]: [-6, 6]
        tilt = params[:, 16:24]
        assert (tilt >= -6.0).all()
        assert (tilt <= 6.0).all()

    def test_gradient_flows_through_voice_params(self, model):
        """Voice source parameters should receive gradients during training."""
        model.train()
        mel = torch.randn(1, N_MELS, 10)
        params, _ = model(mel)
        # Loss on voice source params only
        loss = params[:, N_IR_PARAMS:].sum()
        loss.backward()
        # Check that MLP final layer has gradients
        assert model.mlp[-1].weight.grad is not None
        assert model.mlp[-1].weight.grad.abs().sum() > 0


# ======================================================================
# VoiceSourceStatsTracker tests
# ======================================================================


class TestVoiceSourceStatsTracker:
    """Tests for VoiceSourceStatsTracker."""

    def test_empty_tracker(self):
        """Empty tracker returns None for any speaker."""
        tracker = VoiceSourceStatsTracker()
        assert tracker.get_speaker_mean("nonexistent") is None
        assert tracker.get_all_means() == {}

    def test_single_update(self):
        """Single update should return that exact value as the mean."""
        tracker = VoiceSourceStatsTracker()
        params = torch.zeros(1, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = torch.tensor([0.5, 0.6, 0.1, -0.2, 0.01, 0.02, 0.3, 0.7])
        tracker.update(params, ["spk_a"])
        mean = tracker.get_speaker_mean("spk_a")
        assert mean is not None
        np.testing.assert_allclose(mean, [0.5, 0.6, 0.1, -0.2, 0.01, 0.02, 0.3, 0.7], atol=1e-6)

    def test_running_mean(self):
        """Multiple updates should produce correct running mean."""
        tracker = VoiceSourceStatsTracker()
        v1 = [0.2, 0.4, 0.0, 0.0, 0.01, 0.01, 0.0, 0.5]
        v2 = [0.8, 0.6, 0.2, 0.2, 0.05, 0.03, 0.4, 0.9]
        expected = [(a + b) / 2.0 for a, b in zip(v1, v2)]

        p1 = torch.zeros(1, N_ACOUSTIC_PARAMS)
        p1[0, N_IR_PARAMS:] = torch.tensor(v1)
        p2 = torch.zeros(1, N_ACOUSTIC_PARAMS)
        p2[0, N_IR_PARAMS:] = torch.tensor(v2)

        tracker.update(p1, ["spk_a"])
        tracker.update(p2, ["spk_a"])

        mean = tracker.get_speaker_mean("spk_a")
        np.testing.assert_allclose(mean, expected, atol=1e-6)

    def test_batch_update(self):
        """Batch with multiple speakers separates correctly."""
        tracker = VoiceSourceStatsTracker()
        params = torch.zeros(2, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = torch.tensor([1.0] * 8)
        params[1, N_IR_PARAMS:] = torch.tensor([2.0] * 8)

        tracker.update(params, ["spk_a", "spk_b"])

        mean_a = tracker.get_speaker_mean("spk_a")
        mean_b = tracker.get_speaker_mean("spk_b")
        np.testing.assert_allclose(mean_a, [1.0] * 8, atol=1e-6)
        np.testing.assert_allclose(mean_b, [2.0] * 8, atol=1e-6)

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve means."""
        tracker = VoiceSourceStatsTracker()
        params = torch.zeros(1, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.05, 0.06, 0.7, 0.8])
        tracker.update(params, ["spk_a"])

        path = tmp_path / "stats.json"
        tracker.save(path)

        loaded = VoiceSourceStatsTracker.load(path)
        mean_orig = tracker.get_speaker_mean("spk_a")
        mean_loaded = loaded.get_speaker_mean("spk_a")
        np.testing.assert_allclose(mean_loaded, mean_orig, atol=1e-7)


# ======================================================================
# compute_group_preset tests
# ======================================================================


class TestComputeGroupPreset:
    """Tests for compute_group_preset."""

    def _make_stats(self, tmp_path, speakers: dict[str, list[float]]):
        """Helper to create a stats JSON from speaker means."""
        tracker = VoiceSourceStatsTracker()
        for sid, vals in speakers.items():
            params = torch.zeros(1, N_ACOUSTIC_PARAMS)
            params[0, N_IR_PARAMS:] = torch.tensor(vals)
            tracker.update(params, [sid])
        path = tmp_path / "stats.json"
        tracker.save(path)
        return path

    def test_single_pattern(self, tmp_path):
        """Pattern matching with fnmatch glob."""
        stats_path = self._make_stats(tmp_path, {
            "moe/alice": [0.1] * 8,
            "moe/bob": [0.3] * 8,
            "other/carol": [0.9] * 8,
        })
        result = compute_group_preset(stats_path, ["moe/*"])
        assert result["n_speakers"] == 2
        assert set(result["matched_speakers"]) == {"moe/alice", "moe/bob"}
        np.testing.assert_allclose(result["preset"], [0.2] * 8, atol=1e-6)

    def test_no_match_raises(self, tmp_path):
        """ValueError when no speakers match."""
        stats_path = self._make_stats(tmp_path, {"other/carol": [0.5] * 8})
        with pytest.raises(ValueError, match="No speakers matched"):
            compute_group_preset(stats_path, ["moe/*"])


# ======================================================================
# Voice source blending tests
# ======================================================================


class TestVoiceSourceBlend:
    """Tests for voice source preset blending logic."""

    @staticmethod
    def _blend(acoustic_params, preset, alpha):
        """Reference blend: matches audio_engine._blend_voice_source logic."""
        blended = acoustic_params.copy()
        if preset is not None and alpha > 0.0:
            blended[0, N_IR_PARAMS:] = (
                (1.0 - alpha) * acoustic_params[0, N_IR_PARAMS:]
                + alpha * preset
            )
        return blended

    def test_alpha_zero_identity(self):
        """alpha=0 should return identical acoustic params."""
        params = np.random.randn(1, N_ACOUSTIC_PARAMS).astype(np.float32)
        preset = np.random.randn(N_VOICE_SOURCE_PARAMS).astype(np.float32)
        blended = self._blend(params, preset, 0.0)
        np.testing.assert_array_equal(blended, params)

    def test_alpha_one_full_preset(self):
        """alpha=1 should replace voice source params with preset."""
        params = np.random.randn(1, N_ACOUSTIC_PARAMS).astype(np.float32)
        preset = np.array([0.5, 0.5, 0.0, 0.0, 0.05, 0.05, 0.0, 0.8], dtype=np.float32)
        blended = self._blend(params, preset, 1.0)
        # IR params (0-23) should be unchanged
        np.testing.assert_array_equal(blended[0, :N_IR_PARAMS], params[0, :N_IR_PARAMS])
        # Voice source params (24-31) should equal preset
        np.testing.assert_allclose(blended[0, N_IR_PARAMS:], preset, atol=1e-7)
