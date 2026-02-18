"""Tests for VocoderStudent."""

import math

import torch
import pytest

from tmrvc_core.constants import D_CONTENT, D_VOCODER_FEATURES, VOCODER_STATE_FRAMES
from tmrvc_train.models.vocoder import VocoderStudent


class TestVocoderStudent:
    """Tests for VocoderStudent."""

    @pytest.fixture
    def model(self):
        return VocoderStudent()

    def test_training_mode_shape(self, model):
        features = torch.randn(2, D_VOCODER_FEATURES, 50)
        mag, phase, state = model(features)
        assert mag.shape == (2, D_VOCODER_FEATURES, 50)
        assert phase.shape == (2, D_VOCODER_FEATURES, 50)
        assert state is None

    def test_streaming_mode_shape(self, model):
        features = torch.randn(1, D_VOCODER_FEATURES, 1)
        state_in = model.init_state()

        mag, phase, state_out = model(features, state_in)
        assert mag.shape == (1, D_VOCODER_FEATURES, 1)
        assert phase.shape == (1, D_VOCODER_FEATURES, 1)
        assert state_out.shape == (1, D_CONTENT, VOCODER_STATE_FRAMES)

    def test_state_size_matches_contract(self, model):
        assert model._total_state == VOCODER_STATE_FRAMES  # 14

    def test_magnitude_non_negative(self, model):
        """Magnitude output should always be >= 0 due to ReLU."""
        model.eval()
        features = torch.randn(4, D_VOCODER_FEATURES, 20)
        with torch.no_grad():
            mag, _, _ = model(features)
        assert (mag >= 0).all()

    def test_phase_range(self, model):
        """Phase output should be in [-pi, pi] (atan2 range)."""
        model.eval()
        features = torch.randn(4, D_VOCODER_FEATURES, 20)
        with torch.no_grad():
            _, phase, _ = model(features)
        assert (phase >= -math.pi - 1e-6).all()
        assert (phase <= math.pi + 1e-6).all()

    def test_streaming_matches_training(self, model):
        model.eval()
        T = 10
        features = torch.randn(1, D_VOCODER_FEATURES, T)

        with torch.no_grad():
            mag_full, phase_full, _ = model(features)

        state = model.init_state()
        mag_frames, phase_frames = [], []
        with torch.no_grad():
            for t in range(T):
                f = features[:, :, t:t+1]
                m, p, state = model(f, state)
                mag_frames.append(m)
                phase_frames.append(p)

        mag_stream = torch.cat(mag_frames, dim=-1)
        phase_stream = torch.cat(phase_frames, dim=-1)
        torch.testing.assert_close(mag_full, mag_stream, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(phase_full, phase_stream, atol=1e-5, rtol=1e-4)
