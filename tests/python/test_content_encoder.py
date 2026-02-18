"""Tests for ContentEncoderStudent."""

import torch
import pytest

from tmrvc_core.constants import CONTENT_ENCODER_STATE_FRAMES, D_CONTENT, N_MELS
from tmrvc_train.models.content_encoder import ContentEncoderStudent


class TestContentEncoderStudent:
    """Tests for ContentEncoderStudent."""

    @pytest.fixture
    def model(self):
        return ContentEncoderStudent()

    def test_training_mode_shape(self, model):
        mel = torch.randn(2, N_MELS, 50)
        f0 = torch.randn(2, 1, 50)
        content, state = model(mel, f0)
        assert content.shape == (2, D_CONTENT, 50)
        assert state is None

    def test_streaming_mode_shape(self, model):
        mel = torch.randn(1, N_MELS, 1)
        f0 = torch.randn(1, 1, 1)
        state_in = model.init_state(batch_size=1)

        content, state_out = model(mel, f0, state_in)
        assert content.shape == (1, D_CONTENT, 1)
        assert state_out.shape == (1, D_CONTENT, CONTENT_ENCODER_STATE_FRAMES)

    def test_state_size_matches_contract(self, model):
        assert model._total_state == CONTENT_ENCODER_STATE_FRAMES  # 28

    def test_init_state_zeros(self, model):
        state = model.init_state(batch_size=2)
        assert state.shape == (2, D_CONTENT, CONTENT_ENCODER_STATE_FRAMES)
        assert (state == 0).all()

    def test_streaming_matches_training(self, model):
        """Frame-by-frame streaming should match full-sequence training."""
        model.eval()
        T = 10
        mel = torch.randn(1, N_MELS, T)
        f0 = torch.randn(1, 1, T)

        # Training mode
        with torch.no_grad():
            out_full, _ = model(mel, f0)

        # Streaming mode
        state = model.init_state()
        frames = []
        with torch.no_grad():
            for t in range(T):
                m = mel[:, :, t:t+1]
                f = f0[:, :, t:t+1]
                c, state = model(m, f, state)
                frames.append(c)

        out_streaming = torch.cat(frames, dim=-1)
        torch.testing.assert_close(out_full, out_streaming, atol=1e-5, rtol=1e-4)

    def test_causality(self, model):
        """Output at frame t should not depend on future frames."""
        model.eval()
        T = 20
        mel1 = torch.randn(1, N_MELS, T)
        f01 = torch.randn(1, 1, T)
        mel2 = mel1.clone()
        f02 = f01.clone()
        # Modify future frames
        mel2[:, :, 15:] = torch.randn(1, N_MELS, 5)

        with torch.no_grad():
            out1, _ = model(mel1, f01)
            out2, _ = model(mel2, f02)

        # First 15 frames should match
        torch.testing.assert_close(out1[:, :, :15], out2[:, :, :15])
