"""Tests for tmrvc_train.models.style_encoder module."""

from __future__ import annotations

import torch

from tmrvc_core.constants import D_STYLE, N_EMOTION_CATEGORIES, N_MELS
from tmrvc_train.models.style_encoder import (
    AudioStyleEncoder,
    StyleEncoder,
)


class TestAudioStyleEncoder:
    def test_forward_shape(self):
        B, T = 2, 100
        encoder = AudioStyleEncoder(n_mels=N_MELS, d_style=D_STYLE)
        mel = torch.randn(B, N_MELS, T)
        style = encoder(mel)
        assert style.shape == (B, D_STYLE)

    def test_variable_length(self):
        encoder = AudioStyleEncoder()
        for T in [50, 100, 200]:
            style = encoder(torch.randn(1, N_MELS, T))
            assert style.shape == (1, D_STYLE)

    def test_custom_channels(self):
        encoder = AudioStyleEncoder(channels=[16, 32, 64])
        style = encoder(torch.randn(1, N_MELS, 100))
        assert style.shape == (1, D_STYLE)


class TestStyleEncoder:
    def test_forward(self):
        B, T = 4, 100
        encoder = StyleEncoder()
        mel = torch.randn(B, N_MELS, T)
        style = encoder(mel)
        assert style.shape == (B, D_STYLE)

    def test_predict_emotion(self):
        encoder = StyleEncoder()
        style = torch.randn(2, D_STYLE)
        predictions = encoder.predict_emotion(style)

        assert "emotion_logits" in predictions
        assert "vad" in predictions
        assert "prosody" in predictions
        assert predictions["emotion_logits"].shape == (2, N_EMOTION_CATEGORIES)
        assert predictions["vad"].shape == (2, 3)
        assert predictions["prosody"].shape == (2, 3)

    def test_combine_style_params(self):
        acoustic = torch.randn(2, 32)
        emotion = torch.randn(2, 32)
        combined = StyleEncoder.combine_style_params(acoustic, emotion)
        assert combined.shape == (2, 64)
        assert torch.allclose(combined[:, :32], acoustic)
        assert torch.allclose(combined[:, 32:], emotion)

    def test_make_vc_style_params(self):
        acoustic = torch.randn(3, 32)
        style_params = StyleEncoder.make_vc_style_params(acoustic)
        assert style_params.shape == (3, 64)
        assert torch.allclose(style_params[:, :32], acoustic)
        assert style_params[:, 32:].abs().sum() == 0.0

    def test_gradient_flow(self):
        encoder = StyleEncoder()
        mel = torch.randn(1, N_MELS, 50, requires_grad=True)
        style = encoder(mel)
        loss = style.sum()
        loss.backward()
        assert mel.grad is not None
