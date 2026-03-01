"""Tests for tmrvc_train.models.voice_state_encoder module."""

from __future__ import annotations

import pytest
import torch

from tmrvc_train.models.voice_state_encoder import (
    GradientReversal,
    GradientReversalLayer,
    VoiceStateEncoder,
    VoiceStateEncoderForStreaming,
    create_voice_state_encoder,
)


class TestGradientReversalLayer:
    def test_forward_identity(self):
        x = torch.randn(2, 10, requires_grad=True)
        out = GradientReversalLayer.apply(x, 1.0)
        assert torch.allclose(out, x)

    def test_backward_negates(self):
        x = torch.randn(2, 10, requires_grad=True)
        out = GradientReversalLayer.apply(x, 1.0)
        loss = out.sum()
        loss.backward()
        # Gradient should be -1 * ones
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)


class TestGradientReversal:
    def test_forward(self):
        gr = GradientReversal(alpha=0.5)
        x = torch.randn(2, 10)
        out = gr(x)
        assert out.shape == x.shape


class TestVoiceStateEncoder:
    def test_forward_shape_3d(self):
        encoder = VoiceStateEncoder(
            d_voice_state_explicit=8,
            d_voice_state_ssl=128,
            d_voice_state_delta=8,
            d_model=512,
        )

        B, T = 2, 10
        explicit = torch.randn(B, T, 8)
        ssl = torch.randn(B, T, 128)
        delta = torch.randn(B, T, 8)

        state_cond, adv_logits = encoder(explicit, ssl, delta)

        assert state_cond.shape == (B, T, 512)
        assert adv_logits is None  # No GRL without num_speakers/num_phonemes

    def test_forward_shape_2d(self):
        encoder = VoiceStateEncoder(d_model=512)

        B = 2
        explicit = torch.randn(B, 8)
        ssl = torch.randn(B, 128)
        delta = torch.randn(B, 8)

        state_cond, adv_logits = encoder(explicit, ssl, delta)

        # Note: 2D input gets expanded to [B, 1, 512] then squeezed
        assert state_cond.shape == (B, 512) or state_cond.shape == (B, 1, 512)

    def test_with_adversarial_classifier(self):
        encoder = VoiceStateEncoder(
            d_model=512,
            num_speakers=10,
            num_phonemes=50,
            use_grl=True,
        )

        B, T = 2, 10
        explicit = torch.randn(B, T, 8)
        ssl = torch.randn(B, T, 128)
        delta = torch.randn(B, T, 8)

        state_cond, adv_logits = encoder(explicit, ssl, delta)

        assert state_cond.shape == (B, T, 512)
        assert adv_logits is not None
        # adv_logits shape is [B, T, num_classes] when input is 3D
        assert adv_logits.shape == (B, T, 60)  # 10 + 50

    def test_without_grl(self):
        encoder = VoiceStateEncoder(
            d_model=512,
            num_speakers=10,
            use_grl=False,
        )

        B, T = 2, 10
        explicit = torch.randn(B, T, 8)
        ssl = torch.randn(B, T, 128)
        delta = torch.randn(B, T, 8)

        state_cond, adv_logits = encoder(explicit, ssl, delta)

        assert adv_logits is None


class TestVoiceStateEncoderForStreaming:
    def test_forward_shape(self):
        encoder = VoiceStateEncoderForStreaming(
            d_voice_state_explicit=8,
            d_voice_state_ssl=128,
            d_voice_state_delta=8,
            d_model=512,
        )

        B = 2
        explicit = torch.randn(B, 8)
        ssl = torch.randn(B, 128)
        delta = torch.randn(B, 8)

        out = encoder(explicit, ssl, delta)

        assert out.shape == (B, 512)

    def test_single_frame(self):
        encoder = VoiceStateEncoderForStreaming(d_model=512)

        explicit = torch.randn(1, 8)
        ssl = torch.randn(1, 128)
        delta = torch.randn(1, 8)

        out = encoder(explicit, ssl, delta)

        assert out.shape == (1, 512)

    def test_no_temporal_conv(self):
        # Streaming encoder should not have temporal_conv
        encoder = VoiceStateEncoderForStreaming(d_model=512)
        # Check that forward works without temporal dimension
        explicit = torch.randn(2, 8)
        ssl = torch.randn(2, 128)
        delta = torch.randn(2, 8)

        out = encoder(explicit, ssl, delta)
        assert out.shape == (2, 512)


class TestCreateVoiceStateEncoder:
    def test_creates_streaming_version(self):
        encoder = create_voice_state_encoder(d_model=512, for_streaming=True)
        assert isinstance(encoder, VoiceStateEncoderForStreaming)

    def test_creates_training_version(self):
        encoder = create_voice_state_encoder(d_model=512, for_streaming=False)
        assert isinstance(encoder, VoiceStateEncoder)

    def test_default_is_training(self):
        encoder = create_voice_state_encoder(d_model=512)
        assert isinstance(encoder, VoiceStateEncoder)

    def test_with_adversarial(self):
        encoder = create_voice_state_encoder(
            d_model=512,
            use_grl=True,
            num_speakers=10,
        )
        assert isinstance(encoder, VoiceStateEncoder)

        B, T = 1, 5
        explicit = torch.randn(B, T, 8)
        ssl = torch.randn(B, T, 128)
        delta = torch.randn(B, T, 8)

        state_cond, adv_logits = encoder(explicit, ssl, delta)
        assert adv_logits is not None


class TestIntegration:
    def test_full_pipeline(self):
        """Test combining explicit + ssl + delta -> state_cond."""
        # Simulate inputs
        B, T = 2, 100

        # Explicit voice state (breathiness, tension, etc.)
        explicit = torch.randn(B, T, 8)

        # SSL state from WavLM
        ssl = torch.randn(B, T, 128)

        # Delta state (change from previous frame)
        delta = torch.randn(B, T, 8)

        # Create encoder
        encoder = VoiceStateEncoder(d_model=512)

        # Forward
        state_cond, _ = encoder(explicit, ssl, delta)

        # Check output
        assert state_cond.shape == (B, T, 512)

        # Check that different inputs produce different outputs
        state_cond2, _ = encoder(explicit * 2, ssl, delta)
        assert not torch.allclose(state_cond, state_cond2)
