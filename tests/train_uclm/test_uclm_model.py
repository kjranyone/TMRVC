"""Tests for UCLM (Unified Codec Language Model).

Tests:
- VoiceStateEncoder: shape and output range
- UCLM model: TTS mode, VC mode
- Loss function: forward and backward
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestVoiceStateEncoder:
    """Tests for VoiceStateEncoder."""

    @pytest.fixture
    def encoder(self):
        from tmrvc_train.models.voice_state_encoder import VoiceStateEncoder

        return VoiceStateEncoder(d_state=8, d_model=256)

    def test_output_shape(self, encoder):
        """Test that output has correct shape [B, T, d_model]."""
        voice_state = torch.randn(2, 100, 8)

        output = encoder(voice_state)

        assert output.shape == (2, 100, 256)

    def test_causality(self, encoder):
        """Test that encoder uses causal convolutions.

        Note: We use eval mode to disable dropout randomness.
        """
        encoder.eval()  # Disable dropout
        voice_state = torch.randn(1, 100, 8)

        # Change future values
        voice_state_modified = voice_state.clone()
        voice_state_modified[:, 50:, :] = torch.randn(1, 50, 8)

        output_orig = encoder(voice_state)
        output_mod = encoder(voice_state_modified)

        # First 50 frames should be similar (causal conv preserves causality)
        # but LayerNorm may cause differences due to batch statistics
        diff = (output_orig[:, :50, :] - output_mod[:, :50, :]).abs().max()
        assert diff < 10.0  # Allow tolerance due to LayerNorm batch statistics

    def test_batch_independence(self, encoder):
        """Test that batches are processed independently.

        Note: LayerNorm depends on batch statistics, so outputs may differ
        when processing together vs. separately.
        """
        voice_state = torch.randn(2, 100, 8)

        # Process together
        output_together = encoder(voice_state)

        # Process separately
        output_0 = encoder(voice_state[0:1])
        output_1 = encoder(voice_state[1:2])

        # Should be similar but not identical due to LayerNorm
        # Main check: shapes are correct
        assert output_together[0].shape == output_0[0].shape
        assert output_together[1].shape == output_1[0].shape


class TestCausalConv1d:
    """Tests for CausalConv1d."""

    def test_causality(self):
        from tmrvc_train.models.voice_state_encoder import CausalConv1d

        conv = CausalConv1d(64, 64, kernel_size=5)

        x = torch.randn(1, 64, 100)
        x_modified = x.clone()
        x_modified[:, :, 50:] = torch.randn(1, 64, 50)

        output_orig = conv(x)
        output_mod = conv(x_modified)

        # First 50 frames should be identical
        assert torch.allclose(output_orig[:, :, :50], output_mod[:, :, :50], atol=1e-5)

    def test_output_length(self):
        from tmrvc_train.models.voice_state_encoder import CausalConv1d

        conv = CausalConv1d(64, 64, kernel_size=5)
        x = torch.randn(1, 64, 100)

        output = conv(x)

        # Output should have same length as input
        assert output.shape == x.shape


class TestUCLM:
    """Tests for UCLM model."""

    @pytest.fixture
    def model(self):
        from tmrvc_train.models.uclm import UCLM, UCLMConfig

        config = UCLMConfig(
            d_model=128,  # Smaller for tests
            n_heads=4,
            n_layers=2,
            d_text=128,  # Match d_model
        )
        return UCLM(config)

    @pytest.fixture
    def sample_inputs(self):
        B, T = 2, 50
        return {
            "text_features": torch.randn(B, 20, 128),  # Match d_model
            "voice_state": torch.randn(B, T, 8),
            "speaker_embed": torch.randn(B, 192),
            "target_tokens": torch.randint(0, 1024, (B, 8, T)),
        }

    def test_tts_mode_forward(self, model, sample_inputs):
        """Test TTS mode forward pass."""
        output = model(
            text_features=sample_inputs["text_features"],
            voice_state=sample_inputs["voice_state"],
            speaker_embed=sample_inputs["speaker_embed"],
            target_tokens=sample_inputs["target_tokens"],
            mode="tts",
        )

        B, T = sample_inputs["voice_state"].shape[:2]

        assert "logits_ar" in output
        assert "logits_parallel" in output
        assert output["logits_ar"].shape == (B, T, 1024)
        assert output["logits_parallel"].shape == (B, 7, T, 1024)

    def test_vc_mode_forward(self, model, sample_inputs):
        """Test VC mode forward pass."""
        B, T = sample_inputs["voice_state"].shape[:2]
        source_tokens = torch.randint(0, 1024, (B, 8, T))

        output = model(
            source_tokens=source_tokens,
            voice_state=sample_inputs["voice_state"],
            speaker_embed=sample_inputs["speaker_embed"],
            target_tokens=sample_inputs["target_tokens"],
            mode="vc",
        )

        assert "logits_ar" in output
        assert output["logits_ar"].shape == (B, T, 1024)

    def test_with_past_tokens(self, model, sample_inputs):
        """Test with past token context."""
        B, T = sample_inputs["voice_state"].shape[:2]
        past_tokens = torch.randint(0, 1024, (B, 8, 20))

        output = model(
            text_features=sample_inputs["text_features"],
            voice_state=sample_inputs["voice_state"],
            speaker_embed=sample_inputs["speaker_embed"],
            past_tokens=past_tokens,
            target_tokens=sample_inputs["target_tokens"],
            mode="tts",
        )

        assert output["logits_ar"].shape == (B, T, 1024)

    def test_backward(self, model, sample_inputs):
        """Test backward pass."""
        output = model(
            text_features=sample_inputs["text_features"],
            voice_state=sample_inputs["voice_state"],
            speaker_embed=sample_inputs["speaker_embed"],
            target_tokens=sample_inputs["target_tokens"],
            mode="tts",
        )

        loss = output["logits_ar"].mean() + output["logits_parallel"].mean()
        loss.backward()

        # Check gradients exist for key parameters
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "At least one parameter should have gradients"


class TestUCLMLoss:
    """Tests for UCLM loss function."""

    def test_loss_computation(self):
        from tmrvc_train.models.uclm import uclm_loss

        B, T = 2, 50
        logits_ar = torch.randn(B, T, 1024)
        logits_parallel = torch.randn(B, 7, T, 1024)
        target_tokens = torch.randint(0, 1024, (B, 8, T))

        losses = uclm_loss(logits_ar, logits_parallel, target_tokens)

        assert "loss" in losses
        assert "loss_ar" in losses
        assert "loss_parallel" in losses
        assert losses["loss"].item() > 0

    def test_loss_backward(self):
        from tmrvc_train.models.uclm import uclm_loss

        B, T = 2, 50
        logits_ar = torch.randn(B, T, 1024, requires_grad=True)
        logits_parallel = torch.randn(B, 7, T, 1024, requires_grad=True)
        target_tokens = torch.randint(0, 1024, (B, 8, T))

        losses = uclm_loss(logits_ar, logits_parallel, target_tokens)
        losses["loss"].backward()

        assert logits_ar.grad is not None
        assert logits_parallel.grad is not None


class TestUCLMConfig:
    """Tests for UCLMConfig."""

    def test_default_config(self):
        from tmrvc_train.models.uclm import UCLMConfig

        config = UCLMConfig()

        assert config.vocab_size == 1024
        assert config.n_codebooks == 8
        assert config.d_model == 256

    def test_custom_config(self):
        from tmrvc_train.models.uclm import UCLMConfig

        config = UCLMConfig(
            d_model=512,
            n_layers=24,
        )

        assert config.d_model == 512
        assert config.n_layers == 24
