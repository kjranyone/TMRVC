"""Tests for MelDiscriminator."""

import torch
import pytest

from tmrvc_train.models.discriminator import MelDiscriminator


class TestMelDiscriminator:
    """Tests for MelDiscriminator."""

    @pytest.fixture
    def model(self):
        return MelDiscriminator()

    def test_mel_discriminator_shape(self, model):
        """[B, 80, T] → [B, 1]."""
        mel = torch.randn(2, 80, 50)
        logits = model(mel)
        assert logits.shape == (2, 1)

    def test_mel_discriminator_gradient_flow(self, model):
        """Gradients should flow through the discriminator."""
        mel = torch.randn(2, 80, 50, requires_grad=True)
        logits = model(mel)
        loss = logits.mean()
        loss.backward()
        assert mel.grad is not None
        assert mel.grad.abs().sum() > 0

    def test_spectral_norm_applied(self, model):
        """Spectral norm should be applied to conv and linear layers."""
        from torch.nn.utils import spectral_norm

        for module in model.features:
            if isinstance(module, torch.nn.Conv1d):
                # Spectral norm adds a weight_orig parameter
                assert hasattr(module, "weight_orig"), (
                    "Spectral norm not applied to Conv1d"
                )
