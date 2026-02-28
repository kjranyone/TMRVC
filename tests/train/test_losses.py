"""Tests for loss functions."""

import torch
import pytest

from tmrvc_train.losses import (
    DMD2Loss,
    FlowMatchingLoss,
    MultiResolutionSTFTLoss,
    SVLoss,
    SpeakerConsistencyLoss,
)


class TestFlowMatchingLoss:
    """Tests for FlowMatchingLoss."""

    def test_zero_loss_for_identical(self):
        loss_fn = FlowMatchingLoss()
        x = torch.randn(2, 80, 50)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_loss_for_different(self):
        loss_fn = FlowMatchingLoss()
        pred = torch.randn(2, 80, 50)
        target = torch.randn(2, 80, 50)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_scalar_output(self):
        loss_fn = FlowMatchingLoss()
        loss = loss_fn(torch.randn(4, 80, 100), torch.randn(4, 80, 100))
        assert loss.dim() == 0

    def test_with_mask(self):
        loss_fn = FlowMatchingLoss()
        pred = torch.ones(1, 80, 10)
        target = torch.zeros(1, 80, 10)
        # Mask out last 5 frames
        mask = torch.ones(1, 1, 10)
        mask[:, :, 5:] = 0
        loss_masked = loss_fn(pred, target, mask)
        loss_full = loss_fn(pred, target)
        # Masked loss should only consider first 5 frames (loss=1.0)
        assert loss_masked.item() == pytest.approx(1.0, abs=1e-5)


class TestMultiResolutionSTFTLoss:
    """Tests for MultiResolutionSTFTLoss."""

    def test_mel_domain_l1(self):
        """When given 3D mel tensors, falls back to L1."""
        loss_fn = MultiResolutionSTFTLoss()
        pred = torch.randn(2, 80, 50)
        target = torch.randn(2, 80, 50)
        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert loss.dim() == 0

    def test_zero_loss_identical_mel(self):
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(2, 80, 50)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_waveform_domain(self):
        """Test with waveform-like 2D tensors."""
        loss_fn = MultiResolutionSTFTLoss()
        pred = torch.randn(2, 4800)  # 200ms at 24kHz
        target = torch.randn(2, 4800)
        loss = loss_fn(pred, target)
        assert loss.item() > 0


class TestSpeakerConsistencyLoss:
    """Tests for SpeakerConsistencyLoss."""

    def test_zero_loss_for_identical(self):
        loss_fn = SpeakerConsistencyLoss()
        embed = torch.randn(4, 192)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        loss = loss_fn(embed, embed)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_max_loss_for_opposite(self):
        loss_fn = SpeakerConsistencyLoss()
        embed = torch.randn(4, 192)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        loss = loss_fn(embed, -embed)
        # Cosine sim = -1 → loss = 2.0
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_scalar_output(self):
        loss_fn = SpeakerConsistencyLoss()
        loss = loss_fn(torch.randn(8, 192), torch.randn(8, 192))
        assert loss.dim() == 0


class TestDMD2Loss:
    """Tests for DMD2Loss."""

    def test_dmd2_loss_returns_both(self):
        """Should return both generator and discriminator losses."""
        loss_fn = DMD2Loss()
        logits_real = torch.randn(4, 1)
        logits_fake = torch.randn(4, 1)
        loss_gen, loss_disc = loss_fn(logits_real, logits_fake)
        assert loss_gen.dim() == 0
        assert loss_disc.dim() == 0


class TestSVLoss:
    """Tests for SVLoss."""

    def test_sv_loss_identical_zero(self):
        """Identical embeddings should yield zero loss."""
        loss_fn = SVLoss()
        embed = torch.randn(4, 192)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        loss = loss_fn(embed, embed)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)
