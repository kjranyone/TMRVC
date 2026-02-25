"""Tests for tmrvc_train.models.vq_bottleneck module."""

from __future__ import annotations

import pytest
import torch

from tmrvc_core.constants import D_CONTENT, VQ_CODEBOOK_DIM, VQ_CODEBOOK_SIZE, VQ_N_CODEBOOKS
from tmrvc_train.models.vq_bottleneck import FactorizedVQBottleneck


class TestFactorizedVQBottleneck:
    def test_forward_3d(self):
        B, T = 2, 50
        vq = FactorizedVQBottleneck()
        x = torch.randn(B, D_CONTENT, T)

        quantized, indices, commitment_loss = vq(x)

        assert quantized.shape == (B, D_CONTENT, T)
        assert indices.shape == (B, VQ_N_CODEBOOKS, T)
        assert commitment_loss.ndim == 0

    def test_forward_2d(self):
        B = 4
        vq = FactorizedVQBottleneck()
        x = torch.randn(B, D_CONTENT)

        quantized, indices, commitment_loss = vq(x)

        assert quantized.shape == (B, D_CONTENT)
        assert indices.shape == (B, VQ_N_CODEBOOKS)

    def test_straight_through_gradient(self):
        vq = FactorizedVQBottleneck()
        x = torch.randn(2, D_CONTENT, 10, requires_grad=True)

        quantized, _, _ = vq(x)
        loss = quantized.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_indices_in_range(self):
        vq = FactorizedVQBottleneck()
        x = torch.randn(2, D_CONTENT, 20)
        _, indices, _ = vq(x)

        assert (indices >= 0).all()
        assert (indices < VQ_CODEBOOK_SIZE).all()

    def test_ema_update(self):
        vq = FactorizedVQBottleneck()
        vq.train()

        x = torch.randn(4, D_CONTENT, 10)
        _, indices, _ = vq(x)

        old_weights = vq.ema_weights.clone()
        vq.update_ema(x, indices)
        # EMA weights should change after update
        assert not torch.allclose(vq.ema_weights, old_weights)

    def test_ema_update_noop_in_eval(self):
        vq = FactorizedVQBottleneck()
        vq.eval()

        x = torch.randn(2, D_CONTENT, 10)
        _, indices, _ = vq(x)

        old_weights = vq.ema_weights.clone()
        vq.update_ema(x, indices)
        assert torch.allclose(vq.ema_weights, old_weights)

    def test_commitment_loss_zero_in_eval(self):
        vq = FactorizedVQBottleneck()
        vq.eval()

        x = torch.randn(2, D_CONTENT, 10)
        _, _, commitment_loss = vq(x)
        assert commitment_loss == 0.0

    def test_commitment_loss_nonzero_in_train(self):
        vq = FactorizedVQBottleneck()
        vq.train()

        x = torch.randn(2, D_CONTENT, 10)
        _, _, commitment_loss = vq(x)
        assert commitment_loss > 0

    def test_get_codebook_usage(self):
        vq = FactorizedVQBottleneck()
        usage = vq.get_codebook_usage()
        assert usage.shape == (VQ_N_CODEBOOKS, VQ_CODEBOOK_SIZE)

    def test_invalid_d_input(self):
        with pytest.raises(AssertionError):
            FactorizedVQBottleneck(d_input=100, n_codebooks=2, codebook_dim=128)
