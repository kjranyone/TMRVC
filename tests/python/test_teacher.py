"""Tests for TeacherUNet."""

import torch
import pytest

from tmrvc_core.constants import D_CONTENT_VEC, D_SPEAKER, N_IR_PARAMS, N_MELS
from tmrvc_train.models.teacher_unet import TeacherUNet


class TestTeacherUNet:
    """Tests for TeacherUNet."""

    @pytest.fixture
    def model(self):
        return TeacherUNet()

    def test_forward_shape(self, model):
        B, T = 2, 64
        x_t = torch.randn(B, N_MELS, T)
        t = torch.rand(B)
        content = torch.randn(B, D_CONTENT_VEC, T)
        f0 = torch.randn(B, 1, T)
        spk = torch.randn(B, D_SPEAKER)
        ir = torch.randn(B, N_IR_PARAMS)

        v_pred = model(x_t, t, content, f0, spk, ir)
        assert v_pred.shape == (B, N_MELS, T)

    def test_forward_without_ir(self, model):
        """IR params should default to zeros when None."""
        B, T = 1, 32
        x_t = torch.randn(B, N_MELS, T)
        t = torch.rand(B)
        content = torch.randn(B, D_CONTENT_VEC, T)
        f0 = torch.randn(B, 1, T)
        spk = torch.randn(B, D_SPEAKER)

        v_pred = model(x_t, t, content, f0, spk, ir_params=None)
        assert v_pred.shape == (B, N_MELS, T)

    def test_timestep_2d(self, model):
        """Accept timestep as [B, 1]."""
        B, T = 1, 32
        x_t = torch.randn(B, N_MELS, T)
        t = torch.rand(B, 1)
        content = torch.randn(B, D_CONTENT_VEC, T)
        f0 = torch.randn(B, 1, T)
        spk = torch.randn(B, D_SPEAKER)

        v_pred = model(x_t, t, content, f0, spk)
        assert v_pred.shape == (B, N_MELS, T)

    def test_different_timesteps_give_different_outputs(self, model):
        model.eval()
        B, T = 1, 32
        x_t = torch.randn(B, N_MELS, T)
        content = torch.randn(B, D_CONTENT_VEC, T)
        f0 = torch.randn(B, 1, T)
        spk = torch.randn(B, D_SPEAKER)

        with torch.no_grad():
            v1 = model(x_t, torch.tensor([0.1]), content, f0, spk)
            v2 = model(x_t, torch.tensor([0.9]), content, f0, spk)

        assert not torch.allclose(v1, v2)

    def test_odd_time_length(self, model):
        """Handle T that isn't a power of 2 (U-Net stride alignment)."""
        B, T = 1, 48  # Must be divisible by 16 (4 x stride-2)
        x_t = torch.randn(B, N_MELS, T)
        t = torch.rand(B)
        content = torch.randn(B, D_CONTENT_VEC, T)
        f0 = torch.randn(B, 1, T)
        spk = torch.randn(B, D_SPEAKER)

        v_pred = model(x_t, t, content, f0, spk)
        assert v_pred.shape == (B, N_MELS, T)
