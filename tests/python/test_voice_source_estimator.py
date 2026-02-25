"""Tests for tmrvc_train.models.voice_source_estimator module."""

from __future__ import annotations

import torch

from tmrvc_core.constants import N_MELS, N_VOICE_SOURCE_PARAMS
from tmrvc_train.models.voice_source_estimator import (
    VoiceSourceConfig,
    VoiceSourceDistillationLoss,
    VoiceSourceEstimator,
    create_voice_source_teacher,
)


class TestVoiceSourceConfig:
    def test_defaults(self):
        cfg = VoiceSourceConfig()
        assert cfg.n_mels == N_MELS
        assert cfg.n_voice_params == N_VOICE_SOURCE_PARAMS
        assert cfg.hidden_dim == 128
        assert cfg.n_conv_layers == 4


class TestVoiceSourceEstimator:
    def test_forward_shape(self):
        model = VoiceSourceEstimator()
        mel = torch.randn(2, N_MELS, 100)
        output = model(mel)
        assert output.shape == (2, N_VOICE_SOURCE_PARAMS)

    def test_output_ranges(self):
        model = VoiceSourceEstimator()
        mel = torch.randn(4, N_MELS, 100)
        output = model(mel)

        # breathiness_low/high: [0, 1] (sigmoid)
        assert (output[:, 0] >= 0).all() and (output[:, 0] <= 1).all()
        assert (output[:, 1] >= 0).all() and (output[:, 1] <= 1).all()
        # tension_low/high: [-1, 1] (tanh)
        assert (output[:, 2] >= -1).all() and (output[:, 2] <= 1).all()
        assert (output[:, 3] >= -1).all() and (output[:, 3] <= 1).all()
        # jitter: [0, 0.1] (sigmoid * 0.1)
        assert (output[:, 4] >= 0).all() and (output[:, 4] <= 0.1).all()
        # shimmer: [0, 0.1]
        assert (output[:, 5] >= 0).all() and (output[:, 5] <= 0.1).all()
        # formant_shift: [-1, 1]
        assert (output[:, 6] >= -1).all() and (output[:, 6] <= 1).all()
        # roughness: [0, 1]
        assert (output[:, 7] >= 0).all() and (output[:, 7] <= 1).all()

    def test_variable_length(self):
        model = VoiceSourceEstimator()
        for T in [20, 50, 200]:
            output = model(torch.randn(1, N_MELS, T))
            assert output.shape == (1, N_VOICE_SOURCE_PARAMS)

    def test_custom_config(self):
        cfg = VoiceSourceConfig(hidden_dim=64, n_conv_layers=2)
        model = VoiceSourceEstimator(cfg)
        output = model(torch.randn(1, N_MELS, 50))
        assert output.shape == (1, N_VOICE_SOURCE_PARAMS)

    def test_from_pretrained(self, tmp_path):
        model = VoiceSourceEstimator()
        ckpt_path = tmp_path / "vs.pt"
        torch.save({
            "config": {
                "n_mels": N_MELS,
                "hidden_dim": 128,
                "n_conv_layers": 4,
                "kernel_size": 3,
                "n_voice_params": N_VOICE_SOURCE_PARAMS,
                "dropout": 0.1,
            },
            "model_state_dict": model.state_dict(),
        }, ckpt_path)

        loaded = VoiceSourceEstimator.from_pretrained(str(ckpt_path))
        assert not any(p.requires_grad for p in loaded.parameters())

        mel = torch.randn(1, N_MELS, 50)
        out_orig = model.eval()(mel)
        out_loaded = loaded(mel)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6)


class TestVoiceSourceDistillationLoss:
    def test_forward(self):
        estimator = VoiceSourceEstimator()
        estimator.eval()
        loss_fn = VoiceSourceDistillationLoss(estimator, lambda_voice=0.2)

        mel = torch.randn(2, N_MELS, 100)
        pred = torch.randn(2, N_VOICE_SOURCE_PARAMS)
        loss = loss_fn(mel, pred)

        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_estimator_frozen(self):
        estimator = VoiceSourceEstimator()
        loss_fn = VoiceSourceDistillationLoss(estimator)
        for p in loss_fn.estimator.parameters():
            assert not p.requires_grad


class TestCreateVoiceSourceTeacher:
    def test_none_checkpoint(self):
        result = create_voice_source_teacher(checkpoint_path=None)
        assert result is None

    def test_invalid_checkpoint(self, tmp_path):
        result = create_voice_source_teacher(str(tmp_path / "nonexistent.pt"))
        assert result is None
