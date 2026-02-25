"""Tests for tmrvc_data.augmentation module."""

from __future__ import annotations

import torch

from tmrvc_data.augmentation import AugmentationConfig, Augmenter


class TestAugmentationConfig:
    def test_defaults(self):
        cfg = AugmentationConfig()
        assert cfg.rir_prob == 0.5
        assert cfg.eq_prob == 0.3
        assert cfg.noise_prob == 0.3
        assert cfg.f0_perturbation_prob == 0.3
        assert cfg.content_dropout_prob == 0.1
        assert cfg.rir_dirs == []


class TestAugmenter:
    def test_apply_noise(self):
        aug = Augmenter(AugmentationConfig())
        waveform = torch.randn(1, 24000)
        noised = aug.apply_noise(waveform)
        assert noised.shape == waveform.shape
        # Should not be identical
        assert not torch.allclose(noised, waveform)

    def test_apply_eq(self):
        aug = Augmenter(AugmentationConfig())
        waveform = torch.randn(1, 24000)
        eqed = aug.apply_eq(waveform)
        assert eqed.shape == waveform.shape

    def test_apply_content_dropout(self):
        cfg = AugmentationConfig(content_dropout_rate=0.5)
        aug = Augmenter(cfg)
        content = torch.randn(256, 100)
        dropped = aug.apply_content_dropout(content)
        assert dropped.shape == content.shape
        # Some frames should be zeroed
        zero_frames = (dropped.abs().sum(dim=0) == 0).sum().item()
        assert zero_frames > 0

    def test_apply_content_dropout_zero_rate(self):
        cfg = AugmentationConfig(content_dropout_rate=0.0)
        aug = Augmenter(cfg)
        content = torch.randn(256, 10)
        result = aug.apply_content_dropout(content)
        assert torch.allclose(result, content)

    def test_apply_rir_no_files(self):
        aug = Augmenter(AugmentationConfig())
        waveform = torch.randn(1, 24000)
        result = aug.apply_rir(waveform)
        # No RIR files loaded, should return unchanged
        assert torch.allclose(result, waveform)

    def test_augment_audio_all_disabled(self):
        cfg = AugmentationConfig(
            rir_prob=0.0,
            eq_prob=0.0,
            noise_prob=0.0,
            f0_perturbation_prob=0.0,
        )
        aug = Augmenter(cfg)
        waveform = torch.randn(1, 24000)
        result = aug.augment_audio(waveform)
        assert torch.allclose(result, waveform)
