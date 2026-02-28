"""Tests for TTSDataset and TTS data pipeline."""

import numpy as np
import torch
import pytest

from tmrvc_core.types import TTSFeatureSet, TTSBatch


class TestTTSFeatureSet:
    def test_creation(self):
        fs = TTSFeatureSet(
            mel=torch.randn(80, 100),
            content=torch.randn(768, 100),
            f0=torch.randn(1, 100),
            spk_embed=torch.randn(192),
            phoneme_ids=torch.arange(20),
            durations=torch.ones(20) * 5,
            language_id=0,
            utterance_id="utt_001",
            speaker_id="spk_001",
            n_frames=100,
            n_phonemes=20,
        )
        assert fs.n_frames == 100
        assert fs.n_phonemes == 20
        assert fs.mel.shape == (80, 100)

    def test_content_dim_default(self):
        fs = TTSFeatureSet(
            mel=torch.randn(80, 50),
            content=torch.randn(768, 50),
            f0=torch.randn(1, 50),
            spk_embed=torch.randn(192),
            phoneme_ids=torch.arange(10),
            durations=torch.ones(10) * 5,
        )
        assert fs.content_dim == 768


class TestTTSBatch:
    def test_creation(self):
        B, L, T = 4, 15, 100
        batch = TTSBatch(
            phoneme_ids=torch.zeros(B, L, dtype=torch.long),
            durations=torch.ones(B, L),
            language_ids=torch.zeros(B, dtype=torch.long),
            content=torch.randn(B, 768, T),
            f0=torch.randn(B, 1, T),
            spk_embed=torch.randn(B, 192),
            mel_target=torch.randn(B, 80, T),
            frame_lengths=torch.full((B,), T, dtype=torch.long),
            phoneme_lengths=torch.full((B,), L, dtype=torch.long),
        )
        assert batch.phoneme_ids.shape == (B, L)
        assert batch.mel_target.shape == (B, 80, T)


class TestAdjustDurations:
    def test_scale_down(self):
        from tmrvc_data.tts_dataset import _adjust_durations

        durations = np.array([10, 20, 30], dtype=np.int64)
        adjusted = _adjust_durations(durations, target_frames=30)
        assert adjusted.sum() == 30

    def test_scale_up(self):
        from tmrvc_data.tts_dataset import _adjust_durations

        durations = np.array([5, 5, 5], dtype=np.int64)
        adjusted = _adjust_durations(durations, target_frames=30)
        assert adjusted.sum() == 30

    def test_zero_total(self):
        from tmrvc_data.tts_dataset import _adjust_durations

        durations = np.array([0, 0, 0], dtype=np.int64)
        adjusted = _adjust_durations(durations, target_frames=10)
        # With zero total, should return unchanged
        assert np.array_equal(adjusted, durations)

    def test_identity(self):
        from tmrvc_data.tts_dataset import _adjust_durations

        durations = np.array([10, 10, 10], dtype=np.int64)
        adjusted = _adjust_durations(durations, target_frames=30)
        assert adjusted.sum() == 30


class TestCropOrPad:
    def test_pad(self):
        from tmrvc_data.tts_dataset import _crop_or_pad

        t = torch.randn(80, 50)
        result = _crop_or_pad(t, 100)
        assert result.shape == (80, 100)
        # Original data preserved
        assert torch.allclose(result[:, :50], t)
        # Padded region is zero
        assert (result[:, 50:] == 0).all()

    def test_crop(self):
        from tmrvc_data.tts_dataset import _crop_or_pad

        t = torch.randn(80, 200)
        result = _crop_or_pad(t, 100)
        assert result.shape == (80, 100)
        assert torch.allclose(result, t[:, :100])

    def test_identity(self):
        from tmrvc_data.tts_dataset import _crop_or_pad

        t = torch.randn(80, 100)
        result = _crop_or_pad(t, 100)
        assert torch.allclose(result, t)


class TestTTSCollateFn:
    def _make_feature_set(self, n_frames: int, n_phonemes: int) -> TTSFeatureSet:
        return TTSFeatureSet(
            mel=torch.randn(80, n_frames),
            content=torch.randn(768, n_frames),
            f0=torch.randn(1, n_frames),
            spk_embed=torch.randn(192),
            phoneme_ids=torch.arange(n_phonemes, dtype=torch.long),
            durations=torch.ones(n_phonemes, dtype=torch.float32) * (n_frames // n_phonemes),
            language_id=0,
            utterance_id="test",
            speaker_id="spk",
            n_frames=n_frames,
            n_phonemes=n_phonemes,
        )

    def test_collate_uniform(self):
        from tmrvc_data.tts_dataset import tts_collate_fn

        batch = [self._make_feature_set(100, 10) for _ in range(4)]
        result = tts_collate_fn(batch)

        assert isinstance(result, TTSBatch)
        assert result.mel_target.shape == (4, 80, 100)
        assert result.phoneme_ids.shape == (4, 10)
        assert result.frame_lengths.shape == (4,)

    def test_collate_variable_lengths(self):
        from tmrvc_data.tts_dataset import tts_collate_fn

        batch = [
            self._make_feature_set(80, 8),
            self._make_feature_set(120, 12),
            self._make_feature_set(100, 10),
        ]
        result = tts_collate_fn(batch)

        # Should pad to max lengths
        assert result.mel_target.shape == (3, 80, 120)
        assert result.phoneme_ids.shape == (3, 12)

    def test_collate_with_max_frames(self):
        from tmrvc_data.tts_dataset import tts_collate_fn

        batch = [
            self._make_feature_set(200, 20),
            self._make_feature_set(300, 30),
        ]
        result = tts_collate_fn(batch, max_frames=150)

        assert result.mel_target.shape[2] == 150
        assert result.frame_lengths[0] == 150  # cropped
        assert result.frame_lengths[1] == 150  # cropped

    def test_collate_preserves_speaker_ids(self):
        from tmrvc_data.tts_dataset import tts_collate_fn

        batch = [self._make_feature_set(100, 10) for _ in range(3)]
        for i, f in enumerate(batch):
            f.speaker_id = f"spk_{i}"

        result = tts_collate_fn(batch)
        assert result.speaker_ids == ["spk_0", "spk_1", "spk_2"]
