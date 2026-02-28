"""Tests for tmrvc_data.emotion_dataset module."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tmrvc_core.constants import N_MELS
from tmrvc_data.cache import FeatureCache
from tmrvc_data.emotion_dataset import EmotionDataset


def _create_emotion_entry(
    cache_dir, dataset, split, speaker_id, utt_id, emotion_id=0, n_frames=50
):
    """Helper to create a minimal emotion cache entry."""
    utt_dir = cache_dir / dataset / split / speaker_id / utt_id
    utt_dir.mkdir(parents=True, exist_ok=True)

    np.save(utt_dir / "mel.npy", np.random.randn(N_MELS, n_frames).astype(np.float32))

    emotion_meta = {
        "emotion_id": emotion_id,
        "emotion": "happy",
        "vad": [0.8, 0.6, 0.5],
        "prosody": [1.0, 0.5, 0.3],
    }
    with open(utt_dir / "emotion.json", "w", encoding="utf-8") as f:
        json.dump(emotion_meta, f)

    meta = {
        "speaker_id": speaker_id,
        "utterance_id": utt_id,
        "dataset": dataset,
        "n_frames": n_frames,
        "content_dim": 768,
    }
    with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


class TestEmotionDataset:
    def test_load_single_entry(self, tmp_path):
        _create_emotion_entry(tmp_path, "test_ds", "train", "spk01", "utt001")

        cache = FeatureCache(tmp_path)
        ds = EmotionDataset(cache, datasets=["test_ds"], split="train", max_frames=30)

        assert len(ds) == 1
        sample = ds[0]
        assert sample["mel"].shape == (N_MELS, 30)
        assert sample["emotion_id"].item() == 0
        assert sample["vad"].shape == (3,)
        assert sample["prosody"].shape == (3,)

    def test_pad_short_mel(self, tmp_path):
        _create_emotion_entry(tmp_path, "test_ds", "train", "spk01", "utt001", n_frames=10)

        cache = FeatureCache(tmp_path)
        ds = EmotionDataset(cache, datasets=["test_ds"], split="train", max_frames=50)

        sample = ds[0]
        assert sample["mel"].shape == (N_MELS, 50)
        # Padded region should be zeros
        assert sample["mel"][:, 10:].abs().sum() == 0.0

    def test_crop_long_mel(self, tmp_path):
        _create_emotion_entry(tmp_path, "test_ds", "train", "spk01", "utt001", n_frames=100)

        cache = FeatureCache(tmp_path)
        ds = EmotionDataset(cache, datasets=["test_ds"], split="train", max_frames=30)

        sample = ds[0]
        assert sample["mel"].shape == (N_MELS, 30)

    def test_multiple_datasets(self, tmp_path):
        _create_emotion_entry(tmp_path, "ds_a", "train", "spk01", "utt001")
        _create_emotion_entry(tmp_path, "ds_b", "train", "spk02", "utt002")

        cache = FeatureCache(tmp_path)
        ds = EmotionDataset(cache, datasets=["ds_a", "ds_b"], split="train")

        assert len(ds) == 2

    def test_empty_dataset(self, tmp_path):
        cache = FeatureCache(tmp_path)
        ds = EmotionDataset(cache, datasets=["nonexistent"], split="train")
        assert len(ds) == 0


class TestComputeProsody:
    def test_speech_like_signal(self):
        from tmrvc_data.emotion_features import compute_prosody

        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # Sine wave at 200 Hz (typical F0) with amplitude modulation
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        # Amplitude modulation at ~4 Hz (syllable rate)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
        audio = (audio * envelope).astype(np.float32)

        result = compute_prosody(audio, sr)
        assert len(result) == 3
        rate, energy, pitch_range = result
        assert 0.0 <= rate <= 1.0
        assert 0.0 <= energy <= 1.0
        assert 0.0 <= pitch_range <= 1.0

    def test_silence(self):
        from tmrvc_data.emotion_features import compute_prosody

        audio = np.zeros(24000, dtype=np.float32)
        result = compute_prosody(audio, 24000)
        assert result == [0.0, 0.0, 0.0]

    def test_short_audio(self):
        from tmrvc_data.emotion_features import compute_prosody

        audio = np.random.randn(2400).astype(np.float32) * 0.1  # 100ms
        result = compute_prosody(audio, 24000)
        assert len(result) == 3
        for v in result:
            assert 0.0 <= v <= 1.0
