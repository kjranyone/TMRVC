"""Tests for FeatureCache save/load roundtrip."""

import torch

from tmrvc_core.constants import D_CONTENT_VEC, D_SPEAKER, N_MELS
from tmrvc_core.types import FeatureSet
from tmrvc_data.cache import FeatureCache


class TestFeatureCache:
    def test_save_and_load(self, tmp_cache_dir, mock_feature_set):
        cache = FeatureCache(tmp_cache_dir)
        cache.save(mock_feature_set, "test_ds", "train")

        loaded = cache.load(
            "test_ds",
            "train",
            mock_feature_set.speaker_id,
            mock_feature_set.utterance_id,
            mmap=False,
        )

        assert loaded.utterance_id == mock_feature_set.utterance_id
        assert loaded.speaker_id == mock_feature_set.speaker_id
        assert loaded.n_frames == mock_feature_set.n_frames
        assert torch.allclose(loaded.mel, mock_feature_set.mel)
        assert torch.allclose(loaded.content, mock_feature_set.content)
        assert torch.allclose(loaded.f0, mock_feature_set.f0)
        assert torch.allclose(loaded.spk_embed, mock_feature_set.spk_embed)

    def test_save_and_load_mmap(self, tmp_cache_dir, mock_feature_set):
        cache = FeatureCache(tmp_cache_dir)
        cache.save(mock_feature_set, "test_ds", "train")

        loaded = cache.load(
            "test_ds",
            "train",
            mock_feature_set.speaker_id,
            mock_feature_set.utterance_id,
            mmap=True,
        )
        # mmap should still produce correct data
        assert loaded.mel.shape == (N_MELS, 100)
        assert torch.allclose(loaded.mel, mock_feature_set.mel)

    def test_exists(self, tmp_cache_dir, mock_feature_set):
        cache = FeatureCache(tmp_cache_dir)
        assert not cache.exists(
            "test_ds", "train",
            mock_feature_set.speaker_id,
            mock_feature_set.utterance_id,
        )

        cache.save(mock_feature_set, "test_ds", "train")

        assert cache.exists(
            "test_ds", "train",
            mock_feature_set.speaker_id,
            mock_feature_set.utterance_id,
        )

    def test_iter_entries(self, tmp_cache_dir):
        cache = FeatureCache(tmp_cache_dir)

        for i in range(5):
            fs = FeatureSet(
                mel=torch.randn(N_MELS, 50),
                content=torch.randn(D_CONTENT_VEC, 50),
                f0=torch.randn(1, 50),
                spk_embed=torch.randn(D_SPEAKER),
                utterance_id=f"utt_{i}",
                speaker_id=f"spk_{i % 2}",
                n_frames=50,
            )
            cache.save(fs, "test_ds", "train")

        entries = cache.iter_entries("test_ds", "train")
        assert len(entries) == 5

    def test_verify_valid(self, tmp_cache_dir, mock_feature_set):
        cache = FeatureCache(tmp_cache_dir)
        cache.save(mock_feature_set, "test_ds", "train")

        result = cache.verify("test_ds", "train")
        assert result["total"] == 1
        assert result["valid"] == 1
        assert result["invalid"] == 0

    def test_verify_empty(self, tmp_cache_dir):
        cache = FeatureCache(tmp_cache_dir)
        result = cache.verify("nonexistent", "train")
        assert result["total"] == 0
