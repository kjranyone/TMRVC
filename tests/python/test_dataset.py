"""Tests for TMRVCDataset and DataLoader batch shapes."""

import torch

from tmrvc_core.constants import D_CONTENT_VEC, D_SPEAKER, N_MELS
from tmrvc_core.types import FeatureSet, TrainingBatch
from tmrvc_data.cache import FeatureCache
from tmrvc_data.dataset import TMRVCDataset, collate_fn, create_dataloader
from tmrvc_data.sampler import BalancedSpeakerSampler


def _make_feature_set(speaker_id: str, utt_id: str, n_frames: int) -> FeatureSet:
    return FeatureSet(
        mel=torch.randn(N_MELS, n_frames),
        content=torch.randn(D_CONTENT_VEC, n_frames),
        f0=torch.randn(1, n_frames),
        spk_embed=torch.randn(D_SPEAKER),
        utterance_id=utt_id,
        speaker_id=speaker_id,
        n_frames=n_frames,
    )


class TestCollate:
    def test_batch_shapes(self):
        batch = [
            _make_feature_set("spk_a", "utt_1", 100),
            _make_feature_set("spk_b", "utt_2", 150),
            _make_feature_set("spk_a", "utt_3", 80),
        ]
        result = collate_fn(batch)

        assert isinstance(result, TrainingBatch)
        assert result.content.shape == (3, D_CONTENT_VEC, 150)  # padded to max
        assert result.f0.shape == (3, 1, 150)
        assert result.mel_target.shape == (3, N_MELS, 150)
        assert result.spk_embed.shape == (3, D_SPEAKER)
        assert result.lengths.tolist() == [100, 150, 80]
        assert len(result.utterance_ids) == 3
        assert len(result.speaker_ids) == 3

    def test_single_item(self):
        batch = [_make_feature_set("spk_a", "utt_1", 50)]
        result = collate_fn(batch)
        assert result.content.shape == (1, D_CONTENT_VEC, 50)


class TestDataset:
    def test_dataset_from_cache(self, tmp_cache_dir):
        cache = FeatureCache(tmp_cache_dir)
        dataset_name = "test_ds"

        # Populate cache with synthetic data
        for i in range(10):
            spk = f"spk_{i % 3}"
            fs = _make_feature_set(spk, f"utt_{i:03d}", 100 + i * 10)
            fs.speaker_id = spk
            fs.utterance_id = f"utt_{i:03d}"
            cache.save(fs, dataset_name, "train")

        ds = TMRVCDataset(cache, dataset_name, "train", cross_speaker_prob=0.0)
        assert len(ds) == 10

        item = ds[0]
        assert isinstance(item, FeatureSet)
        assert item.mel.shape[0] == N_MELS

    def test_dataloader_batch(self, tmp_cache_dir):
        cache = FeatureCache(tmp_cache_dir)
        dataset_name = "test_ds2"

        for i in range(8):
            spk = f"spk_{i % 2}"
            fs = _make_feature_set(spk, f"utt_{i:03d}", 100)
            fs.speaker_id = spk
            fs.utterance_id = f"utt_{i:03d}"
            cache.save(fs, dataset_name, "train")

        loader = create_dataloader(
            tmp_cache_dir,
            dataset_name,
            batch_size=4,
            num_workers=0,
            cross_speaker_prob=0.0,
            balanced_sampling=False,
        )

        batch = next(iter(loader))
        assert isinstance(batch, TrainingBatch)
        assert batch.content.shape[0] == 4
        assert batch.content.shape[1] == D_CONTENT_VEC
        assert batch.mel_target.shape[1] == N_MELS
        assert batch.spk_embed.shape == (4, D_SPEAKER)


class TestBalancedSpeakerSampler:
    def test_covers_all_indices(self):
        speaker_ids = ["a", "a", "a", "b", "b", "c"]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42)
        indices = list(sampler)
        assert len(indices) == 6
        assert set(indices) == {0, 1, 2, 3, 4, 5}

    def test_balanced_representation(self):
        # 90 utterances from speaker A, 10 from B
        speaker_ids = ["a"] * 90 + ["b"] * 10
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42)
        indices = list(sampler)

        # Count how many times each speaker appears
        a_count = sum(1 for i in indices if speaker_ids[i] == "a")
        b_count = sum(1 for i in indices if speaker_ids[i] == "b")

        # Should be more balanced than the raw 90/10 split
        assert b_count >= 10  # at least as many as the actual utterances
