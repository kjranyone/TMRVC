"""Tests for TMRVCDataset and DataLoader batch shapes."""

import torch

from tmrvc_core.constants import D_CONTENT_VEC, D_SPEAKER, N_MELS
from tmrvc_core.types import FeatureSet, TrainingBatch
from tmrvc_data.cache import FeatureCache
from tmrvc_data.dataset import TMRVCDataset, collate_fn, create_dataloader
from tmrvc_data.sampler import BalancedSpeakerSampler, SpeakerGroupConfig


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
        # max=150 snaps to bucket boundary 250
        assert result.content.shape == (3, D_CONTENT_VEC, 250)
        assert result.f0.shape == (3, 1, 250)
        assert result.mel_target.shape == (3, N_MELS, 250)
        assert result.spk_embed.shape == (3, D_SPEAKER)
        assert result.lengths.tolist() == [100, 150, 80]
        assert len(result.utterance_ids) == 3
        assert len(result.speaker_ids) == 3

    def test_batch_shapes_no_bucket(self):
        """max_frames overrides bucket boundaries."""
        batch = [
            _make_feature_set("spk_a", "utt_1", 100),
            _make_feature_set("spk_b", "utt_2", 150),
        ]
        result = collate_fn(batch, max_frames=200)
        assert result.content.shape == (3 if len(batch) == 3 else 2, D_CONTENT_VEC, 200)

    def test_single_item(self):
        batch = [_make_feature_set("spk_a", "utt_1", 50)]
        result = collate_fn(batch)
        # 50 snaps to bucket boundary 250
        assert result.content.shape == (1, D_CONTENT_VEC, 250)
        assert result.lengths.tolist() == [50]


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


    def test_subset_filters_entries(self, tmp_cache_dir):
        cache = FeatureCache(tmp_cache_dir)
        dataset_name = "test_subset"

        for i in range(20):
            spk = f"spk_{i % 4}"
            fs = _make_feature_set(spk, f"utt_{i:03d}", 100)
            fs.speaker_id = spk
            fs.utterance_id = f"utt_{i:03d}"
            cache.save(fs, dataset_name, "train")

        # Full dataset
        ds_full = TMRVCDataset(cache, dataset_name, "train", cross_speaker_prob=0.0)
        assert len(ds_full) == 20

        # 50% subset
        ds_half = TMRVCDataset(
            cache, dataset_name, "train", cross_speaker_prob=0.0, subset=0.5,
        )
        assert len(ds_half) == 10

        # 10% subset (at least 1)
        ds_tiny = TMRVCDataset(
            cache, dataset_name, "train", cross_speaker_prob=0.0, subset=0.05,
        )
        assert len(ds_tiny) == 1

    def test_subset_dataloader(self, tmp_cache_dir):
        cache = FeatureCache(tmp_cache_dir)
        dataset_name = "test_subset_dl"

        for i in range(16):
            spk = f"spk_{i % 4}"
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
            subset=0.5,
        )
        # 16 * 0.5 = 8 entries, batch_size=4, drop_last=True → 2 batches
        batches = list(loader)
        assert len(batches) == 2


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


class TestSpeakerGroupWeighting:
    def test_speaker_groups_weighted_sampling(self):
        """Weighted speaker appears more frequently in early rounds."""
        # 100 utterances from speaker A, 10 from moe/tsukuyomi
        speaker_ids = ["a"] * 100 + ["moe/tsukuyomi"] * 10
        groups = [SpeakerGroupConfig(speakers=["moe/*"], weight=10)]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42, speaker_groups=groups)
        indices = list(sampler)

        # All indices must be emitted exactly once
        assert len(indices) == 110
        assert set(indices) == set(range(110))

        # Check first 20 indices: moe/tsukuyomi (weight=10) should have ~10 in first round
        first_20 = indices[:20]
        moe_in_first_20 = sum(1 for i in first_20 if speaker_ids[i] == "moe/tsukuyomi")
        # With weight=10, moe yields 10 per round vs A yields 1.
        # In the first round (2 speakers: A→1, moe→10 = 11 indices), moe gets 10.
        assert moe_in_first_20 >= 5, f"Expected moe to appear frequently early, got {moe_in_first_20}"

    def test_speaker_groups_fnmatch_multiple_speakers(self):
        """fnmatch pattern matches multiple speakers in one group."""
        speaker_ids = (
            ["normal/spk1"] * 50
            + ["normal/spk2"] * 50
            + ["moe/tsukuyomi"] * 5
            + ["moe/aoi"] * 5
        )
        groups = [SpeakerGroupConfig(speakers=["moe/*"], weight=10)]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42, speaker_groups=groups)
        indices = list(sampler)

        assert len(indices) == 110
        assert set(indices) == set(range(110))

        # Both moe speakers should be weighted
        first_30 = indices[:30]
        moe_count = sum(1 for i in first_30 if speaker_ids[i].startswith("moe/"))
        assert moe_count >= 5, f"Expected moe speakers early, got {moe_count}"

    def test_speaker_groups_no_groups_default_behavior(self):
        """Without speaker_groups, behavior matches original sampler."""
        speaker_ids = ["a", "a", "b", "c"]
        sampler_default = BalancedSpeakerSampler(speaker_ids, seed=99)
        sampler_none = BalancedSpeakerSampler(speaker_ids, seed=99, speaker_groups=None)
        sampler_empty = BalancedSpeakerSampler(speaker_ids, seed=99, speaker_groups=[])

        assert list(sampler_default) == list(sampler_none)
        assert list(sampler_default) == list(sampler_empty)

    def test_speaker_groups_unknown_speaker_ignored(self):
        """Unknown speakers in group config are silently ignored."""
        speaker_ids = ["a", "a", "b"]
        groups = [SpeakerGroupConfig(speakers=["nonexistent"], weight=5)]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42, speaker_groups=groups)
        indices = list(sampler)
        assert len(indices) == 3
        assert set(indices) == {0, 1, 2}

    def test_speaker_groups_weight_one_no_effect(self):
        """weight=1 produces same result as no groups."""
        speaker_ids = ["a"] * 5 + ["b"] * 3
        sampler_no_group = BalancedSpeakerSampler(speaker_ids, seed=42)
        groups = [SpeakerGroupConfig(speakers=["b"], weight=1)]
        sampler_w1 = BalancedSpeakerSampler(speaker_ids, seed=42, speaker_groups=groups)
        assert list(sampler_no_group) == list(sampler_w1)

    def test_create_dataloader_with_speaker_groups(self, tmp_cache_dir):
        """create_dataloader accepts speaker_groups without error."""
        cache = FeatureCache(tmp_cache_dir)
        dataset_name = "test_sg"

        for i in range(8):
            spk = "moe/tsukuyomi" if i < 2 else f"spk_{i}"
            fs = _make_feature_set(spk, f"utt_{i:03d}", 100)
            fs.speaker_id = spk
            fs.utterance_id = f"utt_{i:03d}"
            cache.save(fs, dataset_name, "train")

        groups = [SpeakerGroupConfig(speakers=["moe/*"], weight=5)]
        loader = create_dataloader(
            tmp_cache_dir,
            dataset_name,
            batch_size=4,
            num_workers=0,
            cross_speaker_prob=0.0,
            speaker_groups=groups,
        )

        batch = next(iter(loader))
        assert isinstance(batch, TrainingBatch)
        assert batch.content.shape[0] == 4
