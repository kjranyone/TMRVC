"""TMRVCDataset and DataLoader construction."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tmrvc_core.constants import (
    CROSS_SPEAKER_PROB,
    D_CONTENT_VEC,
    D_SPEAKER,
    DEFAULT_BATCH_SIZE,
    N_MELS,
)
from tmrvc_core.types import FeatureSet, TrainingBatch
from tmrvc_data.cache import FeatureCache
from tmrvc_data.sampler import BalancedSpeakerSampler


class TMRVCDataset(Dataset):
    """Lazy-loading dataset backed by :class:`FeatureCache`.

    Supports cross-speaker conditioning augmentation: with probability
    ``cross_speaker_prob``, the speaker embedding is swapped with one
    from a different speaker in the same batch.
    """

    def __init__(
        self,
        cache: FeatureCache,
        dataset: str,
        split: str = "train",
        cross_speaker_prob: float = CROSS_SPEAKER_PROB,
        subset: float = 1.0,
    ) -> None:
        self.cache = cache
        self.dataset = dataset
        self.split = split
        self.cross_speaker_prob = cross_speaker_prob
        self.entries = cache.iter_entries(dataset, split)
        if subset < 1.0:
            k = max(1, int(len(self.entries) * subset))
            self.entries = random.sample(self.entries, k)

        # Build speaker → entry indices mapping for cross-speaker swap
        self._speaker_to_indices: dict[str, list[int]] = {}
        for i, entry in enumerate(self.entries):
            sid = entry["speaker_id"]
            self._speaker_to_indices.setdefault(sid, []).append(i)
        self._all_speakers = list(self._speaker_to_indices.keys())

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> FeatureSet:
        entry = self.entries[idx]
        features = self.cache.load(
            self.dataset,
            self.split,
            entry["speaker_id"],
            entry["utterance_id"],
            mmap=True,
        )

        # Cross-speaker augmentation
        if (
            self.cross_speaker_prob > 0
            and random.random() < self.cross_speaker_prob
            and len(self._all_speakers) > 1
        ):
            # Pick a different speaker
            other_speakers = [
                s for s in self._all_speakers if s != entry["speaker_id"]
            ]
            if other_speakers:
                other_sid = random.choice(other_speakers)
                other_idx = random.choice(self._speaker_to_indices[other_sid])
                other_entry = self.entries[other_idx]
                other_features = self.cache.load(
                    self.dataset,
                    self.split,
                    other_entry["speaker_id"],
                    other_entry["utterance_id"],
                    mmap=True,
                )
                features = FeatureSet(
                    mel=features.mel,
                    content=features.content,
                    f0=features.f0,
                    spk_embed=other_features.spk_embed,
                    utterance_id=features.utterance_id,
                    speaker_id=features.speaker_id,
                    n_frames=features.n_frames,
                )

        return features


def collate_fn(batch: list[FeatureSet]) -> TrainingBatch:
    """Collate variable-length FeatureSets into a padded TrainingBatch.

    Pads along the time axis to the length of the longest item.
    """
    lengths = torch.tensor([f.n_frames for f in batch], dtype=torch.long)

    # Pad along time dimension (dim=-1)
    content_list = [f.content for f in batch]  # each [768, T_i]
    f0_list = [f.f0 for f in batch]            # each [1, T_i]
    mel_list = [f.mel for f in batch]          # each [80, T_i]

    max_t = max(c.shape[-1] for c in content_list)

    def _pad_time(tensors: list[torch.Tensor], max_len: int) -> torch.Tensor:
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[-1]
            if pad_len > 0:
                padded.append(
                    torch.nn.functional.pad(t, (0, pad_len), value=0.0)
                )
            else:
                padded.append(t)
        return torch.stack(padded)

    return TrainingBatch(
        content=_pad_time(content_list, max_t),      # [B, 768, T]
        f0=_pad_time(f0_list, max_t),                 # [B, 1, T]
        spk_embed=torch.stack([f.spk_embed for f in batch]),  # [B, 192]
        mel_target=_pad_time(mel_list, max_t),        # [B, 80, T]
        lengths=lengths,
        utterance_ids=[f.utterance_id for f in batch],
        speaker_ids=[f.speaker_id for f in batch],
    )


def create_dataloader(
    cache_dir: str | Path,
    dataset: str,
    split: str = "train",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    cross_speaker_prob: float = CROSS_SPEAKER_PROB,
    balanced_sampling: bool = True,
    subset: float = 1.0,
) -> DataLoader:
    """Create a DataLoader with balanced speaker sampling.

    Args:
        cache_dir: Root cache directory.
        dataset: Dataset name (e.g. ``"vctk"``).
        split: Split name.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        cross_speaker_prob: Probability of cross-speaker embedding swap.
        balanced_sampling: Use :class:`BalancedSpeakerSampler`.

    Returns:
        Configured DataLoader yielding :class:`TrainingBatch`.
    """
    cache = FeatureCache(cache_dir)
    ds = TMRVCDataset(
        cache=cache,
        dataset=dataset,
        split=split,
        cross_speaker_prob=cross_speaker_prob,
        subset=subset,
    )

    sampler = None
    shuffle = True
    if balanced_sampling and len(ds) > 0:
        speaker_ids = [e["speaker_id"] for e in ds.entries]
        sampler = BalancedSpeakerSampler(speaker_ids)
        shuffle = False  # sampler handles ordering

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
