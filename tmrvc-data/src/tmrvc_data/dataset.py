"""TMRVCDataset and DataLoader construction."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tmrvc_core.audio import compute_mel
from tmrvc_core.constants import (
    CROSS_SPEAKER_PROB,
    D_CONTENT_VEC,
    D_SPEAKER,
    DEFAULT_BATCH_SIZE,
    N_MELS,
)
from tmrvc_core.types import FeatureSet, TrainingBatch
from tmrvc_data.augmentation import Augmenter
from tmrvc_data.cache import FeatureCache
from tmrvc_data.sampler import BalancedSpeakerSampler, SpeakerGroupConfig


class TMRVCDataset(Dataset):
    """Lazy-loading dataset backed by :class:`FeatureCache`.

    Supports cross-speaker conditioning augmentation: with probability
    ``cross_speaker_prob``, the speaker embedding is swapped with one
    from a different speaker in the same batch.

    ``dataset`` can be a single name or a list of names for multi-dataset
    training.  Each entry stores its originating dataset so that
    :meth:`FeatureCache.load` resolves the correct directory.
    """

    def __init__(
        self,
        cache: FeatureCache,
        dataset: str | list[str],
        split: str = "train",
        cross_speaker_prob: float = CROSS_SPEAKER_PROB,
        subset: float = 1.0,
        augmenter: Augmenter | None = None,
    ) -> None:
        self.cache = cache
        self.datasets = [dataset] if isinstance(dataset, str) else list(dataset)
        self.split = split
        self.cross_speaker_prob = cross_speaker_prob
        self.augmenter = augmenter

        # Collect entries from all datasets
        self.entries: list[dict[str, str]] = []
        for ds_name in self.datasets:
            for entry in cache.iter_entries(ds_name, split):
                entry["dataset"] = ds_name
                self.entries.append(entry)

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
        ds_name = entry["dataset"]
        load_waveform = self.augmenter is not None and bool(
            self.augmenter.config.rir_dirs
        )
        features = self.cache.load(
            ds_name,
            self.split,
            entry["speaker_id"],
            entry["utterance_id"],
            mmap=True,
            load_waveform=load_waveform,
        )

        # Feature-level augmentation
        if self.augmenter is not None:
            # Content dropout
            if random.random() < self.augmenter.config.content_dropout_prob:
                features = FeatureSet(
                    mel=features.mel,
                    content=self.augmenter.apply_content_dropout(features.content),
                    f0=features.f0,
                    spk_embed=features.spk_embed,
                    utterance_id=features.utterance_id,
                    speaker_id=features.speaker_id,
                    n_frames=features.n_frames,
                    content_dim=features.content_dim,
                    waveform=features.waveform,
                )

            # F0 perturbation on feature level (semitone shift)
            if random.random() < self.augmenter.config.f0_perturbation_prob:
                shift = random.uniform(
                    *self.augmenter.config.f0_shift_semitones_range
                )
                f0_shifted = features.f0.clone()
                voiced = f0_shifted > 0
                f0_shifted[voiced] = f0_shifted[voiced] * (2.0 ** (shift / 12.0))
                features = FeatureSet(
                    mel=features.mel,
                    content=features.content,
                    f0=f0_shifted,
                    spk_embed=features.spk_embed,
                    utterance_id=features.utterance_id,
                    speaker_id=features.speaker_id,
                    n_frames=features.n_frames,
                    content_dim=features.content_dim,
                    waveform=features.waveform,
                )

            # Audio-level augmentation (RIR/EQ/noise → mel recompute)
            if features.waveform is not None and self.augmenter.config.rir_dirs:
                aug_waveform = self.augmenter.augment_audio(features.waveform)
                aug_mel = compute_mel(aug_waveform)
                # Truncate/pad to match original frame count
                T = features.n_frames
                if aug_mel.shape[-1] > T:
                    aug_mel = aug_mel[..., :T]
                elif aug_mel.shape[-1] < T:
                    aug_mel = torch.nn.functional.pad(
                        aug_mel, (0, T - aug_mel.shape[-1]),
                    )
                features = FeatureSet(
                    mel=aug_mel.squeeze(0) if aug_mel.dim() == 3 else aug_mel,
                    content=features.content,
                    f0=features.f0,
                    spk_embed=features.spk_embed,
                    utterance_id=features.utterance_id,
                    speaker_id=features.speaker_id,
                    n_frames=features.n_frames,
                    content_dim=features.content_dim,
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
                    other_entry["dataset"],
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
                    content_dim=features.content_dim,
                )

        return features


# Default bucket boundaries (frames).  Each value is a max-T that XPU
# will compile a kernel for.  5 buckets → 5 kernel variants at most.
DEFAULT_BUCKET_BOUNDARIES: list[int] = [250, 500, 750, 1000]


def _snap_to_bucket(length: int, boundaries: list[int]) -> int:
    """Round *length* up to the nearest bucket boundary."""
    for b in boundaries:
        if length <= b:
            return b
    return boundaries[-1]


def _crop_or_pad(tensor: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Random-crop if longer than *max_frames*, zero-pad if shorter."""
    T = tensor.shape[-1]
    if T > max_frames:
        start = random.randint(0, T - max_frames)
        return tensor[..., start : start + max_frames]
    if T < max_frames:
        return torch.nn.functional.pad(tensor, (0, max_frames - T), value=0.0)
    return tensor


def collate_fn(
    batch: list[FeatureSet],
    max_frames: int = 0,
    bucket_boundaries: list[int] | None = None,
) -> TrainingBatch:
    """Collate variable-length FeatureSets into a padded TrainingBatch.

    Args:
        batch: List of FeatureSets.
        max_frames: If > 0, crop/pad all items to this fixed length.
            This keeps tensor shapes constant across batches, which is
            critical for XPU performance (avoids kernel recompilation).
            If 0, pads to the longest item in the batch (or snaps to
            a bucket boundary when *bucket_boundaries* is set).
        bucket_boundaries: List of allowed frame lengths, e.g.
            ``[250, 500, 750, 1000]``.  The batch max-T is rounded up
            to the nearest boundary, limiting the number of distinct
            tensor shapes that XPU must compile kernels for.  Utterances
            longer than the largest boundary are random-cropped.
    """
    boundaries = bucket_boundaries or DEFAULT_BUCKET_BOUNDARIES
    max_boundary = boundaries[-1]

    if max_frames > 0:
        # Fixed-length mode: random-crop long utterances, pad short ones
        target_t = max_frames
    else:
        # Bucket mode: snap batch max to nearest boundary
        raw_max = max(f.n_frames for f in batch)
        target_t = _snap_to_bucket(raw_max, boundaries)

    content_list = [_crop_or_pad(f.content, target_t) for f in batch]
    f0_list = [_crop_or_pad(f.f0, target_t) for f in batch]
    mel_list = [_crop_or_pad(f.mel, target_t) for f in batch]
    lengths = torch.tensor(
        [min(f.n_frames, target_t) for f in batch], dtype=torch.long,
    )

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
        content=_pad_time(content_list, target_t),      # [B, D, T]
        f0=_pad_time(f0_list, target_t),                 # [B, 1, T]
        spk_embed=torch.stack([f.spk_embed for f in batch]),  # [B, 192]
        mel_target=_pad_time(mel_list, target_t),        # [B, 80, T]
        lengths=lengths,
        utterance_ids=[f.utterance_id for f in batch],
        speaker_ids=[f.speaker_id for f in batch],
    )


def create_dataloader(
    cache_dir: str | Path,
    dataset: str | list[str],
    split: str = "train",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    cross_speaker_prob: float = CROSS_SPEAKER_PROB,
    balanced_sampling: bool = True,
    subset: float = 1.0,
    speaker_groups: list[SpeakerGroupConfig] | None = None,
    augmenter: Augmenter | None = None,
    max_frames: int = 0,
) -> DataLoader:
    """Create a DataLoader with balanced speaker sampling.

    Args:
        cache_dir: Root cache directory.
        dataset: Dataset name (e.g. ``"vctk"``) or list of names for
            multi-dataset training (e.g. ``["vctk", "jvs", "libritts_r"]``).
        split: Split name.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        cross_speaker_prob: Probability of cross-speaker embedding swap.
        balanced_sampling: Use :class:`BalancedSpeakerSampler`.
        subset: Fraction of data to use.
        speaker_groups: Optional speaker group weight configs.
        augmenter: Optional :class:`Augmenter` for online augmentation.
        max_frames: Fixed frame count for all batches.  If > 0, every
            batch is cropped/padded to this length, keeping tensor shapes
            constant.  **Required for XPU** to avoid kernel recompilation.

    Returns:
        Configured DataLoader yielding :class:`TrainingBatch`.
    """
    from functools import partial

    cache = FeatureCache(cache_dir)
    ds = TMRVCDataset(
        cache=cache,
        dataset=dataset,
        split=split,
        cross_speaker_prob=cross_speaker_prob,
        subset=subset,
        augmenter=augmenter,
    )

    sampler = None
    shuffle = True
    if balanced_sampling and len(ds) > 0:
        speaker_ids = [e["speaker_id"] for e in ds.entries]
        sampler = BalancedSpeakerSampler(
            speaker_ids, speaker_groups=speaker_groups,
        )
        shuffle = False  # sampler handles ordering

    # Bucket batching is always active (uses DEFAULT_BUCKET_BOUNDARIES).
    # max_frames overrides to a single fixed length.
    collate = partial(collate_fn, max_frames=max_frames)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
