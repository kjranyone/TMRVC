"""TTSDataset and DataLoader construction for TTS training.

Extends the VC dataset with phoneme-level alignment data (phoneme_ids, durations)
for training TextEncoder, DurationPredictor, F0Predictor, and ContentSynthesizer.
"""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tmrvc_core.constants import DEFAULT_BATCH_SIZE, N_MELS
from tmrvc_core.types import TTSBatch, TTSFeatureSet
from tmrvc_data.cache import FeatureCache


class TTSDataset(Dataset):
    """Lazy-loading TTS dataset backed by :class:`FeatureCache`.

    Expects additional files in each utterance directory:
    - ``phoneme_ids.npy``: ``[L]`` int64 phoneme index sequence
    - ``durations.npy``: ``[L]`` int64 frames per phoneme

    These are produced by ``scripts/run_forced_alignment.py``.
    """

    def __init__(
        self,
        cache: FeatureCache,
        dataset: str | list[str],
        split: str = "train",
        max_frames: int = 0,
    ) -> None:
        self.cache = cache
        self.datasets = [dataset] if isinstance(dataset, str) else list(dataset)
        self.split = split
        self.max_frames = max_frames

        # Collect entries that have TTS alignment data
        self.entries: list[dict[str, str]] = []
        for ds_name in self.datasets:
            for entry in cache.iter_entries(ds_name, split):
                utt_dir = cache._utt_dir(
                    ds_name, split, entry["speaker_id"], entry["utterance_id"],
                )
                if (utt_dir / "phoneme_ids.npy").exists() and (utt_dir / "durations.npy").exists():
                    entry["dataset"] = ds_name
                    self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> TTSFeatureSet:
        entry = self.entries[idx]
        ds_name = entry["dataset"]
        utt_dir = self.cache._utt_dir(
            ds_name, self.split, entry["speaker_id"], entry["utterance_id"],
        )

        # Load base features
        features = self.cache.load(
            ds_name,
            self.split,
            entry["speaker_id"],
            entry["utterance_id"],
            mmap=True,
            max_frames=self.max_frames,
        )

        # Load TTS alignment data
        phoneme_ids = np.load(utt_dir / "phoneme_ids.npy")
        durations = np.load(utt_dir / "durations.npy")

        # Load language_id from meta if present
        with open(utt_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        language_id = meta.get("language_id", 0)
        text = meta.get("text", "")

        # If max_frames cropping was applied, adjust durations
        actual_frames = features.mel.shape[-1]
        total_dur = int(durations.sum())
        if actual_frames < total_dur:
            durations = _adjust_durations(durations, actual_frames)

        return TTSFeatureSet(
            mel=features.mel,
            content=features.content,
            f0=features.f0,
            spk_embed=features.spk_embed,
            phoneme_ids=torch.from_numpy(phoneme_ids.copy()).long(),
            durations=torch.from_numpy(durations.copy()).float(),
            language_id=language_id,
            utterance_id=features.utterance_id,
            speaker_id=features.speaker_id,
            n_frames=actual_frames,
            n_phonemes=len(phoneme_ids),
            content_dim=features.content_dim,
            text=text,
        )


def _adjust_durations(durations: np.ndarray, target_frames: int) -> np.ndarray:
    """Scale durations to match target frame count after cropping."""
    total = durations.sum()
    if total <= 0:
        return durations
    scale = target_frames / total
    adjusted = np.round(durations * scale).astype(np.int64)
    adjusted = np.maximum(adjusted, 0)
    # Fix rounding error on last non-zero element
    diff = target_frames - adjusted.sum()
    if diff != 0:
        nonzero = np.nonzero(adjusted)[0]
        if len(nonzero) > 0:
            adjusted[nonzero[-1]] = max(1, adjusted[nonzero[-1]] + diff)
    return adjusted


def _crop_or_pad(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    """Pad or crop the last dimension to max_len."""
    T = tensor.shape[-1]
    if T > max_len:
        return tensor[..., :max_len]
    if T < max_len:
        return torch.nn.functional.pad(tensor, (0, max_len - T))
    return tensor


def tts_collate_fn(
    batch: list[TTSFeatureSet],
    max_frames: int = 0,
) -> TTSBatch:
    """Collate TTSFeatureSets into a padded TTSBatch.

    Args:
        batch: List of TTSFeatureSet items.
        max_frames: If > 0, crop/pad all frame-level tensors to this length.
    """
    if max_frames > 0:
        target_t = max_frames
    else:
        target_t = max(f.n_frames for f in batch)

    max_phones = max(f.n_phonemes for f in batch)

    # Frame-level tensors
    mel_list = [_crop_or_pad(f.mel, target_t) for f in batch]
    content_list = [_crop_or_pad(f.content, target_t) for f in batch]
    f0_list = [_crop_or_pad(f.f0, target_t) for f in batch]

    # Phoneme-level tensors (pad to max_phones)
    phone_ids_list = []
    dur_list = []
    for f in batch:
        pad_len = max_phones - f.n_phonemes
        phone_ids_list.append(
            torch.nn.functional.pad(f.phoneme_ids, (0, pad_len), value=0)
        )
        dur_list.append(
            torch.nn.functional.pad(f.durations, (0, pad_len), value=0.0)
        )

    frame_lengths = torch.tensor(
        [min(f.n_frames, target_t) for f in batch], dtype=torch.long,
    )
    phoneme_lengths = torch.tensor(
        [f.n_phonemes for f in batch], dtype=torch.long,
    )

    return TTSBatch(
        phoneme_ids=torch.stack(phone_ids_list),
        durations=torch.stack(dur_list),
        language_ids=torch.tensor([f.language_id for f in batch], dtype=torch.long),
        content=torch.stack(content_list),
        f0=torch.stack(f0_list),
        spk_embed=torch.stack([f.spk_embed for f in batch]),
        mel_target=torch.stack(mel_list),
        frame_lengths=frame_lengths,
        phoneme_lengths=phoneme_lengths,
        utterance_ids=[f.utterance_id for f in batch],
        speaker_ids=[f.speaker_id for f in batch],
        content_dim=batch[0].content_dim,
    )


def create_tts_dataloader(
    cache_dir: str | Path,
    dataset: str | list[str],
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    max_frames: int = 400,
) -> DataLoader:
    """Create a DataLoader for TTS training.

    Args:
        cache_dir: Root cache directory.
        dataset: Dataset name(s).
        split: Split name.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        max_frames: Fixed frame count (0 = variable).

    Returns:
        DataLoader yielding TTSBatch.
    """
    cache = FeatureCache(cache_dir)
    ds = TTSDataset(
        cache=cache,
        dataset=dataset,
        split=split,
        max_frames=max_frames,
    )

    collate = partial(tts_collate_fn, max_frames=max_frames)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
