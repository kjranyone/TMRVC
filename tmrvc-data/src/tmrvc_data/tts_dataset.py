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
from tmrvc_data.events import events_to_tensors, load_events


class TTSDataset(Dataset):
    """Lazy-loading TTS dataset backed by :class:`FeatureCache`.

    Expects additional files in each utterance directory:
    - ``phoneme_ids.npy`` or ``token_ids.npy``: ``[L]`` int64 text unit IDs
    - ``durations.npy``: ``[L]`` int64 frames per text unit

    These are produced by ``scripts/run_forced_alignment.py``.
    """

    def __init__(
        self,
        cache: FeatureCache,
        dataset: str | list[str],
        split: str = "train",
        max_frames: int = 0,
        frontend: str = "auto",
    ) -> None:
        self.cache = cache
        self.datasets = [dataset] if isinstance(dataset, str) else list(dataset)
        self.split = split
        self.max_frames = max_frames
        if frontend not in {"phoneme", "tokenizer", "auto"}:
            raise ValueError(
                f"Unsupported frontend: {frontend}. Expected one of phoneme/tokenizer/auto."
            )
        self.frontend = frontend

        # Collect entries that have TTS alignment data
        self.entries: list[dict[str, str]] = []
        for ds_name in self.datasets:
            for entry in cache.iter_entries(ds_name, split):
                utt_dir = cache._utt_dir(
                    ds_name, split, entry["speaker_id"], entry["utterance_id"],
                )
                id_file = self._resolve_id_file(utt_dir)
                if id_file is not None and (utt_dir / "durations.npy").exists():
                    entry["dataset"] = ds_name
                    self.entries.append(entry)

    def _resolve_id_file(self, utt_dir: Path) -> Path | None:
        token_path = utt_dir / "token_ids.npy"
        phone_path = utt_dir / "phoneme_ids.npy"
        if self.frontend == "tokenizer":
            return token_path if token_path.exists() else None
        if self.frontend == "phoneme":
            return phone_path if phone_path.exists() else None
        # auto: prefer tokenizer path for new E2E pipeline.
        if token_path.exists():
            return token_path
        if phone_path.exists():
            return phone_path
        return None

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
        id_file = self._resolve_id_file(utt_dir)
        if id_file is None:
            raise FileNotFoundError(f"No text unit ID file found in {utt_dir}")
        text_ids = np.load(id_file)
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

        # Load BPEH events if available
        events = load_events(utt_dir)
        event_tensors = events_to_tensors(events, actual_frames) if events else None

        return TTSFeatureSet(
            mel=features.mel,
            content=features.content,
            f0=features.f0,
            spk_embed=features.spk_embed,
            phoneme_ids=torch.from_numpy(text_ids.copy()).long(),
            durations=torch.from_numpy(durations.copy()).float(),
            language_id=language_id,
            utterance_id=features.utterance_id,
            speaker_id=features.speaker_id,
            n_frames=actual_frames,
            n_phonemes=len(text_ids),
            content_dim=features.content_dim,
            text=text,
            breath_onsets=event_tensors["breath_onsets"] if event_tensors else None,
            breath_durations=event_tensors["breath_durations"] if event_tensors else None,
            breath_intensity=event_tensors["breath_intensity"] if event_tensors else None,
            pause_durations=event_tensors["pause_durations"] if event_tensors else None,
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

    # BPEH event tensors (collate if any item has them)
    has_events = any(f.breath_onsets is not None for f in batch)
    breath_onsets = None
    breath_durations = None
    breath_intensity = None
    pause_durations = None
    if has_events:
        breath_onsets = torch.stack([
            _crop_or_pad(f.breath_onsets.unsqueeze(0), target_t).squeeze(0)
            if f.breath_onsets is not None else torch.zeros(target_t)
            for f in batch
        ])
        breath_durations = torch.stack([
            _crop_or_pad(f.breath_durations.unsqueeze(0), target_t).squeeze(0)
            if f.breath_durations is not None else torch.zeros(target_t)
            for f in batch
        ])
        breath_intensity = torch.stack([
            _crop_or_pad(f.breath_intensity.unsqueeze(0), target_t).squeeze(0)
            if f.breath_intensity is not None else torch.zeros(target_t)
            for f in batch
        ])
        pause_durations = torch.stack([
            _crop_or_pad(f.pause_durations.unsqueeze(0), target_t).squeeze(0)
            if f.pause_durations is not None else torch.zeros(target_t)
            for f in batch
        ])

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
        breath_onsets=breath_onsets,
        breath_durations=breath_durations,
        breath_intensity=breath_intensity,
        pause_durations=pause_durations,
    )


def create_tts_dataloader(
    cache_dir: str | Path,
    dataset: str | list[str],
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    max_frames: int = 400,
    frontend: str = "auto",
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
        frontend=frontend,
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
