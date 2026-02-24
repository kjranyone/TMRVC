"""EmotionDataset and DataLoader for StyleEncoder training (Phase 3a).

Loads pre-computed mel spectrograms paired with emotion labels from cache.
Each utterance directory is expected to have:
- ``mel.npy``: [80, T] log-mel spectrogram
- ``emotion.json``: {"emotion_id": int, "vad": [v, a, d], "prosody": [rate, energy, pitch_range]}
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tmrvc_core.constants import N_MELS
from tmrvc_data.cache import FeatureCache

logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """Lazy-loading emotion dataset for StyleEncoder training.

    Each sample returns a dict with:
    - mel: [80, T] float32 tensor
    - emotion_id: int (0..11)
    - vad: [3] float32 tensor (optional, zeros if unavailable)
    - prosody: [3] float32 tensor (optional, zeros if unavailable)
    """

    def __init__(
        self,
        cache: FeatureCache,
        datasets: list[str],
        split: str = "train",
        max_frames: int = 200,
    ) -> None:
        self.cache = cache
        self.split = split
        self.max_frames = max_frames

        self.entries: list[dict] = []
        for ds_name in datasets:
            for entry in cache.iter_entries(ds_name, split):
                utt_dir = cache._utt_dir(
                    ds_name, split, entry["speaker_id"], entry["utterance_id"],
                )
                emotion_path = utt_dir / "emotion.json"
                if emotion_path.exists() and (utt_dir / "mel.npy").exists():
                    entry["dataset"] = ds_name
                    self.entries.append(entry)

        logger.info("EmotionDataset: %d samples from %s", len(self.entries), datasets)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.entries[idx]
        utt_dir = self.cache._utt_dir(
            entry["dataset"], self.split, entry["speaker_id"], entry["utterance_id"],
        )

        mel = np.load(utt_dir / "mel.npy")  # [80, T]

        # Crop or pad to max_frames
        T = mel.shape[1]
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            mel = mel[:, start:start + self.max_frames]
        elif T < self.max_frames:
            pad = np.zeros((N_MELS, self.max_frames - T), dtype=mel.dtype)
            mel = np.concatenate([mel, pad], axis=1)

        # Load emotion metadata
        with open(utt_dir / "emotion.json", encoding="utf-8") as f:
            emotion_meta = json.load(f)

        emotion_id = emotion_meta.get("emotion_id", 6)  # default: neutral
        vad = emotion_meta.get("vad", [0.0, 0.0, 0.0])
        prosody = emotion_meta.get("prosody", [0.0, 0.0, 0.0])

        return {
            "mel": torch.from_numpy(mel).float(),
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
            "vad": torch.tensor(vad, dtype=torch.float32),
            "prosody": torch.tensor(prosody, dtype=torch.float32),
        }


def create_emotion_dataloader(
    cache_dir: str | Path,
    datasets: list[str],
    batch_size: int = 64,
    num_workers: int = 0,
    max_frames: int = 200,
    split: str = "train",
) -> DataLoader:
    """Create a DataLoader for StyleEncoder training.

    Args:
        cache_dir: Path to feature cache root.
        datasets: List of dataset names (e.g. ['expresso', 'jvnv']).
        batch_size: Batch size.
        num_workers: Number of workers.
        max_frames: Fixed mel frame count per sample.
        split: Data split name.

    Returns:
        DataLoader yielding dicts with 'mel', 'emotion_id', 'vad', 'prosody'.
    """
    cache = FeatureCache(Path(cache_dir))
    dataset = EmotionDataset(cache, datasets, split=split, max_frames=max_frames)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
