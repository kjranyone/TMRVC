"""UCLM Dataset for training the Unified Codec Language Model.

Loads pre-extracted features from cache:
- codec_tokens: [n_codebooks, T] from EnCodec
- voice_state: [T, 8] from VoiceStateEstimator
- phoneme_ids: [L] from G2P
- durations: [L] from MFA
- spk_embed: [192] from SpeakerEncoder
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class UCLMBatch:
    """Batch of UCLM training data."""

    codec_tokens: torch.Tensor  # [B, n_codebooks, T]
    voice_state: torch.Tensor  # [B, T, d_voice_state]
    phoneme_ids: torch.Tensor  # [B, L]
    durations: torch.Tensor  # [B, L]
    spk_embed: torch.Tensor  # [B, d_speaker]
    text: list[str]  # [B]
    utterance_ids: list[str]  # [B]
    frame_lengths: torch.Tensor  # [B]
    phoneme_lengths: torch.Tensor  # [B]


class UCLMDataset(Dataset):
    """Dataset for UCLM training.

    Expects cache directory structure:
        data/cache/{dataset}/train/{speaker}/{utterance}/
        ├── codec_tokens.npy     # [n_codebooks, T]
        ├── voice_state.npy      # [T, 8]
        ├── phoneme_ids.npy      # [L]
        ├── durations.npy        # [L]
        ├── spk_embed.npy        # [192]
        └── meta.json

    Args:
        cache_dir: Root cache directory.
        datasets: List of dataset names to include.
        max_frames: Maximum frames per utterance (truncate or skip).
        min_frames: Minimum frames per utterance (skip if shorter).
        include_datasets: If set, only include these datasets.
        exclude_speakers: Speaker IDs to exclude.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        datasets: list[str] | None = None,
        max_frames: int = 400,
        min_frames: int = 20,
        include_datasets: list[str] | None = None,
        exclude_speakers: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.datasets = datasets or []
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.include_datasets = include_datasets
        self.exclude_speakers = set(exclude_speakers or [])

        self.utterances: list[dict[str, Any]] = []
        self._scan_utterances()

        logger.info(
            "UCLMDataset: %d utterances from %d datasets",
            len(self.utterances),
            len(self.datasets),
        )

    def _scan_utterances(self) -> None:
        """Scan cache directory for valid utterances."""
        if not self.cache_dir.exists():
            logger.warning("Cache directory does not exist: %s", self.cache_dir)
            return

        for dataset_dir in self.cache_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if dataset_dir.name.startswith("_"):
                continue  # Skip _manifests etc.
            if self.include_datasets and dataset_dir.name not in self.include_datasets:
                continue
            if self.datasets and dataset_dir.name not in self.datasets:
                continue

            train_dir = dataset_dir / "train"
            if not train_dir.exists():
                continue

            for speaker_dir in train_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                if speaker_dir.name in self.exclude_speakers:
                    continue

                for utt_dir in speaker_dir.iterdir():
                    if not utt_dir.is_dir():
                        continue

                    # Check for required files
                    codec_path = utt_dir / "codec_tokens.npy"
                    voice_path = utt_dir / "voice_state.npy"
                    meta_path = utt_dir / "meta.json"

                    if not all(p.exists() for p in [codec_path, voice_path, meta_path]):
                        continue

                    # Load metadata
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception as e:
                        logger.debug("Failed to load meta.json: %s", e)
                        continue

                    n_frames = meta.get("n_frames", 0)

                    # Filter by frame count
                    if n_frames < self.min_frames:
                        continue
                    if n_frames > self.max_frames:
                        continue

                    self.utterances.append(
                        {
                            "utterance_id": meta.get("utterance_id", utt_dir.name),
                            "speaker_id": meta.get("speaker_id", speaker_dir.name),
                            "dataset": dataset_dir.name,
                            "path": utt_dir,
                            "n_frames": n_frames,
                            "text": meta.get("text", ""),
                            "meta": meta,
                        }
                    )

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        utt = self.utterances[idx]
        utt_path = utt["path"]

        # Load codec tokens
        codec_tokens = np.load(utt_path / "codec_tokens.npy")
        codec_tokens = torch.from_numpy(codec_tokens).long()  # [n_codebooks, T]

        # Load voice state
        voice_state = np.load(utt_path / "voice_state.npy")
        voice_state = torch.from_numpy(voice_state).float()  # [T, 8]

        # Load phoneme ids and durations (optional)
        phoneme_ids_path = utt_path / "phoneme_ids.npy"
        durations_path = utt_path / "durations.npy"

        if phoneme_ids_path.exists():
            phoneme_ids = np.load(phoneme_ids_path)
            phoneme_ids = torch.from_numpy(phoneme_ids).long()
        else:
            phoneme_ids = torch.zeros(1, dtype=torch.long)

        if durations_path.exists():
            durations = np.load(durations_path)
            durations = torch.from_numpy(durations).long()
        else:
            durations = torch.ones(1, dtype=torch.long)

        # Load speaker embedding
        spk_embed_path = utt_path / "spk_embed.npy"
        if spk_embed_path.exists():
            spk_embed = np.load(spk_embed_path)
            spk_embed = torch.from_numpy(spk_embed).float()
        else:
            spk_embed = torch.zeros(192, dtype=torch.float)

        return {
            "codec_tokens": codec_tokens,
            "voice_state": voice_state,
            "phoneme_ids": phoneme_ids,
            "durations": durations,
            "spk_embed": spk_embed,
            "text": utt["text"],
            "utterance_id": utt["utterance_id"],
            "n_frames": utt["n_frames"],
        }


def collate_uclm_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> UCLMBatch:
    """Collate a batch of UCLM samples.

    Pads sequences to the maximum length in the batch.

    Note: codec_tokens use EnCodec frame rate (75 fps), while voice_state
    uses mel frame rate (100 fps). We resample voice_state to match codec_tokens.

    Args:
        batch: List of samples from UCLMDataset.
        pad_id: Padding token ID for phoneme sequences.

    Returns:
        UCLMBatch with padded tensors.
    """
    # Find max lengths (use codec_tokens for frame count)
    max_codec_frames = max(sample["codec_tokens"].shape[1] for sample in batch)
    max_phonemes = max(sample["phoneme_ids"].shape[0] for sample in batch)

    B = len(batch)
    n_codebooks = batch[0]["codec_tokens"].shape[0]
    d_voice_state = batch[0]["voice_state"].shape[1]
    d_speaker = batch[0]["spk_embed"].shape[0]

    # Initialize padded tensors
    codec_tokens = torch.zeros(B, n_codebooks, max_codec_frames, dtype=torch.long)
    voice_state = torch.zeros(B, max_codec_frames, d_voice_state, dtype=torch.float)
    phoneme_ids = torch.full((B, max_phonemes), pad_id, dtype=torch.long)
    durations = torch.zeros(B, max_phonemes, dtype=torch.long)
    spk_embed = torch.zeros(B, d_speaker, dtype=torch.float)
    frame_lengths = torch.zeros(B, dtype=torch.long)
    phoneme_lengths = torch.zeros(B, dtype=torch.long)

    text = []
    utterance_ids = []

    for i, sample in enumerate(batch):
        n_codec_frames = sample["codec_tokens"].shape[1]
        n_voice_frames = sample["voice_state"].shape[0]
        n_phonemes = sample["phoneme_ids"].shape[0]

        # Copy codec tokens
        codec_tokens[i, :, :n_codec_frames] = sample["codec_tokens"]

        # Resample voice_state to match codec frame count if needed
        vs = sample["voice_state"]
        if n_voice_frames != n_codec_frames:
            # Linear interpolation
            vs = (
                torch.nn.functional.interpolate(
                    vs.unsqueeze(0).transpose(1, 2),  # [1, d, T_voice]
                    size=n_codec_frames,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .squeeze(0)
            )  # [T_codec, d]

        voice_state[i, :n_codec_frames, :] = vs
        phoneme_ids[i, :n_phonemes] = sample["phoneme_ids"]
        durations[i, :n_phonemes] = sample["durations"]
        spk_embed[i] = sample["spk_embed"]
        frame_lengths[i] = n_codec_frames  # Use codec frame count
        phoneme_lengths[i] = n_phonemes

        text.append(sample["text"])
        utterance_ids.append(sample["utterance_id"])

    return UCLMBatch(
        codec_tokens=codec_tokens,
        voice_state=voice_state,
        phoneme_ids=phoneme_ids,
        durations=durations,
        spk_embed=spk_embed,
        text=text,
        utterance_ids=utterance_ids,
        frame_lengths=frame_lengths,
        phoneme_lengths=phoneme_lengths,
    )
