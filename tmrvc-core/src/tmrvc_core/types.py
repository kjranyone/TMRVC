"""Shared data types for the TMRVC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Utterance:
    """Metadata for a single utterance in a dataset."""

    utterance_id: str
    speaker_id: str
    dataset: str
    audio_path: Path
    duration_sec: float
    sample_rate: int = 24000
    text: str | None = None
    language: str | None = None


@dataclass
class FeatureSet:
    """Cached features for a single utterance.

    All time-axis tensors share the same frame count ``T``
    (10 ms hop → T = duration / 0.01).

    content dimension is configurable:
    - 768 for ContentVec (Phase 0)
    - 1024 for WavLM-large layer 7 (Phase 1+)
    """

    mel: torch.Tensor  # [80, T]
    content: torch.Tensor  # [content_dim, T] where content_dim ∈ {768, 1024}
    f0: torch.Tensor  # [1, T]
    spk_embed: torch.Tensor  # [192]
    utterance_id: str = ""
    speaker_id: str = ""
    n_frames: int = 0
    content_dim: int = 768  # Track content feature dimension
    waveform: torch.Tensor | None = None  # [1, T_samples] optional for augmentation


@dataclass
class TrainingBatch:
    """Collated batch returned by the DataLoader.

    Shapes follow the convention ``[B, C, T]`` where ``T`` is the
    (padded) number of frames.

    content dimension matches the FeatureSet used for training:
    - 768 for ContentVec (Phase 0)
    - 1024 for WavLM-large layer 7 (Phase 1+)
    """

    content: torch.Tensor  # [B, content_dim, T]
    f0: torch.Tensor  # [B, 1, T]
    spk_embed: torch.Tensor  # [B, 192]
    mel_target: torch.Tensor  # [B, 80, T]
    lengths: torch.Tensor  # [B] — unpadded frame counts
    utterance_ids: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    content_dim: int = 768  # Track content feature dimension
