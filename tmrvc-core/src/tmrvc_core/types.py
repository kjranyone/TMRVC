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


# --- TTS Extension ---


@dataclass
class TTSFeatureSet:
    """Cached features for a single TTS utterance.

    Extends FeatureSet with phoneme-level alignment information.
    """

    mel: torch.Tensor  # [80, T]
    content: torch.Tensor  # [content_dim, T]
    f0: torch.Tensor  # [1, T]
    spk_embed: torch.Tensor  # [192]
    phoneme_ids: torch.Tensor  # [L] — phoneme index sequence
    durations: torch.Tensor  # [L] — frames per phoneme
    language_id: int = 0  # 0=ja, 1=en, 2=zh, 3=ko
    utterance_id: str = ""
    speaker_id: str = ""
    n_frames: int = 0
    n_phonemes: int = 0
    content_dim: int = 768
    text: str = ""

    # BPEH event tensors (frame-level, optional)
    breath_onsets: torch.Tensor | None = None   # [T] binary
    breath_durations: torch.Tensor | None = None  # [T] ms
    breath_intensity: torch.Tensor | None = None  # [T] 0-1
    pause_durations: torch.Tensor | None = None  # [T] ms


@dataclass
class TTSBatch:
    """Collated TTS batch returned by the DataLoader.

    Includes both frame-level features (mel, content, f0) and
    phoneme-level features (phoneme_ids, durations) with separate lengths.
    """

    phoneme_ids: torch.Tensor  # [B, L_max] — padded phoneme sequences
    durations: torch.Tensor  # [B, L_max] — frames per phoneme
    language_ids: torch.Tensor  # [B] — language index
    content: torch.Tensor  # [B, content_dim, T]
    f0: torch.Tensor  # [B, 1, T]
    spk_embed: torch.Tensor  # [B, 192]
    mel_target: torch.Tensor  # [B, 80, T]
    frame_lengths: torch.Tensor  # [B] — unpadded frame counts
    phoneme_lengths: torch.Tensor  # [B] — unpadded phoneme counts
    utterance_ids: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    content_dim: int = 768
    style: torch.Tensor | None = None  # [B, d_style] optional style vec

    # BPEH event tensors (frame-level, optional)
    breath_onsets: torch.Tensor | None = None   # [B, T] binary: 1 at breath onset frames
    breath_durations: torch.Tensor | None = None  # [B, T] ms duration at onset frames, 0 elsewhere
    breath_intensity: torch.Tensor | None = None  # [B, T] intensity (0-1) at onset frames
    pause_durations: torch.Tensor | None = None  # [B, T] ms pause duration at onset frames

    def to(self, device: torch.device | str) -> TTSBatch:
        """Transfer all tensor fields to device."""
        return TTSBatch(
            phoneme_ids=self.phoneme_ids.to(device),
            durations=self.durations.to(device),
            language_ids=self.language_ids.to(device),
            content=self.content.to(device),
            f0=self.f0.to(device),
            spk_embed=self.spk_embed.to(device),
            mel_target=self.mel_target.to(device),
            frame_lengths=self.frame_lengths.to(device),
            phoneme_lengths=self.phoneme_lengths.to(device),
            utterance_ids=self.utterance_ids,
            speaker_ids=self.speaker_ids,
            content_dim=self.content_dim,
            style=self.style.to(device) if self.style is not None else None,
            breath_onsets=self.breath_onsets.to(device) if self.breath_onsets is not None else None,
            breath_durations=self.breath_durations.to(device) if self.breath_durations is not None else None,
            breath_intensity=self.breath_intensity.to(device) if self.breath_intensity is not None else None,
            pause_durations=self.pause_durations.to(device) if self.pause_durations is not None else None,
        )
