"""Shared data types for the TMRVC pipeline (UCLM v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

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
    """Unified cached features for UCLM v2 (dual-stream)."""
    codec_tokens_a: torch.Tensor  # [8, T]
    codec_tokens_b: torch.Tensor  # [4, T]
    voice_state_explicit: torch.Tensor  # [8, T]
    voice_state_ssl: torch.Tensor       # [128, T]
    spk_embed: torch.Tensor  # [192]
    mel: Optional[torch.Tensor] = None
    f0: Optional[torch.Tensor] = None
    utterance_id: str = ""
    speaker_id: str = ""
    n_frames: int = 0
    waveform: Optional[torch.Tensor] = None


@dataclass
class UCLMFeatureSet(FeatureSet):
    """Features for UCLM multi-task training (VC + TTS)."""
    phoneme_ids: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None
    language_id: int = 0
    text: str = ""


@dataclass
class UCLM_Batch:
    """Collated batch for DisentangledUCLM training."""
    target_a: torch.Tensor
    target_b: torch.Tensor
    explicit_state: torch.Tensor
    ssl_state: torch.Tensor
    speaker_embed: torch.Tensor
    speaker_id: torch.Tensor
    lengths: torch.Tensor # Frame counts
    f0_condition: Optional[torch.Tensor] = None
    phoneme_ids: Optional[torch.Tensor] = None
    phoneme_lens: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None
    language_id: Optional[torch.Tensor] = None
    utterance_ids: List[str] = field(default_factory=list)

    def to(self, device: torch.device | str) -> UCLM_Batch:
        res = {}
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device)
            else:
                res[k] = v
        return UCLM_Batch(**res)
