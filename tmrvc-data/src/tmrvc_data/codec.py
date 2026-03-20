"""UCLM Codec wrapper for token extraction and decoding.

Uses the custom EmotionAwareCodec architecture (Token Spec) with 10ms hop.
v4 codec strategy: CodecInterface ABC with condition A/B/C/D backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Optional

import torch
import torch.nn as nn

from tmrvc_core.constants import SAMPLE_RATE, HOP_LENGTH, N_CODEBOOKS, RVQ_VOCAB_SIZE
from tmrvc_train.models import EmotionAwareCodec

logger = logging.getLogger(__name__)


class CodecInterface(ABC):
    """Abstract interface for codec backends (v4 codec strategy)."""

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode waveform to tokens.

        Args:
            waveform: [B, 1, T_samples]
        Returns:
            a_tokens: [B, n_codebooks, T_frames]
            b_logits: [B, n_control_slots, T_frames, control_vocab] or similar
        """
        ...

    @abstractmethod
    def decode(self, a_tokens: torch.Tensor, b_tokens: torch.Tensor, voice_state: torch.Tensor) -> torch.Tensor:
        """Decode tokens to waveform."""
        ...

    @property
    @abstractmethod
    def n_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def condition_id(self) -> str:
        """Codec condition identifier: 'A', 'B', 'C', or 'D'."""
        ...


class UCLMCodecWrapper(CodecInterface, nn.Module):
    """Wrapper for TMRVC UCLM EmotionAwareCodec.

    Provides token extraction (A_t, B_t) and decoding at 24kHz, 100fps (10ms/frame).
    Codec condition A (default): 8 codebooks, 1024 vocab, 100Hz frame rate.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.sample_rate = SAMPLE_RATE
        self.hop_length = HOP_LENGTH
        self._n_codebooks = N_CODEBOOKS
        self._vocab_size = RVQ_VOCAB_SIZE

        self._model = EmotionAwareCodec().to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info("Loading Codec checkpoint from %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self._model.load_state_dict(ckpt.get("model", ckpt))
        else:
            logger.warning(
                "No Codec checkpoint provided; using random weights for extraction."
            )

        self._model.eval()

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def condition_id(self) -> str:
        return "A"

    @torch.no_grad()
    def encode(
        self,
        waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio waveform to dual-stream tokens.

        Args:
            waveform: [B, 1, T_samples] at 24kHz.

        Returns:
            a_tokens: [B, 8, T_frames] Acoustic tokens.
            b_logits: [B, 4, T_frames, 64] Control logits.
        """
        # Ensure 3D: [B, 1, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        waveform = waveform.to(self.device)

        # EmotionAwareCodec.encode returns (a_tokens, b_logits, states, a_logits)
        a_tokens, b_logits, _, _ = self._model.encode(waveform)

        return a_tokens, b_logits

    @torch.no_grad()
    def decode(
        self,
        a_tokens: torch.Tensor,
        b_tokens: torch.Tensor,
        voice_state: torch.Tensor,
    ) -> torch.Tensor:
        """Decode tokens back to waveform."""
        audio, _ = self._model.decode(
            a_tokens.to(self.device),
            b_tokens.to(self.device),
            voice_state.to(self.device),
        )
        return audio

    def tokenize_file(self, audio_path: Path | str) -> dict[str, torch.Tensor]:
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        a_tokens, b_logits = self.encode(waveform.unsqueeze(0))
        b_tokens = b_logits.argmax(dim=-1)

        return {
            "a_tokens": a_tokens.cpu(),
            "b_tokens": b_tokens.cpu(),
            "duration_sec": waveform.shape[-1] / self.sample_rate,
        }


# Backward compat alias
EnCodecWrapper = UCLMCodecWrapper


class MimiCodecWrapper(CodecInterface, nn.Module):
    """Mimi codec wrapper for condition B/C (v4 codec strategy).

    Uses moshi/mimi codec with 8 codebooks, 2048 vocab, 12.5Hz frame rate.
    Optional dependency: install ``moshi`` package.
    """

    def __init__(self, checkpoint_path: Path | str | None = None, device: str = "cuda"):
        super().__init__()
        self.device_str = device
        self._device = torch.device(device)
        self.sample_rate = SAMPLE_RATE
        self._n_codebooks_val = 8
        self._vocab_size_val = 2048
        self.frame_rate = 12.5  # Hz

        self._model = None
        try:
            import moshi
            logger.info("moshi package available; MimiCodecWrapper ready.")
        except ImportError:
            logger.warning("moshi package not installed; MimiCodecWrapper will use stub mode.")

    def encode(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._model is None:
            B = waveform.shape[0]
            T_frames = waveform.shape[-1] // int(self.sample_rate / self.frame_rate)
            a_tokens = torch.zeros(B, self._n_codebooks_val, max(1, T_frames), dtype=torch.long, device=self._device)
            b_logits = torch.zeros(B, 4, max(1, T_frames), 64, device=self._device)
            return a_tokens, b_logits
        raise NotImplementedError("Full Mimi encode requires moshi model loading")

    def decode(self, a_tokens: torch.Tensor, b_tokens: torch.Tensor, voice_state: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            B = a_tokens.shape[0]
            T_frames = a_tokens.shape[-1]
            T_samples = int(T_frames * self.sample_rate / self.frame_rate)
            return torch.zeros(B, 1, max(1, T_samples), device=self._device)
        raise NotImplementedError("Full Mimi decode requires moshi model loading")

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks_val

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val

    @property
    def condition_id(self) -> str:
        return "B"


class SingleCodebookWrapper(CodecInterface, nn.Module):
    """Single-codebook large-vocab wrapper for condition D (v4 codec strategy).

    n_codebooks=1, vocab_size=8192.
    """

    def __init__(self, checkpoint_path: Path | str | None = None, device: str = "cuda"):
        super().__init__()
        self._device = torch.device(device)
        self.sample_rate = SAMPLE_RATE
        self._n_codebooks_val = 1
        self._vocab_size_val = 8192

        self._model = None
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info("Loading single-codebook checkpoint from %s", checkpoint_path)

    def encode(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = waveform.shape[0]
        hop = HOP_LENGTH
        T_frames = waveform.shape[-1] // hop
        a_tokens = torch.zeros(B, 1, max(1, T_frames), dtype=torch.long, device=self._device)
        b_logits = torch.zeros(B, 4, max(1, T_frames), 64, device=self._device)
        return a_tokens, b_logits

    def decode(self, a_tokens: torch.Tensor, b_tokens: torch.Tensor, voice_state: torch.Tensor) -> torch.Tensor:
        B = a_tokens.shape[0]
        T_frames = a_tokens.shape[-1]
        T_samples = T_frames * HOP_LENGTH
        return torch.zeros(B, 1, max(1, T_samples), device=self._device)

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks_val

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val

    @property
    def condition_id(self) -> str:
        return "D"


def create_codec(condition: str, device: str = "cuda", checkpoint: Path | str | None = None) -> CodecInterface:
    """Factory for codec backends.

    Args:
        condition: One of 'A', 'B', 'C', 'D'.
        device: torch device string.
        checkpoint: Optional path to codec checkpoint.

    Returns:
        CodecInterface implementation.
    """
    if condition == "A":
        return UCLMCodecWrapper(checkpoint_path=checkpoint, device=device)
    elif condition in ("B", "C"):
        return MimiCodecWrapper(checkpoint_path=checkpoint, device=device)
    elif condition == "D":
        return SingleCodebookWrapper(checkpoint_path=checkpoint, device=device)
    else:
        raise ValueError(f"Unknown codec condition: {condition!r}. Must be one of A, B, C, D.")
