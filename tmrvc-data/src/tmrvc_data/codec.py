"""UCLM Codec wrapper for token extraction and decoding.

Uses the custom EmotionAwareCodec architecture (Token Spec) with 10ms hop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple, Optional

import torch
import torch.nn as nn

from tmrvc_core.constants import SAMPLE_RATE, HOP_LENGTH, N_CODEBOOKS, RVQ_VOCAB_SIZE
from tmrvc_train.models import EmotionAwareCodec

logger = logging.getLogger(__name__)


class UCLMCodecWrapper(nn.Module):
    """Wrapper for TMRVC UCLM EmotionAwareCodec.

    Provides token extraction (A_t, B_t) and decoding at 24kHz, 100fps (10ms/frame).
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
        self.n_codebooks = N_CODEBOOKS
        self.vocab_size = RVQ_VOCAB_SIZE

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
