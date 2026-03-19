"""Mimi codec wrapper for v4.

Mimi (Kyutai, 2024) is the frozen pre-trained audio codec for v4.
- 24kHz, 8 RVQ quantizers x 2048 bins, 12.5 Hz frame rate
- Encoder and decoder weights are frozen (not fine-tuned)
- Encoder runs at cache generation time
- Decoder runs at inference time

Usage:
    codec = MimiCodecWrapper(device="cuda")
    tokens = codec.encode(waveform)     # [B, 8, T_codec]
    audio = codec.decode(tokens)        # [B, 1, T_samples]
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MimiCodecWrapper:
    """Wrapper around the pre-trained Mimi codec.

    All weights are frozen. This class handles:
    - Lazy model loading from HuggingFace
    - Encode: waveform -> codec tokens
    - Decode: codec tokens -> waveform
    - Device management
    """

    MODEL_ID = "kyutai/mimi"
    SAMPLE_RATE = 24000
    FRAME_RATE = 12.5  # Hz
    N_QUANTIZERS = 8
    CODEBOOK_SIZE = 2048
    HOP_LENGTH = 1920  # = 24000 / 12.5

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import MimiModel

        logger.info("Loading Mimi codec from %s...", self.MODEL_ID)
        self._model = MimiModel.from_pretrained(self.MODEL_ID)
        self._model.eval()
        self._model.to(self.device)

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info("Mimi loaded: %.1fM params, %d quantizers x %d, %.1f Hz",
                     n_params / 1e6, self.N_QUANTIZERS, self.CODEBOOK_SIZE, self.FRAME_RATE)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to codec tokens.

        Args:
            waveform: [B, 1, T_samples] at 24kHz

        Returns:
            tokens: [B, 8, T_codec] where T_codec ~ T_samples / 1920
        """
        self._load()
        waveform = waveform.to(self.device)

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)  # [B, T] -> [B, 1, T]

        encoded = self._model.encode(waveform)
        codes = encoded.audio_codes  # [B, n_q, T_codec]

        # Mimi may return up to 32 quantizers; we use only the first 8
        if codes.size(1) > self.N_QUANTIZERS:
            codes = codes[:, :self.N_QUANTIZERS, :]

        return codes.cpu()

    @torch.inference_mode()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode codec tokens to waveform.

        Args:
            tokens: [B, 8, T_codec]

        Returns:
            waveform: [B, 1, T_samples]
        """
        self._load()
        tokens = tokens.to(self.device)

        # Mimi expects all 32 quantizers; pad with zeros if we only have 8
        if tokens.size(1) < 32:
            padding = torch.zeros(
                tokens.size(0), 32 - tokens.size(1), tokens.size(2),
                dtype=tokens.dtype, device=tokens.device,
            )
            tokens_full = torch.cat([tokens, padding], dim=1)
        else:
            tokens_full = tokens

        decoded = self._model.decode(tokens_full)
        audio = decoded.audio_values  # [B, 1, T_samples]

        return audio.cpu()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        if self._model is not None:
            self._model.to(self.device)
        return self

    def unload(self):
        """Free GPU memory."""
        del self._model
        self._model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
