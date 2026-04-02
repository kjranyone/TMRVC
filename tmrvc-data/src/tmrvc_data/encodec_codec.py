"""EnCodec 24kHz wrapper for v4 condition A baseline.

EnCodec (Meta, 2022): proven codec for AR-LM TTS.
- 24kHz, 14.9M params, pre-trained
- 8 codebooks × 1024 bins at 6kbps (75 Hz)
- CB0 has high temporal autocorrelation (0.48) — good for AR prediction

Usage:
    codec = EnCodecWrapper(device="cuda")
    tokens = codec.encode(waveform)     # [B, 8, T_codec]
    audio = codec.decode(tokens)        # [B, 1, T_samples]
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class EnCodecWrapper:
    """Wrapper around pre-trained EnCodec 24kHz."""

    MODEL_ID = "encodec_24khz"
    SAMPLE_RATE = 24000
    FRAME_RATE = 75.0
    N_QUANTIZERS = 8
    CODEBOOK_SIZE = 1024
    HOP_LENGTH = 320  # 24000 / 75

    def __init__(self, device: str = "cpu", bandwidth: float = 6.0):
        self.device = torch.device(device)
        self.bandwidth = bandwidth
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        from encodec import EncodecModel

        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(self.bandwidth)
        self._model.eval()
        self._model.to(self.device)

        for param in self._model.parameters():
            param.requires_grad = False

        # Remove weight_norm hooks to prevent per-forward memory accumulation
        self._remove_weight_norms(self._model)

        logger.info(
            "EnCodec loaded: %.1fM params, %d codebooks × %d, %.0f Hz, %.1f kbps",
            sum(p.numel() for p in self._model.parameters()) / 1e6,
            self.N_QUANTIZERS, self.CODEBOOK_SIZE,
            self.FRAME_RATE, self.bandwidth,
        )

    @staticmethod
    def _remove_weight_norms(model):
        """Remove all weight_norm hooks from model (safe for inference)."""
        from torch.nn.utils import remove_weight_norm
        removed = 0
        for module in model.modules():
            try:
                remove_weight_norm(module)
                removed += 1
            except ValueError:
                pass
        if removed:
            logger.info("Removed weight_norm from %d modules", removed)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to codec tokens.

        Args:
            waveform: [B, 1, T_samples] at 24kHz

        Returns:
            tokens: [B, 8, T_codec] int64
        """
        self._load()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        waveform = waveform.to(self.device)

        encoded = self._model.encode(waveform)
        # encoded is list of (codes, scale) tuples; codes: [B, n_q, T]
        codes = encoded[0][0]
        return codes.cpu()

    @torch.inference_mode()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode codec tokens to waveform.

        Args:
            tokens: [B, 8, T_codec] int64

        Returns:
            waveform: [B, 1, T_samples]
        """
        self._load()
        tokens = tokens.to(self.device)

        # EnCodec decode expects list of (codes, None) tuples
        decoded = self._model.decode([(tokens, None)])
        return decoded.cpu()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        del self._model
        self._model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
