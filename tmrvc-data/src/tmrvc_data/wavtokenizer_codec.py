"""WavTokenizer wrapper for v4 condition D (single codebook AR).

WavTokenizer (Jisheng Peng et al., 2024): single-codebook neural codec.
- 24kHz, 1 codebook × 4096 bins at 75 Hz
- Single codebook eliminates inter-CB dependency issues
- Suitable for pure AR language model prediction

Usage:
    codec = WavTokenizerWrapper(device="cuda")
    tokens = codec.encode(waveform)     # [B, 1, T_codec]
    audio = codec.decode(tokens)        # [B, 1, T_samples]
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class WavTokenizerWrapper:
    """Wrapper around pre-trained WavTokenizer 24kHz (single codebook)."""

    MODEL_REPO = "novateur/WavTokenizer-medium-speech-75token"
    SAMPLE_RATE = 24000
    FRAME_RATE = 75.0
    N_QUANTIZERS = 1
    CODEBOOK_SIZE = 4096
    HOP_LENGTH = 320  # 24000 / 75

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._model = None
        self._config_path = None
        self._model_path = None

    def _load(self):
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download
        from decoder.pretrained import WavTokenizer

        # Download config and checkpoint
        self._config_path = hf_hub_download(
            self.MODEL_REPO,
            "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        )
        self._model_path = hf_hub_download(
            self.MODEL_REPO,
            "wavtokenizer_medium_speech_320_24k_v2.ckpt",
        )

        self._model = WavTokenizer.from_pretrained0802(
            self._config_path, self._model_path
        )
        self._model.eval()
        self._model.to(self.device)

        logger.info(
            "WavTokenizer loaded: %s, %d codebook × %d, %.0f Hz",
            self.MODEL_REPO, self.N_QUANTIZERS, self.CODEBOOK_SIZE, self.FRAME_RATE,
        )

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to single-codebook tokens.

        Args:
            waveform: [B, 1, T_samples] at 24kHz

        Returns:
            tokens: [B, 1, T_codec] int64
        """
        self._load()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        # WavTokenizer expects [B, T_samples] or [1, T_samples]
        wav = waveform.squeeze(1).to(self.device)  # [B, T]
        bandwidth_id = torch.tensor([0], device=self.device)

        _features, discrete_code = self._model.encode_infer(wav, bandwidth_id=bandwidth_id)
        # discrete_code shape varies by version; ensure [B, 1, T]
        if discrete_code.ndim == 2:
            discrete_code = discrete_code.unsqueeze(1)  # [B, T] -> [B, 1, T]
        elif discrete_code.ndim == 1:
            discrete_code = discrete_code.unsqueeze(0).unsqueeze(0)

        return discrete_code.cpu().long()

    @torch.inference_mode()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode single-codebook tokens to waveform.

        Args:
            tokens: [B, 1, T_codec] int64

        Returns:
            waveform: [B, 1, T_samples]
        """
        self._load()
        tokens = tokens.to(self.device)
        bandwidth_id = torch.tensor([0], device=self.device)

        # WavTokenizer decode expects features from encode, not raw tokens.
        # Use the model's codebook to look up features from tokens.
        if hasattr(self._model, 'codes_to_features'):
            features = self._model.codes_to_features(tokens.squeeze(1))
        else:
            # Fallback: use quantizer codebook lookup
            features = self._model.feature_extractor.encodec.quantizer.decode(tokens)

        audio = self._model.decode(features, bandwidth_id=bandwidth_id)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(1)
        return audio.cpu()

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
