"""Alternative vocoder integration point.

Provides a unified VocoderBase interface so that TMRVC can swap between
different waveform decoders without changing upstream code.

Supported backends:
    - CodecNativeDecoder: wraps EmotionAwareDecoder (default, trains end-to-end)
    - VocosDecoder: Vocos neural vocoder (pretrained, mel-spectrogram based)
    - HiFiGANDecoder: HiFi-GAN vocoder (pretrained, mel-spectrogram based)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from tmrvc_core.constants import D_MODEL

logger = logging.getLogger(__name__)


class VocoderBase(nn.Module, ABC):
    """Abstract base for waveform decoders.

    All vocoders must accept codec tokens (and optional voice state) and
    produce a waveform tensor. The exact input format may vary by backend.
    """

    @abstractmethod
    def forward(
        self,
        codec_tokens: torch.Tensor,
        voice_state: torch.Tensor | None = None,
        control_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode tokens to waveform.

        Args:
            codec_tokens: [B, n_codebooks, T] acoustic codec tokens.
            voice_state: [B, d_vs] optional voice state vector for
                         emotion/style conditioning.
            control_tokens: [B, n_slots, T] optional control stream tokens.

        Returns:
            waveform: [B, 1, T_audio] float32 audio.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate of this vocoder."""
        ...


class CodecNativeDecoder(VocoderBase):
    """Wraps EmotionAwareDecoder as VocoderBase.

    This is the default decoder that uses the trained codec's own decoder
    to convert RVQ tokens back to waveform. It supports the full control
    stream and voice state conditioning.
    """

    def __init__(self, d_model: int = D_MODEL, output_sr: int = 24000):
        super().__init__()
        from .emotion_codec import EmotionAwareDecoder

        self._decoder = EmotionAwareDecoder(d_model=d_model)
        self._sr = output_sr

    def forward(
        self,
        codec_tokens: torch.Tensor,
        voice_state: torch.Tensor | None = None,
        control_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode using the native emotion-aware codec decoder.

        Args:
            codec_tokens: [B, n_codebooks, T] acoustic tokens (a_tokens).
            voice_state: [B, d_vs] voice state vector.
            control_tokens: [B, n_slots, T] control tokens (b_tokens).

        Returns:
            waveform: [B, 1, T_audio].
        """
        if voice_state is None:
            voice_state = torch.zeros(
                codec_tokens.shape[0], 8, device=codec_tokens.device
            )

        if control_tokens is None:
            control_tokens = torch.zeros(
                codec_tokens.shape[0], 4, codec_tokens.shape[-1],
                dtype=torch.long, device=codec_tokens.device,
            )

        audio, _ = self._decoder(codec_tokens, control_tokens, voice_state)
        return audio

    @property
    def sample_rate(self) -> int:
        return self._sr


class VocosDecoder(VocoderBase):
    """Vocos neural vocoder wrapper (loads pretrained).

    Vocos is a fast, high-quality vocoder that operates on mel spectrograms
    or codec features. This wrapper loads a pretrained Vocos model and
    provides the VocoderBase interface.

    Requires: ``pip install vocos``
    """

    def __init__(
        self,
        model_name: str = "charactr/vocos-encodec-24khz",
        output_sr: int = 24000,
    ):
        super().__init__()
        self._model_name = model_name
        self._sr = output_sr
        self._vocos = None

        # Attempt to load at init time; fail gracefully
        self._load_vocos()

    def _load_vocos(self) -> None:
        """Load the Vocos model. Sets self._vocos or logs a warning."""
        try:
            from vocos import Vocos  # type: ignore[import-untyped]

            self._vocos = Vocos.from_pretrained(self._model_name)
            logger.info("Loaded Vocos model: %s", self._model_name)
        except ImportError:
            logger.warning(
                "Vocos is not installed. Install with: pip install vocos\n"
                "The VocosDecoder will raise an error when called."
            )
        except Exception as e:
            logger.warning("Failed to load Vocos model '%s': %s", self._model_name, e)

    def forward(
        self,
        codec_tokens: torch.Tensor,
        voice_state: torch.Tensor | None = None,
        control_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode codec tokens to waveform via Vocos.

        Args:
            codec_tokens: [B, n_codebooks, T] codec tokens.
            voice_state: Ignored by Vocos (no voice state conditioning).
            control_tokens: Ignored by Vocos.

        Returns:
            waveform: [B, 1, T_audio].
        """
        if self._vocos is None:
            raise RuntimeError(
                "Vocos model not loaded. Install vocos: pip install vocos"
            )

        # Vocos expects [B, n_codebooks, T] integer tokens
        audio = self._vocos.decode_from_codes(codec_tokens)

        # Ensure output shape is [B, 1, T_audio]
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)

        return audio

    @property
    def sample_rate(self) -> int:
        return self._sr


class HiFiGANDecoder(VocoderBase):
    """HiFi-GAN vocoder wrapper (loads pretrained).

    HiFi-GAN is a well-established GAN-based vocoder for mel-to-waveform
    synthesis. This wrapper loads a pretrained HiFi-GAN generator.

    This vocoder requires mel spectrogram input. When given codec tokens,
    it first decodes them to a pseudo-mel representation using learned
    embeddings, then runs the HiFi-GAN generator.

    Requires a pretrained HiFi-GAN checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        n_codebooks: int = 8,
        codebook_size: int = 1024,
        n_mels: int = 80,
        d_model: int = D_MODEL,
        output_sr: int = 24000,
    ):
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._sr = output_sr
        self._generator = None
        self.n_mels = n_mels

        # Codec tokens -> pseudo-mel projection
        self.codebook_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model // n_codebooks)
            for _ in range(n_codebooks)
        ])
        self.mel_proj = nn.Sequential(
            nn.Linear(d_model // n_codebooks * n_codebooks, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_mels),
        )

        if checkpoint_path is not None:
            self._load_hifigan(checkpoint_path)

    def _load_hifigan(self, checkpoint_path: str) -> None:
        """Load HiFi-GAN generator from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "generator" in checkpoint:
                # Standard HiFi-GAN checkpoint format
                logger.info("Loaded HiFi-GAN generator from %s", checkpoint_path)
                self._generator = checkpoint["generator"]
            else:
                logger.warning(
                    "HiFi-GAN checkpoint at %s does not contain 'generator' key. "
                    "Available keys: %s", checkpoint_path, list(checkpoint.keys())
                )
        except FileNotFoundError:
            logger.warning(
                "HiFi-GAN checkpoint not found at %s. "
                "The HiFiGANDecoder will use the mel projection fallback.",
                checkpoint_path,
            )
        except Exception as e:
            logger.warning("Failed to load HiFi-GAN checkpoint: %s", e)

    def _tokens_to_mel(self, codec_tokens: torch.Tensor) -> torch.Tensor:
        """Convert codec tokens to pseudo-mel spectrogram.

        Args:
            codec_tokens: [B, n_codebooks, T] integer tokens.

        Returns:
            mel: [B, n_mels, T] pseudo-mel features.
        """
        embeddings = []
        n_cb = min(len(self.codebook_embeds), codec_tokens.shape[1])
        for i in range(n_cb):
            embeddings.append(self.codebook_embeds[i](codec_tokens[:, i, :]))
        x = torch.cat(embeddings, dim=-1)  # [B, T, D]
        mel = self.mel_proj(x)  # [B, T, n_mels]
        return mel.transpose(1, 2)  # [B, n_mels, T]

    def forward(
        self,
        codec_tokens: torch.Tensor,
        voice_state: torch.Tensor | None = None,
        control_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode codec tokens to waveform via HiFi-GAN.

        If no HiFi-GAN generator is loaded, uses the mel projection
        with a simple Griffin-Lim-style approximation (for testing only).

        Args:
            codec_tokens: [B, n_codebooks, T] codec tokens.
            voice_state: Ignored by HiFi-GAN.
            control_tokens: Ignored by HiFi-GAN.

        Returns:
            waveform: [B, 1, T_audio].
        """
        mel = self._tokens_to_mel(codec_tokens)  # [B, n_mels, T]

        if self._generator is not None:
            # Use loaded HiFi-GAN generator
            audio = self._generator(mel)
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)
            return audio

        # Fallback: simple learned upsampling (placeholder for testing)
        # In production, a real HiFi-GAN checkpoint should be loaded
        B, _, T = mel.shape
        # Upsample mel to approximate waveform length (256x for hop_length=256)
        audio = torch.nn.functional.interpolate(
            mel, scale_factor=256, mode="linear", align_corners=False
        )
        # Sum across mel bins and normalize
        audio = audio.mean(dim=1, keepdim=True)  # [B, 1, T_audio]
        audio = torch.tanh(audio)
        return audio

    @property
    def sample_rate(self) -> int:
        return self._sr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

VOCODER_REGISTRY: dict[str, type[VocoderBase]] = {
    "codec_native": CodecNativeDecoder,
    "vocos": VocosDecoder,
    "hifigan": HiFiGANDecoder,
}


def create_vocoder(vocoder_type: str = "codec_native", **kwargs: Any) -> VocoderBase:
    """Create a vocoder instance by type name.

    Args:
        vocoder_type: One of "codec_native", "vocos", "hifigan".
        **kwargs: Forwarded to the vocoder constructor.

    Returns:
        A VocoderBase instance.

    Raises:
        ValueError: If vocoder_type is not registered.
    """
    if vocoder_type not in VOCODER_REGISTRY:
        available = ", ".join(sorted(VOCODER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown vocoder type: {vocoder_type!r}. Available: {available}"
        )
    return VOCODER_REGISTRY[vocoder_type](**kwargs)
