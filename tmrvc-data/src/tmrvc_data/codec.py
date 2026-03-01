"""EnCodec wrapper for neural audio codec token extraction and decoding.

This module provides a unified interface for:
1. Encoding audio to discrete tokens
2. Decoding tokens back to audio
3. Token manipulation for UCLM training/inference

Usage:
    from tmrvc_data.codec import EnCodecWrapper

    codec = EnCodecWrapper(device="cuda")

    # Encode
    tokens = codec.encode(waveform)  # [B, n_codebooks, T]

    # Decode
    audio = codec.decode(tokens)  # [B, 1, T_samples]
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# EnCodec constants
CODEC_SAMPLE_RATE = 24000
CODEC_HOP_LENGTH = 320  # ~13.3ms at 24kHz (75 frames/sec)
CODEC_FRAME_RATE = 75  # 75 fps = ~13.3ms per frame
CODEC_N_CODEBOOKS = 8
CODEC_VOCAB_SIZE = 1024
CODEC_BANDWIDTH = 6.0  # kbps


class EnCodecWrapper(nn.Module):
    """Wrapper for Meta's EnCodec model.

    Provides token extraction and decoding at 24kHz, ~75fps (~13.3ms/frame).
    Uses 8 codebooks with 1024 entries each at 6kbps bandwidth.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to run on ("cuda" or "cpu").
        bandwidth: Target bandwidth in kbps (default: 6.0).
    """

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        device: str = "cuda",
        bandwidth: float = CODEC_BANDWIDTH,
    ) -> None:
        super().__init__()
        self.device = device
        self.bandwidth = bandwidth
        self.model_name = model_name
        self.finetuned_decoder_path = None # Set this after init if needed

        # Lazy loading to avoid importing transformers at module load
        self._model = None
        self._initialized = False

        # Constants
        self.sample_rate = CODEC_SAMPLE_RATE
        self.hop_length = CODEC_HOP_LENGTH
        self.frame_rate = CODEC_FRAME_RATE
        self.n_codebooks = CODEC_N_CODEBOOKS
        self.vocab_size = CODEC_VOCAB_SIZE

    def _load_model(self) -> Any:
        """Lazy load the EnCodec model."""
        if self._model is not None:
            return self._model

        try:
            from transformers import EncodecModel
        except ImportError:
            raise ImportError(
                "transformers is required for EnCodec. "
                "Install with: uv add transformers"
            )

        logger.info("Loading EnCodec model: %s", self.model_name)
        model = EncodecModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        if self.finetuned_decoder_path is not None:
            import torch
            logger.info(f"Loading fine-tuned decoder from {self.finetuned_decoder_path}")
            state_dict = torch.load(self.finetuned_decoder_path, map_location=self.device)
            model.decoder.load_state_dict(state_dict)

        self._model = model
        self._initialized = True
        logger.info("EnCodec loaded successfully")

        return model

    @property
    def model(self) -> Any:
        """Get the underlying EnCodec model (lazy loaded)."""
        return self._load_model()

    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Encode audio waveform to discrete tokens.

        Args:
            waveform: Audio tensor of shape [B, T_samples] or [B, 1, T_samples].
                Expected sample rate is 24kHz. If different, will be resampled.
            sample_rate: Sample rate of input (default: 24000).

        Returns:
            tokens: Discrete tokens of shape [B, n_codebooks, T_frames].
            audio_scales: Scales from encoder (needed for decoding).
        """
        model = self.model

        # Ensure 3D input [B, 1, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            waveform = self._resample(waveform, sample_rate, self.sample_rate)

        # Move to device
        waveform = waveform.to(self.device)

        with torch.no_grad():
            # Encode
            encoded = model.encode(
                waveform,
                bandwidth=self.bandwidth,
            )

            # encoded.audio_codes: [B, 1, n_codebooks, T_frames]
            # We want: [B, n_codebooks, T_frames]
            tokens = encoded.audio_codes.squeeze(1)  # [B, n_codebooks, T_frames]
            audio_scales = encoded.audio_scales

        return tokens, audio_scales

    def encode_simple(
        self,
        waveform: torch.Tensor,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Encode audio waveform to discrete tokens (without scales).

        Convenience method when you only need the tokens.

        Args:
            waveform: Audio tensor of shape [B, T_samples] or [B, 1, T_samples].
            sample_rate: Sample rate of input (default: 24000).

        Returns:
            tokens: Discrete tokens of shape [B, n_codebooks, T_frames].
        """
        tokens, _ = self.encode(waveform, sample_rate)
        return tokens

    def decode(
        self,
        tokens: torch.Tensor,
        audio_scales: list | None = None,
    ) -> torch.Tensor:
        """Decode discrete tokens to audio waveform.

        Args:
            tokens: Discrete tokens of shape [B, n_codebooks, T_frames].
            audio_scales: Scales from encoder (optional, uses None if not provided).

        Returns:
            waveform: Audio tensor of shape [B, 1, T_samples].
        """
        model = self.model
        tokens = tokens.to(self.device)

        # Ensure long type for indexing
        tokens = tokens.long()

        with torch.no_grad():
            # Add back the dimension: [B, 1, n_codebooks, T_frames]
            codes = tokens.unsqueeze(1)

            # Decode (audio_scales can be [None])
            if audio_scales is None:
                audio_scales = [None]

            decoded = model.decode(codes, audio_scales)
            waveform = decoded.audio_values  # [B, 1, T_samples]

        return waveform

    def extract_tokens(
        self,
        waveform: torch.Tensor,
        sample_rate: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract tokens with metadata.

        Args:
            waveform: Audio tensor [B, T] or [B, 1, T].
            sample_rate: Input sample rate.

        Returns:
            Dict with:
                - tokens: [B, n_codebooks, T_frames]
                - n_frames: Number of frames
                - duration_sec: Duration in seconds
        """
        tokens = self.encode_simple(waveform, sample_rate)
        n_frames = tokens.shape[-1]
        duration_sec = n_frames / self.frame_rate

        return {
            "tokens": tokens,
            "n_frames": n_frames,
            "duration_sec": duration_sec,
        }

    def _resample(
        self,
        waveform: torch.Tensor,
        from_sr: int,
        to_sr: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate."""
        if from_sr == to_sr:
            return waveform

        # Use torchaudio for resampling
        try:
            import torchaudio.transforms as T
        except ImportError:
            raise ImportError(
                "torchaudio is required for resampling. Install with: uv add torchaudio"
            )

        resampler = T.Resample(
            orig_freq=from_sr,
            new_freq=to_sr,
        ).to(waveform.device)

        return resampler(waveform)

    def frames_to_samples(self, n_frames: int) -> int:
        """Convert number of frames to number of samples."""
        return n_frames * self.hop_length

    def samples_to_frames(self, n_samples: int) -> int:
        """Convert number of samples to number of frames."""
        return n_samples // self.hop_length


class CodecTokenizer:
    """High-level tokenizer for audio-to-tokens and tokens-to-audio.

    Provides a simplified interface for UCLM training data preparation.

    Args:
        device: Device to run on.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.codec = EnCodecWrapper(device=device)

    def tokenize_file(
        self,
        audio_path: str,
    ) -> dict[str, Any]:
        """Load audio file and extract tokens.

        Args:
            audio_path: Path to audio file.

        Returns:
            Dict with tokens, metadata.
        """
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.unsqueeze(0)  # [1, 1, T]

        result = self.codec.extract_tokens(waveform, sample_rate=sr)
        result["sample_rate"] = self.codec.sample_rate
        result["n_codebooks"] = self.codec.n_codebooks

        return result

    def detokenize_to_file(
        self,
        tokens: torch.Tensor,
        output_path: str,
    ) -> None:
        """Decode tokens and save to audio file.

        Args:
            tokens: [1, n_codebooks, T] tokens.
            output_path: Output file path.
        """
        import torchaudio

        waveform = self.codec.decode(tokens)
        torchaudio.save(
            output_path,
            waveform.squeeze(0),
            self.codec.sample_rate,
        )


def get_codec(device: str = "cuda") -> EnCodecWrapper:
    """Get a cached EnCodec wrapper instance.

    Args:
        device: Device to run on.

    Returns:
        EnCodecWrapper instance.
    """
    return EnCodecWrapper(device=device)
