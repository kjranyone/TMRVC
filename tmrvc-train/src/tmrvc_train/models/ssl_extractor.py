"""Self-Supervised Learning (SSL) feature extractor using WavLM.

Extracts 128-dim latent style representation from audio for voice state conditioning.

Design reference: docs/design/emotion-aware-codec.md
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

HAS_SPEECHBRAIN = False


@dataclass
class SSLConfig:
    d_ssl: int = 128
    wavlm_model: str = "microsoft/wavlm-base-plus"
    wavlm_layer: int = 7
    wavlm_hidden: int = 768
    sample_rate: int = 24000
    frame_size: int = 240


class SSLProjection(nn.Module):
    """Projects WavLM hidden states to 128-dim ssl_state.

    Args:
        wavlm_hidden: WavLM hidden dimension (768 for base, 1024 for large)
        d_ssl: Output dimension (128)
    """

    def __init__(self, wavlm_hidden: int = 768, d_ssl: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(wavlm_hidden, 256),
            nn.GELU(),
            nn.Linear(256, d_ssl),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, wavlm_hidden] WavLM hidden states

        Returns:
            ssl_state: [B, T, d_ssl] projected SSL features
        """
        return self.proj(x)


class WavLMSSLExtractor(nn.Module):
    """Extracts SSL features from audio using WavLM.

    For real-time VC, use extract_frame() for streaming extraction.
    For offline processing, use extract() for batch extraction.
    """

    def __init__(
        self,
        d_ssl: int = 128,
        model_name: str = "microsoft/wavlm-base-plus",
        layer: int = 7,
        freeze_wavlm: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for WavLM. Install with: uv pip install transformers"
            )

        self.model_name = model_name
        self.layer = layer
        self.d_ssl = d_ssl

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.wavlm = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        self.wavlm_hidden = self.wavlm.config.hidden_size

        self.projection = SSLProjection(self.wavlm_hidden, d_ssl)

        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def extract(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """Extract SSL features from audio (batch/offline mode).

        Args:
            audio: [B, T] or [T] audio samples at 24kHz
            sample_rate: Input sample rate (will resample if != 16000)

        Returns:
            ssl_state: [B, T', 128] SSL features at ~50Hz (WavLM native rate)
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if sample_rate != 16000:
            import torchaudio.transforms as T

            resampler = T.Resample(sample_rate, 16000)
            audio = resampler(audio)

        inputs = self.feature_extractor(
            audio.squeeze(0).numpy() if audio.shape[0] == 1 else audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: v.to(next(self.wavlm.parameters()).device) for k, v in inputs.items()
        }

        outputs = self.wavlm(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer]

        ssl_state = self.projection(hidden)

        return ssl_state

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """Forward pass for end-to-end training.

        Args:
            audio: [B, T] audio at sample_rate
            sample_rate: Input sample rate

        Returns:
            ssl_state: [B, T', 128]
        """
        return self.extract(audio, sample_rate)


class StreamingSSLExtractor(nn.Module):
    """Streaming SSL extractor for real-time VC.

    Maintains a buffer of past audio and extracts SSL features
    at a lower rate (e.g., every 50ms) to reduce computation.

    Design:
        - Accumulates 400ms of audio (40 frames at 10ms)
        - Runs WavLM every 200ms (20 frames)
        - Outputs per-frame ssl_state by interpolation
    """

    def __init__(
        self,
        d_ssl: int = 128,
        buffer_frames: int = 40,
        extract_interval: int = 20,
        model_name: str = "microsoft/wavlm-base-plus",
        layer: int = 7,
    ):
        super().__init__()

        self.d_ssl = d_ssl
        self.buffer_frames = buffer_frames
        self.extract_interval = extract_interval
        self.frame_size = 240

        self.extractor = WavLMSSLExtractor(d_ssl, model_name, layer)

        self.register_buffer(
            "audio_buffer", torch.zeros(buffer_frames * self.frame_size)
        )
        self.register_buffer("last_ssl_state", torch.zeros(d_ssl))
        self.register_buffer("prev_ssl_state", torch.zeros(d_ssl))

        self.buffer_pos = 0
        self.frames_since_extract = 0

    @torch.no_grad()
    def process_frame(self, audio_frame: torch.Tensor) -> torch.Tensor:
        """Process one 10ms frame and return ssl_state.

        Args:
            audio_frame: [240] or [1, 240] audio samples

        Returns:
            ssl_state: [128] interpolated SSL features for this frame
        """
        if audio_frame.dim() == 2:
            audio_frame = audio_frame.squeeze(0)

        start = self.buffer_pos * self.frame_size
        end = start + self.frame_size
        self.audio_buffer[start:end] = audio_frame

        self.buffer_pos = (self.buffer_pos + 1) % self.buffer_frames
        self.frames_since_extract += 1

        if self.frames_since_extract >= self.extract_interval:
            ssl_full = self.extractor.extract(
                self.audio_buffer.unsqueeze(0), sample_rate=24000
            )

            self.prev_ssl_state = self.last_ssl_state.clone()
            self.last_ssl_state = ssl_full[0, -1, :]
            self.frames_since_extract = 0

        alpha = self.frames_since_extract / self.extract_interval
        return (1 - alpha) * self.prev_ssl_state + alpha * self.last_ssl_state

    def reset(self):
        """Reset buffer state."""
        self.audio_buffer.zero_()
        self.last_ssl_state.zero_()
        self.prev_ssl_state.zero_()
        self.buffer_pos = 0
        self.frames_since_extract = 0


class MockSSLExtractor(nn.Module):
    """Mock SSL extractor for testing without WavLM.

    Returns zeros or learned embeddings based on voice_state.
    """

    def __init__(self, d_ssl: int = 128, d_voice_state: int = 8):
        super().__init__()
        self.d_ssl = d_ssl
        self.proj = nn.Linear(d_voice_state, d_ssl)

    def forward(self, voice_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voice_state: [B, T, 8] or [B, 8]

        Returns:
            ssl_state: [B, T, 128] or [B, 128]
        """
        if voice_state.dim() == 2:
            return self.proj(voice_state)
        return self.proj(voice_state)

    def process_frame(self, audio_frame: torch.Tensor) -> torch.Tensor:
        """Returns zeros (no audio conditioning)."""
        return torch.zeros(self.d_ssl, device=audio_frame.device)

    def reset(self):
        pass


def create_ssl_extractor(
    d_ssl: int = 128,
    streaming: bool = False,
    mock: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """Factory function to create SSL extractor.

    Args:
        d_ssl: Output dimension (128)
        streaming: Use streaming version for real-time
        mock: Use mock version for testing
        device: Device to load model

    Returns:
        SSL extractor module
    """
    if mock:
        return MockSSLExtractor(d_ssl).to(device)

    if streaming:
        return StreamingSSLExtractor(d_ssl).to(device)

    return WavLMSSLExtractor(d_ssl).to(device)
