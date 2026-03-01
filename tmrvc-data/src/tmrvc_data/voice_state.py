"""Voice State Estimator for frame-level acoustic parameter extraction.

Voice state parameters control acoustic properties of speech generation:
- breathiness: Aspiration noise level
- tension: Vocal fold tension
- arousal: Emotional activation
- valence: Positive/negative emotion
- roughness: Voice quality (creaky/harsh)
- voicing: Voiced vs unvoiced continuum
- energy: Overall loudness
- rate: Speaking rate

Usage:
    from tmrvc_data.voice_state import VoiceStateEstimator

    estimator = VoiceStateEstimator(device="cuda")
    voice_state = estimator.estimate(mel, f0)  # [B, T, 8]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Voice state dimensions
VOICE_STATE_DIM = 8

# Parameter indices
IDX_BREATHINESS = 0
IDX_TENSION = 1
IDX_AROUSAL = 2
IDX_VALENCE = 3
IDX_ROUGHNESS = 4
IDX_VOICING = 5
IDX_ENERGY = 6
IDX_RATE = 7


@dataclass
class VoiceStateConfig:
    """Configuration for voice state estimation."""

    # Breathiness estimation
    breathiness_threshold_low: float = 0.3
    breathiness_threshold_high: float = 0.7

    # Tension estimation
    tension_f0_base: float = 150.0  # Hz, reference F0
    tension_f0_range: float = 200.0  # Hz, range for normalization

    # Roughness estimation
    roughness_jitter_threshold: float = 0.02

    # Energy normalization
    energy_db_min: float = -60.0
    energy_db_max: float = -10.0


class VoiceStateEstimator(nn.Module):
    """Estimate frame-level voice state parameters from audio features.

    Uses heuristics and simple models to estimate acoustic parameters
    from mel spectrogram and F0 contour.

    Args:
        n_mels: Number of mel bands.
        config: Voice state configuration.
        device: Device to run on.
    """

    def __init__(
        self,
        n_mels: int = 80,
        config: VoiceStateConfig | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.config = config or VoiceStateConfig()
        self.device = device

        # Simple learned components for better estimation
        self.breathiness_net = nn.Sequential(
            nn.Conv1d(n_mels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.roughness_net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid(),
        )

        self.arousal_net = nn.Sequential(
            nn.Linear(n_mels + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def estimate(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate voice state from mel and F0.

        Args:
            mel: Mel spectrogram [B, n_mels, T] or [B, T, n_mels].
            f0: F0 contour [B, 1, T] or [B, T].

        Returns:
            voice_state: [B, T, 8] voice state parameters.
        """
        # Ensure correct shapes
        if mel.dim() == 3 and mel.shape[1] != self.n_mels:
            mel = mel.transpose(1, 2)  # [B, T, n_mels] -> [B, n_mels, T]

        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)  # [B, T] -> [B, 1, T]

        B, _, T = mel.shape

        # Move to device
        mel = mel.to(self.device)
        f0 = f0.to(self.device)
        self.to(self.device)

        # 1. Energy from mel (mean across frequency)
        energy = self._estimate_energy(mel)  # [B, T]

        # 2. Voicing from F0
        voicing = self._estimate_voicing(f0)  # [B, T]

        # 3. Breathiness from spectral tilt
        breathiness = self._estimate_breathiness(mel)  # [B, T]

        # 4. Tension from F0 level
        tension = self._estimate_tension(f0)  # [B, T]

        # 5. Roughness from F0 perturbations
        roughness = self._estimate_roughness(f0)  # [B, T]

        # 6. Arousal from energy + F0 dynamics
        arousal = self._estimate_arousal(mel, f0)  # [B, T]

        # 7. Valence (default neutral, can be overridden by external emotion)
        valence = torch.zeros(B, T, device=mel.device)  # [B, T]

        # 8. Speaking rate (default 1.0, should be estimated from duration)
        rate = torch.ones(B, T, device=mel.device)  # [B, T]

        # Stack all parameters
        voice_state = torch.stack(
            [breathiness, tension, arousal, valence, roughness, voicing, energy, rate],
            dim=-1,
        )  # [B, T, 8]

        return voice_state

    def _estimate_energy(self, mel: torch.Tensor) -> torch.Tensor:
        """Estimate energy from mel spectrogram."""
        # Mean energy across mel bands
        energy = mel.mean(dim=1)  # [B, T]

        # Convert to dB-like scale
        energy_db = 20 * torch.log10(energy.abs() + 1e-8)

        # Normalize to [0, 1]
        normalized = (energy_db - self.config.energy_db_min) / (
            self.config.energy_db_max - self.config.energy_db_min
        )

        return torch.clamp(normalized, 0.0, 1.0)

    def _estimate_voicing(self, f0: torch.Tensor) -> torch.Tensor:
        """Estimate voicing from F0."""
        # F0 > 50 Hz indicates voiced
        return (f0.squeeze(1) > 50).float()  # [B, T]

    def _estimate_breathiness(self, mel: torch.Tensor) -> torch.Tensor:
        """Estimate breathiness from spectral features."""
        # Use learned network
        breathiness = self.breathiness_net(mel).squeeze(1)  # [B, T]

        # Additional heuristic: high-frequency energy ratio
        # Breathier voices have more high-frequency energy
        n_mels = mel.shape[1]
        low_band = mel[:, : n_mels // 2, :].mean(dim=1)
        high_band = mel[:, n_mels // 2 :, :].mean(dim=1)

        # High frequency ratio
        hf_ratio = high_band / (low_band + 1e-8)
        hf_ratio = torch.clamp(hf_ratio, 0, 2) / 2  # Normalize

        # Blend network output with heuristic
        breathiness = 0.7 * breathiness + 0.3 * hf_ratio

        return breathiness

    def _estimate_tension(self, f0: torch.Tensor) -> torch.Tensor:
        """Estimate vocal fold tension from F0 level."""
        f0_flat = f0.squeeze(1)  # [B, T]

        # Normalize F0 to [0, 1] based on reference range
        tension = (f0_flat - self.config.tension_f0_base) / self.config.tension_f0_range
        tension = torch.clamp(tension, 0.0, 1.0)

        # Smooth over time
        tension = self._smooth(tension, kernel_size=5)

        return tension

    def _estimate_roughness(self, f0: torch.Tensor) -> torch.Tensor:
        """Estimate roughness from F0 perturbations (jitter)."""
        f0_flat = f0.squeeze(1)  # [B, T]

        # Compute local F0 variation (jitter)
        f0_diff = f0_flat[:, 1:] - f0_flat[:, :-1]
        f0_diff = F.pad(f0_diff, (1, 0), mode="replicate")

        # Relative jitter
        jitter = f0_diff.abs() / (f0_flat + 1e-8)

        # Use learned network
        roughness = self.roughness_net(jitter.unsqueeze(1)).squeeze(1)  # [B, T]

        return roughness

    def _estimate_arousal(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate emotional arousal from energy and F0 dynamics."""
        B, _, T = mel.shape

        # Combine mel and f0 features
        mel_mean = mel.mean(dim=1, keepdim=True).transpose(1, 2)  # [B, T, 1]
        f0_norm = (f0.transpose(1, 2) - 100) / 400  # Normalize F0
        f0_norm = torch.clamp(f0_norm, 0, 1)

        features = torch.cat([mel.transpose(1, 2), f0_norm], dim=-1)  # [B, T, n_mels+1]

        # Use learned network
        arousal = self.arousal_net(features).squeeze(-1)  # [B, T]

        return arousal

    def _smooth(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Apply temporal smoothing."""
        if kernel_size <= 1:
            return x

        # Moving average
        x = x.unsqueeze(1)  # [B, 1, T]
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        smoothed = F.conv1d(x, kernel, padding=kernel_size // 2)

        return smoothed.squeeze(1)


def extract_voice_state(
    mel: torch.Tensor,
    f0: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Convenience function to extract voice state.

    Args:
        mel: Mel spectrogram [B, n_mels, T] or [B, T, n_mels].
        f0: F0 contour [B, 1, T] or [B, T].
        device: Device to run on.

    Returns:
        voice_state: [B, T, 8]
    """
    estimator = VoiceStateEstimator(device=device)
    return estimator.estimate(mel, f0)


def voice_state_to_dict(voice_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert voice state tensor to named dictionary.

    Args:
        voice_state: [B, T, 8] or [T, 8] voice state tensor.

    Returns:
        Dict with named parameters.
    """
    names = [
        "breathiness",
        "tension",
        "arousal",
        "valence",
        "roughness",
        "voicing",
        "energy",
        "rate",
    ]

    return {name: voice_state[..., i] for i, name in enumerate(names)}

from .wavlm_extractor import WavLMFeatureExtractor

class SSLVoiceStateEstimator(nn.Module):
    """Extract both explicit (8-dim) and SSL (128-dim) voice state.
    
    Combines the heuristic VoiceStateEstimator with a WavLM feature extractor.
    """
    def __init__(self, n_mels: int = 80, device: str = "cuda"):
        super().__init__()
        self.explicit_estimator = VoiceStateEstimator(n_mels=n_mels, device=device)
        self.wavlm_extractor = WavLMFeatureExtractor(d_output=128).to(device)
        self.device = device

    def forward(
        self,
        audio_16k: torch.Tensor,
        audio_24k: torch.Tensor,
        mel: torch.Tensor,
        f0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            audio_16k: Audio waveform [B, T_16k] for WavLM.
            audio_24k: Audio waveform [B, T_24k] for frame alignment.
            mel: Mel spectrogram [B, n_mels, T].
            f0: F0 contour [B, 1, T].
            
        Returns:
            dict with:
                explicit_state: [B, T, 8]
                ssl_state: [B, 128, T] -> transposed to [B, T, 128]
        """
        explicit_state = self.explicit_estimator.estimate(mel, f0)
        
        # WavLM extracts [B, 128, T] aligned to mel frames
        ssl_state = self.wavlm_extractor.extract_for_distillation(audio_16k, audio_24k)
        ssl_state = ssl_state.transpose(1, 2)  # [B, T, 128]
        
        return {
            "explicit_state": explicit_state,
            "ssl_state": ssl_state,
        }
