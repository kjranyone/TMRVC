"""Voice State Estimator for frame-level acoustic parameter extraction.

Canonical UCLM v3 voice_state dimensions:
- pitch_level: Fundamental frequency (F0) level
- pitch_range: Local melodic variation / F0 spread
- energy_level: Overall loudness
- pressedness: Phonation compression proxy
- spectral_tilt: Spectral slope / brightness
- breathiness: Aspiration noise level
- voice_irregularity: Jitter + shimmer style perturbation proxy
- openness: Vocal-tract openness proxy

Usage:
    from tmrvc_data.voice_state import VoiceStateEstimator

    estimator = VoiceStateEstimator(device="cuda")
    voice_state = estimator.estimate(mel, f0)  # [B, T, 8]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS

logger = logging.getLogger(__name__)

# Voice state dimensions
VOICE_STATE_DIM = 8

# Parameter indices (canonical 8-D physical registry)
IDX_PITCH_LEVEL = 0
IDX_PITCH_RANGE = 1
IDX_ENERGY_LEVEL = 2
IDX_PRESSEDNESS = 3
IDX_SPECTRAL_TILT = 4
IDX_BREATHINESS = 5
IDX_VOICE_IRREGULARITY = 6
IDX_OPENNESS = 7


@dataclass
class VoiceStateConfig:
    """Configuration for voice state estimation."""

    # Breathiness estimation
    breathiness_threshold_low: float = 0.3
    breathiness_threshold_high: float = 0.7

    # Pitch-level normalization
    tension_f0_base: float = 150.0  # Hz, reference F0
    tension_f0_range: float = 200.0  # Hz, range for normalization

    # Voice-irregularity estimation
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

        self.voice_irregularity_net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
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

        # Canonical physical factors
        pitch_level = self._estimate_pitch_level(f0)  # [B, T]
        pitch_range = self._estimate_pitch_range(f0)  # [B, T]
        energy_level = self._estimate_energy(mel)  # [B, T]
        spectral_tilt = self._estimate_spectral_tilt(mel)  # [B, T]
        breathiness = self._estimate_breathiness(mel)  # [B, T]
        voice_irregularity = self._estimate_voice_irregularity(f0)  # [B, T]
        openness = self._estimate_openness(mel)  # [B, T]

        # Pressedness is a coarse proxy built from source intensity and inverse breathiness.
        pressedness = self._estimate_pressedness(
            pitch_level, energy_level, spectral_tilt, breathiness
        )

        voice_state = torch.stack(
            [
                pitch_level,
                pitch_range,
                energy_level,
                pressedness,
                spectral_tilt,
                breathiness,
                voice_irregularity,
                openness,
            ],
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

    def _estimate_pitch_level(self, f0: torch.Tensor) -> torch.Tensor:
        """Estimate normalised F0 level."""
        f0_flat = f0.squeeze(1)
        voiced = f0_flat > 50.0
        log_f0 = torch.zeros_like(f0_flat)
        log_f0[voiced] = torch.log2(torch.clamp(f0_flat[voiced], min=50.0) / 50.0)
        return torch.clamp(log_f0 / 4.0, 0.0, 1.0)

    def _estimate_pitch_range(self, f0: torch.Tensor) -> torch.Tensor:
        """Estimate local melodic variation from short-term F0 dynamics."""
        f0_flat = f0.squeeze(1)
        voiced = f0_flat > 50.0
        f0_safe = torch.where(voiced, f0_flat, torch.zeros_like(f0_flat))
        mean = self._smooth(f0_safe, kernel_size=9)
        sq_mean = self._smooth(f0_safe * f0_safe, kernel_size=9)
        variance = torch.clamp(sq_mean - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        norm = std / (mean + 1e-6)
        norm = torch.where(voiced, norm, torch.zeros_like(norm))
        return torch.clamp(norm / 0.35, 0.0, 1.0)

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

    def _estimate_spectral_tilt(self, mel: torch.Tensor) -> torch.Tensor:
        """Estimate spectral tilt from low/high band energy balance."""
        n_mels = mel.shape[1]
        low_band = mel[:, : max(1, n_mels // 3), :].mean(dim=1)
        high_band = mel[:, max(1, 2 * n_mels // 3) :, :].mean(dim=1)
        ratio = high_band / (low_band + 1e-8)
        return torch.clamp(ratio / 2.0, 0.0, 1.0)

    def _estimate_voice_irregularity(self, f0: torch.Tensor) -> torch.Tensor:
        """Estimate voice irregularity from F0 perturbations (jitter proxy)."""
        f0_flat = f0.squeeze(1)  # [B, T]

        # Compute local F0 variation (jitter)
        f0_diff = f0_flat[:, 1:] - f0_flat[:, :-1]
        f0_diff = F.pad(f0_diff, (1, 0), mode="replicate")

        # Relative jitter
        jitter = f0_diff.abs() / (f0_flat + 1e-8)

        # Use learned network
        irregularity = self.voice_irregularity_net(jitter.unsqueeze(1)).squeeze(1)  # [B, T]

        return irregularity

    def _estimate_pressedness(
        self,
        pitch_level: torch.Tensor,
        energy_level: torch.Tensor,
        spectral_tilt: torch.Tensor,
        breathiness: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate pressedness from source intensity and inverse breathiness."""
        pressedness = (
            0.35 * pitch_level
            + 0.30 * energy_level
            + 0.20 * spectral_tilt
            + 0.15 * (1.0 - breathiness)
        )
        return torch.clamp(pressedness, 0.0, 1.0)

    def _estimate_openness(self, mel: torch.Tensor) -> torch.Tensor:
        """Estimate articulation openness from lower-mid spectral emphasis."""
        n_mels = mel.shape[1]
        low_mid_start = max(0, n_mels // 6)
        low_mid_end = max(low_mid_start + 1, n_mels // 2)
        low_mid = mel[:, low_mid_start:low_mid_end, :].mean(dim=1)
        full = mel.mean(dim=1)
        openness = low_mid / (full + 1e-8)
        return torch.clamp(openness / 1.5, 0.0, 1.0)

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
    names = list(CANONICAL_VOICE_STATE_IDS)

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
