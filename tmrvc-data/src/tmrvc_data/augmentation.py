"""Online augmentation for training: RIR convolution, EQ, noise, F0 perturbation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from tmrvc_core.constants import SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Probabilities and ranges for each augmentation."""

    rir_prob: float = 0.5
    rir_dirs: list[Path] = field(default_factory=list)

    eq_prob: float = 0.3
    eq_gain_db_range: tuple[float, float] = (-6.0, 6.0)

    noise_prob: float = 0.3
    noise_snr_range: tuple[float, float] = (15.0, 40.0)

    f0_perturbation_prob: float = 0.3
    f0_shift_semitones_range: tuple[float, float] = (-2.0, 2.0)

    content_dropout_prob: float = 0.1
    content_dropout_rate: float = 0.1  # fraction of frames to zero


class Augmenter:
    """Apply online augmentations to audio or features."""

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        self._rir_cache: list[torch.Tensor] = []

    def _load_rir_files(self) -> None:
        """Lazily load RIR files from configured directories."""
        if self._rir_cache:
            return
        for rir_dir in self.config.rir_dirs:
            rir_path = Path(rir_dir)
            if not rir_path.exists():
                logger.warning("RIR directory %s not found", rir_dir)
                continue
            for wav_path in rir_path.rglob("*.wav"):
                try:
                    rir, sr = torchaudio.load(str(wav_path))
                    if sr != SAMPLE_RATE:
                        rir = torchaudio.functional.resample(rir, sr, SAMPLE_RATE)
                    if rir.shape[0] > 1:
                        rir = rir.mean(dim=0, keepdim=True)
                    # Normalise RIR
                    rir = rir / rir.abs().max().clamp(min=1e-8)
                    self._rir_cache.append(rir)
                except Exception:
                    logger.warning("Failed to load RIR: %s", wav_path, exc_info=True)
        logger.info("Loaded %d RIR files", len(self._rir_cache))

    def apply_rir(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convolve waveform with a random RIR.

        Args:
            waveform: ``[1, T]`` audio.

        Returns:
            Convolved ``[1, T]`` audio (same length as input).
        """
        self._load_rir_files()
        if not self._rir_cache:
            return waveform

        rir = random.choice(self._rir_cache).to(waveform.device)
        # FFT-based convolution
        convolved = torchaudio.functional.fftconvolve(waveform, rir)
        # Trim to original length
        convolved = convolved[..., : waveform.shape[-1]]
        # Normalise to prevent clipping
        peak = convolved.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            convolved = convolved / peak
        return convolved

    def apply_eq(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random low/high shelf EQ using proper shelf filters.

        Args:
            waveform: ``[1, T]`` audio at 24 kHz.

        Returns:
            EQ'd ``[1, T]`` audio.
        """
        low_gain = random.uniform(*self.config.eq_gain_db_range)
        high_gain = random.uniform(*self.config.eq_gain_db_range)

        # Low shelf: bass_biquad accepts gain in dB directly, Q=0.707 (Butterworth)
        waveform = torchaudio.functional.bass_biquad(
            waveform, SAMPLE_RATE, gain=low_gain, central_freq=300.0, Q=0.707,
        )
        # High shelf: treble_biquad accepts gain in dB directly
        waveform = torchaudio.functional.treble_biquad(
            waveform, SAMPLE_RATE, gain=high_gain, central_freq=4000.0, Q=0.707,
        )

        return waveform

    def apply_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add random white/brown noise at a random SNR.

        Args:
            waveform: ``[1, T]`` audio.

        Returns:
            Noised ``[1, T]`` audio.
        """
        snr_db = random.uniform(*self.config.noise_snr_range)
        noise = torch.randn_like(waveform)

        # Brown noise: cumulative sum of white noise
        if random.random() < 0.5:
            noise = noise.cumsum(dim=-1)
            noise = noise / noise.abs().max().clamp(min=1e-8)

        # Scale noise to target SNR
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = (signal_power / (noise_power * snr_linear + 1e-10)).sqrt()
        return waveform + noise * scale

    def apply_f0_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Shift pitch by a random number of semitones.

        Uses torchaudio's pitch_shift for simplicity.

        Args:
            waveform: ``[1, T]`` audio at 24 kHz.

        Returns:
            Pitch-shifted ``[1, T]`` audio.
        """
        semitones = random.uniform(*self.config.f0_shift_semitones_range)
        n_steps = int(round(semitones))
        if n_steps == 0:
            return waveform
        return torchaudio.functional.pitch_shift(
            waveform, SAMPLE_RATE, n_steps=n_steps
        )

    def apply_content_dropout(self, content: torch.Tensor) -> torch.Tensor:
        """Zero out random frames of content features.

        Args:
            content: ``[D, T]`` content features.

        Returns:
            Content with random frames zeroed.
        """
        n_frames = content.shape[-1]
        n_drop = int(n_frames * self.config.content_dropout_rate)
        if n_drop == 0:
            return content
        indices = random.sample(range(n_frames), min(n_drop, n_frames))
        content = content.clone()
        content[:, indices] = 0.0
        return content

    def augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply all audio-domain augmentations stochastically.

        Args:
            waveform: ``[1, T]`` audio at 24 kHz.

        Returns:
            Augmented ``[1, T]`` audio.
        """
        if random.random() < self.config.rir_prob and self.config.rir_dirs:
            waveform = self.apply_rir(waveform)
        if random.random() < self.config.eq_prob:
            waveform = self.apply_eq(waveform)
        if random.random() < self.config.noise_prob:
            waveform = self.apply_noise(waveform)
        if random.random() < self.config.f0_perturbation_prob:
            waveform = self.apply_f0_perturbation(waveform)
        return waveform
