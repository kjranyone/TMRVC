"""Mel spectrogram computation with C++ parity guarantees.

Uses torch.stft with causal (no center) padding and a hand-built HTK mel
filterbank so the exact same maths can be reproduced in C++ with no
library-specific quirks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tmrvc_core.constants import (
    HOP_LENGTH,
    LOG_FLOOR,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)


def _hz_to_mel(freq: float) -> float:
    """Convert Hz to HTK mel scale."""
    return 2595.0 * torch.log10(torch.tensor(1.0 + freq / 700.0)).item()


def _mel_to_hz(mel: float) -> float:
    """Convert HTK mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
    fmin: float = MEL_FMIN,
    fmax: float = MEL_FMAX,
) -> torch.Tensor:
    """Build an HTK mel filterbank from first principles.

    Returns:
        Tensor of shape ``[n_mels, n_fft // 2 + 1]``.
    """
    n_freq = n_fft // 2 + 1
    mel_low = _hz_to_mel(fmin)
    mel_high = _hz_to_mel(fmax)

    # Equally spaced mel points (n_mels + 2 edges)
    mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # FFT bin frequencies
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freq)

    filterbank = torch.zeros(n_mels, n_freq)
    for i in range(n_mels):
        low = hz_points[i]
        center = hz_points[i + 1]
        high = hz_points[i + 2]

        # Rising slope
        up = (fft_freqs - low) / (center - low + 1e-10)
        # Falling slope
        down = (high - fft_freqs) / (high - center + 1e-10)

        filterbank[i] = torch.clamp(torch.minimum(up, down), min=0.0)

    return filterbank


class MelSpectrogram(torch.nn.Module):
    """Causal mel spectrogram extractor with deterministic output.

    Designed for exact parity with the C++ ONNX inference pipeline:
    - ``torch.stft(center=False)`` with manual left-padding
    - Hann window (``periodic=True``)
    - HTK mel filterbank built from first principles
    - ``log(clamp(mel, min=1e-10))``
    """

    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        window_length: int = WINDOW_LENGTH,
        n_mels: int = N_MELS,
        sample_rate: int = SAMPLE_RATE,
        fmin: float = MEL_FMIN,
        fmax: float = MEL_FMAX,
        log_floor: float = LOG_FLOOR,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.n_mels = n_mels
        self.log_floor = log_floor
        # Causal padding: pad left so the first frame sees only past samples
        self.pad_length = window_length - hop_length

        self.register_buffer(
            "window", torch.hann_window(window_length, periodic=True)
        )
        self.register_buffer(
            "mel_basis",
            create_mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        Args:
            waveform: ``[B, 1, T]`` or ``[B, T]`` audio tensor at 24 kHz.

        Returns:
            ``[B, n_mels, T_frames]`` log-mel spectrogram.
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, T]

        # Causal left-padding
        x = F.pad(waveform, (self.pad_length, 0))

        # STFT — center=False because we padded manually
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        # Power spectrogram
        power = stft.abs().pow(2)  # [B, n_freq, T_frames]

        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis.to(power.device), power)

        # Log scale
        return mel.clamp(min=self.log_floor).log()


def compute_mel(
    waveform: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window_length: int = WINDOW_LENGTH,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
    fmin: float = MEL_FMIN,
    fmax: float = MEL_FMAX,
    log_floor: float = LOG_FLOOR,
) -> torch.Tensor:
    """Functional interface to compute log-mel spectrogram.

    Args:
        waveform: ``[B, 1, T]`` or ``[B, T]`` audio at 24 kHz.

    Returns:
        ``[B, n_mels, T_frames]`` log-mel spectrogram.
    """
    mel_fn = MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_length=window_length,
        n_mels=n_mels,
        sample_rate=sample_rate,
        fmin=fmin,
        fmax=fmax,
        log_floor=log_floor,
    )
    mel_fn.eval()
    with torch.no_grad():
        return mel_fn(waveform)


def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window_length: int = WINDOW_LENGTH,
) -> torch.Tensor:
    """Compute causal STFT magnitude.

    Args:
        waveform: ``[B, T]`` audio at 24 kHz.

    Returns:
        ``[B, n_fft//2+1, T_frames]`` STFT magnitude (linear).
    """
    pad_length = window_length - hop_length
    x = F.pad(waveform, (pad_length, 0))
    window = torch.hann_window(window_length, periodic=True, device=waveform.device)
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return stft.abs()
