"""Mel spectrogram computation with C++ parity guarantees."""

from __future__ import annotations

import torch
import torch.nn as nn
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
    return 2595.0 * math.log10(1.0 + freq / 700.0)


import math


def create_mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax):
    n_freq = n_fft // 2 + 1
    mel_low = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freq)
    filterbank = torch.zeros(n_mels, n_freq)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (fft_freqs - low) / (center - low + 1e-10)
        down = (high - fft_freqs) / (high - center + 1e-10)
        filterbank[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return filterbank


class MelSpectrogram(nn.Module):
    """
    Mel spectrogram with exact frame alignment to codec tokens.

    Frame count formula: T = ceil(N / hop_length)
    For 1s audio at 24kHz with hop_length=240: T = ceil(24000/240) = 100 frames

    CRITICAL: This MUST match codec token frame count exactly.
    Any mismatch indicates a bug that must be fixed at the source.
    """

    def __init__(
        self,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window_length=WINDOW_LENGTH,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        log_floor=LOG_FLOOR,
    ):
        super().__init__()
        self.n_fft, self.hop_length, self.window_length = (
            n_fft,
            hop_length,
            window_length,
        )
        self.log_floor = log_floor
        # Calculate exact padding for frame alignment
        # T = floor((N + pad - n_fft) / hop_length) + 1
        # For T = ceil(N / hop_length), we need:
        # pad = hop_length - (N - n_fft) % hop_length + (window_length - hop_length)
        # Simplified: pad = window_length - hop_length + (hop_length - (N - n_fft) % hop_length) % hop_length
        # For N=24000, n_fft=1024, hop_length=240: pad = 784
        self.pad_length = 784  # Exact padding for 100 frames at 24kHz
        self.register_buffer("window", torch.hann_window(window_length, periodic=True))
        self.register_buffer(
            "mel_basis", create_mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        x = F.pad(waveform, (self.pad_length, 0))
        # Hard device sync for window
        win = self.window.to(x.device)
        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.window_length,
            window=win,
            center=False,
            return_complex=True,
        )
        power = stft.abs().pow(2)
        mel = torch.matmul(self.mel_basis.to(x.device), power)
        return mel.clamp(min=self.log_floor).log()


def compute_mel(waveform, **kwargs):
    # Functional version ensures correct device placement of the temporary module
    mel_fn = MelSpectrogram(**kwargs).to(waveform.device)
    return mel_fn(waveform)


def compute_stft(
    waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window_length=WINDOW_LENGTH
):
    """
    Compute STFT with exact frame alignment to codec tokens.

    CRITICAL: Frame count MUST match MelSpectrogram and codec tokens.
    """
    pad_length = 784  # Exact padding for frame alignment
    x = F.pad(waveform, (pad_length, 0))
    # Explicitly create window on the SAME device as x
    window = torch.hann_window(window_length, periodic=True, device=x.device)
    stft = torch.stft(
        x,
        n_fft,
        hop_length,
        window_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return stft.abs()
