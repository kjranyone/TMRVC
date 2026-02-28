"""Shared test fixtures: synthetic audio, mock extractors."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tmrvc_core.constants import (
    D_CONTENT_VEC,
    D_SPEAKER,
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
)
from tmrvc_core.types import FeatureSet


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def synth_waveform() -> torch.Tensor:
    """1 second of 440 Hz sine wave at 24 kHz, shape [1, 24000]."""
    t = torch.linspace(0, 1.0, SAMPLE_RATE)
    sine = 0.5 * torch.sin(2 * torch.pi * 440.0 * t)
    return sine.unsqueeze(0)


@pytest.fixture
def synth_waveform_short() -> torch.Tensor:
    """100 ms of 440 Hz sine wave at 24 kHz, shape [1, 2400]."""
    n_samples = int(0.1 * SAMPLE_RATE)
    t = torch.linspace(0, 0.1, n_samples)
    sine = 0.5 * torch.sin(2 * torch.pi * 440.0 * t)
    return sine.unsqueeze(0)


@pytest.fixture
def expected_frames_1s() -> int:
    """Expected number of mel frames for 1 second of audio."""
    from tmrvc_core.constants import N_FFT, WINDOW_LENGTH
    # center=False with left padding of (window_length - hop_length):
    # padded_len = SAMPLE_RATE + (WINDOW_LENGTH - HOP_LENGTH)
    # torch.stft uses n_fft (not window_length) for frame count:
    # n_frames = (padded_len - N_FFT) // HOP_LENGTH + 1
    padded = SAMPLE_RATE + (WINDOW_LENGTH - HOP_LENGTH)
    return (padded - N_FFT) // HOP_LENGTH + 1


@pytest.fixture
def mock_feature_set() -> FeatureSet:
    """A FeatureSet with random data, 100 frames (1 second)."""
    n_frames = 100
    return FeatureSet(
        mel=torch.randn(N_MELS, n_frames),
        content=torch.randn(D_CONTENT_VEC, n_frames),
        f0=torch.randn(1, n_frames),
        spk_embed=torch.randn(D_SPEAKER),
        utterance_id="test_utt_001",
        speaker_id="test_spk_001",
        n_frames=n_frames,
    )


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for feature cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache
