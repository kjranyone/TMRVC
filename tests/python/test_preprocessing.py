"""Tests for audio preprocessing: resample, loudness, trim, segment."""

import tempfile
from pathlib import Path

import soundfile as sf
import torch

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_data.preprocessing import (
    load_and_resample,
    normalize_loudness,
    segment_utterance,
    trim_silence,
)


class TestLoadAndResample:
    def test_resample_to_24k(self, tmp_path: Path):
        # Create a 48 kHz wav file
        sr_in = 48000
        t = torch.linspace(0, 1.0, sr_in)
        audio = (0.5 * torch.sin(2 * torch.pi * 440 * t)).numpy()
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr_in)

        waveform, sr = load_and_resample(wav_path)
        assert sr == SAMPLE_RATE
        assert waveform.shape[0] == 1  # mono
        # Duration should be approximately 1 second
        assert abs(waveform.shape[1] / sr - 1.0) < 0.01

    def test_stereo_to_mono(self, tmp_path: Path):
        sr_in = 24000
        audio = torch.randn(2, sr_in).numpy().T  # [T, 2] for soundfile
        wav_path = tmp_path / "stereo.wav"
        sf.write(str(wav_path), audio, sr_in)

        waveform, sr = load_and_resample(wav_path)
        assert waveform.shape[0] == 1


class TestNormalizeLoudness:
    def test_returns_same_shape(self, synth_waveform):
        normed = normalize_loudness(synth_waveform)
        assert normed.shape == synth_waveform.shape

    def test_silence_passthrough(self):
        silence = torch.zeros(1, SAMPLE_RATE)
        normed = normalize_loudness(silence)
        assert normed.shape == silence.shape


class TestTrimSilence:
    def test_trims_leading_trailing(self):
        # 0.5s silence + 0.5s tone + 0.5s silence
        silence = torch.zeros(1, SAMPLE_RATE // 2)
        t = torch.linspace(0, 0.5, SAMPLE_RATE // 2)
        tone = (0.5 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
        waveform = torch.cat([silence, tone, silence], dim=-1)

        trimmed = trim_silence(waveform)
        # Should be shorter than the original
        assert trimmed.shape[-1] < waveform.shape[-1]
        # Should still have content
        assert trimmed.shape[-1] > 0

    def test_all_silence_passthrough(self):
        silence = torch.zeros(1, SAMPLE_RATE)
        trimmed = trim_silence(silence)
        # Should return original if entirely silent
        assert trimmed.shape == silence.shape


class TestSegmentation:
    def test_short_audio_single_segment(self):
        # 3 seconds < min_sec (5s) → no segments
        waveform = torch.randn(1, SAMPLE_RATE * 3)
        segments = list(segment_utterance(waveform))
        assert len(segments) == 0

    def test_medium_audio_single_segment(self):
        # 10 seconds → within [5, 15] → one segment
        waveform = torch.randn(1, SAMPLE_RATE * 10)
        segments = list(segment_utterance(waveform))
        assert len(segments) == 1
        assert segments[0].shape == waveform.shape

    def test_long_audio_multiple_segments(self):
        # 35 seconds → 2 segments of 15s + 1 of 5s
        waveform = torch.randn(1, SAMPLE_RATE * 35)
        segments = list(segment_utterance(waveform))
        assert len(segments) >= 2
        for seg in segments:
            dur = seg.shape[-1] / SAMPLE_RATE
            assert 5.0 <= dur <= 15.0 + 0.01
