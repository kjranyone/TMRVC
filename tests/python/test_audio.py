"""Tests for mel spectrogram computation."""

import torch

from tmrvc_core.audio import MelSpectrogram, compute_mel, compute_stft, create_mel_filterbank
from tmrvc_core.constants import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, WINDOW_LENGTH


class TestMelFilterbank:
    def test_shape(self):
        fb = create_mel_filterbank()
        assert fb.shape == (N_MELS, N_FFT // 2 + 1)

    def test_non_negative(self):
        fb = create_mel_filterbank()
        assert (fb >= 0).all()

    def test_each_bin_has_nonzero(self):
        fb = create_mel_filterbank()
        # Each mel bin should have at least one nonzero weight
        assert (fb.sum(dim=1) > 0).all()


class TestMelSpectrogram:
    def test_output_shape(self, synth_waveform, expected_frames_1s):
        mel_fn = MelSpectrogram()
        mel = mel_fn(synth_waveform)
        assert mel.dim() == 3
        assert mel.shape[0] == 1  # batch
        assert mel.shape[1] == N_MELS
        assert mel.shape[2] == expected_frames_1s

    def test_output_shape_batched(self, synth_waveform):
        mel_fn = MelSpectrogram()
        batch = synth_waveform.repeat(4, 1)  # [4, T]
        mel = mel_fn(batch)
        assert mel.shape[0] == 4
        assert mel.shape[1] == N_MELS

    def test_3d_input(self, synth_waveform):
        mel_fn = MelSpectrogram()
        wav_3d = synth_waveform.unsqueeze(0)  # [1, 1, T]
        mel = mel_fn(wav_3d)
        assert mel.shape[1] == N_MELS

    def test_deterministic(self, synth_waveform):
        mel_fn = MelSpectrogram()
        mel_fn.eval()
        with torch.no_grad():
            mel1 = mel_fn(synth_waveform)
            mel2 = mel_fn(synth_waveform)
        assert torch.equal(mel1, mel2)

    def test_finite_values(self, synth_waveform):
        mel = compute_mel(synth_waveform)
        assert torch.isfinite(mel).all()

    def test_silence_input(self):
        silence = torch.zeros(1, SAMPLE_RATE)
        mel = compute_mel(silence)
        assert torch.isfinite(mel).all()
        # Log of floor value
        import math

        assert mel.max().item() <= math.log(1e-10) + 1.0


class TestComputeStft:
    def test_output_shape(self, synth_waveform, expected_frames_1s):
        stft = compute_stft(synth_waveform)  # [1, T] → [1, n_freq, T_frames]
        assert stft.shape[0] == 1  # batch
        assert stft.shape[1] == N_FFT // 2 + 1
        assert stft.shape[2] == expected_frames_1s

    def test_non_negative(self, synth_waveform):
        stft = compute_stft(synth_waveform)
        assert (stft >= 0).all()
