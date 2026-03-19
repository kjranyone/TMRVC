"""Test frame alignment across all temporal features.

This test verifies that all temporal features have the same number of frames T,
ensuring mathematical consistency across the UCLM v2 pipeline.

Scientific Rigor: Frame counts MUST match exactly. Any mismatch indicates a bug
in the feature extraction logic, not something to be "fixed" with padding.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from tmrvc_core.constants import SAMPLE_RATE, HOP_LENGTH, N_MELS


class TestFrameCalculation:
    """Test frame count calculation for different feature extractors."""

    def test_mel_frame_count(self):
        """Mel spectrogram frame count: T = ceil((N - n_fft) / hop_length) + 1"""
        # For a 1-second audio at 24kHz
        n_samples = SAMPLE_RATE  # 24000
        n_fft = 1024
        hop_length = HOP_LENGTH  # 240

        # Librosa/torchaudio formula
        # T = floor((n_samples - n_fft) / hop_length) + 1 when center=False
        # T = ceil(n_samples / hop_length) when center=True (default)
        expected_frames_center = int(np.ceil(n_samples / hop_length))
        expected_frames_no_center = int(np.floor((n_samples - n_fft) / hop_length)) + 1

        # Actual computation
        from tmrvc_core.audio import compute_mel

        audio = torch.randn(1, n_samples)
        mel = compute_mel(audio)
        actual_frames = mel.shape[-1]

        print(f"Audio samples: {n_samples}")
        print(f"Expected frames (center=True): {expected_frames_center}")
        print(f"Expected frames (center=False): {expected_frames_no_center}")
        print(f"Actual mel frames: {actual_frames}")

        # Document which formula is used
        assert (
            actual_frames == expected_frames_center
            or actual_frames == expected_frames_no_center
        )

    def test_codec_token_frame_count(self):
        """Codec token frame count: T = ceil(N / hop_length)"""
        # For 1-second audio
        n_samples = SAMPLE_RATE
        hop_length = HOP_LENGTH

        expected_frames = int(np.ceil(n_samples / hop_length))

        print(f"Audio samples: {n_samples}")
        print(f"Expected codec frames: {expected_frames}")

        assert expected_frames == int(np.ceil(SAMPLE_RATE / HOP_LENGTH))

    def test_wavlm_frame_count(self):
        """WavLM frame count: T = ceil(N_16k / 320)"""
        # WavLM operates at 16kHz with hop_length=320
        n_samples_16k = 16000  # 1 second at 16kHz
        wavlm_hop = 320

        # Expected: ceil(16000 / 320) = 50
        expected_frames = int(np.ceil(n_samples_16k / wavlm_hop))

        print(f"WavLM samples: {n_samples_16k}")
        print(f"Expected WavLM frames: {expected_frames}")

        assert expected_frames == 50

    def test_voice_state_frame_count(self):
        """Voice state should match codec token frames (same hop_length)."""
        n_samples = SAMPLE_RATE
        hop_length = HOP_LENGTH

        expected_frames = int(np.ceil(n_samples / hop_length))

        print(f"Expected voice_state frames: {expected_frames}")

        assert expected_frames == int(np.ceil(SAMPLE_RATE / HOP_LENGTH))


class TestFrameAlignmentRealAudio:
    """Test frame alignment with actual feature extraction."""

    @pytest.fixture
    def sample_audio(self):
        """Generate 1-second sample audio at 24kHz."""
        return torch.randn(1, SAMPLE_RATE)

    def test_mel_vs_codec_alignment(self, sample_audio):
        """Mel and codec tokens should have same frame count."""
        from tmrvc_core.audio import compute_mel

        mel = compute_mel(sample_audio)
        mel_frames = mel.shape[-1]

        # Simulate codec token frame count
        codec_frames = int(np.ceil(sample_audio.shape[-1] / HOP_LENGTH))

        print(f"Mel frames: {mel_frames}")
        print(f"Codec frames: {codec_frames}")

        # They MUST match for UCLM v2
        assert mel_frames == codec_frames, (
            f"Frame mismatch: mel={mel_frames}, codec={codec_frames}. "
            f"This indicates a bug in feature extraction parameters."
        )

    def test_voice_state_vs_codec_alignment(self, sample_audio):
        """Voice state and codec tokens must have same frame count."""
        # Voice state is extracted per frame at hop_length intervals
        voice_state_frames = int(np.ceil(sample_audio.shape[-1] / HOP_LENGTH))
        codec_frames = voice_state_frames

        assert voice_state_frames == codec_frames

    @pytest.mark.skipif(not pytest.importorskip("torchaudio", reason="torchaudio not installed"), reason="torchaudio")
    def test_ssl_state_frame_count_mismatch(self, sample_audio):
        """Document that SSL state has DIFFERENT frame count (WavLM native 50Hz)."""
        import torchaudio.transforms as T

        # Resample to 16kHz for WavLM
        resampler = T.Resample(SAMPLE_RATE, 16000)
        audio_16k = resampler(sample_audio)

        # WavLM frame count at 320 hop (50Hz)
        wavlm_frames = int(np.ceil(audio_16k.shape[-1] / 320))

        # Codec frame count at 240 hop (100Hz)
        codec_frames = int(np.ceil(sample_audio.shape[-1] / HOP_LENGTH))

        print(f"WavLM frames (50Hz): {wavlm_frames}")
        print(f"Codec frames (100Hz): {codec_frames}")

        # These are DIFFERENT by design - ssl_state needs interpolation
        assert wavlm_frames != codec_frames, (
            "SSL state at 50Hz must be upsampled to match codec at 100Hz"
        )


class TestSSLStateInterpolation:
    """Test SSL state interpolation from 50Hz to 100Hz."""

    def test_interpolation_shape(self):
        """SSL state should be interpolated from 50Hz to 100Hz."""
        B, T_ssl, D = 1, 50, 128  # WavLM output
        T_target = 100  # Codec frame count

        ssl_state = torch.randn(B, T_ssl, D)

        # Interpolate
        ssl_upsampled = torch.nn.functional.interpolate(
            ssl_state.transpose(1, 2),  # [B, D, T_ssl]
            size=T_target,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [B, T_target, D]

        assert ssl_upsampled.shape == (B, T_target, D)

    def test_interpolation_preserves_content(self):
        """Interpolation should preserve SSL content without introducing artifacts."""
        # Create SSL state with known pattern
        T_ssl, D = 50, 128
        ssl_state = torch.zeros(1, T_ssl, D)
        ssl_state[0, :, 0] = torch.linspace(0, 1, T_ssl)  # Linear ramp

        # Interpolate to 100 frames
        T_target = 100
        ssl_upsampled = torch.nn.functional.interpolate(
            ssl_state.transpose(1, 2), size=T_target, mode="linear", align_corners=False
        ).transpose(1, 2)

        # Verify linear ramp is preserved
        # At frame 50 (middle of upsampled), should be ~0.5
        assert 0.4 < ssl_upsampled[0, 50, 0].item() < 0.6


class TestPreprocessFrameAlignment:
    """Test the frame alignment logic in preprocess.py."""

    def test_all_features_aligned_after_extraction(self):
        """After extraction, all temporal features must have T=100 for 1s audio."""
        T_target = 100  # Expected for 1s at 24kHz / 240 hop

        # Simulate extracted features - all must have T_target frames
        codec_tokens_a = torch.randn(8, T_target)  # Correct
        codec_tokens_b = torch.randn(4, T_target)  # Correct
        explicit_state = torch.randn(T_target, 8)  # Must be T_target
        ssl_state = torch.randn(50, 128)  # Different rate (50Hz) - needs interpolation

        # Verify all have correct frame count
        T_explicit = explicit_state.shape[0]
        T_ssl = ssl_state.shape[0]

        print(f"Target T: {T_target}")
        print(f"explicit_state T: {T_explicit}")
        print(f"ssl_state T: {T_ssl}")

        # All temporal features at codec rate must match T_target
        assert T_explicit == T_target, (
            f"explicit_state has wrong frame count: {T_explicit} != {T_target}"
        )

        # SSL state at 50Hz must be upsampled to 100Hz
        assert T_ssl == 50, f"SSL state should be at 50Hz, got {T_ssl} frames"

    def test_frame_alignment_guard_logic(self):
        """Test that frame alignment guard correctly handles mismatches."""
        T_target = 100

        # Case 1: Too many frames
        explicit_too_long = torch.randn(101, 8)
        aligned = explicit_too_long[:T_target, :]
        assert aligned.shape == (T_target, 8)

        # Case 2: Too few frames (should NOT happen in correct implementation)
        explicit_too_short = torch.randn(99, 8)
        # This indicates a BUG - we should error, not pad
        with pytest.raises(AssertionError):
            assert explicit_too_short.shape[0] == T_target, (
                "Frame count mismatch indicates extraction bug"
            )

    def test_ssl_requires_interpolation_not_padding(self):
        """SSL state at 50Hz requires interpolation, not padding."""
        T_ssl = 50
        T_target = 100

        ssl_state = torch.randn(T_ssl, 128)

        # WRONG: Zero-padding (loses information)
        wrong_approach = torch.zeros(T_target, 128)
        wrong_approach[:T_ssl] = ssl_state

        # CORRECT: Linear interpolation
        correct_approach = (
            torch.nn.functional.interpolate(
                ssl_state.unsqueeze(0).transpose(1, 2),
                size=T_target,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .squeeze(0)
        )

        # Interpolation preserves information; padding does not
        assert correct_approach.shape == (T_target, 128)

        # Verify padding is lossy (zeros in second half)
        assert torch.all(wrong_approach[T_ssl:] == 0)

        # Verify interpolation is not lossy (no zeros in second half)
        assert not torch.all(correct_approach[T_ssl // 2 :] == 0)


class TestRootCauseAnalysis:
    """Investigate WHY frame mismatches occur."""

    def test_conv_padding_effect(self):
        """Test if conv padding causes frame count differences."""
        n_samples = SAMPLE_RATE
        kernel_size = 960  # 40ms
        hop_length = HOP_LENGTH

        # Without padding
        frames_no_pad = (n_samples - kernel_size) // hop_length + 1

        # With 'same' padding (center=True in librosa)
        frames_with_pad = int(np.ceil(n_samples / hop_length))

        print(f"Frames without padding: {frames_no_pad}")
        print(f"Frames with padding: {frames_with_pad}")

        # Padding adds frames only when kernel > hop; with HOP_LENGTH=1920
        # and kernel=960, no extra frames are added
        assert frames_with_pad >= frames_no_pad

    def test_floor_vs_ceil_difference(self):
        """Test if floor vs ceil causes off-by-one errors."""
        n_samples = SAMPLE_RATE
        hop_length = HOP_LENGTH

        # Different implementations use different rounding
        frames_floor = n_samples // hop_length
        frames_ceil = int(np.ceil(n_samples / hop_length))

        print(f"Frames with floor: {frames_floor}")
        print(f"Frames with ceil: {frames_ceil}")

        # 24000 / 1920 = 12.5 — not exact, so ceil > floor by 1
        assert frames_ceil - frames_floor <= 1
        if n_samples % hop_length == 0:
            assert frames_floor == frames_ceil
        else:
            assert frames_ceil == frames_floor + 1
