"""Tests for UCLM data pipeline components.

Tests:
- EnCodec wrapper: encode/decode roundtrip
- VoiceStateEstimator: output shape and range
- prepare_dataset.py integration
"""

from __future__ import annotations

import pytest
import torch
import numpy as np


class TestEnCodecWrapper:
    """Tests for EnCodec wrapper."""

    @pytest.fixture
    def codec(self):
        from tmrvc_data.codec import EnCodecWrapper

        return EnCodecWrapper(device="cpu")

    def test_encode_shape(self, codec):
        """Test that encoding produces correct shape."""
        # Generate 1 second of random audio at 24kHz
        audio = torch.randn(1, 24000)

        tokens = codec.encode_simple(audio)

        # Expected: [B=1, n_codebooks=8, T≈75]
        # EnCodec produces ~75 frames per second
        assert tokens.shape[0] == 1
        assert tokens.shape[1] == 8
        assert tokens.shape[2] >= 70 and tokens.shape[2] <= 80

    def test_decode_shape(self, codec):
        """Test that decoding produces correct shape."""
        # Create random tokens
        tokens = torch.randint(0, 1024, (1, 8, 75))

        audio = codec.decode(tokens)

        # Expected: [B=1, 1, T≈24000]
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] >= 23000  # Approximately 1 second

    def test_roundtrip_quality(self, codec):
        """Test that encode-decode preserves audio structure."""
        # Generate simple sine wave
        t = torch.linspace(0, 1, 24000)
        audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

        # Encode and decode
        tokens = codec.encode_simple(audio)
        reconstructed = codec.decode(tokens)

        # Should be roughly the same length
        assert abs(reconstructed.shape[2] - audio.shape[1]) < 200

    def test_extract_tokens(self, codec):
        """Test extract_tokens convenience method."""
        audio = torch.randn(1, 24000)

        result = codec.extract_tokens(audio)

        assert "tokens" in result
        assert "n_frames" in result
        assert "duration_sec" in result
        assert 70 <= result["n_frames"] <= 80
        assert abs(result["duration_sec"] - 1.0) < 0.01


class TestVoiceStateEstimator:
    """Tests for voice state estimator."""

    @pytest.fixture
    def estimator(self):
        from tmrvc_data.voice_state import VoiceStateEstimator

        return VoiceStateEstimator(device="cpu")

    def test_output_shape(self, estimator):
        """Test that output has correct shape [B, T, 8]."""
        mel = torch.randn(2, 80, 100)  # 2 samples, 100 frames
        f0 = torch.randn(2, 1, 100).abs() * 200  # F0 around 0-200 Hz

        voice_state = estimator.estimate(mel, f0)

        assert voice_state.shape == (2, 100, 8)

    def test_output_range(self, estimator):
        """Test that output is in valid range [0, 1]."""
        mel = torch.randn(1, 80, 100)
        f0 = torch.randn(1, 1, 100).abs() * 200

        voice_state = estimator.estimate(mel, f0)

        # All values should be in [0, 1] except valence which is in [-1, 1]
        assert (voice_state[..., [0, 1, 2, 4, 5, 6]] >= 0).all()
        assert (voice_state[..., [0, 1, 2, 4, 5, 6]] <= 1).all()

    def test_voicing_detection(self, estimator):
        """Test that voicing is detected from F0."""
        mel = torch.randn(1, 80, 100)

        # Create F0 with clear voiced/unvoiced regions
        f0 = torch.zeros(1, 1, 100)
        f0[:, :, :50] = 200  # First half voiced
        # Second half unvoiced (0 Hz)

        voice_state = estimator.estimate(mel, f0)
        voicing = voice_state[..., 5]  # IDX_VOICING

        # First half should be voiced
        assert voicing[0, :50].mean() > voicing[0, 50:].mean()

    def test_voice_state_to_dict(self, estimator):
        """Test voice_state_to_dict utility."""
        from tmrvc_data.voice_state import voice_state_to_dict

        voice_state = torch.zeros(2, 100, 8)
        voice_state[..., 0] = 0.5  # breathiness

        result = voice_state_to_dict(voice_state)

        assert "breathiness" in result
        assert "tension" in result
        assert "arousal" in result
        assert result["breathiness"].shape == (2, 100)


class TestPipelineIntegration:
    """Integration tests for prepare_dataset.py."""

    def test_utterance_meta_new_fields(self):
        """Test that UtteranceMeta includes new UCLM fields."""
        from scripts.data.prepare_dataset import UtteranceMeta

        meta = UtteranceMeta(
            utterance_id="test_001",
            speaker_id="test_speaker",
            n_frames=100,
            duration_sec=1.0,
        )

        assert hasattr(meta, "phonemes")
        assert hasattr(meta, "phoneme_ids")
        assert hasattr(meta, "durations")
        assert hasattr(meta, "voice_state_mean")
        assert hasattr(meta, "has_codec_tokens")
        assert meta.has_codec_tokens == False
        assert len(meta.voice_state_mean) == 8
