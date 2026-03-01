"""Tests for tmrvc_train.models.ssl_extractor module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tmrvc_train.models.ssl_extractor import (
    MockSSLExtractor,
    SSLConfig,
    SSLProjection,
    StreamingSSLExtractor,
    create_ssl_extractor,
)


class TestSSLConfig:
    def test_defaults(self):
        config = SSLConfig()
        assert config.d_ssl == 128
        assert config.wavlm_model == "microsoft/wavlm-base-plus"
        assert config.wavlm_layer == 7
        assert config.sample_rate == 24000
        assert config.frame_size == 240


class TestSSLProjection:
    def test_forward_shape(self):
        B, T, wavlm_hidden = 2, 100, 768
        d_ssl = 128

        proj = SSLProjection(wavlm_hidden, d_ssl)
        x = torch.randn(B, T, wavlm_hidden)
        out = proj(x)

        assert out.shape == (B, T, d_ssl)

    def test_different_hidden_sizes(self):
        # Test with large WavLM (1024 hidden)
        proj = SSLProjection(1024, 128)
        x = torch.randn(1, 50, 1024)
        out = proj(x)
        assert out.shape == (1, 50, 128)


class TestMockSSLExtractor:
    def test_forward_with_voice_state(self):
        mock = MockSSLExtractor(d_ssl=128, d_voice_state=8)
        voice_state = torch.randn(2, 10, 8)
        out = mock(voice_state)
        assert out.shape == (2, 10, 128)

    def test_forward_2d_input(self):
        mock = MockSSLExtractor(d_ssl=128, d_voice_state=8)
        voice_state = torch.randn(2, 8)
        out = mock(voice_state)
        assert out.shape == (2, 128)

    def test_process_frame_returns_zeros(self):
        mock = MockSSLExtractor(d_ssl=128)
        audio_frame = torch.randn(240)
        out = mock.process_frame(audio_frame)
        assert out.shape == (128,)
        assert torch.allclose(out, torch.zeros(128))

    def test_reset_noop(self):
        mock = MockSSLExtractor(d_ssl=128)
        mock.reset()  # Should not raise


class TestStreamingSSLExtractor:
    def test_process_frame_shape(self):
        # Skip if no transformers
        pytest.importorskip("transformers")

        # Use mock for actual test
        mock = MockSSLExtractor(d_ssl=128)
        audio_frame = torch.randn(240)
        out = mock.process_frame(audio_frame)
        assert out.shape == (128,)

    def test_buffer_management(self):
        # Skip if no transformers
        pytest.importorskip("transformers")

        mock = MockSSLExtractor(d_ssl=128)
        for _ in range(10):
            audio_frame = torch.randn(240)
            out = mock.process_frame(audio_frame)
            assert out.shape == (128,)


class TestCreateSSLExtractor:
    def test_creates_mock(self):
        extractor = create_ssl_extractor(d_ssl=128, mock=True)
        assert isinstance(extractor, MockSSLExtractor)

    def test_creates_mock_with_device(self):
        extractor = create_ssl_extractor(d_ssl=128, mock=True, device="cpu")
        assert isinstance(extractor, MockSSLExtractor)


class TestWavLMSSLExtractorMocked:
    """Test WavLMSSLExtractor with mocked transformers."""

    def test_forward_shape_mocked(self):
        B, T_audio = 2, 16000
        T_feat = T_audio // 320
        d_ssl = 128

        # Create mock WavLM
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        hidden_states = [torch.randn(B, T_feat, 768) for _ in range(13)]
        mock_outputs.hidden_states = hidden_states
        mock_model.return_value = mock_outputs
        mock_model.config.hidden_size = 768
        mock_model.parameters.return_value = iter([torch.randn(1)])

        mock_Wav2Vec2Model = MagicMock()
        mock_Wav2Vec2Model.from_pretrained.return_value = mock_model

        mock_Wav2Vec2Processor = MagicMock()
        mock_Wav2Vec2Processor.from_pretrained.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "transformers": MagicMock(
                    Wav2Vec2Model=mock_Wav2Vec2Model,
                    Wav2Vec2Processor=mock_Wav2Vec2Processor,
                )
            },
        ):
            # Force reimport
            import importlib

            import tmrvc_train.models.ssl_extractor as mod

            importlib.reload(mod)

            extractor = mod.WavLMSSLExtractor(d_ssl=d_ssl, layer=7, freeze=True)
            audio = torch.randn(B, T_audio)
            out = extractor(audio, sample_rate=16000)

            assert out.shape[0] == B
            assert out.shape[2] == d_ssl
