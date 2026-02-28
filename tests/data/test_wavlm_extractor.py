"""Tests for tmrvc_data.wavlm_extractor module (mocked, no model download)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from tmrvc_data.wavlm_extractor import (
    WAVLM_LARGE_DIM,
    get_content_teacher,
)


class TestGetContentTeacher:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown teacher_type"):
            get_content_teacher("nonexistent")


def _make_mock_wavlm(B: int, T_feat: int, d: int = 1024, n_layers: int = 26):
    """Create a mock WavLM model that returns correct shapes."""
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    hidden_states = [torch.randn(B, T_feat, d) for _ in range(n_layers)]
    mock_outputs.hidden_states = hidden_states
    mock_model.return_value = mock_outputs
    mock_model.training = False
    mock_model.parameters.return_value = iter([])
    mock_model.eval.return_value = mock_model
    return mock_model


class TestWavLMFeatureExtractor:
    def test_forward_shape(self):
        B, T_audio = 2, 16000
        T_feat = T_audio // 320

        mock_WavLMModel = MagicMock()
        mock_WavLMModel.from_pretrained.return_value = _make_mock_wavlm(B, T_feat)

        with patch.dict(sys.modules, {"transformers": MagicMock(WavLMModel=mock_WavLMModel)}):
            # Force re-import to pick up mock
            import importlib
            import tmrvc_data.wavlm_extractor as wmod
            importlib.reload(wmod)

            extractor = wmod.WavLMFeatureExtractor(layer=7, d_output=1024, freeze=True)
            features = extractor.forward(torch.randn(B, T_audio))
            assert features.shape == (B, 1024, T_feat)

    def test_forward_with_projection(self):
        B, T_audio = 1, 16000
        T_feat = T_audio // 320

        mock_WavLMModel = MagicMock()
        mock_WavLMModel.from_pretrained.return_value = _make_mock_wavlm(B, T_feat)

        with patch.dict(sys.modules, {"transformers": MagicMock(WavLMModel=mock_WavLMModel)}):
            import importlib
            import tmrvc_data.wavlm_extractor as wmod
            importlib.reload(wmod)

            extractor = wmod.WavLMFeatureExtractor(layer=7, d_output=256, freeze=True)
            assert extractor.projection is not None
            features = extractor.forward(torch.randn(B, T_audio))
            assert features.shape == (B, 256, T_feat)

    def test_extract_for_distillation(self):
        B = 1
        T_16k = 16000
        T_24k = 24000
        T_feat = T_16k // 320
        T_mel = T_24k // 240

        mock_WavLMModel = MagicMock()
        mock_WavLMModel.from_pretrained.return_value = _make_mock_wavlm(B, T_feat)

        with patch.dict(sys.modules, {"transformers": MagicMock(WavLMModel=mock_WavLMModel)}):
            import importlib
            import tmrvc_data.wavlm_extractor as wmod
            importlib.reload(wmod)

            extractor = wmod.WavLMFeatureExtractor(layer=7)
            features = extractor.extract_for_distillation(
                torch.randn(B, T_16k), torch.randn(B, T_24k),
            )
            assert features.shape == (B, 1024, T_mel)

    def test_sample_rate_and_hop(self):
        mock_WavLMModel = MagicMock()
        mock_WavLMModel.from_pretrained.return_value = _make_mock_wavlm(1, 50)

        with patch.dict(sys.modules, {"transformers": MagicMock(WavLMModel=mock_WavLMModel)}):
            import importlib
            import tmrvc_data.wavlm_extractor as wmod
            importlib.reload(wmod)

            extractor = wmod.WavLMFeatureExtractor()
            assert extractor.sample_rate == 16000
            assert extractor.hop_length == 320


class TestContentVecFeatureExtractor:
    def test_forward_shape(self):
        B, T_audio = 1, 16000
        T_feat = T_audio // 320

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(B, T_feat, 768)
        mock_model.return_value = mock_outputs
        mock_model.training = False
        mock_model.parameters.return_value = iter([])
        mock_model.eval.return_value = mock_model

        mock_HubertModel = MagicMock()
        mock_HubertModel.from_pretrained.return_value = mock_model

        with patch.dict(sys.modules, {"transformers": MagicMock(HubertModel=mock_HubertModel)}):
            import importlib
            import tmrvc_data.wavlm_extractor as wmod
            importlib.reload(wmod)

            extractor = wmod.ContentVecFeatureExtractor(d_output=768, freeze=True)
            features = extractor.forward(torch.randn(B, T_audio))
            assert features.shape == (B, 768, T_feat)
