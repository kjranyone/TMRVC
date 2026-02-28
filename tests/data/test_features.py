"""Tests for feature extractors (ContentVec, F0).

These tests use mock extractors to avoid downloading large models in CI.
For full integration testing, run with --run-slow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from tmrvc_core.constants import D_CONTENT_VEC, HOP_LENGTH, SAMPLE_RATE


class TestContentVecExtractor:
    def test_extract_shape_mock(self, synth_waveform):
        """Test ContentVec extraction with a mocked model."""
        from tmrvc_data.features import ContentVecExtractor

        extractor = ContentVecExtractor(device="cpu")

        n_samples = synth_waveform.shape[-1]
        # ContentVec at 16 kHz, hop=320 → ~T_cv frames
        n_samples_16k = int(n_samples * 16000 / SAMPLE_RATE)
        t_cv = n_samples_16k // 320

        # Mock the HuBERT model
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, t_cv, D_CONTENT_VEC)

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        extractor._model = mock_model

        result = extractor.extract(synth_waveform, SAMPLE_RATE)

        assert result.shape[0] == D_CONTENT_VEC
        # After 2x interpolation, should have ~2*t_cv frames
        assert result.shape[1] == t_cv * 2


class TestF0Extractor:
    def test_torchcrepe_extract_shape_mock(self, synth_waveform):
        """Test torchcrepe F0 extraction with a mock."""
        import sys

        from tmrvc_data.features import TorchCrepeF0Extractor

        extractor = TorchCrepeF0Extractor(device="cpu")

        expected_frames = synth_waveform.shape[-1] // HOP_LENGTH

        # Mock the torchcrepe module (local import in extract method)
        mock_crepe = MagicMock()
        mock_crepe.predict.return_value = torch.full(
            (1, expected_frames), 440.0
        )

        with patch.dict(sys.modules, {"torchcrepe": mock_crepe}):
            result = extractor.extract(synth_waveform, SAMPLE_RATE)

        assert result.shape[0] == 1
        assert result.shape[1] == expected_frames
        # log(440 + 1) ≈ 6.09
        assert result.max().item() > 0

    def test_factory(self):
        from tmrvc_data.features import TorchCrepeF0Extractor, create_f0_extractor

        ext = create_f0_extractor("torchcrepe")
        assert isinstance(ext, TorchCrepeF0Extractor)

    def test_factory_unknown(self):
        from tmrvc_data.features import create_f0_extractor

        with pytest.raises(ValueError, match="Unknown F0 method"):
            create_f0_extractor("nonexistent")
