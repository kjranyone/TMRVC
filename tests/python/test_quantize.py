"""Tests for tmrvc_export.quantize module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tmrvc_export.quantize import QUANTIZE_TARGETS, SKIP_QUANTIZE, quantize_all


class TestQuantizeConstants:
    def test_quantize_targets(self):
        assert "content_encoder" in QUANTIZE_TARGETS
        assert "converter" in QUANTIZE_TARGETS
        assert "vocoder" in QUANTIZE_TARGETS
        assert "ir_estimator" in QUANTIZE_TARGETS

    def test_skip_quantize(self):
        assert "speaker_encoder" in SKIP_QUANTIZE

    def test_no_overlap(self):
        assert not set(QUANTIZE_TARGETS) & set(SKIP_QUANTIZE)


class TestQuantizeAll:
    @patch("tmrvc_export.quantize.quantize_model")
    def test_quantize_all_found_models(self, mock_quantize, tmp_path):
        fp32_dir = tmp_path / "fp32"
        int8_dir = tmp_path / "int8"
        fp32_dir.mkdir()

        # Create dummy ONNX files
        for name in QUANTIZE_TARGETS:
            (fp32_dir / f"{name}.onnx").write_bytes(b"dummy")

        mock_quantize.side_effect = lambda inp, out: Path(out)

        result = quantize_all(fp32_dir, int8_dir)
        assert len(result) == len(QUANTIZE_TARGETS)
        assert mock_quantize.call_count == len(QUANTIZE_TARGETS)

    @patch("tmrvc_export.quantize.quantize_model")
    def test_quantize_all_missing_models(self, mock_quantize, tmp_path):
        fp32_dir = tmp_path / "fp32"
        int8_dir = tmp_path / "int8"
        fp32_dir.mkdir()

        # Only create one model
        (fp32_dir / "converter.onnx").write_bytes(b"dummy")
        mock_quantize.side_effect = lambda inp, out: Path(out)

        result = quantize_all(fp32_dir, int8_dir)
        assert len(result) == 1
        assert "converter" in result

    @patch("tmrvc_export.quantize.quantize_model")
    def test_quantize_all_creates_output_dir(self, mock_quantize, tmp_path):
        fp32_dir = tmp_path / "fp32"
        int8_dir = tmp_path / "int8" / "nested"
        fp32_dir.mkdir()

        result = quantize_all(fp32_dir, int8_dir)
        assert int8_dir.exists()
        assert len(result) == 0  # No models found
