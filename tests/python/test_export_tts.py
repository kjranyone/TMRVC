"""Tests for TTS ONNX export and parity verification."""

import pytest
import torch
from pathlib import Path

from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_export.export_tts import (
    export_text_encoder,
    export_duration_predictor,
    export_f0_predictor,
    export_content_synthesizer,
    export_tts_all,
)


class TestTTSExport:
    def test_export_text_encoder(self, tmp_path):
        model = TextEncoder()
        path = export_text_encoder(model, tmp_path / "text_encoder.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_duration_predictor(self, tmp_path):
        model = DurationPredictor()
        path = export_duration_predictor(model, tmp_path / "duration_predictor.onnx")
        assert path.exists()

    def test_export_f0_predictor(self, tmp_path):
        model = F0Predictor()
        path = export_f0_predictor(model, tmp_path / "f0_predictor.onnx")
        assert path.exists()

    def test_export_content_synthesizer(self, tmp_path):
        model = ContentSynthesizer()
        path = export_content_synthesizer(model, tmp_path / "content_synth.onnx")
        assert path.exists()

    def test_export_tts_all(self, tmp_path):
        paths = export_tts_all(
            TextEncoder(),
            DurationPredictor(),
            F0Predictor(),
            ContentSynthesizer(),
            tmp_path / "tts",
        )
        assert len(paths) == 4
        for name, path in paths.items():
            assert path.exists(), f"{name} not exported"


class TestTTSParity:
    @pytest.fixture
    def onnx_dir(self, tmp_path):
        return tmp_path / "tts_onnx"

    def test_text_encoder_parity(self, onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        model = TextEncoder()
        path = export_text_encoder(model, onnx_dir / "text_encoder.onnx")

        from tmrvc_export.export_tts import verify_text_encoder
        results = verify_text_encoder(model, path)
        for r in results:
            assert r["passed"], f"Failed: {r}"

    def test_duration_predictor_parity(self, onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        model = DurationPredictor()
        path = export_duration_predictor(model, onnx_dir / "duration_predictor.onnx")

        from tmrvc_export.export_tts import verify_duration_predictor
        results = verify_duration_predictor(model, path)
        for r in results:
            assert r["passed"], f"Failed: {r}"

    def test_f0_predictor_parity(self, onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        model = F0Predictor()
        path = export_f0_predictor(model, onnx_dir / "f0_predictor.onnx")

        from tmrvc_export.export_tts import verify_f0_predictor
        results = verify_f0_predictor(model, path)
        for r in results:
            assert r["passed"], f"Failed: {r}"

    def test_content_synthesizer_parity(self, onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        model = ContentSynthesizer()
        path = export_content_synthesizer(model, onnx_dir / "content_synth.onnx")

        from tmrvc_export.export_tts import verify_content_synthesizer
        results = verify_content_synthesizer(model, path)
        for r in results:
            assert r["passed"], f"Failed: {r}"
