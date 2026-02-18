"""Tests for ONNX export and parity verification."""

import numpy as np
import torch
import pytest

from tmrvc_core.constants import (
    CONVERTER_HQ_STATE_FRAMES,
    D_CONTENT,
    D_CONVERTER_HIDDEN,
    D_SPEAKER,
    D_VOCODER_FEATURES,
    LORA_DELTA_SIZE,
    MAX_LOOKAHEAD_HOPS,
    N_IR_PARAMS,
)
from tmrvc_export.speaker_file import HEADER_SIZE
from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.converter import ConverterStudent, ConverterStudentHQ
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.vocoder import VocoderStudent


class TestONNXExport:
    """Test ONNX export of student models."""

    @pytest.fixture
    def tmp_onnx_dir(self, tmp_path):
        d = tmp_path / "onnx"
        d.mkdir()
        return d

    def test_export_content_encoder(self, tmp_onnx_dir):
        from tmrvc_export.export_onnx import export_content_encoder

        model = ContentEncoderStudent()
        path = export_content_encoder(model, tmp_onnx_dir / "content_encoder.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_converter(self, tmp_onnx_dir):
        from tmrvc_export.export_onnx import export_converter

        model = ConverterStudent()
        path = export_converter(model, tmp_onnx_dir / "converter.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_vocoder(self, tmp_onnx_dir):
        from tmrvc_export.export_onnx import export_vocoder

        model = VocoderStudent()
        path = export_vocoder(model, tmp_onnx_dir / "vocoder.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_ir_estimator(self, tmp_onnx_dir):
        from tmrvc_export.export_onnx import export_ir_estimator

        model = IREstimator()
        path = export_ir_estimator(model, tmp_onnx_dir / "ir_estimator.onnx")
        assert path.exists()
        assert path.stat().st_size > 0


class TestONNXParity:
    """Test PyTorch vs ONNX Runtime parity."""

    @pytest.fixture
    def tmp_onnx_dir(self, tmp_path):
        d = tmp_path / "onnx"
        d.mkdir()
        return d

    def test_content_encoder_parity(self, tmp_onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        from tmrvc_export.export_onnx import export_content_encoder
        from tmrvc_export.verify_parity import verify_content_encoder

        model = ContentEncoderStudent()
        path = export_content_encoder(model, tmp_onnx_dir / "content_encoder.onnx")
        results = verify_content_encoder(model, path)
        assert all(r.passed for r in results), [str(r) for r in results if not r.passed]

    def test_converter_parity(self, tmp_onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        from tmrvc_export.export_onnx import export_converter
        from tmrvc_export.verify_parity import verify_converter

        model = ConverterStudent()
        path = export_converter(model, tmp_onnx_dir / "converter.onnx")
        results = verify_converter(model, path)
        assert all(r.passed for r in results), [str(r) for r in results if not r.passed]

    def test_vocoder_parity(self, tmp_onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        from tmrvc_export.export_onnx import export_vocoder
        from tmrvc_export.verify_parity import verify_vocoder

        model = VocoderStudent()
        path = export_vocoder(model, tmp_onnx_dir / "vocoder.onnx")
        results = verify_vocoder(model, path)
        assert all(r.passed for r in results), [str(r) for r in results if not r.passed]

    def test_ir_estimator_parity(self, tmp_onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        from tmrvc_export.export_onnx import export_ir_estimator
        from tmrvc_export.verify_parity import verify_ir_estimator

        model = IREstimator()
        path = export_ir_estimator(model, tmp_onnx_dir / "ir_estimator.onnx")
        results = verify_ir_estimator(model, path)
        assert all(r.passed for r in results), [str(r) for r in results if not r.passed]


class TestONNXExportHQ:
    """Test ONNX export of HQ converter."""

    @pytest.fixture
    def tmp_onnx_dir(self, tmp_path):
        d = tmp_path / "onnx"
        d.mkdir()
        return d

    def test_export_converter_hq(self, tmp_onnx_dir):
        from tmrvc_export.export_onnx import export_converter_hq

        model = ConverterStudentHQ()
        path = export_converter_hq(model, tmp_onnx_dir / "converter_hq.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_converter_hq_onnx_parity(self, tmp_onnx_dir):
        ort = pytest.importorskip("onnxruntime")
        from tmrvc_export.export_onnx import export_converter_hq

        model = ConverterStudentHQ()
        model.eval()
        path = export_converter_hq(model, tmp_onnx_dir / "converter_hq.onnx")

        # PyTorch inference (via the same wrapper used for export)
        from tmrvc_export.export_onnx import _ConverterHQWrapper

        wrapper = _ConverterHQWrapper(model).eval()
        content = torch.randn(1, D_CONTENT, 1 + MAX_LOOKAHEAD_HOPS)
        spk = torch.randn(1, D_SPEAKER)
        lora = torch.zeros(1, LORA_DELTA_SIZE)
        ir = torch.randn(1, N_IR_PARAMS)
        state_in = torch.zeros(1, D_CONVERTER_HIDDEN, CONVERTER_HQ_STATE_FRAMES)

        with torch.no_grad():
            pt_out, pt_state = wrapper(content, spk, lora, ir, state_in)

        # ONNX Runtime inference
        sess = ort.InferenceSession(str(path))
        ort_inputs = {
            "content": content.numpy(),
            "spk_embed": spk.numpy(),
            "lora_delta": lora.numpy(),
            "ir_params": ir.numpy(),
            "state_in": state_in.numpy(),
        }
        ort_out = sess.run(None, ort_inputs)

        np.testing.assert_allclose(
            ort_out[0], pt_out.numpy(), atol=5e-5, rtol=1e-4,
        )
        np.testing.assert_allclose(
            ort_out[1], pt_state.numpy(), atol=5e-5, rtol=1e-4,
        )


class TestSpeakerFile:
    """Test .tmrvc_speaker v2 file I/O."""

    def test_write_and_read(self, tmp_path):
        from tmrvc_export.speaker_file import read_speaker_file, write_speaker_file

        spk_embed = np.random.randn(D_SPEAKER).astype(np.float32)
        lora_delta = np.random.randn(LORA_DELTA_SIZE).astype(np.float32)

        path = write_speaker_file(tmp_path / "test.tmrvc_speaker", spk_embed, lora_delta)
        loaded_spk, loaded_lora, meta, thumb = read_speaker_file(path)

        np.testing.assert_array_equal(loaded_spk, spk_embed)
        np.testing.assert_array_equal(loaded_lora, lora_delta)
        assert meta["training_mode"] == "embedding"
        assert thumb == b""

    def test_corrupted_checksum(self, tmp_path):
        from tmrvc_export.speaker_file import read_speaker_file, write_speaker_file

        spk_embed = np.zeros(D_SPEAKER, dtype=np.float32)
        lora_delta = np.zeros(LORA_DELTA_SIZE, dtype=np.float32)

        path = write_speaker_file(tmp_path / "test.tmrvc_speaker", spk_embed, lora_delta)

        # Corrupt one byte in the embed region
        data = bytearray(path.read_bytes())
        data[HEADER_SIZE + 10] ^= 0xFF
        path.write_bytes(bytes(data))

        with pytest.raises(ValueError, match="Checksum mismatch"):
            read_speaker_file(path)

    def test_invalid_magic(self, tmp_path):
        from tmrvc_export.speaker_file import read_speaker_file

        path = tmp_path / "bad.tmrvc_speaker"
        # Write enough bytes to pass minimum size check
        path.write_bytes(b"\x00" * (HEADER_SIZE + D_SPEAKER * 4 + LORA_DELTA_SIZE * 4 + 32))

        with pytest.raises(ValueError, match="Invalid magic"):
            read_speaker_file(path)

    def test_wrong_size(self, tmp_path):
        from tmrvc_export.speaker_file import read_speaker_file

        path = tmp_path / "short.tmrvc_speaker"
        path.write_bytes(b"TMSP\x02\x00")

        with pytest.raises(ValueError, match="Invalid file size"):
            read_speaker_file(path)

    def test_metadata_roundtrip(self, tmp_path):
        from tmrvc_export.speaker_file import read_speaker_file, write_speaker_file

        spk_embed = np.random.randn(D_SPEAKER).astype(np.float32)
        lora_delta = np.zeros(LORA_DELTA_SIZE, dtype=np.float32)
        metadata = {
            "profile_name": "Test Speaker",
            "author_name": "Test Author",
            "co_author_name": "Co-Author",
            "licence_url": "https://example.com/licence",
            "created_at": "2026-02-18T12:00:00Z",
            "description": "A test voice",
            "source_audio_files": ["ref1.wav", "ref2.wav"],
            "source_sample_count": 480000,
            "training_mode": "finetune",
            "checkpoint_name": "distill.pt",
        }

        path = write_speaker_file(
            tmp_path / "meta.tmrvc_speaker", spk_embed, lora_delta, metadata=metadata,
        )
        _, _, loaded_meta, _ = read_speaker_file(path)

        assert loaded_meta["profile_name"] == "Test Speaker"
        assert loaded_meta["author_name"] == "Test Author"
        assert loaded_meta["co_author_name"] == "Co-Author"
        assert loaded_meta["licence_url"] == "https://example.com/licence"
        assert loaded_meta["created_at"] == "2026-02-18T12:00:00Z"
        assert loaded_meta["source_audio_files"] == ["ref1.wav", "ref2.wav"]
        assert loaded_meta["source_sample_count"] == 480000
        assert loaded_meta["training_mode"] == "finetune"
        assert loaded_meta["checkpoint_name"] == "distill.pt"

    def test_thumbnail_roundtrip(self, tmp_path):
        from tmrvc_export.speaker_file import read_speaker_file, write_speaker_file

        spk_embed = np.random.randn(D_SPEAKER).astype(np.float32)
        lora_delta = np.zeros(LORA_DELTA_SIZE, dtype=np.float32)
        # Fake PNG data for testing
        thumbnail = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        path = write_speaker_file(
            tmp_path / "thumb.tmrvc_speaker", spk_embed, lora_delta,
            metadata={"profile_name": "Thumb Test"},
            thumbnail_png=thumbnail,
        )
        _, _, meta, loaded_thumb = read_speaker_file(path)

        assert loaded_thumb == thumbnail
        assert meta["profile_name"] == "Thumb Test"
        # Verify thumbnail_b64 is in metadata
        import base64
        assert meta["thumbnail_b64"] == base64.b64encode(thumbnail).decode("ascii")
