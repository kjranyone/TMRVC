"""Tests for .tmrvc_speaker v2 binary file format."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from tmrvc_core.constants import D_SPEAKER, LORA_DELTA_SIZE
from tmrvc_export.speaker_file import (
    CHECKSUM_SIZE,
    HEADER_SIZE,
    MAGIC,
    VERSION,
    read_speaker_file,
    write_speaker_file,
)


@pytest.fixture
def sample_arrays():
    rng = np.random.default_rng(99)
    return {
        "spk_embed": rng.standard_normal(D_SPEAKER).astype(np.float32),
        "lora_delta": rng.standard_normal(LORA_DELTA_SIZE).astype(np.float32),
    }


class TestWriteRead:
    def test_roundtrip(self, tmp_path, sample_arrays):
        meta = {"profile_name": "TestSpeaker", "author_name": "tester"}
        path = write_speaker_file(
            tmp_path / "test.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
            metadata=meta,
        )
        assert path.exists()

        spk, lora, meta_out, thumb = read_speaker_file(path)
        np.testing.assert_array_almost_equal(spk, sample_arrays["spk_embed"])
        np.testing.assert_array_almost_equal(lora, sample_arrays["lora_delta"])
        assert meta_out["profile_name"] == "TestSpeaker"
        assert meta_out["author_name"] == "tester"
        assert thumb == b""

    def test_default_metadata(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "default.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        _, _, meta, _ = read_speaker_file(path)
        assert meta["profile_name"] == ""
        assert meta["training_mode"] == "embedding"

    def test_thumbnail(self, tmp_path, sample_arrays):
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        path = write_speaker_file(
            tmp_path / "thumb.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
            thumbnail_png=fake_png,
        )
        _, _, meta, thumb = read_speaker_file(path)
        assert thumb == fake_png
        assert meta["thumbnail_b64"] != ""

    def test_unicode_metadata(self, tmp_path, sample_arrays):
        meta = {"profile_name": "桜の声", "description": "日本語テスト"}
        path = write_speaker_file(
            tmp_path / "unicode.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
            metadata=meta,
        )
        _, _, meta_out, _ = read_speaker_file(path)
        assert meta_out["profile_name"] == "桜の声"
        assert meta_out["description"] == "日本語テスト"

    def test_magic_and_version(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "check.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = path.read_bytes()
        assert data[:4] == MAGIC
        assert struct.unpack("<I", data[4:8])[0] == VERSION


class TestValidation:
    def test_invalid_magic(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "bad.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = bytearray(path.read_bytes())
        data[0:4] = b"XXXX"
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Invalid magic"):
            read_speaker_file(path)

    def test_corrupted_checksum(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "corrupt.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Checksum mismatch"):
            read_speaker_file(path)

    def test_truncated_file(self, tmp_path):
        path = tmp_path / "truncated.tmrvc_speaker"
        path.write_bytes(b"TMSP" + b"\x00" * 10)
        with pytest.raises(ValueError, match="Invalid file size"):
            read_speaker_file(path)

    def test_wrong_version(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "v99.tmrvc_speaker",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = bytearray(path.read_bytes())
        struct.pack_into("<I", data, 4, 99)
        # Recompute checksum
        payload = bytes(data[:-CHECKSUM_SIZE])
        data[-CHECKSUM_SIZE:] = hashlib.sha256(payload).digest()
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Unsupported version"):
            read_speaker_file(path)

    def test_wrong_spk_embed_shape(self, sample_arrays):
        bad_spk = np.zeros(100, dtype=np.float32)
        with pytest.raises(AssertionError):
            write_speaker_file(
                "dummy.tmrvc_speaker",
                bad_spk,
                sample_arrays["lora_delta"],
            )
