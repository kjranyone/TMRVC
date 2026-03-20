"""Tests for .tmrvc_speaker v4 binary file format."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from tmrvc_core.constants import D_ACTING_LATENT, D_SPEAKER, D_VOICE_STATE_SSL, LORA_DELTA_SIZE
from tmrvc_export.speaker_file import (
    CHECKSUM_SIZE,
    HEADER_SIZE,
    MAGIC,
    VERSION,
    D_STYLE,
    read_speaker_file,
    write_speaker_file,
)


@pytest.fixture
def sample_arrays():
    rng = np.random.default_rng(99)
    return {
        "spk_embed": rng.standard_normal(D_SPEAKER).astype(np.float32),
        "style_embed": rng.standard_normal(D_STYLE).astype(np.float32),
        "lora_delta": rng.standard_normal(LORA_DELTA_SIZE).astype(np.float32),
        "acting_latent": rng.standard_normal(D_ACTING_LATENT).astype(np.float32),
        "ssl_state": rng.standard_normal(D_VOICE_STATE_SSL).astype(np.float32),
    }


class TestWriteRead:
    def test_roundtrip(self, tmp_path, sample_arrays):
        meta = {"profile_name": "TestSpeaker", "author_name": "tester"}
        path = write_speaker_file(
            tmp_path / "test.tmrvc_speaker",
            sample_arrays["spk_embed"],
            style_embed=sample_arrays["style_embed"],
            lora_delta=sample_arrays["lora_delta"],
            metadata=meta,
        )
        assert path.exists()

        result = read_speaker_file(path)
        np.testing.assert_array_almost_equal(
            result.spk_embed, sample_arrays["spk_embed"]
        )
        np.testing.assert_array_almost_equal(
            result.style_embed, sample_arrays["style_embed"]
        )
        np.testing.assert_array_almost_equal(
            result.lora_delta, sample_arrays["lora_delta"]
        )
        assert result.metadata["profile_name"] == "TestSpeaker"
        assert result.metadata["author_name"] == "tester"

    def test_light_level(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "light.tmrvc_speaker",
            sample_arrays["spk_embed"],
        )
        result = read_speaker_file(path)
        assert result.adaptation_level == "light"
        assert result.style_embed is None
        assert result.lora_delta is None

    def test_standard_level(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "standard.tmrvc_speaker",
            sample_arrays["spk_embed"],
            style_embed=sample_arrays["style_embed"],
        )
        result = read_speaker_file(path)
        assert result.adaptation_level == "standard"

    def test_full_level(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "full.tmrvc_speaker",
            sample_arrays["spk_embed"],
            lora_delta=sample_arrays["lora_delta"],
        )
        result = read_speaker_file(path)
        assert result.adaptation_level == "full"

    def test_unicode_metadata(self, tmp_path, sample_arrays):
        meta = {"profile_name": "桜の声", "description": "日本語テスト"}
        path = write_speaker_file(
            tmp_path / "unicode.tmrvc_speaker",
            sample_arrays["spk_embed"],
            metadata=meta,
        )
        result = read_speaker_file(path)
        assert result.metadata["profile_name"] == "桜の声"
        assert result.metadata["description"] == "日本語テスト"

    def test_magic_and_version(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "check.tmrvc_speaker",
            sample_arrays["spk_embed"],
        )
        data = path.read_bytes()
        assert data[:4] == MAGIC
        assert struct.unpack("<I", data[4:8])[0] == VERSION
        assert VERSION == 4

    def test_header_size_is_40(self, tmp_path, sample_arrays):
        assert HEADER_SIZE == 40
        path = write_speaker_file(
            tmp_path / "hdr.tmrvc_speaker",
            sample_arrays["spk_embed"],
        )
        data = path.read_bytes()
        # acting_latent_size at bytes 32..36 should be 0 (no acting_latent)
        assert struct.unpack("<I", data[32:36])[0] == 0
        # reserved/pad at bytes 36..40 should be 0
        assert struct.unpack("<I", data[36:40])[0] == 0

    def test_roundtrip_with_acting_latent(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "acting.tmrvc_speaker",
            sample_arrays["spk_embed"],
            acting_latent=sample_arrays["acting_latent"],
        )
        result = read_speaker_file(path)
        assert result.acting_latent is not None
        np.testing.assert_array_almost_equal(
            result.acting_latent, sample_arrays["acting_latent"]
        )

    def test_roundtrip_with_acting_latent_and_ssl_state(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "full_v4.tmrvc_speaker",
            sample_arrays["spk_embed"],
            style_embed=sample_arrays["style_embed"],
            lora_delta=sample_arrays["lora_delta"],
            ssl_state=sample_arrays["ssl_state"],
            acting_latent=sample_arrays["acting_latent"],
            metadata={"profile_name": "FullV4"},
        )
        result = read_speaker_file(path)
        np.testing.assert_array_almost_equal(
            result.spk_embed, sample_arrays["spk_embed"]
        )
        np.testing.assert_array_almost_equal(
            result.style_embed, sample_arrays["style_embed"]
        )
        np.testing.assert_array_almost_equal(
            result.lora_delta, sample_arrays["lora_delta"]
        )
        np.testing.assert_array_almost_equal(
            result.acting_latent, sample_arrays["acting_latent"]
        )
        assert result.ssl_state is not None
        np.testing.assert_array_almost_equal(
            result.ssl_state, sample_arrays["ssl_state"]
        )
        assert result.metadata["profile_name"] == "FullV4"

    def test_acting_latent_flag_in_header(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "flag.tmrvc_speaker",
            sample_arrays["spk_embed"],
            acting_latent=sample_arrays["acting_latent"],
        )
        data = path.read_bytes()
        flags = struct.unpack("<I", data[8:12])[0]
        assert flags & (1 << 3)  # FLAG_HAS_ACTING_LATENT
        acting_latent_size = struct.unpack("<I", data[32:36])[0]
        assert acting_latent_size == D_ACTING_LATENT


class TestValidation:
    def test_invalid_magic(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "bad.tmrvc_speaker",
            sample_arrays["spk_embed"],
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
        )
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Checksum mismatch"):
            read_speaker_file(path)

    def test_truncated_file(self, tmp_path):
        path = tmp_path / "truncated.tmrvc_speaker"
        path.write_bytes(b"TMSP" + b"\x00" * 10)
        with pytest.raises(ValueError):
            read_speaker_file(path)

    def test_wrong_version(self, tmp_path, sample_arrays):
        path = write_speaker_file(
            tmp_path / "v99.tmrvc_speaker",
            sample_arrays["spk_embed"],
        )
        data = bytearray(path.read_bytes())
        struct.pack_into("<I", data, 4, 99)
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
            )
