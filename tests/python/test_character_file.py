"""Tests for .tmrvc_character file I/O."""

import hashlib
import json
import struct

import numpy as np
import pytest

from tmrvc_core.constants import D_SPEAKER, D_STYLE, LORA_DELTA_SIZE, N_VOICE_SOURCE_PARAMS
from tmrvc_export.character_file import (
    CHECKSUM_SIZE,
    HEADER_SIZE,
    MAGIC,
    VERSION,
    read_character_file,
    write_character_file,
)


@pytest.fixture
def sample_arrays():
    rng = np.random.default_rng(42)
    return {
        "spk_embed": rng.standard_normal(D_SPEAKER).astype(np.float32),
        "lora_delta": rng.standard_normal(LORA_DELTA_SIZE).astype(np.float32),
        "voice_source": rng.standard_normal(N_VOICE_SOURCE_PARAMS).astype(np.float32),
        "style": rng.standard_normal(D_STYLE).astype(np.float32),
    }


class TestWriteRead:
    def test_roundtrip(self, tmp_path, sample_arrays):
        profile = {"name": "Test Character", "personality": "Happy", "language": "ja"}
        path = write_character_file(
            tmp_path / "test.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
            sample_arrays["voice_source"],
            sample_arrays["style"],
            profile,
        )
        assert path.exists()

        spk, lora, vs, sty, prof = read_character_file(path)
        np.testing.assert_array_almost_equal(spk, sample_arrays["spk_embed"])
        np.testing.assert_array_almost_equal(lora, sample_arrays["lora_delta"])
        np.testing.assert_array_almost_equal(vs, sample_arrays["voice_source"])
        np.testing.assert_array_almost_equal(sty, sample_arrays["style"])
        assert prof["name"] == "Test Character"
        assert prof["language"] == "ja"

    def test_defaults_when_none(self, tmp_path, sample_arrays):
        path = write_character_file(
            tmp_path / "minimal.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        spk, lora, vs, sty, prof = read_character_file(path)
        np.testing.assert_array_equal(vs, np.zeros(N_VOICE_SOURCE_PARAMS, dtype=np.float32))
        np.testing.assert_array_equal(sty, np.zeros(D_STYLE, dtype=np.float32))
        assert prof["name"] == ""
        assert prof["language"] == "ja"

    def test_unicode_profile(self, tmp_path, sample_arrays):
        profile = {"name": "桜", "personality": "明るい性格", "language": "ja"}
        path = write_character_file(
            tmp_path / "unicode.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
            profile=profile,
        )
        _, _, _, _, prof = read_character_file(path)
        assert prof["name"] == "桜"
        assert prof["personality"] == "明るい性格"

    def test_magic_and_version(self, tmp_path, sample_arrays):
        path = write_character_file(
            tmp_path / "check.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = path.read_bytes()
        assert data[:4] == MAGIC
        assert struct.unpack("<I", data[4:8])[0] == VERSION


class TestValidation:
    def test_invalid_magic(self, tmp_path, sample_arrays):
        path = write_character_file(
            tmp_path / "bad.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = bytearray(path.read_bytes())
        data[0:4] = b"XXXX"
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Invalid magic"):
            read_character_file(path)

    def test_corrupted_checksum(self, tmp_path, sample_arrays):
        path = write_character_file(
            tmp_path / "corrupt.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF  # flip last checksum byte
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Checksum mismatch"):
            read_character_file(path)

    def test_truncated_file(self, tmp_path):
        path = tmp_path / "truncated.tmrvc_character"
        path.write_bytes(b"TMCH" + b"\x00" * 10)
        with pytest.raises(ValueError, match="Invalid file size"):
            read_character_file(path)

    def test_wrong_version(self, tmp_path, sample_arrays):
        path = write_character_file(
            tmp_path / "v99.tmrvc_character",
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
        )
        data = bytearray(path.read_bytes())
        struct.pack_into("<I", data, 4, 99)
        # Recompute checksum
        payload = bytes(data[:-CHECKSUM_SIZE])
        new_checksum = hashlib.sha256(payload).digest()
        data[-CHECKSUM_SIZE:] = new_checksum
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Unsupported version"):
            read_character_file(path)


class TestFromSpeakerFile:
    def test_conversion(self, tmp_path, sample_arrays):
        from tmrvc_export.speaker_file import write_speaker_file

        spk_path = tmp_path / "test.tmrvc_speaker"
        write_speaker_file(
            spk_path,
            sample_arrays["spk_embed"],
            sample_arrays["lora_delta"],
            metadata={"profile_name": "TestSpk"},
        )

        from tmrvc_export.character_file import from_speaker_file

        char_path = from_speaker_file(
            spk_path,
            tmp_path / "test.tmrvc_character",
            profile={"name": "Converted", "personality": "Calm"},
        )
        spk, lora, vs, sty, prof = read_character_file(char_path)
        np.testing.assert_array_almost_equal(spk, sample_arrays["spk_embed"])
        assert prof["name"] == "Converted"
