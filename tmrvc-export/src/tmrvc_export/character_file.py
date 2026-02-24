""".tmrvc_character binary file format: create and load character profiles.

Extends .tmrvc_speaker v2 with voice_source_preset, default_style, and character profile.

Binary layout (v1):
    Magic: 4 bytes "TMCH"
    Version: uint32_le = 1
    spk_embed_size: uint32_le (192)
    lora_delta_size: uint32_le (15872)
    voice_source_size: uint32_le (8)
    style_size: uint32_le (32)
    profile_size: uint32_le (JSON byte count)
    spk_embed: float32[192]
    lora_delta: float32[15872]
    voice_source_preset: float32[8]
    default_style: float32[32]
    profile_json: UTF-8 JSON (name, personality, voice_description, language)
    checksum: SHA-256 = 32 bytes
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path

import numpy as np

from tmrvc_core.constants import (
    D_SPEAKER,
    D_STYLE,
    LORA_DELTA_SIZE,
    N_VOICE_SOURCE_PARAMS,
)

logger = logging.getLogger(__name__)

MAGIC = b"TMCH"
VERSION = 1

HEADER_SIZE = 28  # magic(4) + ver(4) + spk(4) + lora(4) + vs(4) + style(4) + prof(4)
SPK_EMBED_BYTES = D_SPEAKER * 4
LORA_DELTA_BYTES = LORA_DELTA_SIZE * 4
VOICE_SOURCE_BYTES = N_VOICE_SOURCE_PARAMS * 4
STYLE_BYTES = D_STYLE * 4
CHECKSUM_SIZE = 32


def write_character_file(
    output_path: str | Path,
    spk_embed: np.ndarray,
    lora_delta: np.ndarray,
    voice_source_preset: np.ndarray | None = None,
    default_style: np.ndarray | None = None,
    profile: dict | None = None,
) -> Path:
    """Write a .tmrvc_character binary file.

    Args:
        output_path: Output file path.
        spk_embed: Speaker embedding, shape (192,), float32.
        lora_delta: LoRA delta, shape (15872,), float32.
        voice_source_preset: Voice source params, shape (8,), float32. Zeros if None.
        default_style: Default emotion style, shape (32,), float32. Zeros if None.
        profile: Character profile dict with keys: name, personality,
            voice_description, language. Missing keys use empty defaults.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    assert spk_embed.shape == (D_SPEAKER,), f"Expected ({D_SPEAKER},), got {spk_embed.shape}"
    assert lora_delta.shape == (LORA_DELTA_SIZE,), f"Expected ({LORA_DELTA_SIZE},), got {lora_delta.shape}"
    assert spk_embed.dtype == np.float32
    assert lora_delta.dtype == np.float32

    if voice_source_preset is None:
        voice_source_preset = np.zeros(N_VOICE_SOURCE_PARAMS, dtype=np.float32)
    assert voice_source_preset.shape == (N_VOICE_SOURCE_PARAMS,)

    if default_style is None:
        default_style = np.zeros(D_STYLE, dtype=np.float32)
    assert default_style.shape == (D_STYLE,)

    profile = profile or {}
    profile_dict = {
        "name": profile.get("name", ""),
        "personality": profile.get("personality", ""),
        "voice_description": profile.get("voice_description", ""),
        "language": profile.get("language", "ja"),
    }
    profile_bytes = json.dumps(profile_dict, ensure_ascii=False).encode("utf-8")

    data = bytearray()
    data += MAGIC
    data += struct.pack("<I", VERSION)
    data += struct.pack("<I", D_SPEAKER)
    data += struct.pack("<I", LORA_DELTA_SIZE)
    data += struct.pack("<I", N_VOICE_SOURCE_PARAMS)
    data += struct.pack("<I", D_STYLE)
    data += struct.pack("<I", len(profile_bytes))
    data += spk_embed.tobytes()
    data += lora_delta.tobytes()
    data += voice_source_preset.astype(np.float32).tobytes()
    data += default_style.astype(np.float32).tobytes()
    data += profile_bytes

    checksum = hashlib.sha256(bytes(data)).digest()
    data += checksum

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(data))
    logger.info("Wrote character file to %s (%d bytes)", output_path, len(data))
    return output_path


def read_character_file(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read and validate a .tmrvc_character binary file.

    Args:
        path: Path to the character file.

    Returns:
        Tuple of (spk_embed, lora_delta, voice_source_preset, default_style, profile).
        - spk_embed: shape (192,) float32
        - lora_delta: shape (15872,) float32
        - voice_source_preset: shape (8,) float32
        - default_style: shape (32,) float32
        - profile: dict with name, personality, voice_description, language

    Raises:
        ValueError: If the file is invalid or corrupted.
    """
    path = Path(path)
    data = path.read_bytes()

    min_size = HEADER_SIZE + SPK_EMBED_BYTES + LORA_DELTA_BYTES + VOICE_SOURCE_BYTES + STYLE_BYTES + CHECKSUM_SIZE
    if len(data) < min_size:
        raise ValueError(f"Invalid file size: expected at least {min_size}, got {len(data)}")

    magic = data[:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: expected {MAGIC!r}, got {magic!r}")

    version = struct.unpack("<I", data[4:8])[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    spk_size = struct.unpack("<I", data[8:12])[0]
    lora_size = struct.unpack("<I", data[12:16])[0]
    vs_size = struct.unpack("<I", data[16:20])[0]
    style_size = struct.unpack("<I", data[20:24])[0]
    profile_size = struct.unpack("<I", data[24:28])[0]

    if spk_size != D_SPEAKER:
        raise ValueError(f"Invalid spk_embed_size: expected {D_SPEAKER}, got {spk_size}")
    if lora_size != LORA_DELTA_SIZE:
        raise ValueError(f"Invalid lora_delta_size: expected {LORA_DELTA_SIZE}, got {lora_size}")

    expected_size = (
        HEADER_SIZE + spk_size * 4 + lora_size * 4
        + vs_size * 4 + style_size * 4 + profile_size + CHECKSUM_SIZE
    )
    if len(data) != expected_size:
        raise ValueError(f"File size mismatch: expected {expected_size}, got {len(data)}")

    payload = data[:-CHECKSUM_SIZE]
    stored_checksum = data[-CHECKSUM_SIZE:]
    computed_checksum = hashlib.sha256(payload).digest()
    if stored_checksum != computed_checksum:
        raise ValueError("Checksum mismatch: file is corrupted")

    offset = HEADER_SIZE
    spk_embed = np.frombuffer(data[offset:offset + SPK_EMBED_BYTES], dtype=np.float32).copy()
    offset += SPK_EMBED_BYTES

    lora_delta = np.frombuffer(data[offset:offset + LORA_DELTA_BYTES], dtype=np.float32).copy()
    offset += LORA_DELTA_BYTES

    voice_source_preset = np.frombuffer(data[offset:offset + VOICE_SOURCE_BYTES], dtype=np.float32).copy()
    offset += VOICE_SOURCE_BYTES

    default_style_arr = np.frombuffer(data[offset:offset + STYLE_BYTES], dtype=np.float32).copy()
    offset += STYLE_BYTES

    if profile_size > 0:
        profile = json.loads(data[offset:offset + profile_size])
    else:
        profile = {"name": "", "personality": "", "voice_description": "", "language": "ja"}

    return spk_embed, lora_delta, voice_source_preset, default_style_arr, profile


def from_speaker_file(
    speaker_path: str | Path,
    output_path: str | Path,
    profile: dict | None = None,
    voice_source_preset: np.ndarray | None = None,
    default_style: np.ndarray | None = None,
) -> Path:
    """Convert a .tmrvc_speaker file to .tmrvc_character.

    Args:
        speaker_path: Path to existing .tmrvc_speaker file.
        output_path: Path for the new .tmrvc_character file.
        profile: Character profile dict.
        voice_source_preset: Voice source params, or None for zeros.
        default_style: Default emotion style, or None for zeros.

    Returns:
        Path to the written character file.
    """
    from tmrvc_export.speaker_file import read_speaker_file

    spk_embed, lora_delta, metadata, _thumbnail = read_speaker_file(speaker_path)

    # Extract voice_source_preset from speaker metadata if available
    if voice_source_preset is None:
        vs_preset = metadata.get("voice_source_preset")
        if vs_preset is not None:
            voice_source_preset = np.array(vs_preset, dtype=np.float32)

    # Auto-populate profile from speaker metadata
    if profile is None:
        profile = {}
    if not profile.get("name"):
        profile.setdefault("name", metadata.get("profile_name", ""))

    return write_character_file(
        output_path=output_path,
        spk_embed=spk_embed,
        lora_delta=lora_delta,
        voice_source_preset=voice_source_preset,
        default_style=default_style,
        profile=profile,
    )
