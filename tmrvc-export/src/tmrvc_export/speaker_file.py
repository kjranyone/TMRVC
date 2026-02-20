""".tmrvc_speaker v2 binary file format: create and load speaker profiles."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import struct
from pathlib import Path

import numpy as np

from tmrvc_core.constants import D_SPEAKER, LORA_DELTA_SIZE

logger = logging.getLogger(__name__)

MAGIC = b"TMSP"
VERSION = 2

# v2 binary layout:
#   Magic: 4 bytes "TMSP"
#   Version: uint32_le = 2
#   spk_embed_size: uint32_le = 192
#   lora_delta_size: uint32_le = 24576
#   metadata_size: uint32_le (JSON UTF-8 byte count)
#   thumbnail_size: uint32_le (PNG byte count, 0 = none)
#   spk_embed: float32[192] = 768 bytes
#   lora_delta: float32[24576] = 98304 bytes
#   metadata_json: UTF-8 JSON (variable)
#   thumbnail_png: PNG bytes (variable)
#   checksum: SHA-256 = 32 bytes

HEADER_SIZE = 24  # magic(4) + version(4) + spk_size(4) + lora_size(4) + meta_size(4) + thumb_size(4)
SPK_EMBED_BYTES = D_SPEAKER * 4
LORA_DELTA_BYTES = LORA_DELTA_SIZE * 4
CHECKSUM_SIZE = 32

# Default metadata template
_DEFAULT_METADATA = {
    "profile_name": "",
    "author_name": "",
    "co_author_name": "",
    "licence_url": "",
    "thumbnail_b64": "",
    "created_at": "",
    "description": "",
    "source_audio_files": [],
    "source_sample_count": 0,
    "training_mode": "embedding",
    "checkpoint_name": "",
    "voice_source_preset": None,
    "voice_source_param_names": [],
}


def write_speaker_file(
    output_path: str | Path,
    spk_embed: np.ndarray,
    lora_delta: np.ndarray,
    metadata: dict | None = None,
    thumbnail_png: bytes | None = None,
) -> Path:
    """Write a .tmrvc_speaker v2 binary file.

    Args:
        output_path: Output file path.
        spk_embed: Speaker embedding array, shape ``(192,)``, float32.
        lora_delta: LoRA delta array, shape ``(24576,)``, float32.
        metadata: Optional metadata dict. Missing keys use defaults.
        thumbnail_png: Optional PNG thumbnail bytes. ``None`` means no thumbnail.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    assert spk_embed.shape == (D_SPEAKER,), f"Expected ({D_SPEAKER},), got {spk_embed.shape}"
    assert lora_delta.shape == (LORA_DELTA_SIZE,), f"Expected ({LORA_DELTA_SIZE},), got {lora_delta.shape}"
    assert spk_embed.dtype == np.float32
    assert lora_delta.dtype == np.float32

    # Build metadata JSON
    meta = {**_DEFAULT_METADATA, **(metadata or {})}
    # Encode thumbnail as base64 in metadata (not in binary thumbnail section)
    if thumbnail_png:
        meta["thumbnail_b64"] = base64.b64encode(thumbnail_png).decode("ascii")

    metadata_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    # Build data (without checksum)
    data = bytearray()
    data += MAGIC
    data += struct.pack("<I", VERSION)
    data += struct.pack("<I", D_SPEAKER)
    data += struct.pack("<I", LORA_DELTA_SIZE)
    data += struct.pack("<I", len(metadata_bytes))
    data += struct.pack("<I", 0)  # thumbnail_size = 0 (stored as base64 in metadata)
    data += spk_embed.tobytes()
    data += lora_delta.tobytes()
    data += metadata_bytes

    # Compute SHA-256 checksum
    checksum = hashlib.sha256(bytes(data)).digest()
    data += checksum

    output_path.write_bytes(bytes(data))
    logger.info("Wrote speaker file to %s (%d bytes)", output_path, len(data))
    return output_path


def read_speaker_file(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict, bytes]:
    """Read and validate a .tmrvc_speaker v2 binary file.

    Args:
        path: Path to the speaker file.

    Returns:
        Tuple of ``(spk_embed, lora_delta, metadata_dict, thumbnail_bytes)``.
        ``spk_embed`` is shape ``(192,)`` float32,
        ``lora_delta`` is shape ``(24576,)`` float32,
        ``metadata_dict`` is a dict,
        ``thumbnail_bytes`` is raw PNG bytes (empty if no thumbnail).

    Raises:
        ValueError: If the file is invalid or corrupted.
    """
    path = Path(path)
    data = path.read_bytes()

    min_size = HEADER_SIZE + SPK_EMBED_BYTES + LORA_DELTA_BYTES + CHECKSUM_SIZE
    if len(data) < min_size:
        raise ValueError(
            f"Invalid file size: expected at least {min_size}, got {len(data)}"
        )

    # Validate magic
    magic = data[:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: expected {MAGIC!r}, got {magic!r}")

    # Validate version
    version = struct.unpack("<I", data[4:8])[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    # Validate sizes
    spk_size = struct.unpack("<I", data[8:12])[0]
    lora_size = struct.unpack("<I", data[12:16])[0]
    if spk_size != D_SPEAKER:
        raise ValueError(f"Invalid spk_embed_size: expected {D_SPEAKER}, got {spk_size}")
    if lora_size != LORA_DELTA_SIZE:
        raise ValueError(f"Invalid lora_delta_size: expected {LORA_DELTA_SIZE}, got {lora_size}")

    # Variable-length sizes
    metadata_size = struct.unpack("<I", data[16:20])[0]
    thumbnail_size = struct.unpack("<I", data[20:24])[0]

    # Verify total size
    expected_size = (
        HEADER_SIZE + SPK_EMBED_BYTES + LORA_DELTA_BYTES
        + metadata_size + thumbnail_size + CHECKSUM_SIZE
    )
    if len(data) != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size}, got {len(data)}"
        )

    # Validate checksum
    payload = data[:-CHECKSUM_SIZE]
    stored_checksum = data[-CHECKSUM_SIZE:]
    computed_checksum = hashlib.sha256(payload).digest()
    if stored_checksum != computed_checksum:
        raise ValueError("Checksum mismatch: file is corrupted")

    # Extract arrays
    spk_offset = HEADER_SIZE
    spk_embed = np.frombuffer(
        data[spk_offset:spk_offset + SPK_EMBED_BYTES], dtype=np.float32,
    ).copy()

    lora_offset = spk_offset + SPK_EMBED_BYTES
    lora_delta = np.frombuffer(
        data[lora_offset:lora_offset + LORA_DELTA_BYTES], dtype=np.float32,
    ).copy()

    # Extract metadata JSON
    meta_offset = lora_offset + LORA_DELTA_BYTES
    if metadata_size > 0:
        metadata_dict = json.loads(data[meta_offset:meta_offset + metadata_size])
    else:
        metadata_dict = dict(_DEFAULT_METADATA)

    # Decode thumbnail from metadata base64 (preferred) or legacy raw section
    thumbnail_b64 = metadata_dict.get("thumbnail_b64", "")
    if thumbnail_b64:
        thumbnail_bytes = base64.b64decode(thumbnail_b64)
    elif thumbnail_size > 0:
        thumb_offset = meta_offset + metadata_size
        thumbnail_bytes = bytes(data[thumb_offset:thumb_offset + thumbnail_size])
    else:
        thumbnail_bytes = b""

    return spk_embed, lora_delta, metadata_dict, thumbnail_bytes
