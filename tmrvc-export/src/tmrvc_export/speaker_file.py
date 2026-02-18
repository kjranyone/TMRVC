""".tmrvc_speaker binary file format: create and load speaker profiles."""

from __future__ import annotations

import hashlib
import logging
import struct
from pathlib import Path

import numpy as np

from tmrvc_core.constants import D_SPEAKER, LORA_DELTA_SIZE

logger = logging.getLogger(__name__)

MAGIC = b"TMSP"
VERSION = 1

# Binary layout:
#   Magic: 4 bytes "TMSP"
#   Version: uint32_le = 1
#   spk_embed_size: uint32_le = 192
#   lora_delta_size: uint32_le = 24576
#   spk_embed: float32[192] = 768 bytes
#   lora_delta: float32[24576] = 98304 bytes
#   checksum: SHA-256 = 32 bytes
# Total: 4 + 4 + 4 + 4 + 768 + 98304 + 32 = 99120 bytes

HEADER_SIZE = 16  # magic(4) + version(4) + spk_size(4) + lora_size(4)
SPK_EMBED_BYTES = D_SPEAKER * 4
LORA_DELTA_BYTES = LORA_DELTA_SIZE * 4
CHECKSUM_SIZE = 32
TOTAL_SIZE = HEADER_SIZE + SPK_EMBED_BYTES + LORA_DELTA_BYTES + CHECKSUM_SIZE


def write_speaker_file(
    output_path: str | Path,
    spk_embed: np.ndarray,
    lora_delta: np.ndarray,
) -> Path:
    """Write a .tmrvc_speaker binary file.

    Args:
        output_path: Output file path.
        spk_embed: Speaker embedding array, shape ``(192,)``, float32.
        lora_delta: LoRA delta array, shape ``(24576,)``, float32.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    assert spk_embed.shape == (D_SPEAKER,), f"Expected ({D_SPEAKER},), got {spk_embed.shape}"
    assert lora_delta.shape == (LORA_DELTA_SIZE,), f"Expected ({LORA_DELTA_SIZE},), got {lora_delta.shape}"
    assert spk_embed.dtype == np.float32
    assert lora_delta.dtype == np.float32

    # Build data (without checksum)
    data = bytearray()
    data += MAGIC
    data += struct.pack("<I", VERSION)
    data += struct.pack("<I", D_SPEAKER)
    data += struct.pack("<I", LORA_DELTA_SIZE)
    data += spk_embed.tobytes()
    data += lora_delta.tobytes()

    # Compute SHA-256 checksum
    checksum = hashlib.sha256(bytes(data)).digest()
    data += checksum

    output_path.write_bytes(bytes(data))
    logger.info("Wrote speaker file to %s (%d bytes)", output_path, len(data))
    return output_path


def read_speaker_file(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate a .tmrvc_speaker binary file.

    Args:
        path: Path to the speaker file.

    Returns:
        Tuple of (spk_embed [192], lora_delta [24576]) as float32 numpy arrays.

    Raises:
        ValueError: If the file is invalid or corrupted.
    """
    path = Path(path)
    data = path.read_bytes()

    if len(data) != TOTAL_SIZE:
        raise ValueError(
            f"Invalid file size: expected {TOTAL_SIZE}, got {len(data)}"
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

    return spk_embed, lora_delta
