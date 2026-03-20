""".tmrvc_speaker v4 binary file format: create and load speaker profiles.

Format supports hierarchical adaptation:
- Light: spk_embed only
- Standard: + style_embed + reference_tokens
- Full: + lora_delta + acting_latent
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmrvc_core.constants import D_ACTING_LATENT, D_SPEAKER, D_VOICE_STATE_SSL, LORA_DELTA_SIZE, N_CODEBOOKS

logger = logging.getLogger(__name__)

MAGIC = b"TMSP"
VERSION = 4

HEADER_SIZE = 40
CHECKSUM_SIZE = 32

D_STYLE = 128

# Flags (matches Rust speaker.rs)
FLAG_HAS_STYLE = 1 << 0
FLAG_HAS_REF_TOKENS = 1 << 1
FLAG_HAS_LORA = 1 << 2
FLAG_HAS_ACTING_LATENT = 1 << 3

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
    "adaptation_level": "light",
    "checkpoint_name": "",
    "voice_source_preset": None,
    "voice_source_param_names": [],
}


@dataclass
class SpeakerFile:
    """Speaker file v4 data container."""

    spk_embed: np.ndarray  # [192]
    f0_mean: float = 220.0  # Mean F0 frequency in Hz
    style_embed: np.ndarray | None = None  # [128]
    reference_tokens: np.ndarray | None = None  # [T, N_CODEBOOKS]
    lora_delta: np.ndarray | None = None  # [LORA_DELTA_SIZE]
    ssl_state: np.ndarray | None = None  # [128] default SSL state from WavLM (stored in metadata JSON)
    acting_latent: np.ndarray | None = None  # [D_ACTING_LATENT]
    metadata: dict | None = None

    @property
    def adaptation_level(self) -> str:
        """Determine adaptation level from loaded data."""
        if self.lora_delta is not None:
            return "full"
        if self.style_embed is not None or self.reference_tokens is not None:
            return "standard"
        return "light"


def write_speaker_file(
    output_path: str | Path,
    spk_embed: np.ndarray,
    f0_mean: float = 220.0,
    style_embed: np.ndarray | None = None,
    reference_tokens: np.ndarray | None = None,
    lora_delta: np.ndarray | None = None,
    ssl_state: np.ndarray | None = None,
    acting_latent: np.ndarray | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write a .tmrvc_speaker v4 binary file.

    Args:
        output_path: Output file path.
        spk_embed: Speaker embedding array, shape ``(192,)``, float32. (required)
        f0_mean: Mean F0 frequency in Hz. Default 220.0.
        style_embed: Style embedding array, shape ``(128,)``, float32. (optional)
        reference_tokens: Reference codec tokens, shape ``(T, N_CODEBOOKS)``, int32. (optional)
        lora_delta: LoRA delta array, shape ``(LORA_DELTA_SIZE,)``, float32. (optional)
        ssl_state: Default SSL state from WavLM, shape ``(128,)``, float32.
            Stored in metadata JSON, not in the binary section. (optional)
        acting_latent: Acting latent array, shape ``(D_ACTING_LATENT,)``, float32. (optional)
        metadata: Optional metadata dict.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    assert spk_embed.shape == (D_SPEAKER,), (
        f"Expected ({D_SPEAKER},), got {spk_embed.shape}"
    )
    assert spk_embed.dtype == np.float32

    flags = 0
    if style_embed is not None:
        assert style_embed.shape == (D_STYLE,), (
            f"Expected ({D_STYLE},), got {style_embed.shape}"
        )
        assert style_embed.dtype == np.float32
        flags |= FLAG_HAS_STYLE
    if reference_tokens is not None:
        assert reference_tokens.ndim == 2 and reference_tokens.shape[1] == N_CODEBOOKS
        assert reference_tokens.dtype == np.int32
        flags |= FLAG_HAS_REF_TOKENS
    if lora_delta is not None:
        assert lora_delta.shape == (LORA_DELTA_SIZE,), (
            f"Expected ({LORA_DELTA_SIZE},), got {lora_delta.shape}"
        )
        assert lora_delta.dtype == np.float32
        flags |= FLAG_HAS_LORA
    if acting_latent is not None:
        assert acting_latent.shape == (D_ACTING_LATENT,), (
            f"Expected ({D_ACTING_LATENT},), got {acting_latent.shape}"
        )
        assert acting_latent.dtype == np.float32
        flags |= FLAG_HAS_ACTING_LATENT
    if ssl_state is not None:
        assert ssl_state.shape == (D_VOICE_STATE_SSL,), (
            f"Expected ({D_VOICE_STATE_SSL},), got {ssl_state.shape}"
        )
        assert ssl_state.dtype == np.float32

    # Build metadata (ssl_state is stored in metadata JSON, not in binary section)
    meta = {**_DEFAULT_METADATA, **(metadata or {})}
    if ssl_state is not None:
        meta["ssl_state"] = ssl_state.tolist()
    if acting_latent is not None:
        meta["acting_latent"] = acting_latent.tolist()
    # NOTE: f0_mean is stored in binary section, NOT in metadata (single source of truth)
    if "adaptation_level" not in meta:
        if flags & FLAG_HAS_LORA:
            meta["adaptation_level"] = "full"
        elif flags & (FLAG_HAS_STYLE | FLAG_HAS_REF_TOKENS):
            meta["adaptation_level"] = "standard"
        else:
            meta["adaptation_level"] = "light"

    metadata_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    acting_latent_size = D_ACTING_LATENT if acting_latent is not None else 0

    # Build header (40 bytes)
    data = bytearray()
    data += MAGIC
    data += struct.pack("<I", VERSION)  # version
    data += struct.pack("<I", flags)  # flags
    data += struct.pack("<I", D_SPEAKER)  # spk_embed_size
    data += struct.pack(
        "<I", D_STYLE if style_embed is not None else 0
    )  # style_embed_size
    data += struct.pack(
        "<I", len(reference_tokens) if reference_tokens is not None else 0
    )  # ref_tokens_frames
    data += struct.pack(
        "<I", LORA_DELTA_SIZE if lora_delta is not None else 0
    )  # lora_size
    data += struct.pack("<I", len(metadata_bytes))  # metadata_size
    data += struct.pack("<I", acting_latent_size)  # acting_latent_size (NEW in v4)
    data += struct.pack("<I", 0)  # reserved/pad

    # Build data section
    data += spk_embed.tobytes()
    data += struct.pack("<f", f0_mean)  # f0_mean as float32
    if style_embed is not None:
        data += style_embed.tobytes()
    if reference_tokens is not None:
        data += reference_tokens.tobytes()
    if lora_delta is not None:
        data += lora_delta.tobytes()
    if acting_latent is not None:
        data += acting_latent.tobytes()
    data += metadata_bytes

    # Compute SHA-256 checksum
    checksum = hashlib.sha256(bytes(data)).digest()
    data += checksum

    output_path.write_bytes(bytes(data))
    logger.info(
        "Wrote speaker file v4 to %s (%d bytes, level=%s)",
        output_path,
        len(data),
        meta["adaptation_level"],
    )
    return output_path


def read_speaker_file(path: str | Path) -> SpeakerFile:
    """Read and validate a .tmrvc_speaker v4 binary file.

    Args:
        path: Path to the speaker file.

    Returns:
        SpeakerFile data container.

    Raises:
        ValueError: If the file is invalid or corrupted.
    """
    path = Path(path)
    data = path.read_bytes()

    # Validate magic
    magic = data[:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: expected {MAGIC!r}, got {magic!r}")

    # Validate version
    version = struct.unpack("<I", data[4:8])[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}, expected v{VERSION}")

    # Parse header (40 bytes)
    flags = struct.unpack("<I", data[8:12])[0]
    spk_size = struct.unpack("<I", data[12:16])[0]
    style_size = struct.unpack("<I", data[16:20])[0]
    ref_frames = struct.unpack("<I", data[20:24])[0]
    lora_size = struct.unpack("<I", data[24:28])[0]
    metadata_size = struct.unpack("<I", data[28:32])[0]
    acting_latent_size = struct.unpack("<I", data[32:36])[0]
    # bytes 36..40 are reserved/pad

    # Verify spk_embed size
    if spk_size != D_SPEAKER:
        raise ValueError(
            f"Invalid spk_embed_size: expected {D_SPEAKER}, got {spk_size}"
        )

    # Verify checksum
    checksum = data[-CHECKSUM_SIZE:]
    payload = data[:-CHECKSUM_SIZE]
    if hashlib.sha256(payload).digest() != checksum:
        raise ValueError("Checksum mismatch: file is corrupted")

    offset = HEADER_SIZE

    # Extract spk_embed
    spk_embed = np.frombuffer(
        data[offset : offset + spk_size * 4], dtype=np.float32
    ).copy()
    offset += spk_size * 4

    # Extract f0_mean (4 bytes float32)
    f0_mean = struct.unpack("<f", data[offset : offset + 4])[0]
    offset += 4

    # Extract style_embed (optional)
    style_embed = None
    if flags & FLAG_HAS_STYLE:
        style_embed = np.frombuffer(
            data[offset : offset + style_size * 4], dtype=np.float32
        ).copy()
        offset += style_size * 4

    # Extract reference_tokens (optional)
    reference_tokens = None
    if flags & FLAG_HAS_REF_TOKENS:
        reference_tokens = (
            np.frombuffer(data[offset : offset + ref_frames * N_CODEBOOKS * 4], dtype=np.int32)
            .copy()
            .reshape(ref_frames, N_CODEBOOKS)
        )
        offset += ref_frames * N_CODEBOOKS * 4

    # Extract lora_delta (optional)
    lora_delta = None
    if flags & FLAG_HAS_LORA:
        lora_delta = np.frombuffer(
            data[offset : offset + lora_size * 4], dtype=np.float32
        ).copy()
        offset += lora_size * 4

    # Extract acting_latent (optional)
    acting_latent = None
    if flags & FLAG_HAS_ACTING_LATENT and acting_latent_size > 0:
        acting_latent = np.frombuffer(
            data[offset : offset + acting_latent_size * 4], dtype=np.float32
        ).copy()
        offset += acting_latent_size * 4

    # Extract metadata
    metadata_dict = (
        json.loads(data[offset : offset + metadata_size]) if metadata_size > 0 else {}
    )

    # Extract ssl_state from metadata if present
    ssl_state = None
    if "ssl_state" in metadata_dict:
        ssl_state = np.array(metadata_dict["ssl_state"], dtype=np.float32)
        if ssl_state.shape != (D_VOICE_STATE_SSL,):
            logger.warning(
                "ssl_state has unexpected shape: %s, expected (%d,)",
                ssl_state.shape,
                D_VOICE_STATE_SSL,
            )
            ssl_state = None

    return SpeakerFile(
        spk_embed=spk_embed,
        f0_mean=f0_mean,
        style_embed=style_embed,
        reference_tokens=reference_tokens,
        lora_delta=lora_delta,
        ssl_state=ssl_state,
        acting_latent=acting_latent,
        metadata=metadata_dict,
    )
