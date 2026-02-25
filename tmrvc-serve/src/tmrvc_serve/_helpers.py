"""Shared utility functions for the TTS server."""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Generator

import numpy as np

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import DialogueTurn

logger = logging.getLogger(__name__)


def _audio_to_wav_base64(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Encode float32 audio to base64 WAV string."""
    import struct

    buf = io.BytesIO()
    n_samples = len(audio)
    data_size = n_samples * 4  # float32 = 4 bytes

    # WAV header (float32 format, IEEE)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 3))   # format = IEEE float
    buf.write(struct.pack("<H", 1))   # channels
    buf.write(struct.pack("<I", sr))  # sample rate
    buf.write(struct.pack("<I", sr * 4))  # byte rate
    buf.write(struct.pack("<H", 4))   # block align
    buf.write(struct.pack("<H", 32))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio.astype(np.float32).tobytes())

    return base64.b64encode(buf.getvalue()).decode("ascii")


def _append_silence(
    audio: np.ndarray,
    leading_ms: int,
    trailing_ms: int,
) -> np.ndarray:
    lead_samples = max(0, int(SAMPLE_RATE * leading_ms / 1000))
    trail_samples = max(0, int(SAMPLE_RATE * trailing_ms / 1000))
    if lead_samples == 0 and trail_samples == 0:
        return audio

    parts: list[np.ndarray] = []
    if lead_samples > 0:
        parts.append(np.zeros(lead_samples, dtype=np.float32))
    parts.append(audio.astype(np.float32, copy=False))
    if trail_samples > 0:
        parts.append(np.zeros(trail_samples, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _iter_silence_chunks(
    total_ms: int,
    chunk_duration_ms: int,
) -> Generator[np.ndarray, None, None]:
    total_samples = max(0, int(SAMPLE_RATE * total_ms / 1000))
    chunk_samples = max(1, int(SAMPLE_RATE * chunk_duration_ms / 1000))
    while total_samples > 0:
        n = min(chunk_samples, total_samples)
        yield np.zeros(n, dtype=np.float32)
        total_samples -= n


def _to_dialogue_turns(
    context: list[object] | None,
) -> list[DialogueTurn]:
    if not context:
        return []
    return [
        DialogueTurn(
            speaker=str(getattr(turn, "speaker")),
            text=str(getattr(turn, "text")),
            emotion=getattr(turn, "emotion", None),
        )
        for turn in context
    ]


def _load_speaker_embed(character: object) -> "torch.Tensor":
    """Load speaker embedding from character's speaker file."""
    import torch

    if character.speaker_file and character.speaker_file.exists():
        from tmrvc_export.speaker_file import read_speaker_file
        spk_embed, _lora, _meta, _thumb = read_speaker_file(character.speaker_file)
        return torch.from_numpy(spk_embed).float()

    # Fallback: zero embedding
    logger.warning("No speaker file for '%s', using zero embedding", character.name)
    return torch.zeros(192)
