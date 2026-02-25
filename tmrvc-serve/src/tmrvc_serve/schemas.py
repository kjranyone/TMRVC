"""Pydantic schemas for the TTS API."""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field

StylePreset = Literal["default", "asmr_soft", "asmr_intimate"]


# ---------------------------------------------------------------------------
# Priority enum for real-time WebSocket
# ---------------------------------------------------------------------------


class Priority(int, enum.Enum):
    """Speak request priority (lower value = higher priority)."""

    URGENT = 0
    NORMAL = 1
    LOW = 2


class DialogueTurnSchema(BaseModel):
    """A single dialogue turn for context."""

    speaker: str
    text: str
    emotion: str | None = None


class TTSRequest(BaseModel):
    """Request body for POST /tts."""

    text: str = Field(..., min_length=1, max_length=10000)
    character_id: str = Field(..., min_length=1)
    emotion: str | None = Field(None, description="Emotion override (e.g. 'happy', 'sad').")
    hint: str | None = Field(
        None,
        description="Optional acting/style hint. Used as soft style guidance, not mandatory.",
    )
    style_preset: StylePreset = Field(
        "default",
        description="High-level style preset (default/asmr_soft/asmr_intimate).",
    )
    context: list[DialogueTurnSchema] | None = Field(None, description="Conversation history.")
    situation: str | None = Field(None, description="Scene/situation description.")
    speed: float = Field(1.0, ge=0.5, le=2.0)


class TTSResponse(BaseModel):
    """Response for POST /tts."""

    audio_base64: str = Field(..., description="WAV audio encoded as base64.")
    sample_rate: int = 24000
    duration_sec: float = 0.0
    style_used: dict = Field(default_factory=dict)


class CharacterInfo(BaseModel):
    """Character info returned by GET /characters."""

    id: str
    name: str
    personality: str = ""
    voice_description: str = ""
    language: Literal["ja", "en", "zh", "ko"] = "ja"


class CharacterCreateRequest(BaseModel):
    """Request body for POST /characters."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str
    personality: str = ""
    voice_description: str = ""
    language: Literal["ja", "en", "zh", "ko"] = "ja"
    speaker_file: str | None = Field(None, description="Path to .tmrvc_speaker file.")


class TTSStreamRequest(BaseModel):
    """Request body for POST /tts/stream.

    Same as TTSRequest but returns chunked PCM audio instead of full WAV.
    """

    text: str = Field(..., min_length=1, max_length=10000)
    character_id: str = Field(..., min_length=1)
    emotion: str | None = Field(None, description="Emotion override.")
    hint: str | None = Field(
        None,
        description="Optional acting/style hint. Used as soft style guidance, not mandatory.",
    )
    style_preset: StylePreset = Field(
        "default",
        description="High-level style preset (default/asmr_soft/asmr_intimate).",
    )
    context: list[DialogueTurnSchema] | None = Field(None, description="Conversation history.")
    situation: str | None = Field(None, description="Scene/situation description.")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    chunk_duration_ms: int = Field(100, ge=20, le=500)


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    models_loaded: bool = False
    characters_count: int = 0


# ---------------------------------------------------------------------------
# WebSocket schemas
# ---------------------------------------------------------------------------

# Client → Server


class WSSpeakRequest(BaseModel):
    """Client → Server: request speech synthesis with priority and interrupt."""

    type: str = "speak"
    text: str = Field(..., min_length=1)
    character_id: str = ""
    emotion: str | None = None
    hint: str | None = None
    situation: str | None = None
    style_preset: StylePreset = "default"
    priority: Priority = Priority.NORMAL
    interrupt: bool = False
    speed: float | None = Field(None, ge=0.5, le=2.0)


class WSCancelRequest(BaseModel):
    """Client → Server: cancel all queued speak requests."""

    type: str = "cancel"


class WSConfigureRequest(BaseModel):
    """Client → Server: update session configuration."""

    type: str = "configure"
    character_id: str | None = None
    situation: str | None = None
    style_preset: StylePreset | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    scene_reset: bool = Field(
        False,
        description="Reset scene state to initial (zeroed) state.",
    )


# Server → Client


class WSStyleMessage(BaseModel):
    """Server → Client: emotion/VAD sent before audio for avatar expression sync."""

    type: str = "style"
    emotion: str = "neutral"
    vad: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    reasoning: str = ""
    seq: int = 0


class WSAudioMessage(BaseModel):
    """Server → Client: PCM float32 chunk (base64-encoded)."""

    type: str = "audio"
    data: str = ""
    seq: int = 0
    chunk_index: int = 0
    is_last: bool = False


class WSQueueStatus(BaseModel):
    """Server → Client: queue depth and speaking state."""

    type: str = "queue_status"
    pending: int = 0
    speaking: bool = False


class WSSkipped(BaseModel):
    """Server → Client: notification that a queued item was skipped."""

    type: str = "skipped"
    text: str = ""
    reason: str = ""


class WSError(BaseModel):
    """Server → Client: error message."""

    type: str = "error"
    detail: str = ""
