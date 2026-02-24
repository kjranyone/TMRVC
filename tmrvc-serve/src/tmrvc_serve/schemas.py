"""Pydantic schemas for the TTS API."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    language: str = "ja"


class CharacterCreateRequest(BaseModel):
    """Request body for POST /characters."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str
    personality: str = ""
    voice_description: str = ""
    language: str = "ja"
    speaker_file: str | None = Field(None, description="Path to .tmrvc_speaker file.")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    models_loaded: bool = False
    characters_count: int = 0


# WebSocket message types

class WSCommentMessage(BaseModel):
    """Client → Server: chat comment."""

    type: str = "comment"
    text: str
    user: str = ""
    priority: int = Field(2, ge=0, le=2)


class WSResponseMessage(BaseModel):
    """Client → Server: manual response text."""

    type: str = "response"
    text: str
    character_id: str


class WSAudioChunk(BaseModel):
    """Server → Client: audio chunk."""

    type: str = "audio_chunk"
    data: str = ""
    frame_index: int = 0
    is_last: bool = False


class WSStyleInfo(BaseModel):
    """Server → Client: predicted style info."""

    type: str = "style_info"
    emotion: str = "neutral"
    vad: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    reasoning: str = ""
