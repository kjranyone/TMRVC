"""FastAPI application for TMRVC TTS/VC server.

Endpoints:
- POST /tts            Generate audio from text (batch)
- POST /tts/stream     Streaming audio generation (chunked PCM)
- WS   /ws/chat        WebSocket for live chat TTS (real-time priority queue)
- WS   /vc/stream      WebSocket for real-time VC streaming (Codec-Latent)
- GET  /characters     List available characters
- POST /characters     Register a new character
- GET  /health         Health check
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from tmrvc_core.dialogue_types import CharacterProfile
from tmrvc_serve.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TMRVC TTS/VC Server",
    description="""
Real-time Text-to-Speech and Voice Conversion API using Codec-Latent paradigm.

## Authentication

This API supports two authentication methods:

### 1. API Key (Service-to-Service)

Include your API key in the `X-API-Key` header:

```
X-API-Key: tmrvc_a1b2c3d4e5f6...
```

### 2. JWT Token (User Sessions)

Include your JWT token in the `Authorization` header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## Demo Credentials

For testing, use these credentials with `POST /auth/token`:

| Email | Password | Role |
|-------|----------|------|
| admin@tmrvc.example.com | admin123 | admin |
| enterprise@tmrvc.example.com | enterprise123 | enterprise |
| pro@tmrvc.example.com | pro12345 | pro |
| free@tmrvc.example.com | free12345 | free |

## Features

- **TTS**: Text-to-speech with emotion/style control
- **VC Streaming**: Real-time voice conversion via WebSocket
- **VC Batch**: Process pre-recorded audio files
- **Session Isolation**: Per-connection state management
- **Concurrent Users**: Support for multiple simultaneous sessions

## Rate Limits

| Role | Requests/min | Concurrent | Daily Quota |
|------|-------------|------------|-------------|
| admin | 1000 | 50 | Unlimited |
| enterprise | 300 | 20 | 10 hours |
| pro | 120 | 10 | 2 hours |
| free | 30 | 3 | 10 minutes |
""",
    version="0.1.0",
    openapi_tags=[
        {"name": "auth", "description": "Authentication and API key management"},
        {"name": "health", "description": "Health check endpoints"},
        {"name": "tts", "description": "Text-to-Speech endpoints"},
        {"name": "vc", "description": "Voice Conversion endpoints"},
        {"name": "characters", "description": "Character management"},
        {"name": "websocket", "description": "WebSocket streaming endpoints"},
    ],
)

_engine: TTSEngine | None = None
_characters: dict[str, CharacterProfile] = {}
_context_predictor = None


def get_engine() -> TTSEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized.")
    return _engine


def init_app(
    tts_checkpoint: str | Path | None = None,
    vc_checkpoint: str | Path | None = None,
    device: str = "cpu",
    api_key: str | None = None,
    text_frontend: str = "tokenizer",
) -> None:
    global _engine, _context_predictor

    _engine = TTSEngine(
        tts_checkpoint=tts_checkpoint,
        vc_checkpoint=vc_checkpoint,
        device=device,
        text_frontend=text_frontend,
    )
    _engine.load_models()
    _engine.warmup()

    try:
        from tmrvc_train.context_predictor import ContextStylePredictor

        _context_predictor = ContextStylePredictor(api_key=api_key)
    except Exception as e:
        logger.warning(
            "Context predictor unavailable; using local fallback only: %s", e
        )
        _context_predictor = None


from tmrvc_serve.routes.health import router as health_router
from tmrvc_serve.routes.characters import router as characters_router
from tmrvc_serve.routes.tts import router as tts_router
from tmrvc_serve.routes.ws_chat import router as ws_router
from tmrvc_serve.routes.vc_streaming import router as vc_router
from tmrvc_serve.routes.auth import router as auth_router

app.include_router(health_router)
app.include_router(characters_router)
app.include_router(tts_router)
app.include_router(ws_router)
app.include_router(vc_router)
app.include_router(auth_router)

from tmrvc_serve.style_resolver import (
    StylePresetConfig,
    _INLINE_STAGE_BLEND_WEIGHT,
    _apply_dialogue_dynamics,
    _apply_inline_stage_overlay,
    _blend_styles,
    _resolve_effective_speed,
    _resolve_sentence_pause,
    _resolve_style_preset,
)
from tmrvc_serve._helpers import (
    _append_silence,
    _audio_to_wav_base64,
)
from tmrvc_serve.routes.ws_chat import SpeakItem

from tmrvc_serve.style_resolver import (
    _predict_style_from_inputs as _predict_style_impl,
)


async def _predict_style_from_inputs(**kwargs):
    if "context_predictor" not in kwargs:
        kwargs["context_predictor"] = _context_predictor
    return await _predict_style_impl(**kwargs)
