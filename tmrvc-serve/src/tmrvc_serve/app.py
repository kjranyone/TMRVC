"""FastAPI application for TMRVC TTS server.

Endpoints:
- POST /tts            Generate audio from text (batch)
- POST /tts/stream     Streaming audio generation (chunked PCM)
- WS   /ws/chat        WebSocket for live chat TTS (real-time priority queue)
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
    title="TMRVC TTS Server",
    description="Text-to-Speech API with emotion/style control and WebSocket chat support.",
    version="0.1.0",
)

# Global state (initialized in lifespan or startup)
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
    """Initialize the TTS engine and context predictor.

    Called by the CLI or manually before serving.
    """
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
        logger.warning("Context predictor unavailable; using local fallback only: %s", e)
        _context_predictor = None


# Mount routers (lazy imports in route handlers avoid circular deps)
from tmrvc_serve.routes.health import router as health_router  # noqa: E402
from tmrvc_serve.routes.characters import router as characters_router  # noqa: E402
from tmrvc_serve.routes.tts import router as tts_router  # noqa: E402
from tmrvc_serve.routes.ws_chat import router as ws_router  # noqa: E402

app.include_router(health_router)
app.include_router(characters_router)
app.include_router(tts_router)
app.include_router(ws_router)

# Backward-compatible re-exports (tests import from tmrvc_serve.app)
from tmrvc_serve.style_resolver import (  # noqa: F401, E402
    StylePresetConfig,
    _INLINE_STAGE_BLEND_WEIGHT,
    _apply_dialogue_dynamics,
    _apply_inline_stage_overlay,
    _blend_styles,
    _resolve_effective_speed,
    _resolve_sentence_pause,
    _resolve_style_preset,
)
from tmrvc_serve._helpers import (  # noqa: F401, E402
    _append_silence,
    _audio_to_wav_base64,
)
from tmrvc_serve.routes.ws_chat import SpeakItem  # noqa: F401, E402

# Compat wrapper: injects module-global _context_predictor when not passed
from tmrvc_serve.style_resolver import (  # noqa: E402
    _predict_style_from_inputs as _predict_style_impl,
)


async def _predict_style_from_inputs(**kwargs):  # noqa: E302
    if "context_predictor" not in kwargs:
        kwargs["context_predictor"] = _context_predictor
    return await _predict_style_impl(**kwargs)
