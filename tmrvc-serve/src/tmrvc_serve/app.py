"""FastAPI application for TMRVC TTS/VC server.

Endpoints:
- POST /tts            Generate audio from text (batch)
- POST /tts/stream     Streaming audio generation (chunked PCM)
- WS   /ws/chat        WebSocket for live chat TTS (real-time priority queue)
- WS   /vc/stream      WebSocket for real-time VC streaming (UCLM dual-stream)
- GET  /characters     List available characters
- POST /characters     Register a new character
- GET  /health         Health check
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from tmrvc_core.dialogue_types import CharacterProfile
from tmrvc_serve.uclm_engine import UCLMEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TMRVC TTS/VC Server",
    description="Real-time Unified TTS/VC using UCLM v2 architecture.",
    version="0.2.0",
)

_engine: UCLMEngine | None = None
_characters: dict[str, CharacterProfile] = {}
_context_predictor = None


def _load_persisted_characters() -> None:
    global _characters
    from pathlib import Path
    import json
    
    char_json = Path("configs/characters.json")
    if not char_json.exists():
        return
    
    try:
        with open(char_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for cid, p in data.items():
            speaker_file = Path(p["speaker_file"]) if p.get("speaker_file") else None
            _characters[cid] = CharacterProfile(
                name=p.get("name", cid),
                personality=p.get("personality", ""),
                voice_description=p.get("voice_description", ""),
                language=p.get("language", "ja"),
                speaker_file=speaker_file,
            )
        logger.info("Loaded %d character(s) from %s", len(_characters), char_json)
    except Exception as e:
        logger.error("Failed to load characters from %s: %s", char_json, e)


def _save_persisted_characters() -> None:
    from pathlib import Path
    import json
    
    char_json = Path("configs/characters.json")
    char_json.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data = {}
        for cid, p in _characters.items():
            data[cid] = {
                "name": p.name,
                "speaker_file": str(p.speaker_file) if p.speaker_file else None,
                "adaptation_level": "standard", # Default for now
                "personality": p.personality,
                "voice_description": p.voice_description,
                "language": p.language
            }
        with open(char_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save characters to %s: %s", char_json, e)


def get_engine() -> UCLMEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="UCLM engine not initialized.")
    return _engine


def init_app(
    uclm_checkpoint: str | Path | None = None,
    codec_checkpoint: str | Path | None = None,
    device: str = "cpu",
    api_key: str | None = None,
) -> None:
    global _engine, _context_predictor

    # Load persisted characters from configs/characters.json
    _load_persisted_characters()

    if uclm_checkpoint and codec_checkpoint:
        try:
            _engine = UCLMEngine(
                uclm_checkpoint=uclm_checkpoint,
                codec_checkpoint=codec_checkpoint,
                device=device,
            )
            _engine.load_models()
            logger.info("UCLM engine initialized on %s", device)
        except Exception as e:
            logger.error("Failed to initialize UCLM engine: %s", e)
            _engine = None
    else:
        logger.warning("UCLM or Codec checkpoint missing; engine not initialized.")

    try:
        from tmrvc_train.context_predictor import ContextStylePredictor
        _context_predictor = ContextStylePredictor(api_key=api_key)
    except Exception as e:
        logger.warning(
            "Context predictor unavailable; using local fallback only: %s", e
        )
        _context_predictor = None


# Include routers
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


# Helper for style prediction (async)
async def predict_style_from_inputs(**kwargs):
    from tmrvc_serve.style_resolver import _predict_style_from_inputs as _predict_style_impl
    if "context_predictor" not in kwargs:
        kwargs["context_predictor"] = _context_predictor
    return await _predict_style_impl(**kwargs)
