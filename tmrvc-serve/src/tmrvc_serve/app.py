"""FastAPI application for TMRVC TTS/VC server.

Endpoints:
- POST /tts            Generate audio from text (batch)
- POST /tts/stream     Streaming audio generation (chunked PCM)
- WS   /ws/chat        WebSocket for live chat TTS (real-time priority queue)
- WS   /vc/stream      WebSocket for real-time VC streaming (UCLM dual-stream)
- GET  /characters     List available characters
- POST /characters     Register a new character
- GET  /health         Health check
- /admin/*             System administration and telemetry
- /ui/*                WebUI orchestration and event streams
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from tmrvc_core.dialogue_types import CharacterProfile
from tmrvc_serve.uclm_engine import UCLMEngine
from tmrvc_data.curation.data_service import CurationDataService

logger = logging.getLogger(__name__)

from tmrvc_serve.middleware import IdempotencyMiddleware

app = FastAPI(
    title="TMRVC TTS/VC Server",
    description="Real-time Unified TTS/VC using UCLM v3 pointer-based architecture.",
    version="0.3.0",
)

# Idempotency middleware for UI-originated write endpoints (Worker 04, task 21)
app.add_middleware(IdempotencyMiddleware, ttl=300, max_cache_size=4096)

_engine: UCLMEngine | None = None
_data_service: Optional[CurationDataService] = None
_orchestrator = None
_characters: dict[str, CharacterProfile] = {}
_context_predictor = None


def _load_persisted_characters() -> None:
    global _characters
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


def get_engine() -> UCLMEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="UCLM engine not initialized.")
    return _engine


def init_app(
    uclm_checkpoint: str | Path | None = None,
    codec_checkpoint: str | Path | None = None,
    device: str = "cpu",
    api_key: str | None = None,
    curation_db: str | Path | None = "data/curation/curation.db",
    curation_dir: str | Path | None = "data/curation",
) -> None:
    global _engine, _context_predictor, _data_service, _orchestrator

    # 1. Initialize Curation Data Service (Worker 07 requirement)
    if curation_db:
        from tmrvc_data.curation.data_service import CurationDataService
        _data_service = CurationDataService(curation_db)
        
        from tmrvc_data.curation.orchestrator import CurationOrchestrator
        _orchestrator = CurationOrchestrator(curation_dir, data_service=_data_service)
        
        # Supply to ui router
        import tmrvc_serve.routes.ui as ui_module
        ui_module._data_service = _data_service
        ui_module._orchestrator = _orchestrator
        logger.info("Curation orchestrator & SQLite initialized at %s", curation_db)

    # 2. Load persisted characters
    _load_persisted_characters()

    # 3. Initialize UCLM Engine
    if uclm_checkpoint and codec_checkpoint:
        try:
            _engine = UCLMEngine(
                uclm_checkpoint=uclm_checkpoint,
                codec_checkpoint=codec_checkpoint,
                device=device,
            )
            _engine.load_models()
            # Supply to admin router
            import tmrvc_serve.routes.admin as admin_module
            admin_module._engine = _engine
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
from tmrvc_serve.routes.admin import router as admin_router
from tmrvc_serve.routes.ui import router as ui_router

app.include_router(health_router)
app.include_router(characters_router)
app.include_router(tts_router)
app.include_router(ws_router)
app.include_router(vc_router)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(ui_router)
