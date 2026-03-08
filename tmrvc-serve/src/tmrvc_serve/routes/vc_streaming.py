"""Production WebSocket routes for real-time VC streaming with session isolation."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.auth import (
    AuthContext,
    OptionalAuth,
    get_rate_limiter,
    require_role,
    UserRole,
    RequestContext,
)
from tmrvc_serve.vc_engine_pool import VCEnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vc", tags=["vc"])

# Global engine pool (initialized on startup)
_engine_pool: VCEnginePool | None = None


def get_engine_pool() -> VCEnginePool:
    global _engine_pool
    if _engine_pool is None:
        # Default checkpoint paths
        uclm_path = Path("checkpoints/uclm/uclm_latest.pt")
        codec_path = Path("checkpoints/codec/codec_latest.pt")
        
        _engine_pool = VCEnginePool(
            uclm_checkpoint=uclm_path,
            codec_checkpoint=codec_path,
            max_concurrent_sessions=20,
            max_gpu_inference=2,
            session_timeout_sec=300.0,
        )
        _engine_pool.load_models()
        asyncio.create_task(_engine_pool.start_cleanup_task())
    return _engine_pool


@router.on_event("startup")
async def startup():
    get_engine_pool()
    logger.info("UCLM VC engine pool initialized")


@router.on_event("shutdown")
async def shutdown():
    global _engine_pool
    if _engine_pool:
        await _engine_pool.stop_cleanup_task()


@router.get("/stats")
async def vc_stats(ctx: AuthContext):
    """Get engine pool statistics (enterprise/admin only)."""
    if ctx.role not in (UserRole.ADMIN, UserRole.ENTERPRISE):
        raise HTTPException(status_code=403, detail="Enterprise or admin role required")

    pool = get_engine_pool()
    return {
        "active_sessions": pool.active_sessions,
        "max_sessions": pool.max_concurrent_sessions,
        "is_ready": pool.is_ready,
    }


class VCRequest(BaseModel):
    audio_base64: str
    character_id: str
    explicit_voice_state: Optional[list[float]] = None
    pitch_shift: float = 0.0


class VCResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 24000


@router.post("", response_model=VCResponse)
async def convert_vc(req: VCRequest):
    """Batch VC conversion endpoint (Worker 04, for single conversion in UI)."""
    from tmrvc_serve.app import get_engine, _characters
    from tmrvc_serve._helpers import _audio_to_wav_base64, _load_speaker_embed

    engine = get_engine()
    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found.")

    # 1. Decode audio
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        import io
        import soundfile as sf
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
        audio_np = audio_np.astype(np.float32)
        if sr != 24000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    # 2. Extract speaker embed
    spk_embed = _load_speaker_embed(character)
    spk_t = spk_embed.to(device=engine.device, dtype=torch.float32).unsqueeze(0)

    # 3. Build style
    style = StyleParams.neutral()
    if req.explicit_voice_state:
        # Map list to StyleParams
        style.breathiness = req.explicit_voice_state[0]
        style.tension = req.explicit_voice_state[1]
        style.arousal = req.explicit_voice_state[2]
        style.valence = req.explicit_voice_state[3]
        style.roughness = req.explicit_voice_state[4]
        style.voicing = req.explicit_voice_state[5]
        style.energy = req.explicit_voice_state[6]
        style.speech_rate = req.explicit_voice_state[7]

    # 4. Perform conversion (Frame-by-frame simulation for streaming-model consistency)
    from tmrvc_serve.uclm_engine import EngineState
    state = EngineState()
    output_chunks = []
    
    FRAME_SIZE = 240
    for i in range(0, len(audio_np), FRAME_SIZE):
        chunk = audio_np[i : i + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
        
        chunk_t = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(engine.device)
        out_audio, state = engine.vc_frame(
            chunk_t, spk_t, style, state, pitch_shift=req.pitch_shift
        )
        output_chunks.append(out_audio.cpu().numpy())

    final_audio = np.concatenate(output_chunks)
    audio_b64 = _audio_to_wav_base64(final_audio)

    return VCResponse(audio_base64=audio_b64)


@router.websocket("/stream")
async def vc_stream(
    websocket: WebSocket,
    api_key: str | None = Query(None, description="API key for authentication"),
):
    """Real-time VC streaming with session isolation.

    Protocol:
    1. Client sends: [192 float32 spk_embed]
    2. Server responds: [8 bytes session_id] or error
    3. Client sends: [N float32 audio chunks]
    4. Server sends: [240 float32 audio frames] (10ms)
    """
    await websocket.accept()

    # Validate API key
    from tmrvc_serve.auth import get_key_store

    if not api_key:
        await websocket.close(code=1008, reason="API key required")
        return

    store = get_key_store()
    key_meta = store.validate_key(api_key)

    if not key_meta:
        await websocket.close(code=1008, reason="Invalid or expired API key")
        return

    # Check rate limits
    limiter = get_rate_limiter()
    allowed, info = limiter.check_rate_limit(key_meta.tenant_id, key_meta.rate_limits)

    if not allowed:
        await websocket.close(code=1008, reason=f"Rate limit exceeded: {info}")
        return

    pool = get_engine_pool()
    if not pool.is_ready:
        await websocket.close(code=1011, reason="Engine not ready")
        return

    session_id = f"{key_meta.tenant_id}:{uuid.uuid4().hex[:8]}"
    session = None
    total_audio_seconds = 0.0

    # Track concurrent sessions
    limiter.start_session(key_meta.tenant_id)

    try:
        data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)

        if len(data) < 768:
            await websocket.close(
                code=1007, reason="Need 768 bytes for speaker embedding"
            )
            return

        spk_embed = np.frombuffer(data[:768], dtype=np.float32).copy()

        try:
            session = await asyncio.wait_for(
                pool.create_session(session_id, spk_embed), timeout=30.0
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1013, reason="Server busy, max sessions reached")
            return

        await websocket.send_bytes(
            session_id.split(":")[1].encode("utf-8").ljust(8, b"\x00")
        )
        logger.info("Session %s started (tenant=%s)", session_id, key_meta.tenant_id)

        audio_buffer = np.array([], dtype=np.float32)
        FRAME_SIZE = 240 # 10ms @ 24kHz

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.info("Session %s timeout (no data)", session_id)
                break

            chunk = np.frombuffer(data, dtype=np.float32).copy()
            audio_buffer = np.concatenate([audio_buffer, chunk])

            while len(audio_buffer) >= FRAME_SIZE:
                frame = audio_buffer[:FRAME_SIZE]
                audio_buffer = audio_buffer[FRAME_SIZE:]

                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    pool.process_frame,
                    session,
                    frame,
                    None, # Default StyleParams
                )

                total_audio_seconds += FRAME_SIZE / 24000

                await websocket.send_bytes(output.tobytes())

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception as e:
        logger.error("Session %s error: %s", session_id, e)
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        if session:
            pool.close_session(session_id)
            logger.info(
                "Session %s closed (audio: %.1fs)", session_id, total_audio_seconds
            )

        limiter.end_session(key_meta.tenant_id, total_audio_seconds)
        store.record_usage(api_key, total_audio_seconds)
