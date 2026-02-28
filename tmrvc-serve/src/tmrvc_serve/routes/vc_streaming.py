"""Production WebSocket routes for real-time VC streaming with session isolation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

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
        model_dir = Path("models/fp32")
        _engine_pool = VCEnginePool(
            model_dir=model_dir,
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
    logger.info("VC engine pool initialized")


@router.on_event("shutdown")
async def shutdown():
    global _engine_pool
    if _engine_pool:
        await _engine_pool.stop_cleanup_task()


@router.get("/stats")
async def vc_stats(ctx: AuthContext):
    """Get engine pool statistics (enterprise/admin only)."""
    from tmrvc_serve.auth import UserRole
    from fastapi import HTTPException

    if ctx.role not in (UserRole.ADMIN, UserRole.ENTERPRISE):
        raise HTTPException(status_code=403, detail="Enterprise or admin role required")

    pool = get_engine_pool()
    return {
        "active_sessions": pool.active_sessions,
        "max_sessions": pool.max_concurrent_sessions,
        "is_ready": pool.is_ready,
    }


@router.websocket("/stream")
async def vc_stream(
    websocket: WebSocket,
    api_key: str | None = Query(None, description="API key for authentication"),
):
    """Real-time VC streaming with session isolation.

    Authentication via query param (WebSocket limitation).

    Protocol:
    1. Client sends: [192 float32 spk_embed]
    2. Server responds: [8 bytes session_id] or error
    3. Client sends: [N float32 audio chunks]
    4. Server sends: [480 float32 audio frames]
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
        FRAME_SIZE = 480

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


@router.get("/keys")
async def list_api_keys(ctx: AuthContext):
    """List API keys for tenant (admin only)."""
    from tmrvc_serve.auth import get_key_store, require_role, UserRole

    # Check role manually since AuthContext doesn't enforce it
    if ctx.role not in (UserRole.ADMIN,):
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role {ctx.role} not authorized",
        )

    store = get_key_store()
    keys = store.get_tenant_keys(ctx.tenant_id)

    return {
        "keys": [
            {
                "prefix": k.key_prefix,
                "role": k.role.value,
                "enabled": k.enabled,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "total_requests": k.total_requests,
                "total_audio_seconds": k.total_audio_seconds,
            }
            for k in keys
        ]
    }


@router.post("/keys")
async def create_api_key(
    ctx: AuthContext,
    user_id: str = Query(...),
    role: UserRole = Query(UserRole.PRO),
    expires_days: int | None = Query(None),
):
    """Create new API key (admin only)."""
    from tmrvc_serve.auth import get_key_store, UserRole
    from fastapi import HTTPException, status

    if ctx.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required"
        )

    store = get_key_store()
    api_key = store.create_key(
        tenant_id=ctx.tenant_id,
        user_id=user_id,
        role=role,
        expires_days=expires_days,
    )

    return {"api_key": api_key, "role": role.value}


@router.delete("/keys/{key_prefix}")
async def revoke_api_key(
    key_prefix: str,
    ctx: AuthContext,
):
    """Revoke API key (admin only)."""
    from tmrvc_serve.auth import get_key_store, UserRole

    if ctx.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required")

    store = get_key_store()
    success = store.revoke_key(key_prefix)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"revoked": key_prefix}
