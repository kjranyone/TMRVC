"""GET /health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from tmrvc_serve.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from tmrvc_serve.app import _engine, _characters

    engine = _engine
    return HealthResponse(
        status="ok",
        models_loaded=engine.models_loaded if engine else False,
        characters_count=len(_characters),
    )
