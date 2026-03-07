"""Admin management routes for Worker 12 Gradio control plane."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminHealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: bool = False
    device: str = "cpu"
    cuda_available: bool = False
    cuda_memory_allocated_mb: float = 0.0
    cuda_memory_reserved_mb: float = 0.0
    uptime_seconds: float = 0.0


class TelemetryResponse(BaseModel):
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    avg_tts_latency_ms: float = 0.0
    avg_vc_latency_ms: float = 0.0
    tts_mode: str = "pointer"
    model_checkpoint: str = ""


class ModelInfo(BaseModel):
    name: str
    path: str
    loaded: bool = False


class LoadModelRequest(BaseModel):
    uclm_checkpoint: str
    codec_checkpoint: str


class RuntimeContractResponse(BaseModel):
    tts_mode: str = "pointer"
    pointer_fields: list[str] = ["text_index", "progress", "finished", "stall_frames"]
    voice_state_dims: int = 8
    supports_few_shot: bool = True
    supports_dialogue_context: bool = True
    supports_acting_intent: bool = True
    pacing_controls: list[str] = ["pace", "hold_bias", "boundary_bias", "phrase_pressure", "breath_tendency"]


_start_time = time.time()


@router.get("/health", response_model=AdminHealthResponse)
async def admin_health() -> AdminHealthResponse:
    from tmrvc_serve.app import _engine

    resp = AdminHealthResponse(
        models_loaded=_engine is not None and _engine.models_loaded,
        cuda_available=torch.cuda.is_available(),
        uptime_seconds=time.time() - _start_time,
    )
    if _engine is not None:
        resp.device = str(_engine.device)
    if torch.cuda.is_available():
        resp.cuda_memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        resp.cuda_memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    return resp


@router.get("/telemetry", response_model=TelemetryResponse)
async def admin_telemetry() -> TelemetryResponse:
    from tmrvc_serve.app import _engine

    resp = TelemetryResponse()
    if torch.cuda.is_available():
        resp.vram_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        resp.vram_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    if _engine is not None:
        resp.tts_mode = _engine.tts_mode
    return resp


@router.post("/load_model")
async def admin_load_model(req: LoadModelRequest):
    from tmrvc_serve.app import _engine

    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    try:
        _engine.load_models(uclm_path=req.uclm_checkpoint, codec_path=req.codec_checkpoint)
        return {"status": "ok", "message": "Model loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def admin_list_models():
    checkpoints_dir = Path("checkpoints")
    models = []
    if checkpoints_dir.exists():
        for p in checkpoints_dir.rglob("*.pt"):
            models.append(ModelInfo(name=p.stem, path=str(p)))
    return models


@router.get("/runtime_contract", response_model=RuntimeContractResponse)
async def admin_runtime_contract() -> RuntimeContractResponse:
    from tmrvc_serve.app import _engine

    resp = RuntimeContractResponse()
    if _engine is not None:
        resp.tts_mode = _engine.tts_mode
        resp.supports_few_shot = hasattr(_engine.uclm_core_model, 'speaker_prompt_encoder') if _engine.uclm_core_model else False
        resp.supports_dialogue_context = _engine.scene_state_available
    return resp
