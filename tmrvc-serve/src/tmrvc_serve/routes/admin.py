"""Admin and Monitoring routes for TMRVC Serve (Worker 04).

Provides:
- GET  /admin/health           — detailed system health with VRAM, latency
- GET  /admin/telemetry        — runtime telemetry
- POST /admin/load_model       — switch checkpoints
- GET  /admin/models           — list available checkpoints
- GET  /admin/runtime_contract — expose pointer/voice_state contract
- GET  /admin/contract         — legacy alias (kept for backward compat)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tmrvc_core.voice_state import VOICE_STATE_REGISTRY, get_voice_state_dimension_names
from ..uclm_engine import UCLMEngine

router = APIRouter(prefix="/admin", tags=["Admin"])

# ---------------------------------------------------------------------------
# Startup timestamp for uptime calculation
# ---------------------------------------------------------------------------
_startup_time: float = time.time()

# Global engine instance dependency (provided by app.py)
_engine: Optional[UCLMEngine] = None


def get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


# ---------------------------------------------------------------------------
# v3 Response / Request schemas (Worker 04, task 13)
# ---------------------------------------------------------------------------


class AdminHealthResponse(BaseModel):
    """Detailed system health response for Worker 12 control plane."""

    status: str = "ok"
    models_loaded: bool = False
    device: str = "cpu"
    cuda_available: bool = False
    cuda_memory_allocated_mb: float = 0.0
    cuda_memory_reserved_mb: float = 0.0
    uptime_seconds: float = 0.0


class TelemetryResponse(BaseModel):
    """Runtime telemetry snapshot for Worker 12 dashboard."""

    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    avg_tts_latency_ms: float = 0.0
    avg_vc_latency_ms: float = 0.0
    recent_rtf: float = 0.0
    stall_events: int = 0
    forced_advance_count: int = 0
    skip_protection_count: int = 0
    active_sessions: int = 0
    tts_mode: str = "pointer"
    model_checkpoint: str = ""
    cfg_mode_active: str = "off"


class RuntimeContractResponse(BaseModel):
    """Introspection of the frozen runtime contract (pointer / voice_state)."""

    tts_mode: str = "pointer"
    pointer_step_ms: float = 10.0
    pointer_fields: List[str] = Field(
        default_factory=lambda: [
            "text_index",
            "progress",
            "finished",
            "boundary_confidence",
            "stall_frames",
            "max_frames_per_unit",
            "frames_on_current_unit",
            "skip_protection_threshold",
            "forced_advance_count",
            "skip_protection_count",
        ]
    )
    voice_state_dims: int = 8
    voice_state_names: List[str] = Field(
        default_factory=get_voice_state_dimension_names
    )
    pacing_controls: List[str] = Field(
        default_factory=lambda: [
            "pace",
            "hold_bias",
            "boundary_bias",
            "phrase_pressure",
            "breath_tendency",
        ]
    )
    cfg_modes: List[str] = Field(
        default_factory=lambda: ["off", "full", "lazy", "distilled"]
    )
    max_prompt_frames: int = 0
    supports_few_shot: bool = True
    supports_dialogue_context: bool = True
    supports_acting_intent: bool = True
    sample_rate: int = 24000
    hop_length: int = 240
    frame_index_start_inclusive: bool = True
    frame_index_end_exclusive: bool = True


class LoadModelRequest(BaseModel):
    """Request body for POST /admin/load_model."""

    uclm_checkpoint: str
    codec_checkpoint: str


class ModelInfo(BaseModel):
    """Information about an available checkpoint."""

    name: str
    path: str
    size_mb: float = 0.0


# ---------------------------------------------------------------------------
# Legacy schemas (kept for backward compatibility)
# ---------------------------------------------------------------------------


class SystemHealth(BaseModel):
    status: str
    uptime_sec: float
    device: str
    vram_used_gb: float
    models_loaded: bool


class RuntimeContract(BaseModel):
    pointer_step_ms: float = 10.0
    voice_state_dims: int = 8
    voice_state_names: List[str]
    max_prompt_frames: int
    cfg_modes: List[str] = ["off", "full", "lazy", "distilled"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=AdminHealthResponse)
async def get_health():
    """Detailed system health with VRAM and uptime."""
    engine = _engine
    models_loaded = engine.models_loaded if engine is not None else False
    device = str(engine.device) if engine is not None else "cpu"

    cuda_available = torch.cuda.is_available()
    cuda_mem_alloc = 0.0
    cuda_mem_reserved = 0.0
    if cuda_available:
        cuda_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        cuda_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    uptime = time.time() - _startup_time

    return AdminHealthResponse(
        status="ok",
        models_loaded=models_loaded,
        device=device,
        cuda_available=cuda_available,
        cuda_memory_allocated_mb=round(cuda_mem_alloc, 2),
        cuda_memory_reserved_mb=round(cuda_mem_reserved, 2),
        uptime_seconds=round(uptime, 2),
    )


@router.get("/telemetry", response_model=TelemetryResponse)
async def get_telemetry():
    """Detailed runtime telemetry for Worker 12 dashboard."""
    engine = _engine

    vram_alloc = 0.0
    vram_reserved = 0.0
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    tts_mode = engine.tts_mode if engine is not None else "pointer"
    ckpt = ""
    if engine is not None and engine._uclm_checkpoint is not None:
        ckpt = str(engine._uclm_checkpoint)

    return TelemetryResponse(
        vram_allocated_mb=round(vram_alloc, 2),
        vram_reserved_mb=round(vram_reserved, 2),
        tts_mode=tts_mode,
        model_checkpoint=ckpt,
    )


@router.get("/runtime_contract", response_model=RuntimeContractResponse)
async def get_runtime_contract():
    """Expose pointer/voice_state runtime contract for introspection."""
    from tmrvc_core.constants import MAX_PROMPT_FRAMES, SAMPLE_RATE, HOP_LENGTH

    engine = _engine
    tts_mode = engine.tts_mode if engine is not None else "pointer"

    return RuntimeContractResponse(
        tts_mode=tts_mode,
        voice_state_names=get_voice_state_dimension_names(),
        max_prompt_frames=MAX_PROMPT_FRAMES,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )


@router.post("/load_model")
async def load_model(req: LoadModelRequest):
    """Switch checkpoints dynamically."""
    engine = _engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    try:
        engine.load_models(
            uclm_path=req.uclm_checkpoint,
            codec_path=req.codec_checkpoint,
        )
        return {"status": "ok", "message": f"Loaded model from {req.uclm_checkpoint}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available checkpoints in the models directory."""
    models_dir = Path("models")
    results: List[ModelInfo] = []

    if models_dir.exists():
        for pt_file in sorted(models_dir.rglob("*.pt")):
            try:
                size_mb = pt_file.stat().st_size / (1024 ** 2)
            except OSError:
                size_mb = 0.0
            results.append(
                ModelInfo(
                    name=pt_file.stem,
                    path=str(pt_file),
                    size_mb=round(size_mb, 2),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Legacy alias — kept for backward compatibility
# ---------------------------------------------------------------------------


@router.get("/contract", response_model=RuntimeContract)
async def get_contract():
    """Expose the frozen runtime contract (legacy alias)."""
    from tmrvc_core.constants import MAX_PROMPT_FRAMES

    return RuntimeContract(
        voice_state_names=get_voice_state_dimension_names(),
        max_prompt_frames=MAX_PROMPT_FRAMES,
    )
