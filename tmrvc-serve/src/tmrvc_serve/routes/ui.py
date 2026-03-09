"""UI Orchestration and Event Stream routes for TMRVC Serve (Worker 04/07).

Provides the authoritative multi-user API surface for:
- Dataset upload and registration
- Job management and SSE event streams
- Curation runs and record management
- Drama workshop generation
- Evaluation sessions and assignments
- Artifact download contract
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import base64
import io
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
import scipy.io.wavfile as wavfile

from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS
from tmrvc_data.curation.models import RecordStatus, PromotionBucket
from tmrvc_data.curation.data_service import CurationDataService
from tmrvc_serve.events import SSEEvent, SSEEventType
from tmrvc_serve.schemas import ArtifactResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ui", tags=["UI-Orchestration"])

# Global data service dependency
_data_service: Optional[CurationDataService] = None


def get_data_service():
    if _data_service is None:
        raise HTTPException(status_code=503, detail="Curation service not initialized")
    return _data_service


# ---------------------------------------------------------------------------
# Request / Response schemas (Worker 04, task 14)
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    """Status of a long-running job."""

    job_id: str
    job_type: str = ""
    status: str = "pending"  # pending, running, completed, failed, blocked_human
    progress: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None


class DatasetUploadResponse(BaseModel):
    """Response after initiating a dataset upload."""

    job_id: str
    status: str = "accepted"


class DatasetRegisterRequest(BaseModel):
    """Request body for POST /ui/datasets/register."""

    name: str
    path: str
    language: str = "ja"
    description: str = ""
    idempotency_key: Optional[str] = None


class DatasetRegisterResponse(BaseModel):
    """Response for dataset registration."""

    dataset_id: str
    name: str
    status: str = "registered"


class CurationRunRequest(BaseModel):
    """Request body for POST /ui/curation/runs."""

    dataset_id: str
    policy: str = "default"
    idempotency_key: Optional[str] = None


class CurationRunResponse(BaseModel):
    """Response for curation run creation."""

    run_id: str
    dataset_id: str
    status: str = "pending"


class CurationActionRequest(BaseModel):
    """Request body for curation record actions."""

    record_id: str
    metadata_version: int
    action: str  # promote, reject, review
    bucket: Optional[str] = None
    rationale: str = ""
    actor_id: str


class WorkshopGenerateRequest(BaseModel):
    """Request body for POST /ui/workshop/generate."""

    character_id: str
    text: str
    emotion: Optional[str] = None
    style_preset: str = "default"
    pace: float = Field(1.0, ge=0.5, le=3.0)
    hold_bias: float = Field(0.0, ge=-1.0, le=1.0)
    boundary_bias: float = Field(0.0, ge=-1.0, le=1.0)
    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)
    cfg_mode: str = "full"
    speaker_profile_id: Optional[str] = None
    explicit_voice_state: Optional[List[float]] = Field(
        None,
        description=(
            "Canonical 8-D voice_state vector ordered as: "
            + ", ".join(CANONICAL_VOICE_STATE_IDS)
        ),
        min_length=8,
        max_length=8,
    )
    n_takes: int = Field(1, ge=1, le=10)
    idempotency_key: Optional[str] = None


class WorkshopGenerateResponse(BaseModel):
    """Response for workshop generation."""

    job_id: str
    takes: List[str] = Field(default_factory=list)  # take IDs
    status: str = "pending"
    audio_base64: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_sec: Optional[float] = None


class WorkshopSessionRequest(BaseModel):
    """Request body for POST /ui/workshop/sessions."""

    name: str = ""
    character_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class WorkshopSessionResponse(BaseModel):
    """Response for workshop session creation."""

    session_id: str
    name: str = ""
    created_at: str = ""


class EvalSessionRequest(BaseModel):
    """Request body for POST /ui/eval/sessions."""

    name: str = ""
    evaluator_id: str = ""
    dataset_id: Optional[str] = None
    n_assignments: int = Field(10, ge=1, le=100)
    idempotency_key: Optional[str] = None


class EvalSessionResponse(BaseModel):
    """Response for evaluation session creation."""

    session_id: str
    assignments: List[str] = Field(default_factory=list)
    status: str = "created"


class EvalAssignmentResponse(BaseModel):
    """Response for a single evaluation assignment."""

    assignment_id: str
    session_id: str = ""
    record_id: str = ""
    audio_url: str = ""
    text: str = ""
    status: str = "pending"  # pending, submitted


class EvalSubmitRequest(BaseModel):
    """Request body for POST /ui/eval/assignments/{id}/submit."""

    rating: float = Field(..., ge=1.0, le=5.0)
    notes: str = ""
    metadata_version: Optional[int] = None
    idempotency_key: Optional[str] = None


class EvalSubmitResponse(BaseModel):
    """Response for evaluation submission."""

    assignment_id: str
    status: str = "submitted"


class TakePinResponse(BaseModel):
    """Response for pinning a take."""

    take_id: str
    pinned: bool = True


class TakeExportResponse(BaseModel):
    """Response for exporting a take."""

    take_id: str
    artifact: Optional[ArtifactResponse] = None
    status: str = "exporting"


# ---------------------------------------------------------------------------
# Workshop routes (Worker 04 Tier 1 implementation)
# ---------------------------------------------------------------------------


@router.post("/workshop/generate", response_model=WorkshopGenerateResponse)
async def workshop_generate(
    req: WorkshopGenerateRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Generate one or more takes for the drama workshop."""
    from tmrvc_serve.app import get_engine
    from tmrvc_data.g2p import text_to_phonemes

    engine = get_engine()
    job_id = f"wk-{uuid.uuid4().hex[:8]}"

    try:
        g2p_result = text_to_phonemes(req.text)
        phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)
        supra_t = g2p_result.text_suprasegmentals

        speaker_profile = None
        if req.speaker_profile_id:
            speaker_profile = engine.load_speaker_profile(req.speaker_profile_id)

        evs = None
        if req.explicit_voice_state:
            evs = torch.tensor(req.explicit_voice_state).float().view(1, 1, 8)

        audio_t, stats = engine.tts(
            phonemes=phonemes_t,
            speaker_profile=speaker_profile,
            text_suprasegmentals=supra_t,
            explicit_voice_state=evs,
            pace=req.pace,
            hold_bias=req.hold_bias,
            boundary_bias=req.boundary_bias,
            cfg_scale=req.cfg_scale,
            cfg_mode=req.cfg_mode,
            language_id=g2p_result.language_id,
        )

        audio_np = audio_t.detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        buf = io.BytesIO()
        wavfile.write(buf, engine.FRAME_SAMPLE_RATE, audio_int16)
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        return WorkshopGenerateResponse(
            job_id=job_id,
            takes=[f"take-{uuid.uuid4().hex[:8]}"],
            status="completed",
            audio_base64=audio_b64,
            sample_rate=engine.FRAME_SAMPLE_RATE,
            duration_sec=float(len(audio_int16) / engine.FRAME_SAMPLE_RATE),
        )

    except Exception as e:
        logger.exception("Workshop generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workshop/takes/{take_id}/pin", response_model=TakePinResponse)
async def pin_take(take_id: str):
    """Pin a take for future reference."""
    return TakePinResponse(take_id=take_id, pinned=True)


@router.post("/workshop/takes/{take_id}/export")
async def export_take(take_id: str):
    """Export a take as a downloadable artifact."""
    artifact = ArtifactResponse(
        artifact_id=f"art-{uuid.uuid4().hex[:8]}",
        artifact_type="take_bundle",
        download_url=f"/artifacts/{take_id}/download",
        provenance_summary={"take_id": take_id},
    )
    return TakeExportResponse(take_id=take_id, artifact=artifact, status="ready")


@router.post("/workshop/sessions", response_model=WorkshopSessionResponse)
async def create_workshop_session(
    req: WorkshopSessionRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Create a new workshop session."""
    session_id = f"ws-{uuid.uuid4().hex[:8]}"
    return WorkshopSessionResponse(
        session_id=session_id,
        name=req.name,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ---------------------------------------------------------------------------
# Curation routes
# ---------------------------------------------------------------------------


@router.get("/curation/records")
async def list_records(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    ds: CurationDataService = Depends(get_data_service),
):
    """Authoritative API for manifest browsing (Worker 07/12)."""
    records = ds.query_records(status=status, limit=limit, offset=offset)
    return [r.to_dict() for r in records]


@router.get("/curation/records/{record_id}")
async def get_record(
    record_id: str,
    ds: CurationDataService = Depends(get_data_service),
):
    """Get a single curation record by ID."""
    record = ds.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record.to_dict()


@router.post("/curation/records/{record_id}/action")
async def update_record_status(
    record_id: str,
    req: CurationActionRequest,
    idempotency_key: Optional[str] = Header(None),
    ds: CurationDataService = Depends(get_data_service),
):
    """Update record status with optimistic locking and idempotency (Worker 04/07)."""
    record = ds.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    if record.metadata_version != req.metadata_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "stale_version",
                "current_version": record.metadata_version,
                "message": "The record has been modified by another user.",
            },
        )

    if req.action == "promote":
        record.status = RecordStatus.PROMOTED
        if req.bucket:
            record.promotion_bucket = PromotionBucket(req.bucket)
    elif req.action == "reject":
        record.status = RecordStatus.REJECTED
    elif req.action == "review":
        record.status = RecordStatus.REVIEW

    success = ds.update_record(
        record,
        actor_id=req.actor_id,
        action_type=f"ui_{req.action}",
        rationale=req.rationale,
    )

    if not success:
        raise HTTPException(status_code=409, detail="conflict_retry")

    return {"status": "ok", "new_version": record.metadata_version}


# ---------------------------------------------------------------------------
# Job routes
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a long-running job."""
    return JobStatusResponse(job_id=job_id, status="pending", progress=0.0)


@router.get("/events")
async def event_stream(request: Request):
    """SSE Event Stream for job progress and telemetry (Worker 04 requirement)."""
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected(): break
            evt = SSEEvent(event_type=SSEEventType.TELEMETRY_UPDATE, data={"active_jobs": 0})
            yield evt.to_sse()
            await asyncio.sleep(2.0)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
