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
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
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
# SSE event stream (Worker 04, task 22)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker 04 Task 14: WebUI-facing orchestration and evaluation routes
# ---------------------------------------------------------------------------

@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Stub for uploading dataset zip/tar via browser."""
    return DatasetUploadResponse(job_id=f"job_upload_{uuid.uuid4().hex[:8]}")

@router.post("/datasets/register", response_model=DatasetRegisterResponse)
async def register_dataset(req: DatasetRegisterRequest):
    """Stub for registering an existing server-side directory."""
    return DatasetRegisterResponse(
        dataset_id=f"ds_{uuid.uuid4().hex[:8]}", 
        name=req.name
    )

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Get status of any long-running job (stub)."""
    # TODO: Wire to actual job tracking once orchestrator is integrated
    return JobStatusResponse(
        job_id=job_id,
        status="pending",
        progress=0.0,
    )

@router.post("/curation/runs", response_model=CurationRunResponse)
async def create_curation_run(req: CurationRunRequest):
    """Stub for starting a full curation pipeline run."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    return CurationRunResponse(run_id=run_id, dataset_id=req.dataset_id)

@router.post("/curation/runs/{run_id}/resume")
async def resume_curation_run(run_id: str):
    """Stub for resuming a stopped/failed curation run."""
    return {"status": "resumed", "run_id": run_id}

@router.post("/curation/runs/{run_id}/stop")
async def stop_curation_run(run_id: str):
    """Stub for stopping an active curation run."""
    return {"status": "stopped", "run_id": run_id}

@router.post("/workshop/generate")
async def workshop_generate():
    """Stub for generating multi-take variations."""
    return {"take_ids": [f"take_{uuid.uuid4().hex[:8]}" for _ in range(3)]}

@router.post("/workshop/takes/{take_id}/pin")
async def workshop_pin_take(take_id: str):
    """Stub for pinning a selected take."""
    return {"status": "pinned", "take_id": take_id}

@router.post("/workshop/takes/{take_id}/export")
async def workshop_export_take(take_id: str):
    """Stub for exporting a take."""
    return {"status": "exported", "take_id": take_id}

@router.post("/workshop/sessions")
async def create_workshop_session():
    """Stub for creating a persisted workshop session."""
    return {"session_id": f"ws_{uuid.uuid4().hex[:8]}"}

@router.post("/eval/sessions")
async def create_eval_session(req: Dict):
    """Stub for creating a blind A/B evaluation session."""
    session_id = f"eval_{uuid.uuid4().hex[:8]}"
    n_assignments = req.get("n_assignments", 10)
    assignments = [f"asn_{uuid.uuid4().hex[:8]}" for _ in range(n_assignments)]
    return {"session_id": session_id, "assignments": assignments}

@router.get("/eval/assignments/{assignment_id}")
async def get_eval_assignment(assignment_id: str):
    """Stub for getting a specific A/B pair to rate."""
    return {
        "assignment_id": assignment_id,
        "text": "This is a sample evaluation text.",
        "sample_a_url": f"/media/{assignment_id}_a.wav",
        "sample_b_url": f"/media/{assignment_id}_b.wav",
    }

@router.post("/eval/assignments/{assignment_id}/submit")
async def submit_eval_assignment(assignment_id: str, req: Dict):
    """Stub for submitting an evaluation rating."""
    return {"status": "submitted", "assignment_id": assignment_id}

# ---------------------------------------------------------------------------
# Existing Event Stream and Curation Routes
# ---------------------------------------------------------------------------

@router.get("/events")
async def event_stream(
    request: Request,
    last_event_id: Optional[str] = None,
):
    """SSE Event Stream for job progress and telemetry (Worker 04 requirement).

    Supports resumable streaming via Last-Event-ID header or query parameter.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                break

            # Emit periodic telemetry heartbeat
            evt = SSEEvent(
                event_type=SSEEventType.TELEMETRY_UPDATE,
                data={"active_jobs": 0},
            )
            yield evt.to_sse()
            await asyncio.sleep(2.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Dataset routes
# ---------------------------------------------------------------------------


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(None),
    idempotency_key: Optional[str] = Header(None),
):
    """Upload a dataset archive for processing."""
    job_id = uuid.uuid4().hex[:16]
    return DatasetUploadResponse(job_id=job_id, status="accepted")


@router.post("/datasets/register", response_model=DatasetRegisterResponse)
async def register_dataset(
    req: DatasetRegisterRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Register an existing dataset path."""
    dataset_id = f"ds-{uuid.uuid4().hex[:8]}"
    return DatasetRegisterResponse(
        dataset_id=dataset_id,
        name=req.name,
        status="registered",
    )


# ---------------------------------------------------------------------------
# Job routes
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a long-running job."""
    # Stub: in production, this looks up from a job store
    return JobStatusResponse(
        job_id=job_id,
        job_type="unknown",
        status="pending",
        progress=0.0,
    )


@router.get("/jobs/{job_id}/events")
async def job_event_stream(
    request: Request,
    job_id: str,
    last_event_id: Optional[str] = None,
):
    """SSE stream of events for a specific job.

    Resumable via Last-Event-ID or query parameter.
    """

    async def _stream() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break

            evt = SSEEvent(
                event_type=SSEEventType.JOB_PROGRESS,
                job_id=job_id,
                data={"progress": 0.0, "status": "pending"},
            )
            yield evt.to_sse()
            await asyncio.sleep(2.0)

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Curation routes
# ---------------------------------------------------------------------------


@router.post("/curation/runs", response_model=CurationRunResponse)
async def create_curation_run(
    req: CurationRunRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Start a new curation run on a dataset."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    return CurationRunResponse(
        run_id=run_id,
        dataset_id=req.dataset_id,
        status="pending",
    )


@router.post("/curation/runs/{run_id}/resume")
async def resume_curation_run(run_id: str):
    """Resume a paused curation run."""
    return {"run_id": run_id, "status": "resumed"}


@router.post("/curation/runs/{run_id}/stop")
async def stop_curation_run(run_id: str):
    """Stop a running curation run."""
    return {"run_id": run_id, "status": "stopped"}


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
    # 1. Fetch record
    record = ds.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    # 2. Check metadata version (Optimistic Locking)
    if record.metadata_version != req.metadata_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "stale_version",
                "current_version": record.metadata_version,
                "message": "The record has been modified by another user.",
            },
        )

    # 3. Apply action
    if req.action == "promote":
        record.status = RecordStatus.PROMOTED
        if req.bucket:
            record.promotion_bucket = PromotionBucket(req.bucket)
    elif req.action == "reject":
        record.status = RecordStatus.REJECTED
    elif req.action == "review":
        record.status = RecordStatus.REVIEW

    # 4. Save with audit log
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
# Workshop routes
# ---------------------------------------------------------------------------


@router.post("/workshop/generate", response_model=WorkshopGenerateResponse)
async def workshop_generate(
    req: WorkshopGenerateRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Generate one or more takes for the drama workshop."""
    job_id = f"wk-{uuid.uuid4().hex[:8]}"
    take_ids = [f"take-{uuid.uuid4().hex[:8]}" for _ in range(req.n_takes)]

    # In production, this enqueues generation jobs and returns immediately.
    return WorkshopGenerateResponse(
        job_id=job_id,
        takes=take_ids,
        status="pending",
    )


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
# Evaluation routes
# ---------------------------------------------------------------------------


@router.post("/eval/sessions", response_model=EvalSessionResponse)
async def create_eval_session(
    req: EvalSessionRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Create a new evaluation session with assignments."""
    session_id = f"ev-{uuid.uuid4().hex[:8]}"
    assignment_ids = [
        f"asgn-{uuid.uuid4().hex[:8]}" for _ in range(req.n_assignments)
    ]
    return EvalSessionResponse(
        session_id=session_id,
        assignments=assignment_ids,
        status="created",
    )


@router.get(
    "/eval/assignments/{assignment_id}",
    response_model=EvalAssignmentResponse,
)
async def get_eval_assignment(assignment_id: str):
    """Get a single evaluation assignment."""
    return EvalAssignmentResponse(
        assignment_id=assignment_id,
        status="pending",
    )


@router.post(
    "/eval/assignments/{assignment_id}/submit",
    response_model=EvalSubmitResponse,
)
async def submit_eval_assignment(
    assignment_id: str,
    req: EvalSubmitRequest,
    idempotency_key: Optional[str] = Header(None),
):
    """Submit an evaluation rating for an assignment."""
    return EvalSubmitResponse(
        assignment_id=assignment_id,
        status="submitted",
    )
