"""WebUI-specific data APIs for Curation Auditor, Dataset Manager, Workshop,
and Evaluation (Worker 04, tasks 13/20/21/22).

All /ui/* routes are the authoritative multi-user contract.  Direct manifest
file access is not a mainline API.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tmrvc_data.curation.models import RecordStatus, PromotionBucket
from tmrvc_serve.events import SSEEvent, SSEEventType
from tmrvc_serve.middleware import ConflictType, raise_conflict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])

# ---------------------------------------------------------------------------
# In-memory registries (simple dict-based stores for job, take, session state)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_takes: dict[str, dict[str, Any]] = {}
_datasets: dict[str, dict[str, Any]] = {}
_workshop_sessions: dict[str, dict[str, Any]] = {}
_eval_sessions: dict[str, dict[str, Any]] = {}
_eval_assignments: dict[str, dict[str, Any]] = {}
_curation_runs: dict[str, dict[str, Any]] = {}
_job_events: dict[str, list[SSEEvent]] = {}  # job_id -> ordered events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_job(job_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a new job entry and return it."""
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    _jobs[job_id] = job
    _job_events[job_id] = []
    return job


def _emit_event(
    job_id: str,
    event_type: SSEEventType,
    object_type: str = "",
    object_id: str = "",
    data: dict[str, Any] | None = None,
) -> SSEEvent:
    """Create an SSE event and append it to the job's event log."""
    evt = SSEEvent(
        event_type=event_type,
        job_id=job_id,
        object_type=object_type,
        object_id=object_id,
        data=data or {},
    )
    _job_events.setdefault(job_id, []).append(evt)
    return evt


# ---------------------------------------------------------------------------
# Pydantic models for request/response
# ---------------------------------------------------------------------------


class CurationRecordSummary(BaseModel):
    record_id: str
    transcript: str
    language: str
    quality_score: float
    status: str
    speaker_cluster: str
    source_legality: str
    metadata_version: int


class CurationRecordUpdate(BaseModel):
    transcript: Optional[str] = None
    status: Optional[RecordStatus] = None
    promotion_bucket: Optional[PromotionBucket] = None
    source_legality: Optional[str] = None
    expected_version: int


class ActionRequest(BaseModel):
    record_id: str
    role: str
    actor_id: str
    rationale: str = ""
    expected_version: int


class DatasetRegisterRequest(BaseModel):
    name: str
    path: str = Field(..., description="Server-side directory path containing the dataset.")
    language: str = "ja"
    description: str = ""
    idempotency_key: str | None = None


class CurationRunRequest(BaseModel):
    dataset_id: str
    policy: str = "default"
    idempotency_key: str | None = None


class WorkshopGenerateRequest(BaseModel):
    character_id: str
    text: str = Field(..., min_length=1, max_length=10000)
    emotion: str | None = None
    hint: str | None = None
    style_preset: str = "default"
    session_id: str | None = None
    idempotency_key: str | None = None


class WorkshopSessionRequest(BaseModel):
    name: str = ""
    character_id: str = ""
    idempotency_key: str | None = None


class EvalSessionRequest(BaseModel):
    name: str = ""
    eval_type: str = "ab_test"
    dataset_id: str | None = None
    idempotency_key: str | None = None


class EvalSubmitRequest(BaseModel):
    rating: float = Field(..., ge=1.0, le=5.0)
    notes: str = ""
    expected_version: int = 1


class ArtifactResponse(BaseModel):
    """Artifact download contract (Worker 04, task 22)."""
    artifact_id: str
    artifact_type: str = Field(
        ...,
        description='One of "training_bundle", "eval_bundle", "take_bundle".',
    )
    download_url: str
    expires_at: datetime | None = None
    provenance_summary: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Existing curation record endpoints
# ---------------------------------------------------------------------------


@router.get("/curation/records", response_model=List[CurationRecordSummary])
async def list_curation_records(
    status: Optional[str] = None,
    bucket: Optional[str] = None,
):
    from tmrvc_serve.app import _curation_orchestrator

    if _curation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Curation service not initialized.")

    records = []
    for r in _curation_orchestrator.records.values():
        if status and status != "all" and r.status.value != status:
            continue
        if bucket and bucket != "all" and r.promotion_bucket.value != bucket:
            continue

        records.append(CurationRecordSummary(
            record_id=r.record_id,
            transcript=r.transcript or "",
            language=r.language or "",
            quality_score=r.quality_score,
            status=r.status.value,
            speaker_cluster=r.speaker_cluster or "",
            source_legality=r.source_legality,
            metadata_version=r.metadata_version,
        ))
    return records


@router.get("/curation/records/{record_id}")
async def get_curation_record(record_id: str):
    from tmrvc_serve.app import _curation_orchestrator

    if _curation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Curation service not initialized.")

    record = _curation_orchestrator.records.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")

    return record.to_dict()


@router.patch("/curation/records/{record_id}")
async def update_curation_record(record_id: str, req: CurationRecordUpdate):
    from tmrvc_serve.app import _curation_orchestrator

    if _curation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Curation service not initialized.")

    record = _curation_orchestrator.records.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")

    # Update fields
    if req.transcript is not None:
        record.transcript = req.transcript
    if req.status is not None:
        record.status = req.status
    if req.promotion_bucket is not None:
        record.promotion_bucket = req.promotion_bucket
    if req.source_legality is not None:
        record.source_legality = req.source_legality

    try:
        _curation_orchestrator.update_record(record, expected_version=req.expected_version)
        _curation_orchestrator.save_manifest()
        return {"status": "ok", "metadata_version": record.metadata_version}
    except ValueError as e:
        raise_conflict(
            ConflictType.STALE_VERSION,
            str(e),
            current_version=record.metadata_version,
        )


@router.post("/curation/actions/promote")
async def promote_record(req: ActionRequest):
    from tmrvc_serve.app import _curation_orchestrator

    if _curation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Curation service not initialized.")

    record = _curation_orchestrator.records.get(req.record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record {req.record_id} not found.")

    record.status = RecordStatus.PROMOTED
    # Default bucket if none
    if record.promotion_bucket == PromotionBucket.NONE:
        record.promotion_bucket = PromotionBucket.TTS_MAINLINE

    try:
        _curation_orchestrator.update_record(record, expected_version=req.expected_version)
        _curation_orchestrator.save_manifest()
        return {"status": "ok", "metadata_version": record.metadata_version}
    except ValueError as e:
        raise_conflict(
            ConflictType.STALE_VERSION,
            str(e),
            current_version=record.metadata_version,
        )


# ---------------------------------------------------------------------------
# Dataset management (task 13)
# ---------------------------------------------------------------------------


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = None,
    language: str = "ja",
):
    """Accept a dataset archive upload (zip/tar.gz) and register it."""
    dataset_id = uuid.uuid4().hex[:12]
    dataset_name = name or (file.filename or f"upload-{dataset_id}")

    # Save uploaded file to a staging directory
    staging_dir = Path(tempfile.gettempdir()) / "tmrvc_datasets" / dataset_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    dest_path = staging_dir / (file.filename or "upload.bin")

    try:
        with open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Create a background job for processing
    job = _create_job("dataset_upload", {"dataset_id": dataset_id, "filename": file.filename})

    _datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "language": language,
        "path": str(staging_dir),
        "status": "uploaded",
        "job_id": job["job_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Emit a progress event
    _emit_event(
        job["job_id"],
        SSEEventType.JOB_PROGRESS,
        object_type="dataset",
        object_id=dataset_id,
        data={"progress": 0.0, "message": "Upload received, processing queued."},
    )

    # Mark completed immediately for now (real processing would be async)
    job["status"] = "completed"
    job["progress"] = 1.0
    _emit_event(
        job["job_id"],
        SSEEventType.JOB_COMPLETED,
        object_type="dataset",
        object_id=dataset_id,
        data={"message": "Dataset uploaded successfully."},
    )

    return {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "job_id": job["job_id"],
        "status": "uploaded",
    }


@router.post("/datasets/register")
async def register_dataset(req: DatasetRegisterRequest):
    """Register a server-side directory as a dataset."""
    dataset_path = Path(req.path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.path}")
    if not dataset_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {req.path}")

    dataset_id = uuid.uuid4().hex[:12]
    job = _create_job("dataset_register", {"dataset_id": dataset_id, "path": req.path})

    _datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "name": req.name,
        "language": req.language,
        "description": req.description,
        "path": req.path,
        "status": "registered",
        "job_id": job["job_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    job["status"] = "completed"
    job["progress"] = 1.0
    _emit_event(
        job["job_id"],
        SSEEventType.JOB_COMPLETED,
        object_type="dataset",
        object_id=dataset_id,
        data={"message": "Dataset registered."},
    )

    return {
        "dataset_id": dataset_id,
        "name": req.name,
        "job_id": job["job_id"],
        "status": "registered",
    }


# ---------------------------------------------------------------------------
# Job management (task 13/20)
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the current status of a job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(**job)


@router.get("/jobs/{job_id}/events")
async def get_job_events(
    job_id: str,
    last_event_id: str | None = Query(None, alias="Last-Event-ID"),
):
    """SSE event stream for a job (task 20).

    Supports resumption via ``Last-Event-ID`` query parameter or header.
    Emits all past events immediately, then keeps the connection open for
    new events (long-poll style with periodic keepalive).
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    async def _event_generator():
        events = _job_events.get(job_id, [])

        # Determine the starting point for resumption
        start_idx = 0
        if last_event_id:
            for i, evt in enumerate(events):
                if evt.event_id == last_event_id:
                    start_idx = i + 1
                    break

        # Emit buffered events
        for evt in events[start_idx:]:
            yield evt.to_sse()

        # Keep-alive loop: emit a comment every 15 seconds.
        # In a production system this would await new events from a queue.
        job = _jobs.get(job_id)
        if job and job.get("status") in ("completed", "failed"):
            # Job is terminal -- no need to keep the stream open.
            return

        for _ in range(60):  # keep-alive for up to ~15 minutes
            await asyncio.sleep(15)
            yield ": keepalive\n\n"
            # Check for new events appended since we started streaming
            current_events = _job_events.get(job_id, [])
            if len(current_events) > len(events):
                for evt in current_events[len(events):]:
                    yield evt.to_sse()
                events = current_events
            # Break if job reached terminal state
            job = _jobs.get(job_id)
            if job and job.get("status") in ("completed", "failed"):
                return

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Curation runs (task 13)
# ---------------------------------------------------------------------------


@router.post("/curation/runs")
async def start_curation_run(req: CurationRunRequest):
    """Start a new curation run for a dataset."""
    if req.dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id} not found.")

    run_id = uuid.uuid4().hex[:12]
    job = _create_job("curation_run", {"run_id": run_id, "dataset_id": req.dataset_id})

    _curation_runs[run_id] = {
        "run_id": run_id,
        "dataset_id": req.dataset_id,
        "policy": req.policy,
        "status": "running",
        "job_id": job["job_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    job["status"] = "running"
    _emit_event(
        job["job_id"],
        SSEEventType.JOB_PROGRESS,
        object_type="curation_run",
        object_id=run_id,
        data={"progress": 0.0, "message": "Curation run started."},
    )

    return {"run_id": run_id, "job_id": job["job_id"], "status": "running"}


@router.post("/curation/runs/{run_id}/resume")
async def resume_curation_run(run_id: str):
    """Resume a paused curation run."""
    run = _curation_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Curation run {run_id} not found.")
    if run["status"] not in ("paused", "blocked"):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} is in state '{run['status']}', cannot resume.",
        )

    run["status"] = "running"
    job = _jobs.get(run["job_id"])
    if job:
        job["status"] = "running"
        _emit_event(
            job["job_id"],
            SSEEventType.JOB_PROGRESS,
            object_type="curation_run",
            object_id=run_id,
            data={"message": "Curation run resumed."},
        )

    return {"run_id": run_id, "status": "running"}


@router.post("/curation/runs/{run_id}/stop")
async def stop_curation_run(run_id: str):
    """Stop a running curation run."""
    run = _curation_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Curation run {run_id} not found.")

    run["status"] = "stopped"
    job = _jobs.get(run["job_id"])
    if job:
        job["status"] = "completed"
        job["progress"] = job.get("progress", 0.0)
        _emit_event(
            job["job_id"],
            SSEEventType.JOB_COMPLETED,
            object_type="curation_run",
            object_id=run_id,
            data={"message": "Curation run stopped by user."},
        )

    return {"run_id": run_id, "status": "stopped"}


# ---------------------------------------------------------------------------
# Workshop (task 13)
# ---------------------------------------------------------------------------


@router.post("/workshop/generate")
async def workshop_generate(req: WorkshopGenerateRequest):
    """Generate a TTS take in the workshop."""
    take_id = uuid.uuid4().hex[:12]
    job = _create_job("workshop_generate", {"take_id": take_id, "character_id": req.character_id})

    _takes[take_id] = {
        "take_id": take_id,
        "character_id": req.character_id,
        "text": req.text,
        "emotion": req.emotion,
        "hint": req.hint,
        "style_preset": req.style_preset,
        "session_id": req.session_id,
        "status": "generating",
        "pinned": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job["job_id"],
        "audio_available": False,
        "metadata_version": 1,
    }

    # In a real implementation this would call the TTS engine asynchronously.
    # For now, mark as completed immediately.
    _takes[take_id]["status"] = "ready"
    _takes[take_id]["audio_available"] = True
    job["status"] = "completed"
    job["progress"] = 1.0

    _emit_event(
        job["job_id"],
        SSEEventType.TAKE_READY,
        object_type="take",
        object_id=take_id,
        data={"message": "Take generated.", "character_id": req.character_id},
    )

    return {
        "take_id": take_id,
        "job_id": job["job_id"],
        "status": "ready",
    }


@router.post("/workshop/takes/{take_id}/pin")
async def pin_take(take_id: str):
    """Pin a workshop take for later use."""
    take = _takes.get(take_id)
    if take is None:
        raise HTTPException(status_code=404, detail=f"Take {take_id} not found.")

    take["pinned"] = True
    take["metadata_version"] = take.get("metadata_version", 1) + 1
    return {"take_id": take_id, "pinned": True, "metadata_version": take["metadata_version"]}


@router.post("/workshop/takes/{take_id}/export")
async def export_take(take_id: str):
    """Export a workshop take as a downloadable artifact (task 22)."""
    take = _takes.get(take_id)
    if take is None:
        raise HTTPException(status_code=404, detail=f"Take {take_id} not found.")

    artifact_id = uuid.uuid4().hex[:12]
    artifact = ArtifactResponse(
        artifact_id=artifact_id,
        artifact_type="take_bundle",
        download_url=f"/artifacts/{artifact_id}/download",
        expires_at=None,
        provenance_summary={
            "take_id": take_id,
            "character_id": take.get("character_id", ""),
            "text": take.get("text", ""),
            "created_at": take.get("created_at", ""),
        },
    )

    return artifact.model_dump(mode="json")


@router.post("/workshop/sessions")
async def create_workshop_session(req: WorkshopSessionRequest):
    """Create a new workshop session."""
    session_id = uuid.uuid4().hex[:12]
    _workshop_sessions[session_id] = {
        "session_id": session_id,
        "name": req.name or f"session-{session_id}",
        "character_id": req.character_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "takes": [],
    }
    return {
        "session_id": session_id,
        "name": _workshop_sessions[session_id]["name"],
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Evaluation (task 13)
# ---------------------------------------------------------------------------


@router.post("/eval/sessions")
async def create_eval_session(req: EvalSessionRequest):
    """Create a new evaluation session."""
    session_id = uuid.uuid4().hex[:12]
    _eval_sessions[session_id] = {
        "session_id": session_id,
        "name": req.name or f"eval-{session_id}",
        "eval_type": req.eval_type,
        "dataset_id": req.dataset_id,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assignments": [],
    }
    return {
        "session_id": session_id,
        "name": _eval_sessions[session_id]["name"],
        "eval_type": req.eval_type,
        "status": "active",
    }


@router.get("/eval/assignments/{assignment_id}")
async def get_eval_assignment(assignment_id: str):
    """Get an evaluation assignment."""
    assignment = _eval_assignments.get(assignment_id)
    if assignment is None:
        raise HTTPException(status_code=404, detail=f"Assignment {assignment_id} not found.")
    return assignment


@router.post("/eval/assignments/{assignment_id}/submit")
async def submit_eval_assignment(assignment_id: str, req: EvalSubmitRequest):
    """Submit a rating for an evaluation assignment."""
    assignment = _eval_assignments.get(assignment_id)
    if assignment is None:
        raise HTTPException(status_code=404, detail=f"Assignment {assignment_id} not found.")

    current_version = assignment.get("metadata_version", 1)
    if req.expected_version != current_version:
        raise_conflict(
            ConflictType.STALE_VERSION,
            f"Expected version {req.expected_version}, current is {current_version}.",
            current_version=current_version,
        )

    if assignment.get("status") == "submitted":
        raise_conflict(
            ConflictType.ALREADY_SUBMITTED,
            f"Assignment {assignment_id} has already been submitted.",
        )

    assignment["rating"] = req.rating
    assignment["notes"] = req.notes
    assignment["status"] = "submitted"
    assignment["metadata_version"] = current_version + 1
    assignment["submitted_at"] = datetime.now(timezone.utc).isoformat()

    return {
        "assignment_id": assignment_id,
        "status": "submitted",
        "metadata_version": assignment["metadata_version"],
    }
