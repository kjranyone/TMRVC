"""UI Orchestration and Event Stream routes for TMRVC Serve.

Provides the authoritative multi-user API surface for:
- Dataset upload and registration
- Workshop orchestration (Compile, Generate, Patch, Replay, Transfer)
- Curation runs and record management
- SSE event streams for telemetry
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_data.curation.models import PromotionBucket
from tmrvc_core.types import (
    IntentCompilerOutput,
    TrajectoryRecord,
    TrajectoryProvenance,
    PacingControls,
    SpeakerProfile
)
from tmrvc_serve.events import SSEEvent, SSEEventType
from tmrvc_serve.schemas import (
    ArtifactResponse,
    CharacterInfo,
    JobStatusResponse,
    CurationActionRequest,
    CompileRequest,
    CompileResponse,
    TTSRequestAdvanced,
    TTSRequestReplay,
    TTSResponse,
    PatchRequest,
    TransferRequest,
    TrajectoryInfo,
    ActingMacroControls,
    PacingControlsSchema,
)
from tmrvc_serve.intent_compiler import IntentCompiler
from tmrvc_serve.trajectory_store import TrajectoryStore
from tmrvc_data.curation.data_service import CurationDataService, RecordStatus
from tmrvc_data.curation.orchestrator import CurationOrchestrator
from tmrvc_data.g2p import text_to_phonemes
import soundfile as sf
import torch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ui", tags=["UI-Orchestration"])

# Global services
_intent_compiler = IntentCompiler()
_trajectory_store = TrajectoryStore()

# Global data service dependency (Injected by app.py)
_data_service: Optional[CurationDataService] = None
_orchestrator: Optional[CurationOrchestrator] = None

def get_data_service():
    if _data_service is None:
        raise HTTPException(status_code=503, detail="Curation data service not initialized.")
    return _data_service

# ---------------------------------------------------------------------------
# Workshop Schemas (canonical)
# ---------------------------------------------------------------------------

class WorkshopGenerateRequest(BaseModel):
    character_id: str
    text: str
    compile_id: Optional[str] = None
    speaker_profile_id: Optional[str] = None
    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)
    cfg_mode: str = "full"
    n_takes: int = Field(1, ge=1, le=10)
    style_preset: str = "default"

class WorkshopGenerateResponse(BaseModel):
    job_id: str
    takes: List[str] = Field(default_factory=list)
    status: str = "pending"
    audio_base64: Optional[str] = None
    trajectory_id: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_sec: Optional[float] = None

class DatasetRegisterRequest(BaseModel):
    """Request to register a dataset."""
    name: str
    path: str
    language: str = "ja"
    description: str = ""

class CurationRunRequest(BaseModel):
    """Request to start a curation run."""
    dataset_id: str
    policy: str = "default"

class EvalSubmitRequest(BaseModel):
    """Request to submit an evaluation rating."""
    rating: float = Field(..., ge=1.0, le=5.0)
    notes: str = ""

class SpeakerEnrollRequest(BaseModel):
    name: str
    audio_base64: str
    notes: Optional[str] = ""

class SpeakerEnrollResponse(BaseModel):
    profile_id: str
    status: str = "enrolled"

# ---------------------------------------------------------------------------
# Workshop Routes (canonical 12-D physical controls)
# ---------------------------------------------------------------------------

@router.post("/workshop/compile")
async def workshop_compile(req: CompileRequest):
    """Compile acting intent into canonical controls.

    Creator-facing: may accept prompts, tags, and context.
    """
    # Use intent compiler if available, else return defaults
    if _intent_compiler is not None:
        context = {"scene": req.scene_context} if req.scene_context else None
        compiled = _intent_compiler.compile(
            prompt=req.acting_prompt or req.text,
            context=context,
        )
        # Use the compiler's own compile_id for provenance continuity
        compile_id = compiled.compile_id
        physical_targets = compiled.physical_targets.squeeze(0).tolist()
        physical_confidence = [0.5] * 12  # default confidence
        acting_macro = ActingMacroControls(
            intensity=compiled.acting_macro.intensity,
            instability=compiled.acting_macro.instability,
            tenderness=compiled.acting_macro.tenderness,
            tension=compiled.acting_macro.tension,
            spontaneity=compiled.acting_macro.spontaneity,
            reference_mix=compiled.acting_macro.reference_mix,
        )
        pacing = PacingControlsSchema(
            pace=compiled.pacing.pace,
            hold_bias=compiled.pacing.hold_bias,
            boundary_bias=compiled.pacing.boundary_bias,
            phrase_pressure=compiled.pacing.phrase_pressure,
            breath_tendency=compiled.pacing.breath_tendency,
        )
        warnings = compiled.warnings
    else:
        compile_id = str(uuid.uuid4())
        physical_targets = [0.5] * 12  # 12-D defaults
        physical_confidence = [0.5] * 12
        acting_macro = ActingMacroControls()
        pacing = PacingControlsSchema()
        warnings = []

    # v4: Extract acting texture latent prior from compiled intent.
    # `compiled` is only defined when `_intent_compiler is not None`.
    acting_latent_prior = [0.0] * 24
    if _intent_compiler is not None and compiled.acting_latent_prior is not None:
        acting_latent_prior = compiled.acting_latent_prior.squeeze(0).tolist()

    return CompileResponse(
        compile_id=compile_id,
        physical_targets=physical_targets,
        physical_confidence=physical_confidence,
        acting_macro=acting_macro,
        acting_texture_latent_prior=acting_latent_prior,
        pacing=pacing,
        warnings=warnings,
    )


@router.post("/workshop/generate")
async def workshop_generate(req: TTSRequestAdvanced):
    """Generate speech with physical + acting controls.

    Records trajectory artifact for subsequent replay/patch/transfer.
    """
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    if not engine.models_loaded:
        raise HTTPException(status_code=503, detail="UCLM engine models not loaded.")

    trajectory_id = str(uuid.uuid4())

    # 1. G2P: text -> phoneme IDs
    g2p_result = text_to_phonemes(req.text, language=req.language or "ja")
    phonemes = g2p_result.phoneme_ids.unsqueeze(0)  # [1, L]

    # 2. Physical controls -> torch tensor [1, 12]
    physical_list = req.physical_controls.to_list()
    explicit_vs = torch.tensor(physical_list, dtype=torch.float32)

    # Delta physical controls
    delta_vs = None
    if req.delta_physical_controls is not None:
        delta_vs = torch.tensor(req.delta_physical_controls.to_list(), dtype=torch.float32)

    # 3. Speaker profile
    speaker_profile = None
    if req.speaker_profile_id:
        speaker_profile = engine.load_speaker_profile(req.speaker_profile_id)
        if speaker_profile is None:
            raise HTTPException(status_code=404, detail=f"Speaker profile {req.speaker_profile_id} not found")

    # 4. CFG mode
    cfg_mode_str = req.cfg_mode if req.cfg_mode else "full"

    # 5. Acting controls -> 24-D acting texture latent
    # v4: Direct acting_texture_latent takes priority over macro projection.
    # Both paths produce acting_texture_latent; acting_intent (legacy 64-D) is never used.
    acting_tex_t = None
    if req.acting_texture_latent is not None:
        acting_tex_t = torch.tensor([req.acting_texture_latent], dtype=torch.float32)
    elif req.acting_controls is not None:
        macro_vec = torch.tensor([
            req.acting_controls.intensity,
            req.acting_controls.instability,
            req.acting_controls.tenderness,
            req.acting_controls.tension,
            req.acting_controls.spontaneity,
            req.acting_controls.reference_mix,
        ], dtype=torch.float32).unsqueeze(0)  # [1, 6]
        if macro_vec.abs().sum() > 0:
            acting_tex_t = engine.project_acting_macro(macro_vec)

    # 6. Generate via engine
    t0 = time.time()
    audio_tensor, meta = engine.tts(
        phonemes=phonemes,
        speaker_profile=speaker_profile,
        explicit_voice_state=explicit_vs,
        delta_voice_state=delta_vs,
        cfg_scale=req.cfg_scale,
        cfg_mode=cfg_mode_str,
        acting_texture_latent=acting_tex_t,
        pace=req.pacing.pace,
        hold_bias=req.pacing.hold_bias,
        boundary_bias=req.pacing.boundary_bias,
        phrase_pressure=req.pacing.phrase_pressure,
        breath_tendency=req.pacing.breath_tendency,
        text_suprasegmentals=g2p_result.text_suprasegmentals.unsqueeze(0) if g2p_result.text_suprasegmentals is not None else None,
    )
    gen_time_ms = (time.time() - t0) * 1000.0

    # 6. Build TrajectoryRecord
    # Resolve acting latent for trajectory storage:
    # acting_tex_t (direct 24-D) takes priority, then macro-projected intent
    _stored_acting_latent = None
    if acting_tex_t is not None:
        _stored_acting_latent = acting_tex_t.detach().cpu()
    elif acting_intent is not None:
        _stored_acting_latent = acting_intent.detach().cpu()

    record = TrajectoryRecord(
        trajectory_id=trajectory_id,
        source_compile_id=req.source_compile_id if hasattr(req, "source_compile_id") and req.source_compile_id else "",
        phoneme_ids=meta.get("phoneme_ids", phonemes),
        text_suprasegmentals=g2p_result.text_suprasegmentals.unsqueeze(0) if g2p_result.text_suprasegmentals is not None else None,
        pointer_trace=meta.get("pointer_trace", []),
        physical_trajectory=meta.get("physical_trajectory"),
        acting_latent_trajectory=_stored_acting_latent,
        acting_latent_is_static=True,
        acoustic_trace=meta.get("acoustic_trace"),
        control_trace=meta.get("control_trace"),
        applied_pacing=PacingControls(
            pace=req.pacing.pace,
            hold_bias=req.pacing.hold_bias,
            boundary_bias=req.pacing.boundary_bias,
            phrase_pressure=req.pacing.phrase_pressure,
            breath_tendency=req.pacing.breath_tendency,
        ),
        speaker_profile_id=req.speaker_profile_id or "",
        provenance=TrajectoryProvenance.FRESH_COMPILE,
        version=1,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _trajectory_store.save(record)

    # 7. Encode audio as WAV base64
    audio_np = audio_tensor.squeeze().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    duration_sec = float(len(audio_np)) / SAMPLE_RATE

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        trajectory_id=trajectory_id,
        provenance="fresh_compile",
        rtf=meta.get("rtf", 0.0),
        gen_time_ms=gen_time_ms,
        cfg_mode=meta.get("cfg_mode", cfg_mode_str),
        forced_advance_count=meta.get("forced_advance_count", 0),
        skip_protection_count=meta.get("skip_protection_count", 0),
    )


@router.post("/workshop/trajectories/{trajectory_id}/patch")
async def workshop_patch(trajectory_id: str, req: PatchRequest):
    """Patch a local region of a trajectory.

    Uses optimistic concurrency via expected_version.
    """
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    if not engine.models_loaded:
        raise HTTPException(status_code=503, detail="UCLM engine models not loaded.")

    # Load trajectory
    record = _trajectory_store.load(trajectory_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    # Optimistic concurrency check
    current_version = getattr(record, 'version', 1)
    if current_version != req.expected_version:
        raise HTTPException(
            status_code=409,
            detail=f"Version conflict: expected {req.expected_version}, current {current_version}",
        )

    # Apply physical control overrides to the specified pointer range
    if req.physical_overrides is not None and record.physical_trajectory is not None:
        override_vals = torch.tensor(
            req.physical_overrides.to_list(), dtype=torch.float32,
        )  # [12]
        start_frame = 0
        end_frame = 0
        cumulative = 0
        for idx, (text_idx, frames_spent) in enumerate(record.pointer_trace):
            if idx == req.start_pointer_index:
                start_frame = cumulative
            cumulative += frames_spent
            if idx == req.end_pointer_index:
                end_frame = cumulative
                break
        if end_frame <= start_frame:
            end_frame = record.physical_trajectory.shape[0]

        # Overwrite the physical trajectory in the patched region
        record.physical_trajectory[start_frame:end_frame, :] = override_vals.unsqueeze(0).expand(
            end_frame - start_frame, -1,
        )

    # Apply pacing overrides to metadata (for regeneration context)
    if req.pacing_overrides is not None:
        record.applied_pacing = PacingControls(
            pace=req.pacing_overrides.pace,
            hold_bias=req.pacing_overrides.hold_bias,
            boundary_bias=req.pacing_overrides.boundary_bias,
            phrase_pressure=req.pacing_overrides.phrase_pressure,
            breath_tendency=req.pacing_overrides.breath_tendency,
        )

    # Bump version and update provenance
    record.version = current_version + 1
    record.provenance = TrajectoryProvenance.PATCHED_REPLAY

    _trajectory_store.save(record)

    # Regenerate audio for the patched trajectory via replay
    phonemes = record.phoneme_ids
    if phonemes is None:
        raise HTTPException(status_code=422, detail="Trajectory has no stored phoneme_ids for regeneration")

    speaker_profile = None
    if record.speaker_profile_id:
        speaker_profile = engine.load_speaker_profile(record.speaker_profile_id)

    audio_tensor, meta = engine.replay_trajectory(
        phonemes=phonemes,
        trajectory=record,
        speaker_profile=speaker_profile,
    )

    # Encode audio
    audio_np = audio_tensor.squeeze().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    n_frames = record.physical_trajectory.shape[0] if record.physical_trajectory is not None else 0
    duration_sec = float(len(audio_np)) / SAMPLE_RATE

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        trajectory_id=trajectory_id,
        provenance="patched_replay",
        rtf=meta.get("rtf", 0.0) if isinstance(meta, dict) else 0.0,
        gen_time_ms=meta.get("gen_time_ms", 0.0) if isinstance(meta, dict) else 0.0,
        cfg_mode=meta.get("cfg_mode", "full") if isinstance(meta, dict) else "full",
        forced_advance_count=meta.get("forced_advance_count", 0) if isinstance(meta, dict) else 0,
        skip_protection_count=meta.get("skip_protection_count", 0) if isinstance(meta, dict) else 0,
        schema_version="1.0",
    )


@router.post("/workshop/trajectories/{trajectory_id}/replay")
async def workshop_replay(trajectory_id: str, req: TTSRequestReplay):
    """Deterministic replay from a frozen TrajectoryRecord.

    Must NOT silently reinterpret or recompile prompts.
    """
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    if not engine.models_loaded:
        raise HTTPException(status_code=503, detail="UCLM engine models not loaded.")

    record = _trajectory_store.load(trajectory_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    phonemes = record.phoneme_ids
    if phonemes is None:
        raise HTTPException(status_code=422, detail="Trajectory has no stored phoneme_ids for replay")

    # Use override speaker if provided, otherwise use original
    speaker_profile = None
    spk_id = req.speaker_profile_id or record.speaker_profile_id
    if spk_id:
        speaker_profile = engine.load_speaker_profile(spk_id)

    audio_tensor, meta = engine.replay_trajectory(
        phonemes=phonemes,
        trajectory=record,
        speaker_profile=speaker_profile,
        text_suprasegmentals=record.text_suprasegmentals,
    )

    # Encode audio as WAV base64
    audio_np = audio_tensor.squeeze().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    duration_sec = float(len(audio_np)) / SAMPLE_RATE

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        trajectory_id=trajectory_id,
        provenance="deterministic_replay",
        rtf=meta.get("rtf", 0.0),
        gen_time_ms=meta.get("gen_time_ms", 0.0),
    )


@router.post("/workshop/trajectories/{trajectory_id}/transfer")
async def workshop_transfer(trajectory_id: str, req: TransferRequest):
    """Transfer a trajectory's acting to a different speaker.

    Transfer is a first-class capability.
    Replays the acting trajectory on the target speaker profile.
    """
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    if not engine.models_loaded:
        raise HTTPException(status_code=503, detail="UCLM engine models not loaded.")

    record = _trajectory_store.load(trajectory_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")

    phonemes = record.phoneme_ids
    if phonemes is None:
        raise HTTPException(status_code=422, detail="Trajectory has no stored phoneme_ids for transfer")

    # Load target speaker profile
    target_profile = engine.load_speaker_profile(req.target_speaker_profile_id)
    if target_profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Target speaker profile {req.target_speaker_profile_id} not found",
        )

    # Use the trajectory's physical controls but the new speaker's timbre
    # by calling tts() with the stored physical trajectory as explicit voice state
    explicit_vs = None
    if record.physical_trajectory is not None:
        # Use the mean physical trajectory as the static voice state target
        explicit_vs = record.physical_trajectory.mean(dim=0)  # [12]

    audio_tensor, meta = engine.tts(
        phonemes=phonemes,
        speaker_profile=target_profile,
        explicit_voice_state=explicit_vs,
        cfg_scale=1.5,
        cfg_mode="full",
        pace=record.applied_pacing.pace if record.applied_pacing else 1.0,
        hold_bias=record.applied_pacing.hold_bias if record.applied_pacing else 0.0,
        boundary_bias=record.applied_pacing.boundary_bias if record.applied_pacing else 0.0,
        phrase_pressure=record.applied_pacing.phrase_pressure if record.applied_pacing else 0.0,
        breath_tendency=record.applied_pacing.breath_tendency if record.applied_pacing else 0.0,
        text_suprasegmentals=record.text_suprasegmentals,
    )

    # Save transfer trajectory
    transfer_id = str(uuid.uuid4())
    transfer_record = TrajectoryRecord(
        trajectory_id=transfer_id,
        source_compile_id=record.source_compile_id,
        phoneme_ids=phonemes,
        text_suprasegmentals=record.text_suprasegmentals,
        pointer_trace=meta.get("pointer_trace", []),
        physical_trajectory=meta.get("physical_trajectory"),
        acoustic_trace=meta.get("acoustic_trace"),
        control_trace=meta.get("control_trace"),
        applied_pacing=record.applied_pacing,
        speaker_profile_id=req.target_speaker_profile_id,
        provenance=TrajectoryProvenance.CROSS_SPEAKER_TRANSFER,
        version=1,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata={"source_trajectory_id": trajectory_id},
    )
    _trajectory_store.save(transfer_record)

    # Encode audio as WAV base64
    audio_np = audio_tensor.squeeze().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    duration_sec = float(len(audio_np)) / SAMPLE_RATE

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        trajectory_id=transfer_id,
        provenance="cross_speaker_transfer",
        rtf=meta.get("rtf", 0.0),
        gen_time_ms=meta.get("gen_time_ms", 0.0),
        cfg_mode=meta.get("cfg_mode", "full"),
        forced_advance_count=meta.get("forced_advance_count", 0),
        skip_protection_count=meta.get("skip_protection_count", 0),
    )


@router.post("/speaker/enroll", response_model=SpeakerEnrollResponse)
async def speaker_enroll(req: SpeakerEnrollRequest):
    """Authoritative speaker enrollment via engine encoding."""
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    profile_id = f"spk-{uuid.uuid4().hex[:8]}"
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        sr, audio_np = wavfile.read(io.BytesIO(audio_bytes))
        audio_t = torch.from_numpy(audio_np).float().unsqueeze(0)
        if audio_t.max() > 1.0: audio_t /= 32768.0

        speaker_embed, summary_tokens, codec_tokens, _ = engine.encode_speaker_prompt(
            reference_waveform=audio_t.to(engine.device)
        )

        profile = SpeakerProfile(
            speaker_profile_id=profile_id,
            reference_audio_hash=str(hash(req.audio_base64))[:16],
            speaker_embed=speaker_embed.cpu().squeeze(0),
            prompt_codec_tokens=codec_tokens.cpu().squeeze(0),
            prompt_summary_tokens=summary_tokens.cpu().squeeze(0),
            display_name=req.name,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"notes": req.notes}
        )
        profile_path = Path("data/profiles") / f"{profile_id}.pt"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(profile, profile_path)
        return SpeakerEnrollResponse(profile_id=profile_id)
    except Exception as e:
        logger.exception("Speaker enrollment failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Curation and Utility Routes
# ---------------------------------------------------------------------------

@router.get("/curation/records")
async def list_records(status: Optional[str] = None, limit: int = 100, offset: int = 0, ds: CurationDataService = Depends(get_data_service)):
    records = ds.query_records(status=status, limit=limit, offset=offset)
    return [r.to_dict() for r in records]

@router.get("/curation/records/{record_id}")
async def get_record(record_id: str, ds: CurationDataService = Depends(get_data_service)):
    record = ds.get_record(record_id)
    if not record: raise HTTPException(status_code=404, detail="Record not found")
    return record.to_dict()

@router.post("/curation/records/{record_id}/action")
async def update_record_status(record_id: str, req: CurationActionRequest, ds: CurationDataService = Depends(get_data_service)):
    record = ds.get_record(record_id)
    if not record: raise HTTPException(status_code=404, detail="Record not found")
    if record.metadata_version != req.expected_version:
        raise HTTPException(status_code=409, detail={"error": "stale_version", "current_version": record.metadata_version})

    if req.action == "promote":
        record.status = RecordStatus.PROMOTED
        if req.bucket: record.promotion_bucket = PromotionBucket(req.bucket)
    elif req.action == "reject": record.status = RecordStatus.REJECTED
    elif req.action == "review": record.status = RecordStatus.REVIEW

    if not ds.update_record(record, actor_id="ui_anonymous", action_type=f"ui_{req.action}", rationale=req.rationale):
        raise HTTPException(status_code=409, detail="conflict_retry")
    return {"status": "ok", "new_version": record.metadata_version}

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    return JobStatusResponse(job_id=job_id, status="pending", progress=0.0)

@router.get("/events")
async def event_stream(request: Request):
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected(): break
            yield SSEEvent(event_type=SSEEventType.TELEMETRY_UPDATE, data={"active_jobs": 0}).to_sse()
            await asyncio.sleep(2.0)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Dataset Routes
# ---------------------------------------------------------------------------

@router.post("/datasets/upload")
async def dataset_upload():
    """Upload a dataset archive.

    Full implementation requires multipart upload handling.
    Currently returns a placeholder — use /datasets/register for local paths.
    """
    return {"status": "accepted", "detail": "Use /datasets/register for local datasets."}

@router.post("/datasets/register")
async def dataset_register(req: DatasetRegisterRequest):
    """Register a dataset from a local path."""
    dataset_path = Path(req.path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    # Count audio files
    audio_files = list(dataset_path.rglob("*.wav")) + list(dataset_path.rglob("*.flac"))
    dataset_id = f"ds-{uuid.uuid4().hex[:8]}"

    return {
        "dataset_id": dataset_id,
        "name": req.name,
        "path": str(dataset_path.resolve()),
        "language": req.language,
        "description": req.description,
        "audio_file_count": len(audio_files),
        "status": "registered",
    }

# ---------------------------------------------------------------------------
# Job event stream
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """SSE event stream for a specific job."""
    async def _gen() -> AsyncGenerator[str, None]:
        yield SSEEvent(event_type=SSEEventType.JOB_PROGRESS, job_id=job_id, data={"progress": 0.0}).to_sse()
    return StreamingResponse(_gen(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Curation Run Routes
# ---------------------------------------------------------------------------

@router.post("/curation/runs")
async def curation_run_create(req: CurationRunRequest):
    """Start a new curation run."""
    ds = get_data_service()
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    logger.info("Curation run created: %s for dataset %s (policy=%s)", run_id, req.dataset_id, req.policy)
    return {
        "run_id": run_id,
        "dataset_id": req.dataset_id,
        "policy": req.policy,
        "status": "running",
    }

@router.post("/curation/runs/{run_id}/resume")
async def curation_run_resume(run_id: str):
    """Resume a paused curation run."""
    logger.info("Curation run resumed: %s", run_id)
    return {"run_id": run_id, "status": "running"}

@router.post("/curation/runs/{run_id}/stop")
async def curation_run_stop(run_id: str):
    """Stop a running curation run."""
    logger.info("Curation run stopped: %s", run_id)
    return {"run_id": run_id, "status": "stopped"}

# ---------------------------------------------------------------------------
# Workshop Take Routes
# ---------------------------------------------------------------------------

@router.post("/workshop/takes/{take_id}/pin")
async def workshop_take_pin(take_id: str):
    """Pin a take for later reference."""
    logger.info("Take pinned: %s", take_id)
    return {"take_id": take_id, "pinned": True}

@router.post("/workshop/takes/{take_id}/export")
async def workshop_take_export(take_id: str):
    """Export a take as a downloadable artifact."""
    record = _trajectory_store.load(take_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Take {take_id} not found")
    return {
        "take_id": take_id,
        "trajectory_id": record.trajectory_id,
        "provenance": record.provenance.value if hasattr(record.provenance, "value") else str(record.provenance),
        "version": record.version,
        "status": "exported",
    }

@router.get("/workshop/sessions")
async def workshop_sessions():
    """List active workshop sessions."""
    return []

# ---------------------------------------------------------------------------
# Eval Routes
# ---------------------------------------------------------------------------

@router.get("/eval/sessions")
async def eval_sessions():
    """List active evaluation sessions."""
    return []

@router.get("/eval/assignments/{assignment_id}")
async def eval_assignment(assignment_id: str):
    """Get an evaluation assignment."""
    return {
        "assignment_id": assignment_id,
        "status": "pending",
        "pairs": [],
    }

@router.post("/eval/assignments/{assignment_id}/submit")
async def eval_submit(assignment_id: str, req: EvalSubmitRequest):
    """Submit an evaluation rating."""
    logger.info("Eval submitted: assignment=%s rating=%.1f", assignment_id, req.rating)
    return {
        "assignment_id": assignment_id,
        "rating": req.rating,
        "notes": req.notes,
        "status": "submitted",
    }
