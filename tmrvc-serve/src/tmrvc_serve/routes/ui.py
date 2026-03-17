"""UI Orchestration and Event Stream routes for TMRVC Serve (Worker 04/07).

Provides the authoritative multi-user API surface for:
- Dataset upload and registration
- Workshop orchestration (Compile, Generate, Patch, Replay)
- Curation runs and record management
- SSE event streams for telemetry
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
import time
import uuid
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import PromotionBucket
from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS
from tmrvc_core.types import (
    IntentCompilerOutput, 
    TrajectoryRecord, 
    PacingControls,
    SpeakerProfile
)
from tmrvc_serve.events import SSEEvent, SSEEventType
from tmrvc_serve.schemas import (
    ArtifactResponse, 
    CharacterInfo, 
    JobStatusResponse,
    CurationActionRequest
)
from tmrvc_serve.intent_compiler import IntentCompiler
from tmrvc_serve.trajectory_store import TrajectoryStore
from tmrvc_data.curation.data_service import CurationDataService, RecordStatus
from tmrvc_data.curation.orchestrator import CurationOrchestrator
from tmrvc_data.g2p import text_to_phonemes
import torch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ui", tags=["UI-Orchestration"])

# Global services (Worker 04)
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
# Workshop Schemas
# ---------------------------------------------------------------------------

class WorkshopCompileRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None

class WorkshopCompileResponse(BaseModel):
    compile_id: str
    warnings: List[str] = []
    pacing: dict
    explicit_voice_state: Optional[List[float]] = None

class WorkshopGenerateRequest(BaseModel):
    character_id: str
    text: str
    compile_id: Optional[str] = None
    speaker_profile_id: Optional[str] = None
    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)
    cfg_mode: str = "full"
    n_takes: int = Field(1, ge=1, le=10)

class TrajectoryReplayRequest(BaseModel):
    speaker_profile_id: Optional[str] = None
    cfg_scale: Optional[float] = None
    use_exact_tokens: bool = True

# ---------------------------------------------------------------------------
# Workshop Routes (Worker 04)
# ---------------------------------------------------------------------------

@router.post("/workshop/compile", response_model=WorkshopCompileResponse)
async def workshop_compile(req: WorkshopCompileRequest):
    """Compile a prompt into an IntentCompilerOutput."""
    intent = _intent_compiler.compile(req.prompt, req.context)
    return WorkshopCompileResponse(
        compile_id=intent.compile_id,
        warnings=intent.warnings,
        pacing=vars(intent.pacing),
        explicit_voice_state=intent.explicit_voice_state.squeeze(0).tolist() if intent.explicit_voice_state is not None else None
    )

@router.post("/workshop/trajectories/{tid}/replay")
async def trajectory_replay(tid: str, req: TrajectoryReplayRequest, request: Request):
    """SOTA: Replay a stable trajectory with bit-exact reproduction."""
    record = _trajectory_store.load_trajectory(tid)
    if not record:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    
    engine = request.app.state.uclm_engine
    if not engine:
        raise HTTPException(status_code=503, detail="UCLM Engine not initialized")
        
    audio, metrics = engine.replay(record)
    
    # Return as base64 or stream
    return {
        "audio_b64": base64.b64encode(audio.numpy().tobytes()).decode(),
        "metrics": metrics,
        "sample_rate": SAMPLE_RATE
    }

@router.post("/workshop/trajectories/{tid}/patch")
async def trajectory_patch(tid: str, req: TrajectoryPatchRequest, request: Request):
    """SOTA: Create a new trajectory by patching an existing one (Edit Locality)."""
    # Note: Full patch implementation requires start/end frames which should be in req
    # For v0 simplicity, we demonstrate the plumbing
    base_record = _trajectory_store.load_trajectory(tid)
    if not base_record:
        raise HTTPException(status_code=404, detail="Base trajectory not found")
        
    # Implement patching logic here using _trajectory_store.patch_trajectory
    # This is where 'edit_locality_score' from Worker 06 is validated.
    return {"message": "Patching demonstrated, awaiting frame range inputs", "base_tid": tid}

class WorkshopGenerateResponse(BaseModel):
    job_id: str
    takes: List[str] = Field(default_factory=list)
    status: str = "pending"
    audio_base64: Optional[str] = None
    trajectory_id: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_sec: Optional[float] = None

class SpeakerEnrollRequest(BaseModel):
    name: str
    audio_base64: str
    notes: Optional[str] = ""

class SpeakerEnrollResponse(BaseModel):
    profile_id: str
    status: str = "enrolled"

# ---------------------------------------------------------------------------
# Workshop Routes (Worker 04)
# ---------------------------------------------------------------------------

@router.post("/workshop/compile", response_model=WorkshopCompileResponse)
async def workshop_compile(req: WorkshopCompileRequest):
    """Compile natural language prompts into deterministic UCLM controls."""
    try:
        output = _intent_compiler.compile(req.prompt, req.context)
        # Store temporary compile results
        _trajectory_store.root_dir.mkdir(parents=True, exist_ok=True)
        compile_path = _trajectory_store.root_dir / f"compile_{output.compile_id}.json"
        with open(compile_path, "w") as f:
            json.dump({
                "source_prompt": output.source_prompt,
                "pacing": vars(output.pacing),
                "explicit_voice_state": output.explicit_voice_state.tolist() if output.explicit_voice_state is not None else None
            }, f)

        return WorkshopCompileResponse(
            compile_id=output.compile_id,
            warnings=output.warnings,
            pacing=vars(output.pacing),
            explicit_voice_state=output.explicit_voice_state.tolist()[0] if output.explicit_voice_state is not None else None
        )
    except Exception as e:
        logger.exception("Intent compilation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workshop/generate", response_model=WorkshopGenerateResponse)
async def workshop_generate(req: WorkshopGenerateRequest):
    """Generate audio and record the performance trajectory."""
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    trajectory_id = f"tj-{uuid.uuid4().hex[:8]}"

    try:
        pacing = PacingControls()
        evs = None
        if req.compile_id:
            compile_path = _trajectory_store.root_dir / f"compile_{req.compile_id}.json"
            if compile_path.exists():
                with open(compile_path, "r") as f:
                    c_data = json.load(f)
                    pacing = PacingControls(**c_data["pacing"])
                    if c_data["explicit_voice_state"]:
                        evs = torch.tensor(c_data["explicit_voice_state"]).float().to(engine.device)

        g2p_result = text_to_phonemes(req.text)
        phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long, device=engine.device).unsqueeze(0)
        supra_t = g2p_result.text_suprasegmentals.to(engine.device) if g2p_result.text_suprasegmentals is not None else None

        speaker_profile = engine.load_speaker_profile(req.speaker_profile_id) if req.speaker_profile_id else None

        audio_t, stats = engine.tts(
            phonemes=phonemes_t,
            speaker_profile=speaker_profile,
            text_suprasegmentals=supra_t,
            explicit_voice_state=evs,
            pace=pacing.pace,
            hold_bias=pacing.hold_bias,
            boundary_bias=pacing.boundary_bias,
            cfg_scale=req.cfg_scale,
            cfg_mode=req.cfg_mode,
            language_id=g2p_result.language_id,
        )

        record = TrajectoryRecord(
            trajectory_id=trajectory_id,
            source_compile_id=req.compile_id or "direct",
            phoneme_ids=phonemes_t.cpu(),
            text_suprasegmentals=supra_t.cpu() if supra_t is not None else None,
            pointer_trace=stats["pointer_trace"],
            voice_state_trajectory=stats["voice_state_trajectory"].cpu(),
            acoustic_trace=stats["acoustic_trace"].cpu() if stats.get("acoustic_trace") is not None else None,
            control_trace=stats["control_trace"].cpu(),
            applied_pacing=pacing,
            speaker_profile_id=req.speaker_profile_id or "default",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"text": req.text}
        )
        _trajectory_store.save(record)

        buf = io.BytesIO()
        wavfile.write(buf, engine.FRAME_SAMPLE_RATE, (audio_t.cpu().numpy() * 32767).astype(np.int16))
        
        return WorkshopGenerateResponse(
            job_id=f"wk-{uuid.uuid4().hex[:8]}",
            takes=[trajectory_id],
            trajectory_id=trajectory_id,
            status="completed",
            audio_base64=base64.b64encode(buf.getvalue()).decode(),
            sample_rate=engine.FRAME_SAMPLE_RATE,
            duration_sec=float(len(audio_t) / engine.FRAME_SAMPLE_RATE)
        )
    except Exception as e:
        logger.exception("Workshop generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workshop/trajectories/{trajectory_id}/replay", response_model=WorkshopGenerateResponse)
async def trajectory_replay(trajectory_id: str, req: TrajectoryReplayRequest):
    """Replay a specific performance trajectory deterministically."""
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    try:
        trajectory = _trajectory_store.load(trajectory_id)
        if not trajectory: raise HTTPException(status_code=404, detail="Trajectory not found")
            
        g2p_result = text_to_phonemes(trajectory.metadata.get("text", ""))
        speaker_profile = engine.load_speaker_profile(req.speaker_profile_id or trajectory.speaker_profile_id) if (req.speaker_profile_id or trajectory.speaker_profile_id) != "default" else None

        audio_t, stats = engine.replay_trajectory(
            phonemes=g2p_result.phoneme_ids.to(dtype=torch.long, device=engine.device).unsqueeze(0),
            trajectory=trajectory,
            speaker_profile=speaker_profile,
            text_suprasegmentals=g2p_result.text_suprasegmentals,
            language_id=g2p_result.language_id,
            use_exact_tokens=req.use_exact_tokens
        )

        buf = io.BytesIO()
        wavfile.write(buf, engine.FRAME_SAMPLE_RATE, (audio_t.cpu().numpy() * 32767).astype(np.int16))

        return WorkshopGenerateResponse(
            job_id=f"re-{uuid.uuid4().hex[:8]}",
            takes=[trajectory_id],
            trajectory_id=trajectory_id,
            status="completed",
            audio_base64=base64.b64encode(buf.getvalue()).decode(),
            sample_rate=engine.FRAME_SAMPLE_RATE,
            duration_sec=float(len(audio_t) / engine.FRAME_SAMPLE_RATE)
        )
    except Exception as e:
        logger.exception("Trajectory replay failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workshop/trajectories/{trajectory_id}/patch", response_model=WorkshopGenerateResponse)
async def workshop_patch(trajectory_id: str, req: TrajectoryPatchRequest):
    """Patch an existing trajectory with new control values and re-render."""
    from tmrvc_serve.app import get_engine
    engine = get_engine()
    new_trajectory_id = f"tj-patch-{uuid.uuid4().hex[:8]}"
    
    try:
        base = _trajectory_store.load(trajectory_id)
        if not base: raise HTTPException(status_code=404, detail="Trajectory not found")
            
        new_pacing = PacingControls(
            pace=req.pace if req.pace is not None else base.applied_pacing.pace,
            hold_bias=req.hold_bias if req.hold_bias is not None else base.applied_pacing.hold_bias,
            boundary_bias=req.boundary_bias if req.boundary_bias is not None else base.applied_pacing.boundary_bias,
        )
        
        local_plan = {int(idx): torch.tensor(vs_list).float().view(1, 8) for idx, vs_list in req.voice_state_overrides.items()}
        
        audio_t, stats = engine.tts(
            phonemes=base.phoneme_ids.to(engine.device),
            text_suprasegmentals=base.text_suprasegmentals.to(engine.device) if base.text_suprasegmentals is not None else None,
            speaker_profile=engine.load_speaker_profile(base.speaker_profile_id) if base.speaker_profile_id != "default" else None,
            pace=new_pacing.pace,
            hold_bias=new_pacing.hold_bias,
            boundary_bias=new_pacing.boundary_bias,
            local_prosody_plan=local_plan
        )
        
        record = TrajectoryRecord(
            trajectory_id=new_trajectory_id,
            source_compile_id=base.source_compile_id,
            phoneme_ids=base.phoneme_ids,
            text_suprasegmentals=base.text_suprasegmentals,
            pointer_trace=stats["pointer_trace"],
            voice_state_trajectory=stats["voice_state_trajectory"].cpu(),
            acoustic_trace=stats["acoustic_trace"].cpu() if stats.get("acoustic_trace") is not None else None,
            control_trace=stats["control_trace"].cpu(),
            applied_pacing=new_pacing,
            speaker_profile_id=base.speaker_profile_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"text": base.metadata.get("text", ""), "patched_from": trajectory_id}
        )
        _trajectory_store.save(record)
        
        buf = io.BytesIO()
        wavfile.write(buf, engine.FRAME_SAMPLE_RATE, (audio_t.cpu().numpy() * 32767).astype(np.int16))

        return WorkshopGenerateResponse(
            job_id=f"pt-{uuid.uuid4().hex[:8]}",
            takes=[new_trajectory_id],
            trajectory_id=new_trajectory_id,
            status="completed",
            audio_base64=base64.b64encode(buf.getvalue()).decode(),
            sample_rate=engine.FRAME_SAMPLE_RATE,
            duration_sec=float(len(audio_t) / engine.FRAME_SAMPLE_RATE)
        )
    except Exception as e:
        logger.exception("Trajectory patch failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

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
    if record.metadata_version != req.metadata_version:
        raise HTTPException(status_code=409, detail={"error": "stale_version", "current_version": record.metadata_version})

    if req.action == "promote":
        record.status = RecordStatus.PROMOTED
        if req.bucket: record.promotion_bucket = PromotionBucket(req.bucket)
    elif req.action == "reject": record.status = RecordStatus.REJECTED
    elif req.action == "review": record.status = RecordStatus.REVIEW

    if not ds.update_record(record, actor_id=req.actor_id, action_type=f"ui_{req.action}", rationale=req.rationale):
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
