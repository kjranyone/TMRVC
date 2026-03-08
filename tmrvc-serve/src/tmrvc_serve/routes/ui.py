"""WebUI-specific data APIs for Curation Auditor and Dataset Manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tmrvc_data.curation.models import RecordStatus, PromotionBucket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])


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
        raise HTTPException(status_code=409, detail=str(e))


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
        # Audit log would happen here if integrated
        return {"status": "ok", "metadata_version": record.metadata_version}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
