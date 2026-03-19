
"""Trajectory Service for Programmable Expressive Speech (Worker 04).

Handles persistence, replay, and local patching of TrajectoryRecords.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime, timezone

import torch
import numpy as np

from tmrvc_core.types import TrajectoryRecord, PacingControls

logger = logging.getLogger(__name__)

class TrajectoryService:
    """Manages the lifecycle of speech performance trajectories."""

    def __init__(self, storage_dir: str | Path | None = None, **kwargs):
        # SOTA: Handle legacy root_dir from some tests
        s_dir = storage_dir or kwargs.get("root_dir")
        if s_dir is None:
            raise ValueError("storage_dir (or root_dir) must be provided")
            
        self.storage_dir = Path(s_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, TrajectoryRecord] = {}

    def save_trajectory(self, record: TrajectoryRecord) -> str:
        """Persist a trajectory to disk."""
        tid = record.trajectory_id or str(uuid.uuid4())
        record.trajectory_id = tid
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()
        
        path = self.storage_dir / f"{tid}.json"
        
        # We need to handle tensors for JSON serialization
        data = self._record_to_dict(record)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        self._cache[tid] = record
        return tid

    def save(self, record: TrajectoryRecord) -> str:
        """Alias for save_trajectory (test compatibility)."""
        return self.save_trajectory(record)

    def load_trajectory(self, tid: str) -> Optional[TrajectoryRecord]:
        """Load a trajectory from disk or cache."""
        if tid in self._cache:
            return self._cache[tid]
            
        path = self.storage_dir / f"{tid}.json"
        if not path.exists():
            return None
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        record = self._dict_to_record(data)
        self._cache[tid] = record
        return record

    def load(self, tid: str) -> Optional[TrajectoryRecord]:
        """Alias for load_trajectory (test compatibility)."""
        return self.load_trajectory(tid)

    def patch_trajectory(
        self, 
        base_tid: str, 
        start_frame: int, 
        end_frame: int, 
        patch_record: TrajectoryRecord
    ) -> TrajectoryRecord:
        """SOTA: Create a new trajectory by patching a region of an existing one.
        
        This implements the 'Edit Locality' requirement from Worker 06.
        """
        base = self.load_trajectory(base_tid)
        if not base:
            raise ValueError(f"Base trajectory {base_tid} not found")

        # 1. Clone base
        new_tid = f"patch_{base_tid}_{uuid.uuid4().hex[:8]}"
        
        # 2. Stitch traces
        # Note: This requires frame-aligned traces in Stream A/B
        new_acoustic = base.acoustic_trace.clone()
        new_control = base.control_trace.clone()
        new_vs = base.physical_trajectory.clone()
        
        # Replace the target region
        patch_len = end_frame - start_frame
        actual_patch_len = patch_record.acoustic_trace.shape[-1]
        
        # If patch length mismatches, we use the actual patch length
        # SOTA: In a real editor, we'd handle time-stretching, 
        # but for v0 we do direct substitution.
        limit = min(actual_patch_len, patch_len)
        
        new_acoustic[:, start_frame : start_frame + limit] = patch_record.acoustic_trace[:, :limit]
        new_control[:, start_frame : start_frame + limit] = patch_record.control_trace[:, :limit]
        new_vs[start_frame : start_frame + limit, :] = patch_record.physical_trajectory[:limit, :]
        
        # 3. Update pointer trace (approximate for v0)
        # TODO: Refine pointer trace stitching for accurate boundary replay
        
        patched_record = TrajectoryRecord(
            trajectory_id=new_tid,
            source_compile_id=base.source_compile_id,
            phoneme_ids=base.phoneme_ids,
            text_suprasegmentals=base.text_suprasegmentals,
            acoustic_trace=new_acoustic,
            control_trace=new_control,
            physical_trajectory=new_vs,
            applied_pacing=base.applied_pacing,
            speaker_profile_id=base.speaker_profile_id,
            uclm_version=base.uclm_version,
            metadata={
                "patch_source": base_tid,
                "patch_range": [start_frame, end_frame],
                "patch_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        self.save_trajectory(patched_record)
        return patched_record

    def _record_to_dict(self, record: TrajectoryRecord) -> dict:
        """Serialize TrajectoryRecord to JSON-compatible dict."""
        out = {}
        for k, v in vars(record).items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().numpy().tolist()
            elif isinstance(v, PacingControls):
                out[k] = vars(v)
            else:
                out[k] = v
        return out

    def _dict_to_record(self, data: dict) -> TrajectoryRecord:
        """Deserialize JSON-compatible dict to TrajectoryRecord."""
        # Convert lists back to tensors
        tensors = ["phoneme_ids", "text_suprasegmentals", "acoustic_trace", "control_trace", "physical_trajectory"]
        for k in tensors:
            if k in data and data[k] is not None:
                # Determine dtype based on field
                dtype = torch.long if k in ("phoneme_ids", "acoustic_trace", "control_trace") else torch.float32
                data[k] = torch.tensor(data[k], dtype=dtype)
        
        if "applied_pacing" in data:
            data["applied_pacing"] = PacingControls(**data["applied_pacing"])
            
        # Handle pointer_trace which is List[Tuple[int, int]]
        if "pointer_trace" in data:
            data["pointer_trace"] = [tuple(x) for x in data["pointer_trace"]]
            
        return TrajectoryRecord(**data)
