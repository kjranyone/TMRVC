
"""Trajectory Store adapter for TMRVC Serve (Worker 04)."""

from __future__ import annotations

import os
from pathlib import Path
from tmrvc_serve.trajectory_service import TrajectoryService

class TrajectoryStore(TrajectoryService):
    """Wrapper for TrajectoryService with default production storage path."""
    
    def __init__(self, storage_dir: str | Path | None = None):
        if storage_dir is None:
            storage_dir = os.getenv("TMRVC_TRAJECTORY_DIR", "data/trajectories")
        super().__init__(storage_dir)
