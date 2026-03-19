
"""Trajectory Store adapter for TMRVC Serve (Worker 04)."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from tmrvc_serve.trajectory_service import TrajectoryService

class TrajectoryStore(TrajectoryService):
    """Wrapper for TrajectoryService with default production storage path."""

    def __init__(self, storage_dir: str | Path | None = None, **kwargs):
        if storage_dir is None and "root_dir" not in kwargs:
            storage_dir = os.getenv("TMRVC_TRAJECTORY_DIR", "data/trajectories")
        super().__init__(storage_dir, **kwargs)

    def save_with_version_check(self, record, expected_version: int) -> bool:
        """Save a trajectory record with optimistic concurrency check.

        Uses a file lock to prevent TOCTOU race between the version check
        (load) and the subsequent save.  The lock file is per-trajectory so
        concurrent saves to *different* trajectories are not serialised.

        Returns True if save succeeded, False if version conflict.
        """
        lock_path = self.storage_dir / f".{record.trajectory_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                existing = self.load(record.trajectory_id)
                if existing is not None and getattr(existing, 'version', 1) != expected_version:
                    return False

                # Write to a temp file in the same directory, then atomically
                # rename so that readers never see a partially-written file.
                data = self._record_to_dict(record)
                target = self.storage_dir / f"{record.trajectory_id}.json"
                fd, tmp_path = tempfile.mkstemp(
                    dir=self.storage_dir, suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                        json.dump(data, tmp_f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, target)
                except BaseException:
                    # Clean up temp file on failure
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise

                # Update in-memory cache
                self._cache[record.trajectory_id] = record
                return True
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
