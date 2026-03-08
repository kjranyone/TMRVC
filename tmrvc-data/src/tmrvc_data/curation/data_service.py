"""SQLite-backed Curation Data Service (Worker 07).

Provides the operational database for the curation system, supporting:
- Optimistic locking (metadata_version)
- Batch updates
- High-speed querying for Gradio UI
- Auditing
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from tmrvc_data.curation.models import CurationRecord, RecordStatus, PromotionBucket

logger = logging.getLogger(__name__)


class CurationDataService:
    """Manages curation records in a SQLite database with optimistic locking."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the SQLite schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS curation_records (
                    record_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    speaker_cluster TEXT,
                    metadata_version INTEGER NOT NULL DEFAULT 1,
                    data TEXT NOT NULL
                )
                """
            )
            # Create indexes for high-speed queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON curation_records(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker ON curation_records(speaker_cluster)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON curation_records(quality_score)")
            
            # Audit log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    rationale TEXT,
                    FOREIGN KEY(record_id) REFERENCES curation_records(record_id)
                )
                """
            )

    def insert_or_replace(self, record: CurationRecord) -> None:
        """Insert a new record or completely replace an existing one (ignores version)."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO curation_records 
                (record_id, status, quality_score, speaker_cluster, metadata_version, data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.status.value,
                    record.quality_score,
                    record.speaker_cluster,
                    record.metadata_version,
                    json.dumps(record.to_dict())
                )
            )

    def get_record(self, record_id: str) -> Optional[CurationRecord]:
        """Fetch a single record by ID."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT data FROM curation_records WHERE record_id = ?", (record_id,)).fetchone()
            if row:
                return CurationRecord.from_dict(json.loads(row["data"]))
            return None

    def update_record(self, record: CurationRecord, actor_id: str, action_type: str, rationale: str = "") -> bool:
        """Update an existing record using optimistic locking.

        Returns:
            bool: True if the update succeeded, False if the version was stale (conflict).
        """
        # The expected version is the one currently in the record object
        expected_version = record.metadata_version
        
        # Increment version for the update
        record.metadata_version += 1
        
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE curation_records 
                SET status = ?, quality_score = ?, speaker_cluster = ?, metadata_version = ?, data = ?
                WHERE record_id = ? AND metadata_version = ?
                """,
                (
                    record.status.value,
                    record.quality_score,
                    record.speaker_cluster,
                    record.metadata_version,
                    json.dumps(record.to_dict()),
                    record.record_id,
                    expected_version
                )
            )
            
            if cursor.rowcount == 0:
                # Revert version increment since update failed
                record.metadata_version = expected_version
                return False
                
            # Log the action
            conn.execute(
                """
                INSERT INTO audit_logs (record_id, actor_id, action_type, rationale)
                VALUES (?, ?, ?, ?)
                """,
                (record.record_id, actor_id, action_type, rationale)
            )
            return True

    def query_records(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CurationRecord]:
        """Fetch records with basic filtering."""
        query = "SELECT data FROM curation_records"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [CurationRecord.from_dict(json.loads(row["data"])) for row in rows]
