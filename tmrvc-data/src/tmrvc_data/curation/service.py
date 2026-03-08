"""SQLite-backed Curation Data Service (Worker 07).

Provides the operational database for the curation system, supporting:
- Optimistic locking (metadata_version) via ``StaleVersionError``
- Batch create / update operations
- State-transition validation
- Full audit trail (who, when, what changed, previous value)
- WAL mode for concurrent read access
- High-speed indexing for record_id, status, speaker_cluster, quality_score
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import InvalidTransitionError, StaleVersionError
from .models import (
    CurationRecord,
    PromotionBucket,
    RecordStatus,
    VALID_TRANSITIONS,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class CurationDataService:
    """Manages curation records in a SQLite database with optimistic locking.

    Thread-safety: each public method opens its own connection using
    ``_get_conn`` so that the service can be shared across threads.
    SQLite WAL mode is enabled on first access to allow concurrent reads.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection with WAL mode enabled."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the per-thread connection (if any)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indices if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS curation_records (
                    record_id        TEXT PRIMARY KEY,
                    source_path      TEXT NOT NULL,
                    stage            INTEGER NOT NULL DEFAULT 0,
                    status           TEXT NOT NULL DEFAULT 'ingested',
                    metadata_version INTEGER NOT NULL DEFAULT 1,
                    quality_score    REAL NOT NULL DEFAULT 0.0,
                    promotion_bucket TEXT NOT NULL DEFAULT 'none',
                    curation_pass    INTEGER NOT NULL DEFAULT 0,
                    speaker_cluster  TEXT,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL,
                    data             TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status "
                "ON curation_records(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_speaker "
                "ON curation_records(speaker_cluster)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality "
                "ON curation_records(quality_score)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage "
                "ON curation_records(stage)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pass "
                "ON curation_records(curation_pass)"
            )

            # Audit log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id   TEXT NOT NULL,
                    actor_id    TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    before_state TEXT,
                    after_state  TEXT,
                    rationale   TEXT,
                    FOREIGN KEY(record_id)
                        REFERENCES curation_records(record_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_record "
                "ON audit_log(record_id)"
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_record(self, record: CurationRecord) -> None:
        """Insert a new record.  Raises if *record_id* already exists."""
        now = _utc_now_iso()
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO curation_records
                    (record_id, source_path, stage, status,
                     metadata_version, quality_score, promotion_bucket,
                     curation_pass, speaker_cluster, created_at, updated_at,
                     data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.source_path,
                    0,
                    record.status.value,
                    record.metadata_version,
                    record.quality_score,
                    record.promotion_bucket.value,
                    record.pass_index,
                    record.speaker_cluster,
                    now,
                    now,
                    json.dumps(record.to_dict()),
                ),
            )

    def get_record(self, record_id: str) -> Optional[CurationRecord]:
        """Fetch a single record by ID, or ``None`` if not found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM curation_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return CurationRecord.from_dict(json.loads(row["data"]))

    def update_record(
        self,
        record: CurationRecord,
        *,
        actor_id: str = "system",
        action_type: str = "update",
        rationale: str = "",
        validate_transition: bool = True,
    ) -> None:
        """Update an existing record using optimistic locking.

        The ``metadata_version`` on *record* must match the version in
        the database.  On success the version is incremented in both the
        object and the database.

        Raises:
            StaleVersionError: if the version does not match.
            InvalidTransitionError: if the status transition is invalid
                (and *validate_transition* is True).
        """
        expected_version = record.metadata_version
        conn = self._get_conn()

        # Fetch current state for audit + transition check
        row = conn.execute(
            "SELECT status, metadata_version, data "
            "FROM curation_records WHERE record_id = ?",
            (record.record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Record not found: {record.record_id}")

        actual_version = row["metadata_version"]
        if actual_version != expected_version:
            raise StaleVersionError(
                record.record_id, expected_version, actual_version
            )

        # Validate status transition
        if validate_transition:
            old_status = RecordStatus(row["status"])
            new_status = record.status
            if old_status != new_status:
                allowed = VALID_TRANSITIONS.get(old_status, frozenset())
                if new_status not in allowed:
                    raise InvalidTransitionError(
                        record.record_id,
                        old_status.value,
                        new_status.value,
                    )

        before_json = row["data"]
        record.metadata_version = expected_version + 1
        now = _utc_now_iso()

        with conn:
            cursor = conn.execute(
                """
                UPDATE curation_records
                SET source_path = ?,
                    stage = ?,
                    status = ?,
                    metadata_version = ?,
                    quality_score = ?,
                    promotion_bucket = ?,
                    curation_pass = ?,
                    speaker_cluster = ?,
                    updated_at = ?,
                    data = ?
                WHERE record_id = ? AND metadata_version = ?
                """,
                (
                    record.source_path,
                    _max_stage(record),
                    record.status.value,
                    record.metadata_version,
                    record.quality_score,
                    record.promotion_bucket.value,
                    record.pass_index,
                    record.speaker_cluster,
                    now,
                    json.dumps(record.to_dict()),
                    record.record_id,
                    expected_version,
                ),
            )

            if cursor.rowcount == 0:
                # Race: another thread updated between our SELECT and UPDATE
                record.metadata_version = expected_version
                raise StaleVersionError(
                    record.record_id, expected_version
                )

            # Audit trail
            conn.execute(
                """
                INSERT INTO audit_log
                    (record_id, actor_id, action_type, timestamp,
                     before_state, after_state, rationale)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    actor_id,
                    action_type,
                    now,
                    before_json,
                    json.dumps(record.to_dict()),
                    rationale,
                ),
            )

    def list_records(
        self,
        *,
        status: Optional[str] = None,
        stage: Optional[int] = None,
        curation_pass: Optional[int] = None,
        promotion_bucket: Optional[str] = None,
        min_quality: Optional[float] = None,
        max_quality: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CurationRecord]:
        """Query records with optional filtering."""
        clauses: List[str] = []
        params: List[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if curation_pass is not None:
            clauses.append("curation_pass = ?")
            params.append(curation_pass)
        if promotion_bucket is not None:
            clauses.append("promotion_bucket = ?")
            params.append(promotion_bucket)
        if min_quality is not None:
            clauses.append("quality_score >= ?")
            params.append(min_quality)
        if max_quality is not None:
            clauses.append("quality_score <= ?")
            params.append(max_quality)

        query = "SELECT data FROM curation_records"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY record_id LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        return [CurationRecord.from_dict(json.loads(r["data"])) for r in rows]

    def count_records(
        self,
        *,
        status: Optional[str] = None,
    ) -> int:
        """Return the number of records matching the filter."""
        query = "SELECT COUNT(*) AS cnt FROM curation_records"
        params: List[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        conn = self._get_conn()
        row = conn.execute(query, params).fetchone()
        return int(row["cnt"])

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_create(self, records: Sequence[CurationRecord]) -> int:
        """Insert multiple records in a single transaction.

        Returns the number of records inserted.
        """
        now = _utc_now_iso()
        conn = self._get_conn()
        inserted = 0
        with conn:
            for record in records:
                try:
                    conn.execute(
                        """
                        INSERT INTO curation_records
                            (record_id, source_path, stage, status,
                             metadata_version, quality_score,
                             promotion_bucket, curation_pass,
                             speaker_cluster, created_at, updated_at,
                             data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.record_id,
                            record.source_path,
                            0,
                            record.status.value,
                            record.metadata_version,
                            record.quality_score,
                            record.promotion_bucket.value,
                            record.pass_index,
                            record.speaker_cluster,
                            now,
                            now,
                            json.dumps(record.to_dict()),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    logger.debug(
                        "Skipping duplicate record: %s", record.record_id
                    )
        return inserted

    def batch_update(
        self,
        records: Sequence[CurationRecord],
        *,
        actor_id: str = "system",
        action_type: str = "batch_update",
        rationale: str = "",
    ) -> tuple[int, list[str]]:
        """Update multiple records in a single transaction.

        Returns ``(updated_count, list_of_stale_record_ids)``.
        """
        conn = self._get_conn()
        updated = 0
        stale: list[str] = []
        now = _utc_now_iso()

        with conn:
            for record in records:
                expected = record.metadata_version
                new_version = expected + 1

                # Fetch before-state for audit
                row = conn.execute(
                    "SELECT data FROM curation_records "
                    "WHERE record_id = ? AND metadata_version = ?",
                    (record.record_id, expected),
                ).fetchone()
                if row is None:
                    stale.append(record.record_id)
                    continue

                before_json = row["data"]
                record.metadata_version = new_version

                cursor = conn.execute(
                    """
                    UPDATE curation_records
                    SET source_path = ?,
                        stage = ?,
                        status = ?,
                        metadata_version = ?,
                        quality_score = ?,
                        promotion_bucket = ?,
                        curation_pass = ?,
                        speaker_cluster = ?,
                        updated_at = ?,
                        data = ?
                    WHERE record_id = ? AND metadata_version = ?
                    """,
                    (
                        record.source_path,
                        _max_stage(record),
                        record.status.value,
                        new_version,
                        record.quality_score,
                        record.promotion_bucket.value,
                        record.pass_index,
                        record.speaker_cluster,
                        now,
                        json.dumps(record.to_dict()),
                        record.record_id,
                        expected,
                    ),
                )

                if cursor.rowcount == 0:
                    record.metadata_version = expected
                    stale.append(record.record_id)
                    continue

                conn.execute(
                    """
                    INSERT INTO audit_log
                        (record_id, actor_id, action_type, timestamp,
                         before_state, after_state, rationale)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        actor_id,
                        action_type,
                        now,
                        before_json,
                        json.dumps(record.to_dict()),
                        rationale,
                    ),
                )
                updated += 1

        return updated, stale

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_log(
        self,
        record_id: str,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return audit entries for a record, most recent first."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT log_id, record_id, actor_id, action_type,
                   timestamp, before_state, after_state, rationale
            FROM audit_log
            WHERE record_id = ?
            ORDER BY log_id DESC
            LIMIT ?
            """,
            (record_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------

    def status_summary(self) -> Dict[str, Any]:
        """Return aggregate counts by status, bucket, and pass."""
        conn = self._get_conn()

        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM curation_records GROUP BY status"
        ).fetchall()
        status_counts = {r["status"]: r["cnt"] for r in status_rows}

        bucket_rows = conn.execute(
            "SELECT promotion_bucket, COUNT(*) AS cnt "
            "FROM curation_records GROUP BY promotion_bucket"
        ).fetchall()
        bucket_counts = {r["promotion_bucket"]: r["cnt"] for r in bucket_rows}

        total_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM curation_records"
        ).fetchone()
        total = total_row["cnt"] if total_row else 0

        return {
            "total_records": total,
            "status": status_counts,
            "promotion_buckets": bucket_counts,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_stage(record: CurationRecord) -> int:
    """Derive the highest completed stage number from provenance keys."""
    from .stage_framework import STAGE_NAMES

    name_to_num = {v: k for k, v in STAGE_NAMES.items()}
    max_s = 0
    for prov_key in record.providers:
        s = name_to_num.get(prov_key, 0)
        if s > max_s:
            max_s = s
    return max_s
