"""Orchestration layer for AI Curation System (Worker 07).

Manages the curation pipeline: running records through stages, handling
retries, persisting progress to both manifest.jsonl and SQLite, and
supporting resume from checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .errors import StageExecutionError
from .models import CurationRecord, RecordStatus, StageResult
from .stage_framework import CurationStage, StageRegistry, STAGE_NAMES

logger = logging.getLogger(__name__)

# Stage name -> provenance key mapping.  Used by run_named_stage() to look up
# the correct processor function from the built-in stage registry.
_STAGE_ORDER: List[str] = [
    "ingest",                # 0
    "cleanup",               # 1
    "separation",            # 2
    "speaker_recovery",      # 3
    "transcript_recovery",   # 4
    "transcript_refinement", # 5
    "prosody_recovery",      # 6
]


class CurationOrchestrator:
    """Manages the curation pipeline stages and manifest persistence.

    The orchestrator supports two storage back-ends:

    1. **manifest.jsonl** -- portable interchange format, always written.
    2. **CurationDataService** (SQLite) -- optional operational DB for
       concurrent access and audit trails.  Attach via *data_service*.
    """

    def __init__(
        self,
        output_dir: Path | str,
        registry: "ProviderRegistry | None" = None,
        stage_registry: StageRegistry | None = None,
        data_service: "CurationDataService | None" = None,
        *,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.registry = registry
        self.stage_registry = stage_registry
        self.data_service = data_service

        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self.records: Dict[str, CurationRecord] = {}
        self._active_jobs: Dict[str, dict] = {} # Job ID -> status dict (Worker 07 requirement)
        self.load_manifest()

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Fetch real-time progress of an active job (for Worker 12 UI)."""
        return self._active_jobs.get(job_id)

    def _update_job_progress(self, job_id: str, progress: float, message: str = "") -> None:
        """Update job progress and potentially broadcast via SSE."""
        if job_id in self._active_jobs:
            self._active_jobs[job_id].update({
                "progress": progress,
                "message": message,
                "last_update": time.time()
            })
            # Log progress for observability
            if int(progress * 100) % 10 == 0:
                logger.info("Job %s: %.1f%% - %s", job_id, progress * 100, message)

    async def run_named_stage_async(self, stage_name: str, job_id: str) -> None:
        """Run a curation stage across all records asynchronously (Worker 07)."""
        if stage_name not in STAGE_NAMES:
            raise ValueError(f"Unknown stage: {stage_name}")

        self._active_jobs[job_id] = {
            "stage": stage_name,
            "progress": 0.0,
            "status": "running",
            "start_time": time.time()
        }

        records = list(self.records.values())
        total = len(records)
        if total == 0:
            self._active_jobs[job_id]["status"] = "completed"
            return

        try:
            for i, record in enumerate(records):
                # Process one record
                # (Stub: in real use, this calls providers.py via stage_framework)
                await asyncio.sleep(0.01) # Simulate work
                
                # Update progress every 10%
                if (i + 1) % max(1, total // 10) == 0:
                    self._update_job_progress(job_id, (i + 1) / total, f"Processing {i+1}/{total}")

            self._active_jobs[job_id]["status"] = "completed"
            self._active_jobs[job_id]["progress"] = 1.0
            self.save_manifest()
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e)
            self._active_jobs[job_id]["status"] = "failed"
            self._active_jobs[job_id]["error"] = str(e)

    # ------------------------------------------------------------------
    # Manifest persistence
    # ------------------------------------------------------------------

    def load_manifest(self) -> None:
        """Load records from manifest.jsonl if it exists."""
        if not self.manifest_path.exists():
            self.records = {}
            return

        records = {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = CurationRecord.from_dict(data)
                    records[record.record_id] = record
                except Exception as e:
                    logger.error("Failed to parse manifest line: %s", e)

        self.records = records
        logger.info(
            "Loaded %d records from %s", len(self.records), self.manifest_path
        )

    def save_manifest(self) -> None:
        """Save all records to manifest.jsonl atomically."""
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in self.records.values():
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        tmp_path.replace(self.manifest_path)
        self._update_summary()

    def update_record(
        self,
        record: CurationRecord,
        expected_version: int | None = None,
    ) -> None:
        """Update a single record in memory with optimistic lock check.

        Args:
            record: The updated record.
            expected_version: The version the caller originally read.

        Raises:
            ValueError: if expected_version doesn't match current metadata_version.
        """
        current = self.records.get(record.record_id)
        if current is not None and expected_version is not None:
            if current.metadata_version != expected_version:
                raise ValueError(
                    f"Conflict detected for {record.record_id}: "
                    f"current version is {current.metadata_version}, "
                    f"but update expected {expected_version}"
                )

        # Increment version on update
        record.metadata_version += 1
        self.records[record.record_id] = record

    def _update_summary(self) -> None:
        """Update summary.json with current statistics."""
        status_counts: Dict[str, int] = {}
        bucket_counts: Dict[str, int] = {}

        for r in self.records.values():
            status_counts[r.status.value] = (
                status_counts.get(r.status.value, 0) + 1
            )
            bucket_counts[r.promotion_bucket.value] = (
                bucket_counts.get(r.promotion_bucket.value, 0) + 1
            )

        summary = {
            "total_records": len(self.records),
            "status": status_counts,
            "promotion_buckets": bucket_counts,
            "last_updated": time.time(),
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Stage execution (callback-based, legacy API)
    # ------------------------------------------------------------------

    def run_stage(
        self,
        stage_name: str,
        processor: Callable[[CurationRecord], Optional[CurationRecord]],
        force: bool = False,
    ) -> None:
        """Run a processing stage over all records.

        Args:
            stage_name: Name of the stage (for logging and provenance).
            processor: Function that takes a record and returns an updated one
                       (or None if the record should be removed/filtered).
            force: If True, re-process even if the record was already handled
                   by this stage.
        """
        logger.info("Starting stage: %s", stage_name)
        count = 0
        updated = 0

        # Snapshot record IDs to avoid issues if processor modifies dict size
        record_ids = list(self.records.keys())

        for rid in record_ids:
            record = self.records.get(rid)
            if record is None:
                continue

            # Skip if already processed by this stage (based on provenance)
            if not force and any(
                p.stage == stage_name for p in record.providers.values()
            ):
                continue

            new_record = processor(record)
            if new_record is None:
                del self.records[rid]
            else:
                self.records[rid] = new_record
                updated += 1

            count += 1
            # Periodically save manifest for large runs
            if count % 100 == 0:
                self.save_manifest()
                logger.info(
                    "[%s] Progress: %d/%d",
                    stage_name, count, len(record_ids),
                )

        self.save_manifest()
        logger.info(
            "Finished stage: %s (Processed: %d, Updated: %d)",
            stage_name, count, updated,
        )

    # ------------------------------------------------------------------
    # Stage-addressable execution (numeric stages via StageRegistry)
    # ------------------------------------------------------------------

    def run_stage_num(
        self,
        stage_num: int,
        *,
        force: bool = False,
        record_ids: Optional[List[str]] = None,
    ) -> Dict[str, StageResult]:
        """Run a numeric stage on pending records using the StageRegistry.

        Args:
            stage_num: Stage number (0-9).
            force: Re-process records that already have provenance for this
                   stage.
            record_ids: If provided, only process these records.

        Returns:
            Mapping of record_id -> StageResult for each processed record.
        """
        if self.stage_registry is None:
            raise RuntimeError("No StageRegistry attached to orchestrator")

        stage_impl = self.stage_registry.get(stage_num)
        if stage_impl is None:
            raise ValueError(
                f"No implementation registered for stage {stage_num} "
                f"({STAGE_NAMES.get(stage_num, '?')})"
            )

        stage_name = STAGE_NAMES.get(stage_num, f"stage_{stage_num}")
        logger.info("Running stage %d (%s)", stage_num, stage_name)

        targets = record_ids or list(self.records.keys())
        results: Dict[str, StageResult] = {}

        for rid in targets:
            record = self.records.get(rid)
            if record is None:
                continue

            # Skip already-processed unless forced
            if not force and any(
                p.stage == stage_name for p in record.providers.values()
            ):
                continue

            if not stage_impl.can_process(record):
                continue

            result = self._execute_with_retry(stage_impl, record)
            results[rid] = result

            if result.success and result.outputs:
                # Merge outputs into record attributes
                for key, val in result.outputs.items():
                    if hasattr(record, key):
                        setattr(record, key, val)
                    else:
                        record.attributes[key] = val
                self.records[rid] = record

        self.save_manifest()
        logger.info(
            "Stage %d (%s) complete: %d/%d succeeded",
            stage_num,
            stage_name,
            sum(1 for r in results.values() if r.success),
            len(results),
        )
        return results

    def run_all_stages(
        self,
        *,
        start_stage: int = 0,
        end_stage: int = 9,
        force: bool = False,
    ) -> Dict[int, Dict[str, StageResult]]:
        """Run all stages from *start_stage* to *end_stage* inclusive.

        Only stages that have registered implementations will run; others
        are silently skipped.
        """
        all_results: Dict[int, Dict[str, StageResult]] = {}
        for num in range(start_stage, end_stage + 1):
            if self.stage_registry is None or num not in self.stage_registry:
                logger.debug("Skipping stage %d (no implementation)", num)
                continue
            all_results[num] = self.run_stage_num(num, force=force)
        return all_results

    def resume(self) -> Dict[int, Dict[str, StageResult]]:
        """Resume processing from the earliest incomplete stage.

        Scans records to find the lowest stage number that has records
        still needing processing, then runs from there.
        """
        if self.stage_registry is None:
            raise RuntimeError("No StageRegistry attached to orchestrator")

        # Find the lowest stage that has unprocessed records
        start = 0
        for num in sorted(STAGE_NAMES.keys()):
            stage_name = STAGE_NAMES[num]
            has_pending = False
            for record in self.records.values():
                if not any(
                    p.stage == stage_name for p in record.providers.values()
                ):
                    has_pending = True
                    break
            if has_pending:
                start = num
                break

        logger.info("Resuming from stage %d (%s)", start, STAGE_NAMES.get(start, "?"))
        return self.run_all_stages(start_stage=start)

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _execute_with_retry(
        self,
        stage: CurationStage,
        record: CurationRecord,
    ) -> StageResult:
        """Execute a stage with retry on transient failures."""
        last_result: Optional[StageResult] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = stage.process(record)
            except Exception as exc:
                logger.warning(
                    "Stage %d attempt %d/%d failed for %s: %s",
                    stage.stage_num, attempt, self.max_retries,
                    record.record_id, exc,
                )
                last_result = StageResult(
                    success=False,
                    error=str(exc),
                    retryable=True,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                continue

            if result.success:
                return result

            last_result = result
            if not result.retryable:
                return result

            logger.info(
                "Stage %d attempt %d/%d: retryable failure for %s",
                stage.stage_num, attempt, self.max_retries,
                record.record_id,
            )
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)

        # All attempts exhausted
        return last_result or StageResult(
            success=False, error="max retries exceeded"
        )

    # ------------------------------------------------------------------
    # Named-stage convenience API (legacy, uses _STAGE_ORDER)
    # ------------------------------------------------------------------

    @staticmethod
    def available_stages() -> List[str]:
        """Return the ordered list of built-in stage names."""
        return list(_STAGE_ORDER)

    def _get_builtin_processor(
        self, stage_name: str,
    ) -> Callable[[CurationRecord], Optional[CurationRecord]]:
        """Resolve a stage name to its built-in processor function.

        The import is deferred so that orchestrator.py itself has no hard
        dependency on the stage modules (avoids circular imports).
        """
        if stage_name == "cleanup":
            from .stages.cleanup import run_cleanup
            return run_cleanup

        if stage_name == "separation":
            from .stages.separation import run_separation
            reg = self.registry
            return lambda record: run_separation(record, registry=reg)

        if stage_name == "speaker_recovery":
            from .stages.speaker_recovery import run_speaker_recovery
            reg = self.registry
            return lambda record: run_speaker_recovery(record, registry=reg)

        if stage_name == "transcript_recovery":
            from .stages.transcript_recovery import run_transcript_recovery
            reg = self.registry
            return lambda record: run_transcript_recovery(record, registry=reg)

        if stage_name == "transcript_refinement":
            from .stages.transcript_refinement import run_transcript_refinement
            reg = self.registry
            return lambda record: run_transcript_refinement(record, registry=reg)

        if stage_name == "prosody_recovery":
            from .stages.prosody_recovery import run_prosody_recovery
            reg = self.registry
            return lambda record: run_prosody_recovery(record, registry=reg)

        raise ValueError(
            f"Unknown stage '{stage_name}'. "
            f"Available: {', '.join(_STAGE_ORDER)}"
        )

    def run_named_stage(
        self, stage_name: str, *, force: bool = False
    ) -> None:
        """Run a built-in curation stage by name.

        Example::

            orchestrator.run_named_stage("cleanup")
            orchestrator.run_named_stage("separation", force=True)

        Args:
            stage_name: One of the names in ``available_stages()``.
            force: Re-process records even if they already have provenance
                   for the given stage.
        """
        processor = self._get_builtin_processor(stage_name)
        self.run_stage(stage_name, processor, force=force)

    def run_pipeline(
        self,
        stages: Optional[List[str]] = None,
        *,
        force: bool = False,
    ) -> None:
        """Run multiple stages in sequence.

        Args:
            stages: List of stage names.  Defaults to all stages except
                    ``ingest`` (which has its own entry point).
            force: Re-process records even if already handled.
        """
        if stages is None:
            stages = [s for s in _STAGE_ORDER if s != "ingest"]
        for stage_name in stages:
            self.run_named_stage(stage_name, force=force)
