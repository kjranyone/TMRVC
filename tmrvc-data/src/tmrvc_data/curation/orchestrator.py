"""Orchestration layer for AI Curation System."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

from .models import CurationRecord, RecordStatus

logger = logging.getLogger(__name__)

# Stage name -> provenance key mapping.  Used by run_named_stage() to look up
# the correct processor function from the built-in stage registry.
_STAGE_ORDER: List[str] = [
    "ingest",           # 0
    "cleanup",          # 1
    "separation",       # 2
    "speaker_recovery", # 3
    "transcript_recovery",   # 4
    "transcript_refinement", # 5
    "prosody_recovery",      # 6
]


class CurationOrchestrator:
    """Manages the curation pipeline stages and manifest persistence."""

    def __init__(self, output_dir: Path | str, registry: "ProviderRegistry | None" = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.registry = registry

        self.records: Dict[str, CurationRecord] = {}
        self.load_manifest()

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
        logger.info("Loaded %d records from %s", len(self.records), self.manifest_path)

    def save_manifest(self) -> None:
        """Save all records to manifest.jsonl atomically."""
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in self.records.values():
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        tmp_path.replace(self.manifest_path)
        self._update_summary()

    def update_record(self, record: CurationRecord, expected_version: int | None = None) -> None:
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
        status_counts = {}
        bucket_counts = {}
        
        for r in self.records.values():
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1
            bucket_counts[r.promotion_bucket.value] = bucket_counts.get(r.promotion_bucket.value, 0) + 1
            
        summary = {
            "total_records": len(self.records),
            "status": status_counts,
            "promotion_buckets": bucket_counts,
            "last_updated": time.time(),
        }
        
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def run_stage(
        self, 
        stage_name: str, 
        processor: Callable[[CurationRecord], Optional[CurationRecord]],
        force: bool = False
    ) -> None:
        """Run a processing stage over all records.
        
        Args:
            stage_name: Name of the stage (for logging and provenance).
            processor: Function that takes a record and returns an updated one 
                       (or None if the record should be removed/filtered).
            force: If True, re-process even if the record was already handled by this stage.
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
            if not force and any(p.stage == stage_name for p in record.providers.values()):
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
                logger.info("[%s] Progress: %d/%d", stage_name, count, len(record_ids))

        self.save_manifest()
        logger.info("Finished stage: %s (Processed: %d, Updated: %d)", stage_name, count, updated)

    # ------------------------------------------------------------------
    # Named-stage convenience API
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

    def run_named_stage(self, stage_name: str, *, force: bool = False) -> None:
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
