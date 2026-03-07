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


class CurationOrchestrator:
    """Manages the curation pipeline stages and manifest persistence."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        
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

    def update_record(self, record: CurationRecord) -> None:
        """Update a single record in memory (call save_manifest to persist)."""
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
