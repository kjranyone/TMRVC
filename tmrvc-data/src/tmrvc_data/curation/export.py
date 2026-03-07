"""Export curated subsets to TMRVC training-compatible cache format.

Implements Worker 10: converts promoted records into trainable assets.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import CurationRecord, PromotionBucket, RecordStatus

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for curation export."""

    output_dir: Path = field(default_factory=lambda: Path("data/curated_export"))
    export_style_embedding: bool = True
    export_events: bool = True
    export_dialogue_graph: bool = True


class CurationExporter:
    """Exports promoted curation subsets to TMRVC cache-compatible format."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    def export_subset(
        self,
        records: List[CurationRecord],
        bucket: PromotionBucket,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Export records from a specific bucket to cache format.

        Returns export summary dict.
        """
        out = (
            Path(output_dir)
            if output_dir
            else self.config.output_dir / bucket.value
        )
        out.mkdir(parents=True, exist_ok=True)

        eligible = [
            r
            for r in records
            if r.status == RecordStatus.PROMOTED
            and r.promotion_bucket == bucket
        ]

        if not eligible:
            logger.warning("No records eligible for bucket %s", bucket.value)
            return {"bucket": bucket.value, "exported": 0}

        manifest_entries: List[Dict[str, Any]] = []
        for record in eligible:
            entry = self._record_to_export(record, bucket)
            manifest_entries.append(entry)

            # Write per-record meta.json
            record_dir = out / record.record_id
            record_dir.mkdir(parents=True, exist_ok=True)
            meta_path = record_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)

        # Write manifest
        manifest_path = out / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        summary: Dict[str, Any] = {
            "bucket": bucket.value,
            "exported": len(manifest_entries),
            "output_dir": str(out),
        }

        summary_path = out / "export_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Exported %d records to %s (bucket=%s)",
            len(manifest_entries),
            out,
            bucket.value,
        )
        return summary

    def _record_to_export(
        self, record: CurationRecord, bucket: PromotionBucket
    ) -> Dict[str, Any]:
        """Convert a CurationRecord to export-ready dict."""
        entry: Dict[str, Any] = {
            "record_id": record.record_id,
            "source_path": record.source_path,
            "transcript": record.transcript or "",
            "language": record.language or "",
            "speaker_cluster": record.speaker_cluster,
            "quality_score": record.quality_score,
            "source_legality": record.source_legality,
            "promotion_bucket": bucket.value,
            "segment_start_sec": record.segment_start_sec,
            "segment_end_sec": record.segment_end_sec,
            "duration_sec": record.duration_sec,
            # Provenance
            "providers": {
                k: {
                    "provider": v.provider,
                    "version": v.version,
                    "confidence": v.confidence,
                }
                for k, v in record.providers.items()
            },
        }

        # Dialogue graph fields
        if self.config.export_dialogue_graph:
            entry["conversation_id"] = record.conversation_id
            entry["turn_index"] = record.turn_index
            entry["prev_record_id"] = record.prev_record_id
            entry["next_record_id"] = record.next_record_id
            entry["context_window_ids"] = record.context_window_ids

        # Events
        if self.config.export_events:
            entry["pause_events"] = record.attributes.get("pause_events", [])
            entry["breath_events"] = record.attributes.get(
                "breath_events", []
            )

        # Style embedding reference
        if self.config.export_style_embedding:
            entry["has_style_embedding"] = (
                "style_embedding" in record.attributes
            )

        return entry

    def export_all_buckets(
        self,
        records: List[CurationRecord],
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Export all promotion buckets."""
        base = Path(output_dir) if output_dir else self.config.output_dir
        results: Dict[str, Any] = {}
        for bucket in PromotionBucket:
            if bucket == PromotionBucket.NONE:
                continue
            bucket_dir = base / bucket.value
            summary = self.export_subset(records, bucket, bucket_dir)
            results[bucket.value] = summary
        return results

    def export_evaluation_package(
        self,
        records: List[CurationRecord],
        output_dir: Union[Path, str],
    ) -> Dict[str, Any]:
        """Export holdout evaluation subset as a reproducible package."""
        return self.export_subset(
            records, PromotionBucket.HOLDOUT_EVAL, output_dir
        )
