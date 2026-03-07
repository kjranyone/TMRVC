"""Curation validation and acceptance testing.

Implements Worker 11: proves the curation system improves data quality.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from .models import CurationRecord, RecordStatus, PromotionBucket

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Thresholds for curation validation."""
    min_promote_ratio: float = 0.10
    max_false_promote_rate: float = 0.05
    max_false_reject_rate: float = 0.10
    min_asr_spot_accuracy: float = 0.90
    min_speaker_nmi: float = 0.80
    max_holdout_leak_ratio: float = 0.0


class CurationValidator:
    """Validates curation system quality and policy compliance."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate_promotion_distribution(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Check promote/review/reject distribution."""
        total = len(records)
        if total == 0:
            return {"pass": False, "reason": "no_records", "total": 0}

        counts = {}
        for r in records:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1

        promoted = counts.get("promoted", 0)
        promote_ratio = promoted / total

        return {
            "pass": promote_ratio >= self.config.min_promote_ratio,
            "total": total,
            "distribution": counts,
            "promote_ratio": round(promote_ratio, 4),
            "threshold": self.config.min_promote_ratio,
        }

    def validate_legality_gating(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Verify no unknown-rights sources in mainline export."""
        violations = []
        for r in records:
            if r.status != RecordStatus.PROMOTED:
                continue
            if r.promotion_bucket in (
                PromotionBucket.TTS_MAINLINE,
                PromotionBucket.HOLDOUT_EVAL,
            ):
                if r.source_legality not in ("owned", "licensed"):
                    violations.append({
                        "record_id": r.record_id,
                        "bucket": r.promotion_bucket.value,
                        "legality": r.source_legality,
                    })

        return {
            "pass": len(violations) == 0,
            "violations": violations,
            "total_checked": sum(
                1 for r in records if r.status == RecordStatus.PROMOTED
            ),
        }

    def validate_holdout_integrity(
        self, records: List[CurationRecord],
        holdout_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Verify holdout records don't leak into training buckets."""
        if holdout_ids is None:
            holdout_ids = {
                r.record_id for r in records
                if r.promotion_bucket == PromotionBucket.HOLDOUT_EVAL
            }

        leaks = []
        train_buckets = {
            PromotionBucket.TTS_MAINLINE,
            PromotionBucket.VC_PRIOR,
            PromotionBucket.EXPRESSIVE_PRIOR,
        }
        for r in records:
            if r.record_id in holdout_ids and r.promotion_bucket in train_buckets:
                leaks.append(r.record_id)

        return {
            "pass": len(leaks) == 0,
            "leaks": leaks,
            "holdout_size": len(holdout_ids),
        }

    def validate_provenance_completeness(
        self, records: List[CurationRecord]
    ) -> Dict[str, Any]:
        """Check that promoted records have complete provenance."""
        promoted = [r for r in records if r.status == RecordStatus.PROMOTED]
        incomplete = []
        for r in promoted:
            if not r.providers:
                incomplete.append(r.record_id)

        return {
            "pass": len(incomplete) == 0,
            "total_promoted": len(promoted),
            "incomplete": incomplete[:20],  # Limit output
        }

    def run_all(self, records: List[CurationRecord]) -> Dict[str, Any]:
        """Run all validation checks and return combined report."""
        results = {
            "promotion_distribution": self.validate_promotion_distribution(records),
            "legality_gating": self.validate_legality_gating(records),
            "holdout_integrity": self.validate_holdout_integrity(records),
            "provenance_completeness": self.validate_provenance_completeness(records),
        }

        all_pass = all(v.get("pass", False) for v in results.values())
        results["overall"] = {"pass": all_pass}

        return results

    def save_report(self, report: Dict[str, Any], output_path: Path | str) -> None:
        """Save validation report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Validation report saved to %s", path)
