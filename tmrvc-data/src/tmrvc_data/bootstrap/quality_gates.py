"""v4 bootstrap quality gate evaluation.

Implements measurable quality gates for the raw-audio bootstrap pipeline.
These gates are release blockers, not offline curiosities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tmrvc_data.bootstrap.supervision import QualityGateReport


@dataclass
class QualityGateConfig:
    """Configurable thresholds for v4 bootstrap quality gates."""
    min_diarization_purity: float = 0.80
    min_speaker_cluster_consistency: float = 0.75
    min_overlap_rejection_precision: float = 0.85
    max_transcript_wer: float = 0.20
    min_physical_coverage: float = 0.60
    max_physical_calibration_error: float = 0.15
    min_language_count: int = 1
    min_tier_a_ratio: float = 0.10  # At least 10% Tier A
    min_tier_ab_ratio: float = 0.40  # At least 40% Tier A+B


def evaluate_bootstrap_quality(
    report: QualityGateReport,
    config: Optional[QualityGateConfig] = None,
) -> QualityGateReport:
    """Evaluate all quality gates on a bootstrap report.

    Args:
        report: QualityGateReport with populated metrics.
        config: Gate thresholds. Uses defaults if None.

    Returns:
        Updated QualityGateReport with gate evaluation results.
    """
    cfg = config or QualityGateConfig()
    report.failed_gates = []

    # Core gates
    report.evaluate_gates(
        min_diarization_purity=cfg.min_diarization_purity,
        min_overlap_rejection_precision=cfg.min_overlap_rejection_precision,
        max_transcript_wer=cfg.max_transcript_wer,
        min_physical_coverage=cfg.min_physical_coverage,
        max_physical_calibration_error=cfg.max_physical_calibration_error,
    )

    # Additional gates
    if report.speaker_cluster_consistency < cfg.min_speaker_cluster_consistency:
        report.failed_gates.append(
            f"speaker_cluster_consistency {report.speaker_cluster_consistency:.3f} "
            f"< {cfg.min_speaker_cluster_consistency}"
        )

    if len(report.languages_detected) < cfg.min_language_count:
        report.failed_gates.append(
            f"language_count {len(report.languages_detected)} < {cfg.min_language_count}"
        )

    # Tier distribution gates
    total = sum(report.tier_distribution.values()) if report.tier_distribution else 0
    if total > 0:
        tier_a = report.tier_distribution.get("tier_a", 0)
        tier_b = report.tier_distribution.get("tier_b", 0)
        tier_a_ratio = tier_a / total
        tier_ab_ratio = (tier_a + tier_b) / total

        if tier_a_ratio < cfg.min_tier_a_ratio:
            report.failed_gates.append(
                f"tier_a_ratio {tier_a_ratio:.3f} < {cfg.min_tier_a_ratio}"
            )
        if tier_ab_ratio < cfg.min_tier_ab_ratio:
            report.failed_gates.append(
                f"tier_ab_ratio {tier_ab_ratio:.3f} < {cfg.min_tier_ab_ratio}"
            )

    report.gates_passed = len(report.failed_gates) == 0
    return report
