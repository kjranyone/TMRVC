"""Provider comparison metrics (Worker 08).

Utilities for comparing outputs from different providers on the same
input, enabling provider uplift measurement and disagreement analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import ProviderOutput

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two provider outputs."""

    metric_name: str
    provider_a: str
    provider_b: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class ProviderComparisonMetrics:
    """Compute comparison metrics between provider outputs.

    Supports:
    - ASR transcript agreement (word error rate proxy)
    - Diarization segment overlap (Jaccard index)
    - Cross-file speaker clustering purity
    - Voice state dimension-wise agreement
    - Alignment timing deviation
    """

    # ------------------------------------------------------------------
    # ASR comparison
    # ------------------------------------------------------------------

    @staticmethod
    def asr_agreement(
        output_a: ProviderOutput,
        output_b: ProviderOutput,
    ) -> ComparisonResult:
        """Compute transcript agreement between two ASR outputs.

        Uses simple word-level intersection-over-union as a WER proxy.
        """
        text_a = output_a.fields.get("transcript", "")
        text_b = output_b.fields.get("transcript", "")

        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a and not words_b:
            agreement = 1.0
        elif not words_a or not words_b:
            agreement = 0.0
        else:
            intersection = words_a & words_b
            union = words_a | words_b
            agreement = len(intersection) / len(union)

        prov_a = output_a.provenance.provider if output_a.provenance else "unknown"
        prov_b = output_b.provenance.provider if output_b.provenance else "unknown"

        return ComparisonResult(
            metric_name="asr_word_agreement",
            provider_a=prov_a,
            provider_b=prov_b,
            value=round(agreement, 4),
            details={
                "n_words_a": len(words_a),
                "n_words_b": len(words_b),
                "n_intersection": len(words_a & words_b),
            },
        )

    @staticmethod
    def asr_confidence_uplift(
        output_a: ProviderOutput,
        output_b: ProviderOutput,
    ) -> ComparisonResult:
        """Compute confidence uplift of provider B over provider A."""
        conf_a = output_a.confidence
        conf_b = output_b.confidence
        uplift = conf_b - conf_a

        prov_a = output_a.provenance.provider if output_a.provenance else "unknown"
        prov_b = output_b.provenance.provider if output_b.provenance else "unknown"

        return ComparisonResult(
            metric_name="asr_confidence_uplift",
            provider_a=prov_a,
            provider_b=prov_b,
            value=round(uplift, 4),
            details={
                "confidence_a": round(conf_a, 4),
                "confidence_b": round(conf_b, 4),
            },
        )

    # ------------------------------------------------------------------
    # Diarization comparison
    # ------------------------------------------------------------------

    @staticmethod
    def diarization_segment_overlap(
        turns_a: List[Dict[str, Any]],
        turns_b: List[Dict[str, Any]],
        *,
        provider_a: str = "a",
        provider_b: str = "b",
    ) -> ComparisonResult:
        """Compute segment-level Jaccard overlap between diarization outputs.

        Each turn dict must have 'start_sec' and 'end_sec'.
        """
        def _to_intervals(turns: List[Dict[str, Any]]) -> List[tuple]:
            return [(t["start_sec"], t["end_sec"]) for t in turns]

        intervals_a = _to_intervals(turns_a)
        intervals_b = _to_intervals(turns_b)

        if not intervals_a and not intervals_b:
            jaccard = 1.0
        elif not intervals_a or not intervals_b:
            jaccard = 0.0
        else:
            # Discretize at 10ms resolution
            resolution = 0.01
            max_t = max(
                max(e for _, e in intervals_a),
                max(e for _, e in intervals_b),
            )
            n_bins = int(max_t / resolution) + 1

            mask_a = np.zeros(n_bins, dtype=bool)
            mask_b = np.zeros(n_bins, dtype=bool)

            for s, e in intervals_a:
                mask_a[int(s / resolution): int(e / resolution)] = True
            for s, e in intervals_b:
                mask_b[int(s / resolution): int(e / resolution)] = True

            intersection = int(np.sum(mask_a & mask_b))
            union = int(np.sum(mask_a | mask_b))
            jaccard = intersection / union if union > 0 else 0.0

        return ComparisonResult(
            metric_name="diarization_segment_jaccard",
            provider_a=provider_a,
            provider_b=provider_b,
            value=round(jaccard, 4),
            details={
                "n_turns_a": len(turns_a),
                "n_turns_b": len(turns_b),
            },
        )

    # ------------------------------------------------------------------
    # Voice state comparison
    # ------------------------------------------------------------------

    @staticmethod
    def voice_state_agreement(
        state_a: List[float],
        state_b: List[float],
        mask_a: Optional[List[bool]] = None,
        mask_b: Optional[List[bool]] = None,
        *,
        provider_a: str = "a",
        provider_b: str = "b",
    ) -> ComparisonResult:
        """Compute per-dimension agreement between two voice state vectors.

        Only compares dimensions where both masks are True.
        Returns mean absolute difference (lower = more agreement).
        """
        a = np.array(state_a, dtype=np.float32)
        b = np.array(state_b, dtype=np.float32)

        if mask_a is not None and mask_b is not None:
            ma = np.array(mask_a, dtype=bool)
            mb = np.array(mask_b, dtype=bool)
            joint_mask = ma & mb
        else:
            joint_mask = np.ones(len(a), dtype=bool)

        if not np.any(joint_mask):
            return ComparisonResult(
                metric_name="voice_state_mad",
                provider_a=provider_a,
                provider_b=provider_b,
                value=float("nan"),
                details={"n_compared_dims": 0},
            )

        diffs = np.abs(a[joint_mask] - b[joint_mask])
        mad = float(np.mean(diffs))

        return ComparisonResult(
            metric_name="voice_state_mad",
            provider_a=provider_a,
            provider_b=provider_b,
            value=round(mad, 4),
            details={
                "n_compared_dims": int(np.sum(joint_mask)),
                "per_dim_diff": [round(float(d), 4) for d in diffs],
            },
        )

    # ------------------------------------------------------------------
    # Alignment comparison
    # ------------------------------------------------------------------

    @staticmethod
    def alignment_timing_deviation(
        phonemes_a: List[Dict[str, Any]],
        phonemes_b: List[Dict[str, Any]],
        *,
        provider_a: str = "a",
        provider_b: str = "b",
    ) -> ComparisonResult:
        """Compute mean timing deviation between two alignment outputs.

        Matches phonemes by index (assumes same phoneme sequence) and
        computes the mean absolute difference in start/end times.
        """
        n = min(len(phonemes_a), len(phonemes_b))
        if n == 0:
            return ComparisonResult(
                metric_name="alignment_timing_deviation_sec",
                provider_a=provider_a,
                provider_b=provider_b,
                value=float("nan"),
                details={"n_matched": 0},
            )

        start_diffs = []
        end_diffs = []
        for i in range(n):
            start_diffs.append(abs(
                phonemes_a[i].get("start_sec", 0) - phonemes_b[i].get("start_sec", 0)
            ))
            end_diffs.append(abs(
                phonemes_a[i].get("end_sec", 0) - phonemes_b[i].get("end_sec", 0)
            ))

        mean_dev = float(np.mean(start_diffs + end_diffs))

        return ComparisonResult(
            metric_name="alignment_timing_deviation_sec",
            provider_a=provider_a,
            provider_b=provider_b,
            value=round(mean_dev, 4),
            details={
                "n_matched": n,
                "mean_start_deviation": round(float(np.mean(start_diffs)), 4),
                "mean_end_deviation": round(float(np.mean(end_diffs)), 4),
                "n_total_a": len(phonemes_a),
                "n_total_b": len(phonemes_b),
            },
        )

    # ------------------------------------------------------------------
    # Speaker clustering purity
    # ------------------------------------------------------------------

    @staticmethod
    def speaker_clustering_purity(
        predicted_labels: Sequence[str],
        reference_labels: Sequence[str],
        *,
        provider_a: str = "predicted",
        provider_b: str = "reference",
    ) -> ComparisonResult:
        """Compute clustering purity against reference labels.

        Purity = (1/N) * sum_k max_j |cluster_k intersect class_j|
        where k indexes predicted clusters and j indexes reference classes.
        """
        n = min(len(predicted_labels), len(reference_labels))
        if n == 0:
            return ComparisonResult(
                metric_name="speaker_clustering_purity",
                provider_a=provider_a,
                provider_b=provider_b,
                value=0.0,
                details={"n_samples": 0},
            )

        # Group by predicted cluster
        clusters: Dict[str, List[str]] = {}
        for i in range(n):
            pred = predicted_labels[i]
            ref = reference_labels[i]
            if pred not in clusters:
                clusters[pred] = []
            clusters[pred].append(ref)

        # Purity
        correct = 0
        for cluster_refs in clusters.values():
            from collections import Counter
            counts = Counter(cluster_refs)
            correct += counts.most_common(1)[0][1]

        purity = correct / n

        return ComparisonResult(
            metric_name="speaker_clustering_purity",
            provider_a=provider_a,
            provider_b=provider_b,
            value=round(purity, 4),
            details={
                "n_samples": n,
                "n_predicted_clusters": len(clusters),
                "n_reference_classes": len(set(reference_labels[:n])),
            },
        )
