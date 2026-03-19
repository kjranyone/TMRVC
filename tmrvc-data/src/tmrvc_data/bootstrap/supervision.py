"""v4 supervision tier classification.

Each utterance is classified into tiers A-D based on
the confidence and coverage of its labels.

DSP/SSL own: physical voice control targets, confidence, observed mask, speaker timbre anchor
Whisper+LLM own: transcript, punctuation recovery, scene summary, dialogue intent, emotion, acting hint
Whisper+LLM must NOT replace physical supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TierThresholds:
    """Configurable thresholds for supervision tier classification."""
    # Tier A thresholds (all high-confidence)
    tier_a_transcript_confidence: float = 0.90
    tier_a_speaker_confidence: float = 0.85
    tier_a_physical_coverage: float = 0.80
    tier_a_physical_confidence: float = 0.80
    tier_a_semantic_coverage: float = 0.70

    # Tier B thresholds (transcript+speaker high, physical/semantic partial)
    tier_b_transcript_confidence: float = 0.80
    tier_b_speaker_confidence: float = 0.70
    tier_b_physical_coverage: float = 0.50

    # Tier C thresholds (transcript present, physical sparse)
    tier_c_transcript_confidence: float = 0.50
    tier_c_speaker_confidence: float = 0.40


class SupervisionTierClassifier:
    """Classifies utterances into supervision tiers A-D.

    Tier A: speaker / transcript / physical / semantic all high-confidence
    Tier B: transcript and speaker high-confidence, physical or semantic partly pseudo
    Tier C: transcript and basic speaker anchor present, physical supervision sparse
    Tier D: reference-only or auxiliary-only
    """

    def __init__(self, thresholds: Optional[TierThresholds] = None):
        self.thresholds = thresholds or TierThresholds()

    def classify(
        self,
        transcript_confidence: float,
        diarization_confidence: float,
        physical_observed_mask: Optional[np.ndarray] = None,
        physical_confidence: Optional[np.ndarray] = None,
        has_semantic_annotations: bool = False,
    ) -> str:
        """Classify an utterance into a supervision tier.

        Args:
            transcript_confidence: Whisper transcript confidence [0, 1]
            diarization_confidence: Speaker diarization confidence [0, 1]
            physical_observed_mask: [T, 12] bool mask of observed physical dims
            physical_confidence: [T, 12] confidence per physical dimension
            has_semantic_annotations: Whether LLM semantic annotations exist

        Returns:
            Tier string: "tier_a", "tier_b", "tier_c", or "tier_d"
        """
        t = self.thresholds

        # Compute physical coverage and mean confidence
        physical_coverage = 0.0
        mean_physical_confidence = 0.0
        if physical_observed_mask is not None:
            physical_coverage = float(np.mean(physical_observed_mask))
        if physical_confidence is not None and physical_observed_mask is not None:
            observed = physical_confidence[physical_observed_mask]
            if len(observed) > 0:
                mean_physical_confidence = float(np.mean(observed))

        # Tier A: everything high-confidence
        if (transcript_confidence >= t.tier_a_transcript_confidence
                and diarization_confidence >= t.tier_a_speaker_confidence
                and physical_coverage >= t.tier_a_physical_coverage
                and mean_physical_confidence >= t.tier_a_physical_confidence
                and has_semantic_annotations):
            return "tier_a"

        # Tier B: transcript+speaker high, physical/semantic partial
        if (transcript_confidence >= t.tier_b_transcript_confidence
                and diarization_confidence >= t.tier_b_speaker_confidence
                and physical_coverage >= t.tier_b_physical_coverage):
            return "tier_b"

        # Tier C: transcript present, physical sparse
        if (transcript_confidence >= t.tier_c_transcript_confidence
                and diarization_confidence >= t.tier_c_speaker_confidence):
            return "tier_c"

        # Tier D: everything else
        return "tier_d"

    def compute_tier_weights(self, tier: str) -> dict:
        """Return loss weight multipliers for each supervision signal by tier.

        Low-confidence pseudo-labels must not be treated as dense ground truth.
        """
        weights = {
            "tier_a": {
                "codec_loss": 1.0,
                "control_loss": 1.0,
                "pointer_loss": 1.0,
                "physical_loss": 1.0,
                "acting_latent_loss": 1.0,
                "disentanglement_loss": 1.0,
                "speaker_loss": 1.0,
                "prosody_loss": 1.0,
                "semantic_loss": 1.0,
            },
            "tier_b": {
                "codec_loss": 1.0,
                "control_loss": 1.0,
                "pointer_loss": 1.0,
                "physical_loss": 0.5,   # partially pseudo
                "acting_latent_loss": 0.5,
                "disentanglement_loss": 0.5,
                "speaker_loss": 1.0,
                "prosody_loss": 0.8,
                "semantic_loss": 0.5,
            },
            "tier_c": {
                "codec_loss": 1.0,
                "control_loss": 1.0,
                "pointer_loss": 1.0,
                "physical_loss": 0.2,   # sparse
                "acting_latent_loss": 0.2,
                "disentanglement_loss": 0.2,
                "speaker_loss": 0.8,
                "prosody_loss": 0.5,
                "semantic_loss": 0.1,
            },
            "tier_d": {
                "codec_loss": 0.5,
                "control_loss": 0.5,
                "pointer_loss": 0.5,
                "physical_loss": 0.0,   # not used
                "acting_latent_loss": 0.0,
                "disentanglement_loss": 0.0,
                "speaker_loss": 0.5,
                "prosody_loss": 0.2,
                "semantic_loss": 0.0,
            },
        }
        return weights.get(tier, weights["tier_d"])


@dataclass
class QualityGateReport:
    """Report from bootstrap quality gate evaluation.

    Required metrics (from track_validation.md):
    - diarization purity
    - speaker-cluster consistency
    - overlap rejection precision
    - transcript quality proxy (WER/CER)
    - physical-label coverage
    - physical-label confidence calibration
    - language coverage
    """
    corpus_id: str

    # Diarization quality
    diarization_purity: float = 0.0
    speaker_cluster_consistency: float = 0.0

    # Rejection quality
    overlap_rejection_precision: float = 0.0
    overlap_rejection_recall: float = 0.0

    # Transcript quality
    transcript_wer_proxy: float = 1.0    # lower is better
    transcript_cer_proxy: float = 1.0    # lower is better

    # Physical label quality
    physical_label_coverage: float = 0.0   # fraction of frames with observed labels
    physical_confidence_mean: float = 0.0
    physical_confidence_calibration_error: float = 1.0  # ECE, lower is better

    # Language coverage
    languages_detected: list = None
    language_distribution: dict = None

    # Tier distribution
    tier_distribution: dict = None  # {tier_a: N, tier_b: N, ...}

    # Gate pass/fail
    gates_passed: bool = False
    failed_gates: list = None

    def __post_init__(self):
        if self.languages_detected is None:
            self.languages_detected = []
        if self.language_distribution is None:
            self.language_distribution = {}
        if self.tier_distribution is None:
            self.tier_distribution = {}
        if self.failed_gates is None:
            self.failed_gates = []

    def evaluate_gates(
        self,
        min_diarization_purity: float = 0.80,
        min_overlap_rejection_precision: float = 0.85,
        max_transcript_wer: float = 0.20,
        min_physical_coverage: float = 0.60,
        max_physical_calibration_error: float = 0.15,
    ) -> bool:
        """Evaluate whether quality gates pass."""
        self.failed_gates = []

        if self.diarization_purity < min_diarization_purity:
            self.failed_gates.append(
                f"diarization_purity {self.diarization_purity:.3f} < {min_diarization_purity}"
            )
        if self.overlap_rejection_precision < min_overlap_rejection_precision:
            self.failed_gates.append(
                f"overlap_rejection_precision {self.overlap_rejection_precision:.3f} < {min_overlap_rejection_precision}"
            )
        if self.transcript_wer_proxy > max_transcript_wer:
            self.failed_gates.append(
                f"transcript_wer_proxy {self.transcript_wer_proxy:.3f} > {max_transcript_wer}"
            )
        if self.physical_label_coverage < min_physical_coverage:
            self.failed_gates.append(
                f"physical_label_coverage {self.physical_label_coverage:.3f} < {min_physical_coverage}"
            )
        if self.physical_confidence_calibration_error > max_physical_calibration_error:
            self.failed_gates.append(
                f"physical_calibration_error {self.physical_confidence_calibration_error:.3f} > {max_physical_calibration_error}"
            )

        self.gates_passed = len(self.failed_gates) == 0
        return self.gates_passed
