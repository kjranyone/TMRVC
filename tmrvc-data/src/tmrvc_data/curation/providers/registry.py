"""Provider registry with mainline defaults and version pinning (Worker 08).

Every mainline provider is pinned to a specific artifact_id and version.
No ``latest``, ``main``, or implicit default references are allowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from . import BaseProvider, ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class ProviderSpec:
    """Pinned specification for a mainline provider.

    Used for registry documentation and runtime validation.
    """

    stage: str
    provider_id: str
    artifact_id: str
    version: str
    runtime_backend: str
    calibration_version: str = "uncalibrated"
    supported_languages: List[str] = field(default_factory=list)
    license_status: str = "unknown"
    gated_access: bool = False
    fallback_policy: str = "skip"  # "skip" | "downgrade" | "review"
    notes: str = ""


# ------------------------------------------------------------------
# Mainline provider specifications (pinned, no implicit defaults)
# ------------------------------------------------------------------

MAINLINE_PROVIDERS: Dict[str, ProviderSpec] = {
    "asr": ProviderSpec(
        stage="asr",
        provider_id="qwen3_asr",
        artifact_id="Qwen/Qwen3-ASR-1.7B",
        version="1.7b-v1",
        runtime_backend="transformers",
        supported_languages=[
            "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar",
        ],
        license_status="apache-2.0",
        gated_access=False,
        fallback_policy="downgrade",
        notes="Primary ASR. Throughput fallback: faster-whisper.",
    ),
    "diarization": ProviderSpec(
        stage="diarization",
        provider_id="pyannote_community",
        artifact_id="pyannote/speaker-diarization-community-1",
        version="1.0.0",
        runtime_backend="pyannote.audio",
        license_status="mit",
        gated_access=True,
        fallback_policy="downgrade",
        notes="Primary diarization. Requires HuggingFace token for gated access.",
    ),
    "alignment": ProviderSpec(
        stage="alignment",
        provider_id="qwen3_aligner",
        artifact_id="Qwen/Qwen3-ForcedAligner-0.6B",
        version="0.6b-v1",
        runtime_backend="transformers",
        supported_languages=[
            "zh", "en", "ja", "ko", "fr", "de", "es",
        ],
        license_status="apache-2.0",
        gated_access=False,
        fallback_policy="skip",
        notes=(
            "Primary forced aligner. No fallback for unsupported languages; "
            "emit explicit absence and route to review."
        ),
    ),
    "voice_state_estimation": ProviderSpec(
        stage="voice_state_estimation",
        provider_id="voice_state_estimator",
        artifact_id="builtin/voice-state-estimator-v1",
        version="1.0.0",
        runtime_backend="numpy",
        license_status="internal",
        gated_access=False,
        fallback_policy="skip",
        notes="8-D voice state estimator. Uses parselmouth/librosa/FCPE when available.",
    ),
    "speaker_clustering": ProviderSpec(
        stage="speaker_clustering",
        provider_id="cross_file_speaker_clustering",
        artifact_id="builtin/speaker-clustering-v1",
        version="1.0.0",
        runtime_backend="numpy",
        license_status="internal",
        gated_access=False,
        fallback_policy="skip",
        notes="Cross-file speaker ID normalization via embedding similarity.",
    ),
    "asr_throughput_fallback": ProviderSpec(
        stage="asr",
        provider_id="faster_whisper",
        artifact_id="faster-whisper/large-v3",
        version="1.0.0",
        runtime_backend="ctranslate2",
        license_status="mit",
        gated_access=False,
        fallback_policy="skip",
        notes="Throughput fallback ASR. Must record downgrade provenance.",
    ),
    "transcript_refinement": ProviderSpec(
        stage="transcript_refinement",
        provider_id="multi_asr_refiner",
        artifact_id="builtin/multi-asr-refiner-v1",
        version="1.0.0",
        runtime_backend="python",
        license_status="internal",
        gated_access=False,
        fallback_policy="skip",
        notes="Multi-ASR fusion and normalization.",
    ),
}


def create_provider_registry(
    *,
    include_stubs: bool = True,
) -> ProviderRegistry:
    """Create a ``ProviderRegistry`` populated with Worker 08 providers.

    This extends the base ``create_default_registry()`` with the
    mainline providers defined in this module.

    Args:
        include_stubs: If True, register stub providers even when their
            runtime dependencies are missing.  Useful for testing.
    """
    from . import (
        FasterWhisperASR,
        TranscriptRefiner,
    )
    from .asr import Qwen3ASRProvider
    from .alignment import Qwen3AlignerProvider
    from .diarization import PyAnnoteDiarizationProvider
    from .voice_state import VoiceStateEstimator
    from .speaker_clustering import CrossFileSpeakerClustering

    registry = ProviderRegistry()

    # --- Mainline ASR (first = primary) ---
    registry.register(Qwen3ASRProvider())
    # --- Throughput fallback ASR ---
    registry.register(FasterWhisperASR())

    # --- Diarization ---
    registry.register(PyAnnoteDiarizationProvider())

    # --- Alignment ---
    registry.register(Qwen3AlignerProvider())

    # --- Voice state ---
    registry.register(VoiceStateEstimator())

    # --- Speaker clustering ---
    registry.register(CrossFileSpeakerClustering())

    # --- Transcript refinement ---
    registry.register(TranscriptRefiner())

    return registry


def validate_registry(registry: ProviderRegistry) -> List[str]:
    """Validate that a registry satisfies the mainline provider policy.

    Returns a list of warning messages for any violations.
    """
    warnings: List[str] = []
    for stage_key, spec in MAINLINE_PROVIDERS.items():
        if stage_key.endswith("_fallback"):
            continue  # fallbacks are optional
        providers = registry.get_providers(spec.stage)
        if not providers:
            warnings.append(
                f"No providers registered for stage '{spec.stage}'"
            )
            continue
        # Check that the primary matches the mainline spec
        primary = providers[0]
        if primary.name != spec.provider_id:
            warnings.append(
                f"Stage '{spec.stage}' primary is '{primary.name}', "
                f"expected mainline '{spec.provider_id}'"
            )
        if primary.version != spec.version:
            warnings.append(
                f"Stage '{spec.stage}' primary version '{primary.version}' "
                f"does not match mainline pin '{spec.version}'"
            )
    return warnings
