"""Shared data types for the TMRVC pipeline (UCLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

import torch

from tmrvc_core.constants import (
    D_SUPRASEGMENTAL as D_SUPRASEGMENTAL_CONST,
    MAX_FRAMES_PER_UNIT,
    MAX_PROMPT_CACHE_BYTES,
    MAX_PROMPT_FRAMES,
    MAX_PROMPT_KV_TOKENS,
    MAX_PROMPT_SECONDS_ACTIVE,
    SKIP_PROTECTION_THRESHOLD,
)

# Suprasegmental feature dimensionality:
# [accent_upstep, accent_downstep, phrase_break, lexical_tone_id]
D_SUPRASEGMENTAL = D_SUPRASEGMENTAL_CONST


class CFGMode(str, Enum):
    """Classifier-Free Guidance operating modes.

    OFF: No guidance -- single conditional pass only.
    FULL: Two-pass guidance (unconditional + conditional), blended via cfg_scale.
    LAZY: Cached guidance -- unconditional logits are refreshed every N frames
        and reused in between, reducing compute by ~(N-1)/N vs full mode.
    DISTILLED: Single-pass guidance -- the model has been trained to approximate
        the blended output internally given cfg_scale as an input signal.
        Falls back to FULL if distilled weights are not available.
    """
    OFF = "off"
    FULL = "full"
    LAZY = "lazy"
    DISTILLED = "distilled"


@dataclass
class Utterance:
    """Metadata for a single utterance in a dataset."""
    utterance_id: str
    speaker_id: str
    dataset: str
    audio_path: Path
    duration_sec: float
    sample_rate: int = 24000
    text: str | None = None
    language: str | None = None


@dataclass
class PointerState:
    """Tracks the current phoneme pointer position for streaming TTS.

    This is the canonical shared pointer-state contract consumed by
    Python serve, Rust runtime, ONNX export, and VST.

    Attributes:
        text_index: [B] current phoneme index (integer).
        progress: [B] fractional progress within the current phoneme (0-1).
        boundary_confidence: confidence score for the last boundary decision.
        stall_frames: number of consecutive frames without pointer advance.
        is_hard_bootstrapped: if True, the pointer is forced to external alignment
            (Stage 2 Annealing - Hard Phase).
    """

    text_index: torch.Tensor  # [B]
    progress: torch.Tensor  # [B]
    finished: bool = False
    boundary_confidence: float = 0.0
    stall_frames: int = 0
    max_frames_per_unit: int = MAX_FRAMES_PER_UNIT
    frames_on_current_unit: int = 0
    skip_protection_threshold: float = SKIP_PROTECTION_THRESHOLD
    forced_advance_count: int = 0
    skip_protection_count: int = 0
    is_hard_bootstrapped: bool = False
    last_advance_score: float = 0.0

    @property
    def advance_logit(self) -> float:
        """Compatibility accessor for the last pointer advance score."""
        return self.last_advance_score

    @property
    def progress_delta(self) -> float:
        """Alias: fractional progress accumulated so far on current unit."""
        return float(self.progress.item()) if hasattr(self.progress, 'item') else float(self.progress)

    @property
    def progress_value(self) -> torch.Tensor:
        """Alias for progress tensor — canonical name used in pointer contracts."""
        return self.progress

    def clone(self) -> "PointerState":
        return PointerState(
            text_index=self.text_index.clone(),
            progress=self.progress.clone(),
            finished=self.finished,
            boundary_confidence=self.boundary_confidence,
            stall_frames=self.stall_frames,
            max_frames_per_unit=self.max_frames_per_unit,
            frames_on_current_unit=self.frames_on_current_unit,
            skip_protection_threshold=self.skip_protection_threshold,
            forced_advance_count=self.forced_advance_count,
            skip_protection_count=self.skip_protection_count,
            is_hard_bootstrapped=self.is_hard_bootstrapped,
            last_advance_score=self.last_advance_score,
        )


# ---------------------------------------------------------------------------
# CFG Unconditional Mask Contract (Frozen)
# ---------------------------------------------------------------------------

# Fields zeroed during CFG unconditional pass:
CFG_ZEROED_FIELDS = frozenset({
    "explicit_voice_state",   # [B, T, 12] or [B, 12]
    "delta_voice_state",      # [B, T, 12] or [B, 12]
    "ssl_voice_state",        # [B, T, d_ssl]
    "speaker_embed",          # [B, d_speaker]
    "prompt_codec_tokens",    # [B, T_prompt, n_codebooks]
    "prompt_kv_cache",        # [B, T_prompt, d_model]
    "dialogue_context",       # [B, C_ctx, d_model] or [B, d_model]
    "acting_intent",          # [B, d_acting]
    "local_prosody_latent",   # [B, d_prosody] or [B, T, d_prosody]
})

# Fields preserved during CFG unconditional pass:
CFG_PRESERVED_FIELDS = frozenset({
    "phoneme_ids",            # [B, L]
    "language_ids",           # [B, L]
    "acoustic_history",       # (a_ctx, b_ctx) — causal past tokens
    "pointer_state",          # PointerState — current text progression
})


# ---------------------------------------------------------------------------
# Data Quality and Provenance
# ---------------------------------------------------------------------------

class SupervisionTier(str, Enum):
    """Data quality tier classification.

    Tier A: speaker / transcript / physical / semantic all high-confidence
    Tier B: transcript and speaker high-confidence, physical or semantic partly pseudo
    Tier C: transcript and basic speaker anchor present, physical supervision sparse
    Tier D: reference-only or auxiliary-only
    """
    A = "tier_a"
    B = "tier_b"
    C = "tier_c"
    D = "tier_d"


class TrajectoryProvenance(str, Enum):
    """Provenance label for trajectory artifacts."""
    FRESH_COMPILE = "fresh_compile"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    CROSS_SPEAKER_TRANSFER = "cross_speaker_transfer"
    PATCHED_REPLAY = "patched_replay"


# ---------------------------------------------------------------------------
# Voice State Supervision Contract
# ---------------------------------------------------------------------------

@dataclass
class VoiceStateSupervision:
    """Canonical 12-D voice_state supervision artifact from curation/export.

    Tensor Contract:
        targets: [B, T, 12] — 12-D physical voice state targets
        observed_mask: [B, T, 12] — True where dimension has usable evidence
        confidence: [B, T, 12] or [B, T, 1] — numeric confidence per dimension
        supervision_tier: SupervisionTier classification
        provenance: str — estimator identity and calibration version
        target_source: canonical source label such as direct/pseudo/absent

    Missing dimensions must be masked in loss, never treated as zero-valued neutral.
    """
    targets: torch.Tensor             # [B, T, 12]
    observed_mask: torch.Tensor       # [B, T, 12] bool
    confidence: torch.Tensor          # [B, T, 12] or [B, T, 1]
    supervision_tier: SupervisionTier = SupervisionTier.C
    provenance: str = "unknown"
    target_source: str = "unknown"


@dataclass
class SpeakerProfile:
    """Canonical Speaker Profile contract for the Casting Gallery.

    Shared by Gradio UI, Python Serve, Rust Engine, and VST.
    Prompt-based only. Legacy adaptor-related fields remain as
    compatibility placeholders.
    """
    speaker_profile_id: str
    reference_audio_hash: str
    speaker_embed: torch.Tensor  # [d_speaker] (usually 192)
    prompt_codec_tokens: torch.Tensor  # [T_prompt, n_codebooks] (original frames)
    schema_version: int = 1
    metadata_version: int = 1
    prompt_kv_cache: Optional[torch.Tensor] = None
    # Compressed summary tokens from Prompt Resampler (Q-Former)
    prompt_summary_tokens: Optional[torch.Tensor] = None  # [n_prompt_summary_tokens, d_model]
    # Deprecated compatibility placeholders.
    adaptor_id: Optional[str] = None
    adaptor_merged: bool = False
    prompt_text_tokens: Optional[torch.Tensor] = None  # [L_prompt]
    prompt_encoder_fingerprint: str = ""
    display_name: str = ""
    language: str = "ja"
    gender: str = "other"  # male, female, other
    license: str = "unknown"
    created_at: str = ""  # ISO 8601
    tags: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TTSFeatureSet:
    """Per-utterance feature set for TTS training."""
    mel: torch.Tensor               # [n_mels, T]
    content: torch.Tensor           # [D_content, T]
    f0: torch.Tensor                # [1, T] or [T]
    spk_embed: torch.Tensor         # [d_speaker]
    phoneme_ids: torch.Tensor       # [L]
    durations: torch.Tensor         # [L]
    language_id: int = 0
    utterance_id: str = ""
    speaker_id: str = ""
    n_frames: int = 0
    n_phonemes: int = 0
    content_dim: int = 0
    text: str = ""
    breath_onsets: Optional[torch.Tensor] = None      # [T]
    breath_durations: Optional[torch.Tensor] = None    # [T]
    breath_intensity: Optional[torch.Tensor] = None    # [T]
    pause_durations: Optional[torch.Tensor] = None     # [T]


@dataclass
class TTSBatch:
    """Collated batch for TTS training."""
    phoneme_ids: torch.Tensor        # [B, L_max]
    durations: torch.Tensor          # [B, L_max]
    language_ids: torch.Tensor       # [B]
    content: torch.Tensor            # [B, D_content, T_max]
    f0: torch.Tensor                 # [B, 1, T_max] or [B, T_max]
    spk_embed: torch.Tensor          # [B, d_speaker]
    mel_target: torch.Tensor         # [B, n_mels, T_max]
    frame_lengths: torch.Tensor      # [B]
    phoneme_lengths: torch.Tensor    # [B]
    utterance_ids: List[str] = field(default_factory=list)
    speaker_ids: List[str] = field(default_factory=list)
    content_dim: int = 0
    breath_onsets: Optional[torch.Tensor] = None       # [B, T_max]
    breath_durations: Optional[torch.Tensor] = None    # [B, T_max]
    breath_intensity: Optional[torch.Tensor] = None    # [B, T_max]
    pause_durations: Optional[torch.Tensor] = None     # [B, T_max]

    def to(self, device: torch.device | str) -> TTSBatch:
        res = {}
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device)
            else:
                res[k] = v
        return TTSBatch(**res)


@dataclass
class FeatureSet:
    """Unified cached features for UCLM (dual-stream)."""
    codec_tokens_a: torch.Tensor  # [8, T]
    codec_tokens_b: torch.Tensor  # [4, T]
    voice_state_explicit: torch.Tensor  # [12, T]
    voice_state_ssl: torch.Tensor       # [128, T]
    spk_embed: torch.Tensor  # [192]
    mel: Optional[torch.Tensor] = None
    f0: Optional[torch.Tensor] = None
    utterance_id: str = ""
    speaker_id: str = ""
    n_frames: int = 0
    waveform: Optional[torch.Tensor] = None


@dataclass
class UCLMFeatureSet(FeatureSet):
    """Features for UCLM multi-task training (VC + TTS)."""
    phoneme_ids: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None
    language_id: int = 0
    text: str = ""
    text_suprasegmentals: Optional[torch.Tensor] = None  # [L, D_SUPRASEGMENTAL]


@dataclass
class UCLM_Batch:
    """Collated batch for DisentangledUCLM training."""
    target_a: torch.Tensor
    target_b: torch.Tensor
    explicit_state: torch.Tensor
    ssl_state: torch.Tensor
    speaker_embed: torch.Tensor
    speaker_id: torch.Tensor
    lengths: torch.Tensor  # Frame counts
    f0_condition: Optional[torch.Tensor] = None
    phoneme_ids: Optional[torch.Tensor] = None
    phoneme_lens: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None
    language_id: Optional[torch.Tensor] = None
    utterance_ids: List[str] = field(default_factory=list)
    # Expressive conditioning fields
    delta_voice_state: Optional[torch.Tensor] = None
    dialogue_context: Optional[torch.Tensor] = None
    acting_intent: Optional[torch.Tensor] = None
    prosody_targets: Optional[torch.Tensor] = None
    context_groups: Optional[torch.Tensor] = None
    prompt_codec_tokens: Optional[torch.Tensor] = None
    bootstrap_alignment: Optional[dict] = None
    # Suprasegmental features (accent/tone/boundary/stress)
    text_suprasegmentals: Optional[torch.Tensor] = None  # [B, L, D_SUPRASEGMENTAL]
    # Voice state supervision (Worker 01 § Physical-First Control)
    voice_state_targets: Optional[torch.Tensor] = None       # [B, T, 12]
    voice_state_observed_mask: Optional[torch.Tensor] = None  # [B, T, 12] bool
    voice_state_confidence: Optional[torch.Tensor] = None     # [B, T, 12] or [B, T, 1]
    voice_state_target_source: Optional[List[str] | torch.Tensor] = None

    def to(self, device: torch.device | str) -> UCLM_Batch:
        res = {}
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device)
            else:
                res[k] = v
        return UCLM_Batch(**res)


# ---------------------------------------------------------------------------
# Programmable Expressive Speech Contracts (Worker 01)
# ---------------------------------------------------------------------------

@dataclass
class PacingControls:
    """Compiled pacing biases for pointer progression."""
    pace: float = 1.0            # Overall speed multiplier
    hold_bias: float = 0.0       # Negative = advance sooner, Positive = hold longer
    boundary_bias: float = 0.0   # Positive = encourage transitions
    phrase_pressure: float = 0.0 # Urgency within phrases
    breath_tendency: float = 0.0 # Likeliness to pause at boundaries


@dataclass
class ActingTextureMacro:
    """Macro controls for acting texture latent.

    These are user-facing abstract controls, NOT raw latent axes.
    Each maps to a learned projection into the latent space.
    """
    intensity: float = 0.5       # Overall acting intensity [0, 1]
    instability: float = 0.2     # Emotional instability / variance [0, 1]
    tenderness: float = 0.3      # Warmth / softness quality [0, 1]
    tension: float = 0.3         # Internal tension / restraint [0, 1]
    spontaneity: float = 0.5     # Naturalness vs controlled delivery [0, 1]
    reference_mix: float = 0.0   # Blend with reference-derived latent [0, 1]


@dataclass
class ActingTextureLatent:
    """Acting texture latent contract.

    Represents the non-physical residual acting quality that cannot be
    captured by explicit physical controls alone.

    Rules:
    - This is a SEPARATE tensor from physical controls
    - Raw latent axes are NOT exposed to end users
    - Users interact via ActingTextureMacro controls
    - Must support same-speaker replay and cross-speaker reuse
    """
    latent: torch.Tensor             # [B, d_acting_latent] (default 24-D)
    source: str = "prior"            # prior | reference | replay
    macro_controls: ActingTextureMacro = field(default_factory=ActingTextureMacro)
    confidence: float = 1.0
    schema_version: str = "1.0"


@dataclass
class IntentCompilerOutput:
    """Authoritative compiled intent for expressive synthesis.

    Produced by the Intent Compiler from prompts/tags/references.
    Targets the 12-D physical + 24-D acting latent contract.
    """
    compile_id: str
    source_prompt: str
    inline_tags: List[str] = field(default_factory=list)

    # Compiled physical targets (12-D)
    physical_targets: torch.Tensor | None = None      # [1, 12]
    physical_confidence: torch.Tensor | None = None    # [1, 12]

    # Acting texture latent prior
    acting_latent_prior: torch.Tensor | None = None    # [1, d_acting_latent]
    acting_macro: ActingTextureMacro = field(default_factory=ActingTextureMacro)

    # Time-varying / pointer-synchronous targets
    local_prosody_plan: dict[int, torch.Tensor] = field(default_factory=dict)

    # Pacing
    pacing: PacingControls = field(default_factory=PacingControls)

    # Optional dialogue state
    dialogue_state: dict = field(default_factory=dict)

    # Metadata
    warnings: List[str] = field(default_factory=list)
    provenance: str = "intent_compiler"
    schema_version: str = "1.0"
    metadata: dict = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    """Deterministic record of a speech performance.

    Supports 12-D physical controls and acting texture latent trajectories.
    Supports replay, edit (patch), and cross-speaker transfer.

    Rules:
    - Wall-clock-only addressing is forbidden (use pointer-synchronous)
    - Opaque unversioned latent blobs are forbidden
    - Patching a local region must be a first-class use case
    """
    trajectory_id: str
    source_compile_id: str

    # 1. Input context
    phoneme_ids: torch.Tensor | None = None            # [1, L]
    text_suprasegmentals: torch.Tensor | None = None   # [1, L, D_supra]

    # 2. Realized pointer trace
    pointer_trace: list = field(default_factory=list)   # [(text_index, frames_spent), ...]

    # 3. Realized physical trajectory (12-D)
    physical_trajectory: torch.Tensor | None = None    # [T, 12]

    # 4. Realized acting latent trajectory or resolved state
    acting_latent_trajectory: torch.Tensor | None = None  # [T, d_acting_latent] or [1, d_acting_latent]
    acting_latent_is_static: bool = False               # True if single static latent, False if time-varying

    # 5. Realized tokens (for bit-exact replay)
    acoustic_trace: torch.Tensor | None = None         # Stream A [8, T]
    control_trace: torch.Tensor | None = None          # Stream B [4, T]

    # 6. Compiled pacing
    applied_pacing: PacingControls = field(default_factory=PacingControls)

    # 7. Provenance
    speaker_profile_id: str = ""
    provenance: TrajectoryProvenance = TrajectoryProvenance.FRESH_COMPILE
    uclm_version: str = ""

    # 8. Versioning
    version: int = 1                # Optimistic concurrency for patching
    schema_version: str = "1.0"
    created_at: str = ""            # ISO 8601
    metadata: dict = field(default_factory=dict)


@dataclass
class BootstrapCacheEntry:
    """Train-ready cache contract for a single utterance.

    This is the canonical output format of the raw-audio bootstrap pipeline.
    """
    utterance_id: str
    corpus_id: str

    # Audio tokens
    acoustic_tokens: torch.Tensor | None = None     # [8, T]
    control_tokens: torch.Tensor | None = None      # [4, T]

    # Speaker
    pseudo_speaker_id: str = ""
    speaker_embed: torch.Tensor | None = None       # [d_speaker]
    diarization_confidence: float = 0.0

    # Text
    text_transcript: str = ""
    enriched_transcript: str = ""
    phoneme_ids: torch.Tensor | None = None         # [L]
    transcript_confidence: float = 0.0
    language: str = ""

    # Physical supervision (12-D)
    physical_targets: torch.Tensor | None = None     # [T, 12]
    physical_observed_mask: torch.Tensor | None = None  # [T, 12] bool
    physical_confidence: torch.Tensor | None = None  # [T, 12]

    # Semantic / acting annotations
    acting_annotations: dict = field(default_factory=dict)  # scene_summary, dialogue_intent, emotion, acting_hint

    # Quality metadata
    supervision_tier: str = "tier_d"
    quality_score: float = 0.0
    n_frames: int = 0
    duration_sec: float = 0.0

    schema_version: str = "1.0"
