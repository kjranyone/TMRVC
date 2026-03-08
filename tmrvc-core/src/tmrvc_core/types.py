"""Shared data types for the TMRVC pipeline (UCLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List

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

    def step_pointer(
        self,
        advance_prob: float,
        progress_delta: float,
        boundary_confidence: float = 0.0,
    ) -> bool:
        """Canonical pointer state-transition logic.

        Implements forced advance on stall, skip-protection against premature
        advance, and normal advance/hold behaviour.

        Returns:
            ``True`` if the pointer advanced to the next text unit.
        """
        self.last_advance_score = float(advance_prob)
        self.boundary_confidence = float(boundary_confidence)

        # 1. Track time on the current unit
        self.frames_on_current_unit += 1

        # 2. Accumulate progress
        self.progress += progress_delta

        # 3. Forced advance when stuck too long on one unit
        if self.frames_on_current_unit >= self.max_frames_per_unit:
            self.text_index += 1
            self.progress = self.progress * 0 + 0.0  # keep tensor type if applicable
            self.frames_on_current_unit = 0
            self.stall_frames = 0
            self.forced_advance_count += 1
            return True

        # 4. High-confidence advance with skip-protection
        if advance_prob > 0.5 and self.progress >= 1.0:
            if boundary_confidence >= self.skip_protection_threshold:
                # Normal boundary advance
                self.text_index += 1
                self.progress = self.progress * 0 + 0.0
                self.frames_on_current_unit = 0
                self.stall_frames = 0
                return True
            else:
                # Skip-protection blocks the advance
                self.skip_protection_count += 1
                return False

        # 5. Advance on either signal alone
        if advance_prob > 0.5 or self.progress >= 1.0:
            self.text_index += 1
            self.progress = self.progress * 0 + 0.0
            self.frames_on_current_unit = 0
            self.stall_frames = 0
            return True

        # 6. Hold — no advance
        self.stall_frames += 1
        return False


# ---------------------------------------------------------------------------
# CFG Unconditional Mask Contract (Frozen)
# ---------------------------------------------------------------------------

# Fields zeroed during CFG unconditional pass:
CFG_ZEROED_FIELDS = frozenset({
    "explicit_voice_state",   # [B, T, 8] or [B, 8]
    "delta_voice_state",      # [B, T, 8] or [B, 8]
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
# Voice State Supervision Contract
# ---------------------------------------------------------------------------

@dataclass
class VoiceStateSupervision:
    """Canonical voice_state supervision artifact from curation/export.

    Tensor Contract:
        targets: [B, T, 8] — 8-D physical voice state targets
        observed_mask: [B, T, 8] — True where dimension has usable evidence
        confidence: [B, T, 8] or [B, T, 1] — numeric confidence per dimension
        provenance: str — estimator identity and calibration version
        target_source: canonical source label such as direct/pseudo/absent

    Missing dimensions must be masked in loss, never treated as zero-valued neutral.
    """
    targets: torch.Tensor        # [B, T, 8]
    observed_mask: torch.Tensor  # [B, T, 8] bool
    confidence: torch.Tensor     # [B, T, 8] or [B, T, 1]
    provenance: str = "unknown"
    target_source: str = "unknown"


@dataclass
class SpeakerProfile:
    """Canonical Speaker Profile contract for the Casting Gallery.

    Shared by Gradio UI, Python Serve, Rust Engine, and VST.
    v3.0 mainline is prompt-based only. Legacy adaptor-related fields remain as
    compatibility placeholders and are not part of the canonical v3.0 contract.
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
    # Deprecated compatibility placeholders: not canonical in v3.0.
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
class FeatureSet:
    """Unified cached features for UCLM (dual-stream)."""
    codec_tokens_a: torch.Tensor  # [8, T]
    codec_tokens_b: torch.Tensor  # [4, T]
    voice_state_explicit: torch.Tensor  # [8, T]
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
    # v3 expressive conditioning fields
    delta_voice_state: Optional[torch.Tensor] = None
    dialogue_context: Optional[torch.Tensor] = None
    acting_intent: Optional[torch.Tensor] = None
    prosody_targets: Optional[torch.Tensor] = None
    context_groups: Optional[torch.Tensor] = None
    prompt_codec_tokens: Optional[torch.Tensor] = None
    bootstrap_alignment: Optional[dict] = None
    # Suprasegmental features (v3 accent/tone/boundary/stress)
    text_suprasegmentals: Optional[torch.Tensor] = None  # [B, L, D_SUPRASEGMENTAL]
    # Voice state supervision (Worker 01 § Physical-First Control)
    voice_state_targets: Optional[torch.Tensor] = None       # [B, T, 8]
    voice_state_observed_mask: Optional[torch.Tensor] = None  # [B, T, 8] bool
    voice_state_confidence: Optional[torch.Tensor] = None     # [B, T, 8] or [B, T, 1]
    voice_state_target_source: Optional[List[str] | torch.Tensor] = None

    def to(self, device: torch.device | str) -> UCLM_Batch:
        res = {}
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device)
            else:
                res[k] = v
        return UCLM_Batch(**res)
