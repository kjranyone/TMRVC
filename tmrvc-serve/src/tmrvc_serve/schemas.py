"""Pydantic schemas for the TTS API."""

from __future__ import annotations

import enum
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field
from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS

StylePreset = Literal["default", "asmr_soft", "asmr_intimate"]
CFGModeStr = Literal["off", "full", "lazy", "distilled"]


# ---------------------------------------------------------------------------
# v4 Physical Control Names (12-D canonical ordering)
# ---------------------------------------------------------------------------

V4_PHYSICAL_CONTROL_NAMES: list[str] = [
    "pitch_level", "pitch_range", "energy_level", "pressedness",
    "spectral_tilt", "breathiness", "voice_irregularity", "openness",
    "aperiodicity", "formant_shift", "vocal_effort", "creak",
]


# ---------------------------------------------------------------------------
# Priority enum for real-time WebSocket
# ---------------------------------------------------------------------------


class Priority(int, enum.Enum):
    """Speak request priority (lower value = higher priority)."""

    URGENT = 0
    NORMAL = 1
    LOW = 2


class DialogueTurnSchema(BaseModel):
    """A single dialogue turn for context."""

    speaker: str
    text: str
    emotion: str | None = None


# ---------------------------------------------------------------------------
# Shared TTS schemas
# ---------------------------------------------------------------------------


class PointerTelemetry(BaseModel):
    """Pointer state telemetry from a TTS generation run."""

    text_index: int = 0
    progress: float = 0.0
    total_phonemes: int = 0
    frames_generated: int = 0
    stall_frames: int = 0
    max_frames_per_unit: int = 50
    frames_on_current_unit: int = 0
    skip_protection_threshold: float = 0.3
    forced_advance_count: int = 0
    skip_protection_count: int = 0
    cfg_cache_age: int = 0


class SpeakerProfileResponse(BaseModel):
    """Response containing speaker profile information."""

    speaker_profile_id: str
    display_name: str = ""
    language: str = "ja"
    gender: str = "other"
    has_prompt_tokens: bool = False
    has_summary_tokens: bool = False
    has_adaptor: bool = False
    created_at: str = ""
    tags: list[str] = Field(default_factory=list)


class AdminHealthResponse(BaseModel):
    """Detailed system health (used by admin routes)."""

    status: str = "ok"
    models_loaded: bool = False
    device: str = "cpu"
    cuda_available: bool = False
    cuda_memory_allocated_mb: float = 0.0
    cuda_memory_reserved_mb: float = 0.0
    uptime_seconds: float = 0.0


class AdminTelemetryResponse(BaseModel):
    """Runtime telemetry snapshot."""

    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    avg_tts_latency_ms: float = 0.0
    avg_vc_latency_ms: float = 0.0
    recent_rtf: float = 0.0
    tts_mode: str = "pointer"
    model_checkpoint: str = ""


class CharacterInfo(BaseModel):
    """Character info returned by GET /characters."""

    id: str
    name: str
    personality: str = ""
    voice_description: str = ""
    language: Literal["ja", "en", "zh", "ko"] = "ja"


class CharacterCreateRequest(BaseModel):
    """Request body for POST /characters."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str
    personality: str = ""
    voice_description: str = ""
    language: Literal["ja", "en", "zh", "ko"] = "ja"
    speaker_file: str | None = Field(None, description="Path to .tmrvc_speaker file.")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    models_loaded: bool = False
    characters_count: int = 0


# ---------------------------------------------------------------------------
# WebSocket schemas
# ---------------------------------------------------------------------------

# Client → Server


class WSSpeakRequest(BaseModel):
    """Client → Server: request speech synthesis with priority and interrupt."""

    type: str = "speak"
    text: str = Field(..., min_length=1)
    character_id: str = ""
    emotion: str | None = None
    hint: str | None = None
    situation: str | None = None
    style_preset: StylePreset = "default"
    priority: Priority = Priority.NORMAL
    interrupt: bool = False
    speed: float | None = Field(None, ge=0.5, le=2.0)


class WSCancelRequest(BaseModel):
    """Client → Server: cancel all queued speak requests."""

    type: str = "cancel"


class WSConfigureRequest(BaseModel):
    """Client → Server: update session configuration."""

    type: str = "configure"
    character_id: str | None = None
    situation: str | None = None
    style_preset: StylePreset | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    scene_reset: bool = Field(
        False,
        description="Reset scene state to initial (zeroed) state.",
    )


# Server → Client


class WSStyleMessage(BaseModel):
    """Server → Client: emotion/VAD sent before audio for avatar expression sync."""

    type: str = "style"
    emotion: str = "neutral"
    vad: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    reasoning: str = ""
    seq: int = 0


class WSAudioMessage(BaseModel):
    """Server → Client: PCM float32 chunk (base64-encoded)."""

    type: str = "audio"
    data: str = ""
    seq: int = 0
    chunk_index: int = 0
    is_last: bool = False


class WSQueueStatus(BaseModel):
    """Server → Client: queue depth and speaking state."""

    type: str = "queue_status"
    pending: int = 0
    speaking: bool = False


class WSSkipped(BaseModel):
    """Server → Client: notification that a queued item was skipped."""

    type: str = "skipped"
    text: str = ""
    reason: str = ""


class WSError(BaseModel):
    """Server → Client: error message."""

    type: str = "error"
    detail: str = ""


# ---------------------------------------------------------------------------
# VC (Voice Conversion) schemas
# ---------------------------------------------------------------------------


class SpeakerEmbedding(BaseModel):
    """Speaker embedding vector (192 dimensions)."""

    data: list[float] = Field(
        ...,
        min_length=192,
        max_length=192,
        description="L2-normalized speaker embedding vector",
    )


class VCSessionInfo(BaseModel):
    """Active VC session information."""

    session_id: str = Field(..., description="Unique session identifier")
    created_at: float = Field(..., description="Unix timestamp of session creation")
    last_activity: float = Field(..., description="Unix timestamp of last activity")


class VCEngineStats(BaseModel):
    """VC engine pool statistics."""

    active_sessions: int = Field(..., description="Number of active sessions")
    max_sessions: int = Field(..., description="Maximum concurrent sessions allowed")
    is_ready: bool = Field(..., description="Whether engine is ready for inference")
    gpu_instances: int = Field(2, description="Number of GPU model instances")
    cpu_instances: int = Field(4, description="Number of CPU model instances")


class VCBatchRequest(BaseModel):
    """Batch VC processing request."""

    speaker_embedding: SpeakerEmbedding
    audio: list[float] = Field(
        ...,
        min_length=480,
        description="Audio samples (float32, 24kHz mono, minimum 480 samples)",
    )


class VCBatchResponse(BaseModel):
    """Batch VC processing response."""

    rtf: float = Field(
        ...,
        ge=0,
        description="Real-Time Factor (<1 means faster than real-time)",
    )
    elapsed_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )
    audio_duration_ms: float = Field(
        ...,
        ge=0,
        description="Output audio duration in milliseconds",
    )
    audio: list[float] = Field(
        ...,
        description="Output audio samples (float32, 24kHz mono)",
    )


class VCRTFTestResponse(BaseModel):
    """RTF benchmark test response."""

    rtf: float = Field(..., description="Real-Time Factor")
    elapsed_ms: float = Field(..., description="Processing time in milliseconds")
    audio_duration_ms: float = Field(
        ..., description="Input audio duration in milliseconds"
    )
    samples_processed: int = Field(..., description="Number of samples processed")


# ---------------------------------------------------------------------------
# WebSocket Protocol Documentation
# ---------------------------------------------------------------------------

WEBSOCKET_PROTOCOL_VC_STREAM = """
# WebSocket VC Streaming Protocol

## Endpoint: `/ws/vc/stream`

### Connection Flow

1. **Connect**: Client opens WebSocket
2. **Initialize**: Client sends speaker embedding (768 bytes = 192 float32)
3. **Confirm**: Server sends session ID (8 bytes UTF-8, null-padded)
4. **Stream**: Client sends audio chunks, Server returns processed audio

### Binary Message Formats

| Direction | Format | Size |
|-----------|--------|------|
| Client→Server (init) | Speaker embedding | 768 bytes |
| Server→Client (confirm) | Session ID | 8 bytes |
| Client→Server (audio) | PCM samples | N×4 bytes |
| Server→Client (audio) | PCM samples | 1920 bytes (480 samples) |

### Example (Python)

```python
import asyncio
import websockets
import numpy as np

async def stream_vc():
    async with websockets.connect("ws://localhost:8000/ws/vc/stream") as ws:
        # Initialize
        spk_embed = np.random.randn(192).astype(np.float32)
        await ws.send(spk_embed.tobytes())
        
        session_id = await ws.recv()
        
        # Stream
        for _ in range(50):
            audio = np.random.randn(480).astype(np.float32)
            await ws.send(audio.tobytes())
            output = np.frombuffer(await ws.recv(), dtype=np.float32)

asyncio.run(stream_vc())
```

### Concurrency

- Max concurrent sessions: 20
- Session timeout: 5 min
- GPU parallel inference: 2
- CPU fallback: automatic
"""


WEBSOCKET_DOCS = {
    "vc_stream": WEBSOCKET_PROTOCOL_VC_STREAM,
}


# ---------------------------------------------------------------------------
# Artifact download contract (Worker 04, task 22)
# ---------------------------------------------------------------------------


class ArtifactResponse(BaseModel):
    """Artifact download envelope for WebUI-initiated exports."""

    artifact_id: str
    artifact_type: str
    download_url: str
    expires_at: Optional[str] = None
    provenance_summary: dict = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    """Status of an asynchronous backend job (e.g. curation, export)."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    created_at: str = ""


class CurationActionRequest(BaseModel):
    """Request to perform an action on a curation record."""

    action: Literal["promote", "reject", "review", "reset"]
    expected_version: int
    bucket: Optional[str] = None
    rationale: str = ""


# ---------------------------------------------------------------------------
# Physical / Acting / Pacing Control Schemas
# ---------------------------------------------------------------------------


class PhysicalControls(BaseModel):
    """12-D explicit physical control vector."""
    pitch_level: float = Field(0.5, ge=0.0, le=1.0)
    pitch_range: float = Field(0.3, ge=0.0, le=1.0)
    energy_level: float = Field(0.5, ge=0.0, le=1.0)
    pressedness: float = Field(0.35, ge=0.0, le=1.0)
    spectral_tilt: float = Field(0.5, ge=0.0, le=1.0)
    breathiness: float = Field(0.2, ge=0.0, le=1.0)
    voice_irregularity: float = Field(0.15, ge=0.0, le=1.0)
    openness: float = Field(0.5, ge=0.0, le=1.0)
    aperiodicity: float = Field(0.2, ge=0.0, le=1.0)
    formant_shift: float = Field(0.5, ge=0.0, le=1.0)
    vocal_effort: float = Field(0.4, ge=0.0, le=1.0)
    creak: float = Field(0.1, ge=0.0, le=1.0)

    def to_list(self) -> list[float]:
        return [
            self.pitch_level, self.pitch_range, self.energy_level,
            self.pressedness, self.spectral_tilt, self.breathiness,
            self.voice_irregularity, self.openness, self.aperiodicity,
            self.formant_shift, self.vocal_effort, self.creak,
        ]


class PhysicalControlWithConfidence(BaseModel):
    """Single physical control dimension with value and confidence.

    Confidence reflects how reliably the value was estimated during
    bootstrap.  Inference may downweight low-confidence dimensions.
    """
    value: float = Field(0.5, ge=0.0, le=1.0, description="Control value [0, 1]")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Estimation confidence.  1.0 = fully trusted; 0.0 = unknown/absent.",
    )


class ConfidenceBearingPhysicalControls(BaseModel):
    """12-D physical controls where each dimension carries a confidence score.

    Used when the control target originates from the bootstrap pipeline's
    DSP/SSL extraction stage which may be partially observed.
    """
    pitch_level: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    pitch_range: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    energy_level: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    pressedness: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    spectral_tilt: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    breathiness: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    voice_irregularity: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    openness: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    aperiodicity: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    formant_shift: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    vocal_effort: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)
    creak: PhysicalControlWithConfidence = Field(default_factory=PhysicalControlWithConfidence)

    def to_value_list(self) -> list[float]:
        """Return only the 12 values in canonical order."""
        return [
            self.pitch_level.value, self.pitch_range.value,
            self.energy_level.value, self.pressedness.value,
            self.spectral_tilt.value, self.breathiness.value,
            self.voice_irregularity.value, self.openness.value,
            self.aperiodicity.value, self.formant_shift.value,
            self.vocal_effort.value, self.creak.value,
        ]

    def to_confidence_list(self) -> list[float]:
        """Return only the 12 confidence values in canonical order."""
        return [
            self.pitch_level.confidence, self.pitch_range.confidence,
            self.energy_level.confidence, self.pressedness.confidence,
            self.spectral_tilt.confidence, self.breathiness.confidence,
            self.voice_irregularity.confidence, self.openness.confidence,
            self.aperiodicity.confidence, self.formant_shift.confidence,
            self.vocal_effort.confidence, self.creak.confidence,
        ]

    def to_plain_controls(self) -> "PhysicalControls":
        """Downcast to a plain PhysicalControls (drop confidence)."""
        vals = self.to_value_list()
        return PhysicalControls(**dict(zip(V4_PHYSICAL_CONTROL_NAMES, vals)))


class V4SpeakerProfile(BaseModel):
    """Speaker profile derived from the v4 bootstrap pipeline.

    This schema carries bootstrap-derived information that enriches
    the existing ``SpeakerProfile`` core contract for serving.
    """
    pseudo_speaker_id: str = Field(
        ..., description="Cluster-assigned pseudo speaker ID from bootstrap diarization.",
    )
    speaker_embed: list[float] = Field(
        ...,
        min_length=192, max_length=192,
        description="192-dim L2-normalised speaker embedding from ECAPA-TDNN.",
    )
    display_name: str = ""
    language: str = "ja"
    gender: str = "other"

    # Bootstrap quality metadata
    supervision_tier: str = Field(
        "tier_d",
        pattern="^tier_[abcd]$",
        description="Bootstrap supervision tier (tier_a through tier_d).",
    )
    diarization_confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Confidence of diarization cluster assignment.",
    )
    utterance_count: int = Field(
        0, ge=0,
        description="Number of utterances assigned to this speaker in the bootstrap corpus.",
    )

    # Physical control priors (per-speaker average from bootstrap)
    physical_prior: Optional[ConfidenceBearingPhysicalControls] = Field(
        None,
        description="Per-speaker average 12-D physical controls from bootstrap, with confidence.",
    )

    # Reference data paths (resolved server-side)
    prompt_codec_tokens_path: Optional[str] = Field(
        None,
        description="Path to pre-encoded prompt codec tokens (.npy).",
    )
    prompt_summary_tokens_path: Optional[str] = Field(
        None,
        description="Path to pre-encoded summary tokens (.npy).",
    )

    schema_version: str = "v4.0"


class ActingMacroControls(BaseModel):
    """Acting texture macro controls (user-facing, not raw latent)."""
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    instability: float = Field(0.2, ge=0.0, le=1.0)
    tenderness: float = Field(0.3, ge=0.0, le=1.0)
    tension: float = Field(0.3, ge=0.0, le=1.0)
    spontaneity: float = Field(0.5, ge=0.0, le=1.0)
    reference_mix: float = Field(0.0, ge=0.0, le=1.0)


class PacingControlsSchema(BaseModel):
    """Pacing controls for pointer progression."""
    pace: float = Field(1.0, ge=0.3, le=3.0)
    hold_bias: float = Field(0.0, ge=-1.0, le=1.0)
    boundary_bias: float = Field(0.0, ge=-1.0, le=1.0)
    phrase_pressure: float = Field(0.0, ge=-1.0, le=1.0)
    breath_tendency: float = Field(0.0, ge=-1.0, le=1.0)


class TTSRequestSimple(BaseModel):
    """Simple-mode TTS request.

    Minimal interface for basic synthesis without explicit physical/acting controls.
    """
    text: str
    speaker_profile_id: Optional[str] = None
    reference_audio_base64: Optional[str] = Field(None, max_length=67_108_864)
    language: Optional[str] = None
    emotion: Optional[str] = None
    speed: float = Field(1.0, ge=0.3, le=3.0)

    # v4 fields
    pseudo_speaker_id: Optional[str] = Field(
        None,
        description="Bootstrap-derived pseudo speaker ID.  Resolved to a V4SpeakerProfile server-side.",
    )
    enriched_transcript: Optional[str] = Field(
        None,
        description="Enriched transcript with inline acting tags (e.g. '[laugh] hello [emphasis] world').",
    )

    schema_version: str = "1.0"


class TTSRequestAdvanced(BaseModel):
    """Advanced physical-control mode TTS request.

    Full 12-D physical control with optional acting macro controls.
    """
    text: str
    speaker_profile_id: Optional[str] = None
    reference_audio_base64: Optional[str] = Field(None, max_length=67_108_864)
    language: Optional[str] = None

    # 12-D physical controls
    physical_controls: PhysicalControls = Field(default_factory=PhysicalControls)
    delta_physical_controls: Optional[PhysicalControls] = None

    # v4: confidence-bearing physical controls from bootstrap
    confidence_bearing_controls: Optional[ConfidenceBearingPhysicalControls] = Field(
        None,
        description="12-D controls with per-dimension confidence from bootstrap.  "
                    "When provided, takes precedence over plain physical_controls.",
    )

    # v4 fields
    pseudo_speaker_id: Optional[str] = Field(
        None,
        description="Bootstrap-derived pseudo speaker ID.",
    )
    enriched_transcript: Optional[str] = Field(
        None,
        description="Enriched transcript with inline acting tags.",
    )

    # Acting macro controls
    acting_controls: ActingMacroControls = Field(default_factory=ActingMacroControls)

    # Pacing
    pacing: PacingControlsSchema = Field(default_factory=PacingControlsSchema)

    # CFG
    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)
    cfg_mode: str = Field("full", pattern="^(off|full|lazy|distilled)$")

    schema_version: str = "1.0"


class TTSRequestPrompt(BaseModel):
    """Prompt/acting mode TTS request.

    Natural language acting instructions compiled via Intent Compiler.
    """
    text: str
    speaker_profile_id: Optional[str] = None
    reference_audio_base64: Optional[str] = Field(None, max_length=67_108_864)
    language: Optional[str] = None

    # Acting instruction
    acting_prompt: str = ""
    acting_tags: list[str] = Field(default_factory=list)
    scene_context: str = ""

    # Optional physical overrides
    physical_overrides: Optional[PhysicalControls] = None

    # Pacing
    pacing: PacingControlsSchema = Field(default_factory=PacingControlsSchema)

    # CFG
    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)
    cfg_mode: str = Field("full", pattern="^(off|full|lazy|distilled)$")

    schema_version: str = "1.0"


class TTSRequestReplay(BaseModel):
    """Trajectory replay mode TTS request.

    Deterministic replay from a frozen TrajectoryRecord.
    Must NOT silently reinterpret or recompile prompts.
    """
    trajectory_id: str
    speaker_profile_id: Optional[str] = None  # None = same speaker as original
    schema_version: str = "1.0"


class TTSResponse(BaseModel):
    """TTS response with full provenance."""
    audio_base64: str
    sample_rate: int = 24000
    duration_sec: float = 0.0

    # Provenance
    trajectory_id: Optional[str] = None
    provenance: str = "fresh_compile"  # fresh_compile | deterministic_replay | cross_speaker_transfer | patched_replay

    # Telemetry
    rtf: float = 0.0
    gen_time_ms: float = 0.0
    cfg_mode: str = "full"
    forced_advance_count: int = 0
    skip_protection_count: int = 0

    schema_version: str = "1.0"


class CompileRequest(BaseModel):
    """Request to compile acting intent into canonical controls."""
    text: str
    acting_prompt: str = ""
    acting_tags: list[str] = Field(default_factory=list)
    scene_context: str = ""
    reference_audio_base64: Optional[str] = Field(None, max_length=67_108_864)
    schema_version: str = "1.0"


class CompileResponse(BaseModel):
    """Compiled intent output."""
    compile_id: str
    physical_targets: list[float]     # [12]
    physical_confidence: list[float]  # [12]
    acting_macro: ActingMacroControls
    pacing: PacingControlsSchema
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = "1.0"


class PatchRequest(BaseModel):
    """Patch a local region of a trajectory."""
    start_pointer_index: int
    end_pointer_index: int
    physical_overrides: Optional[PhysicalControls] = None
    acting_macro_overrides: Optional[ActingMacroControls] = None
    pacing_overrides: Optional[PacingControlsSchema] = None
    expected_version: int  # Optimistic concurrency
    schema_version: str = "1.0"

    from pydantic import model_validator

    @model_validator(mode="after")
    def _check_start_before_end(self) -> "PatchRequest":
        if self.start_pointer_index >= self.end_pointer_index:
            raise ValueError(
                f"start_pointer_index ({self.start_pointer_index}) must be "
                f"less than end_pointer_index ({self.end_pointer_index})"
            )
        return self


class TransferRequest(BaseModel):
    """Transfer a trajectory's acting to a different speaker."""
    target_speaker_profile_id: str
    schema_version: str = "1.0"


class TrajectoryInfo(BaseModel):
    """Trajectory artifact metadata for API responses."""
    trajectory_id: str
    source_compile_id: str
    speaker_profile_id: str
    provenance: str
    version: int
    schema_version: str
    created_at: str
    n_frames: int = 0
    duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Backward-compatible unified request schemas
# ---------------------------------------------------------------------------


def _validate_12d(v: list[float] | None) -> list[float] | None:
    if v is not None and len(v) != 12:
        raise ValueError("must be exactly 12 elements")
    return v


class TTSRequest(BaseModel):
    """Unified TTS request (backward-compatible with v3 tests)."""
    text: str
    character_id: str = ""
    speaker_profile_id: Optional[str] = None
    reference_audio_base64: Optional[str] = Field(None, max_length=67_108_864)
    reference_text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[str] = None

    # Physical controls (12-D)
    explicit_voice_state: Optional[list[float]] = None
    delta_voice_state: Optional[list[float]] = None

    # Pacing
    pace: float = Field(1.0, ge=0.5, le=3.0)
    hold_bias: float = Field(0.0, ge=-1.0, le=1.0)
    boundary_bias: float = Field(0.0, ge=-1.0, le=1.0)

    # Expressive
    dialogue_context: Optional[list[float]] = None
    acting_intent: Optional[list[float]] = None

    # Style
    style_preset: StylePreset = "default"

    # v4 fields
    pseudo_speaker_id: Optional[str] = None
    enriched_transcript: Optional[str] = None
    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)

    schema_version: str = "1.0"

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

    from pydantic import model_validator

    @model_validator(mode="after")
    def _check_voice_state_lengths(self) -> "TTSRequest":
        _validate_12d(self.explicit_voice_state)
        _validate_12d(self.delta_voice_state)
        return self


class TTSStreamRequest(BaseModel):
    """Unified streaming TTS request (backward-compatible with v3 tests)."""
    text: str
    character_id: str = ""
    speaker_profile_id: Optional[str] = None
    reference_audio_base64: Optional[str] = Field(None, max_length=67_108_864)
    reference_text: Optional[str] = None
    language: Optional[str] = None

    explicit_voice_state: Optional[list[float]] = None
    delta_voice_state: Optional[list[float]] = None

    pace: float = Field(1.0, ge=0.5, le=3.0)
    hold_bias: float = Field(0.0, ge=-1.0, le=1.0)
    boundary_bias: float = Field(0.0, ge=-1.0, le=1.0)

    dialogue_context: Optional[list[float]] = None
    acting_intent: Optional[list[float]] = None

    cfg_scale: float = Field(1.5, ge=0.5, le=5.0)

    schema_version: str = "1.0"

    from pydantic import model_validator

    @model_validator(mode="after")
    def _check_voice_state_lengths(self) -> "TTSStreamRequest":
        _validate_12d(self.explicit_voice_state)
        _validate_12d(self.delta_voice_state)
        return self
