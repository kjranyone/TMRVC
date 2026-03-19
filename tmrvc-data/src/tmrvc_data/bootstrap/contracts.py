"""v4 bootstrap pipeline contracts and data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Dict, Any


class BootstrapStage(IntEnum):
    """Canonical v4 bootstrap pipeline stages."""
    INGEST = 0
    AUDIO_NORMALIZATION = 1
    VAD_SEGMENTATION = 2
    REJECTION = 3            # overlap / music / noise
    DIARIZATION = 4          # speaker clustering
    PSEUDO_SPEAKER = 5       # pseudo speaker assignment
    SPEAKER_EMBEDDING = 6
    TRANSCRIPTION = 7        # Whisper
    TEXT_NORMALIZATION = 8   # norm + G2P
    PHYSICAL_EXTRACTION = 9  # DSP / SSL features
    SEMANTIC_ANNOTATION = 10 # LLM acting / intent
    CONFIDENCE_SCORING = 11  # quality scoring + artifact masking
    CACHE_EXPORT = 12        # train-ready cache


@dataclass
class BootstrapConfig:
    """Configuration for the v4 bootstrap pipeline."""
    corpus_dir: Path = Path("data/raw_corpus")
    output_dir: Path = Path("data/v4_cache")

    # Audio
    target_sample_rate: int = 24000
    loudness_target_lufs: float = -23.0
    segment_min_sec: float = 0.5
    segment_max_sec: float = 30.0

    # VAD
    vad_aggressiveness: int = 2       # 0-3
    min_speech_duration_ms: int = 250

    # Rejection
    overlap_rejection_threshold: float = 0.3
    music_rejection_threshold: float = 0.5
    snr_rejection_threshold_db: float = 10.0

    # Diarization
    min_cluster_size: int = 5
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Transcription
    whisper_model: str = "large-v3"
    whisper_language: Optional[str] = None  # None = auto-detect

    # Physical extraction
    physical_dim: int = 12  # v4 12-D
    ssl_model: str = "wavlm-large"

    # Quality gates
    min_quality_score: float = 0.3
    min_transcript_confidence: float = 0.5
    min_diarization_confidence: float = 0.4

    # Processing
    num_workers: int = 4
    device: str = "cuda"
    batch_size: int = 16


@dataclass
class SegmentInfo:
    """Metadata for a single audio segment after VAD."""
    segment_id: str
    source_file: str
    start_sec: float
    end_sec: float
    duration_sec: float
    sample_rate: int = 24000


@dataclass
class BootstrapUtterance:
    """A single utterance moving through the bootstrap pipeline.

    Fields are populated progressively as stages complete.
    """
    utterance_id: str
    corpus_id: str
    source_file: str

    # Stage 1: Audio normalization
    audio_path: Optional[Path] = None
    sample_rate: int = 24000
    duration_sec: float = 0.0

    # Stage 2-3: VAD + Rejection
    segment: Optional[SegmentInfo] = None
    is_rejected: bool = False
    rejection_reason: str = ""

    # Stage 4-5: Diarization + pseudo speaker
    pseudo_speaker_id: str = ""
    diarization_confidence: float = 0.0
    speaker_cluster_id: int = -1

    # Stage 6: Speaker embedding
    speaker_embed: Optional[Any] = None  # numpy [d_speaker]

    # Stage 7: Transcription (Whisper + LLM owned)
    text_transcript: str = ""
    enriched_transcript: str = ""
    transcript_confidence: float = 0.0
    language: str = ""

    # Stage 8: Text normalization (Whisper + LLM owned)
    phoneme_ids: Optional[Any] = None  # numpy [L]

    # Stage 9: Physical extraction (DSP/SSL owned)
    physical_targets: Optional[Any] = None      # numpy [T, 12]
    physical_observed_mask: Optional[Any] = None # numpy [T, 12]
    physical_confidence: Optional[Any] = None    # numpy [T, 12]

    # Stage 10: Semantic annotation (LLM owned)
    acting_annotations: Dict[str, Any] = field(default_factory=dict)
    # Keys: scene_summary, dialogue_intent, emotion_description, acting_hint

    # Stage 11: Quality scoring
    quality_score: float = 0.0
    supervision_tier: str = "tier_d"

    # Stage 12: Cache export
    acoustic_tokens: Optional[Any] = None   # numpy [8, T]
    control_tokens: Optional[Any] = None    # numpy [4, T]
    n_frames: int = 0

    # Processing metadata
    stage_completed: int = -1
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class BootstrapResult:
    """Result summary from a bootstrap pipeline run."""
    corpus_id: str
    total_files: int = 0
    total_segments: int = 0
    accepted_utterances: int = 0
    rejected_utterances: int = 0

    # Per-tier counts
    tier_a_count: int = 0
    tier_b_count: int = 0
    tier_c_count: int = 0
    tier_d_count: int = 0

    # Quality metrics
    mean_quality_score: float = 0.0
    mean_transcript_confidence: float = 0.0
    mean_diarization_confidence: float = 0.0
    physical_label_coverage: float = 0.0

    # Rejection breakdown
    overlap_rejections: int = 0
    music_rejections: int = 0
    noise_rejections: int = 0
    short_rejections: int = 0

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class TrainReadyCacheContract:
    """v4 train-ready cache contract specification.

    Defines the minimum required fields for each utterance
    in the v4 train-ready cache. This supersedes the v3 dataset contract.
    """
    REQUIRED_FIELDS: tuple = (
        "acoustic_tokens",       # [8, T]
        "control_tokens",        # [4, T]
        "pseudo_speaker_id",     # str
        "speaker_embed",         # [d_speaker]
        "text_transcript",       # str
        "enriched_transcript",   # str (inline acting tags)
        "phoneme_ids",           # [L]
        "language",              # str
        "physical_targets",      # [T, 12]
        "acting_annotations",    # dict
        "quality_metadata",      # dict with supervision_tier, quality_score, etc.
    )

    CACHE_LAYOUT: str = "v4_cache/{corpus_id}/{pseudo_speaker_id}/{utterance_id}/"

    SCHEMA_VERSION: str = "v4.0"

    # File naming convention within each utterance directory
    FILE_NAMES: Dict[str, str] = field(default_factory=lambda: {
        "acoustic_tokens": "acoustic_tokens.npy",
        "control_tokens": "control_tokens.npy",
        "speaker_embed": "spk_embed.npy",
        "phoneme_ids": "phoneme_ids.npy",
        "physical_targets": "physical_targets.npy",
        "physical_observed_mask": "physical_observed_mask.npy",
        "physical_confidence": "physical_confidence.npy",
        "meta": "meta.json",
    })
