"""Data models for the AI Curation System."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RecordStatus(str, Enum):
    INGESTED = "ingested"
    ANNOTATING = "annotating"
    SCORED = "scored"
    PROMOTED = "promoted"
    REVIEW = "review"
    REJECTED = "rejected"
    EXPORTED = "exported"


# Valid status transitions.  Each key maps to the set of statuses it may
# transition *to*.  Any transition not listed here is invalid and should
# be rejected by the service layer.
VALID_TRANSITIONS: Dict[RecordStatus, frozenset[RecordStatus]] = {
    RecordStatus.INGESTED: frozenset({
        RecordStatus.ANNOTATING,
        RecordStatus.REJECTED,
    }),
    RecordStatus.ANNOTATING: frozenset({
        RecordStatus.ANNOTATING,   # re-entering annotating on pass N
        RecordStatus.SCORED,
        RecordStatus.REJECTED,
    }),
    RecordStatus.SCORED: frozenset({
        RecordStatus.PROMOTED,
        RecordStatus.REVIEW,
        RecordStatus.REJECTED,
    }),
    RecordStatus.PROMOTED: frozenset({
        RecordStatus.EXPORTED,
        RecordStatus.REVIEW,       # demotion after re-score
        RecordStatus.ANNOTATING,   # re-enter for refinement pass
    }),
    RecordStatus.REVIEW: frozenset({
        RecordStatus.ANNOTATING,   # re-process
        RecordStatus.PROMOTED,
        RecordStatus.REJECTED,
    }),
    RecordStatus.REJECTED: frozenset({
        RecordStatus.ANNOTATING,   # rescued via re-process
    }),
    RecordStatus.EXPORTED: frozenset({
        RecordStatus.ANNOTATING,   # re-ingest for new pass
    }),
}


class PromotionBucket(str, Enum):
    NONE = "none"
    TTS_MAINLINE = "tts_mainline"
    VC_PRIOR = "vc_prior"
    EXPRESSIVE_PRIOR = "expressive_prior"
    HOLDOUT_EVAL = "holdout_eval"


class LegalityStatus(str, Enum):
    OWNED = "owned"
    LICENSED = "licensed"
    RESEARCH_RESTRICTED = "research-restricted"
    UNKNOWN = "unknown"


@dataclass
class Provenance:
    """Tracks which provider produced which metadata."""
    stage: str
    provider: str
    version: str
    timestamp: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Outcome of running a single stage on a record.

    Returned by ``CurationStage.process()``.
    """
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    error: Optional[str] = None
    retryable: bool = False


@dataclass
class CurationRecord:
    """A single unit of data (audio segment) in the curation pipeline."""
    record_id: str
    source_path: str
    audio_hash: str
    
    # Segment info (default to full file)
    segment_start_sec: float = 0.0
    segment_end_sec: float = 0.0
    duration_sec: float = 0.0
    
    # Core labels
    language: Optional[str] = None
    transcript: Optional[str] = None
    transcript_confidence: Optional[float] = None
    
    # Speaker info
    speaker_cluster: Optional[str] = None
    diarization_confidence: Optional[float] = None
    
    # Quality and Status
    quality_score: float = 0.0
    status: RecordStatus = RecordStatus.INGESTED
    promotion_bucket: PromotionBucket = PromotionBucket.NONE
    
    # Reasons for decisions
    rejection_reasons: List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    
    # Evolution tracking
    providers: Dict[str, Provenance] = field(default_factory=dict)
    pass_index: int = 0
    metadata_version: int = 1
    
    # Source legality
    source_legality: str = "unknown"

    # Conversation / dialogue context
    conversation_id: Optional[str] = None
    turn_index: int = 0
    prev_record_id: Optional[str] = None
    next_record_id: Optional[str] = None
    context_window_ids: List[str] = field(default_factory=list)

    # Custom attributes (pitch, energy, style, etc.)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-serializable dictionary."""
        d = {
            "record_id": self.record_id,
            "source_path": self.source_path,
            "audio_hash": self.audio_hash,
            "segment_start_sec": self.segment_start_sec,
            "segment_end_sec": self.segment_end_sec,
            "duration_sec": self.duration_sec,
            "language": self.language,
            "transcript": self.transcript,
            "transcript_confidence": self.transcript_confidence,
            "speaker_cluster": self.speaker_cluster,
            "diarization_confidence": self.diarization_confidence,
            "quality_score": self.quality_score,
            "status": self.status.value,
            "promotion_bucket": self.promotion_bucket.value,
            "rejection_reasons": self.rejection_reasons,
            "review_reasons": self.review_reasons,
            "pass_index": self.pass_index,
            "metadata_version": self.metadata_version,
            "source_legality": self.source_legality,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "prev_record_id": self.prev_record_id,
            "next_record_id": self.next_record_id,
            "context_window_ids": self.context_window_ids,
            "attributes": self.attributes,
            "providers": {
                k: {
                    "stage": v.stage,
                    "provider": v.provider,
                    "version": v.version,
                    "timestamp": v.timestamp,
                    "confidence": v.confidence,
                    "metadata": v.metadata
                } for k, v in self.providers.items()
            }
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CurationRecord:
        """Create a record from a dictionary (e.g., from manifest.jsonl)."""
        providers_raw = d.pop("providers", {})
        providers = {
            k: Provenance(**v) for k, v in providers_raw.items()
        }
        status = RecordStatus(d.pop("status", "ingested"))
        promotion_bucket = PromotionBucket(d.pop("promotion_bucket", "none"))
        
        return cls(
            status=status,
            promotion_bucket=promotion_bucket,
            providers=providers,
            **d
        )
