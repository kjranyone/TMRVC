"""v4 bootstrap pipeline stages.

Each stage is a class with a ``process(utterances) -> utterances`` method.
Stages populate fields on BootstrapUtterance progressively.
"""

from tmrvc_data.bootstrap.stages.ingest import IngestStage
from tmrvc_data.bootstrap.stages.normalize import NormalizeStage
from tmrvc_data.bootstrap.stages.vad import VADStage
from tmrvc_data.bootstrap.stages.rejection import RejectionStage
from tmrvc_data.bootstrap.stages.diarization import DiarizationStage
from tmrvc_data.bootstrap.stages.pseudo_speaker import PseudoSpeakerStage
from tmrvc_data.bootstrap.stages.speaker_embed import SpeakerEmbedStage
from tmrvc_data.bootstrap.stages.transcription import TranscriptionStage
from tmrvc_data.bootstrap.stages.text_normalize import TextNormalizeStage
from tmrvc_data.bootstrap.stages.physical_extraction import PhysicalExtractionStage
from tmrvc_data.bootstrap.stages.semantic_annotation import SemanticAnnotationStage
from tmrvc_data.bootstrap.stages.enriched_transcript import EnrichedTranscriptStage
from tmrvc_data.bootstrap.stages.confidence import ConfidenceStage
from tmrvc_data.bootstrap.stages.cache_export import CacheExportStage

__all__ = [
    "IngestStage",
    "NormalizeStage",
    "VADStage",
    "RejectionStage",
    "DiarizationStage",
    "PseudoSpeakerStage",
    "SpeakerEmbedStage",
    "TranscriptionStage",
    "TextNormalizeStage",
    "PhysicalExtractionStage",
    "SemanticAnnotationStage",
    "EnrichedTranscriptStage",
    "ConfidenceStage",
    "CacheExportStage",
]
