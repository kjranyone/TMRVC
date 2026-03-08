"""Curation pipeline stages.

Stage 0: Ingest - Identify and register raw audio assets.
Stage 1: Cleanup - VAD, clipping, corruption detection.
Stage 2: Separation - Source separation / enhancement for mixed audio.
Stage 3: Speaker Recovery - Diarization and speaker clustering.
Stage 4: Transcript Recovery - ASR-based transcript generation.
Stage 5: Transcript Refinement - Multi-ASR fusion and normalization.
Stage 6: Prosody Recovery - Prosody/event extraction and voice_state labels.
"""

from .ingest import ingest_directory
from .cleanup import run_cleanup
from .separation import run_separation
from .speaker_recovery import run_speaker_recovery
from .transcript_recovery import run_transcript_recovery
from .transcript_refinement import run_transcript_refinement
from .prosody_recovery import run_prosody_recovery

__all__ = [
    "ingest_directory",
    "run_cleanup",
    "run_separation",
    "run_speaker_recovery",
    "run_transcript_recovery",
    "run_transcript_refinement",
    "run_prosody_recovery",
]
