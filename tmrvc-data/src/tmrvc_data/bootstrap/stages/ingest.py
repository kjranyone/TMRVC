"""Stage 0: Audio file ingestion.

Discovers wav/flac/mp3 files in a corpus directory, validates format,
and extracts basic metadata (sample rate, duration, channels).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
    SegmentInfo,
)

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = (".wav", ".flac", ".mp3")


class IngestStage:
    """Discover and validate audio files from a raw corpus directory.

    Produces one ``BootstrapUtterance`` per source file with metadata
    populated (sample_rate, duration_sec, audio_path).
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()

    def process(
        self,
        utterances: List[BootstrapUtterance],
        *,
        corpus_id: str = "",
        corpus_dir: Optional[Path] = None,
    ) -> List[BootstrapUtterance]:
        """Ingest audio files and create BootstrapUtterance objects.

        If *utterances* is empty, discovers files in *corpus_dir* (or
        ``config.corpus_dir / corpus_id``) and creates new utterances.
        If *utterances* is non-empty, validates existing audio paths.

        Returns:
            List of utterances with audio metadata populated.
        """
        if utterances:
            return self._validate_existing(utterances)

        search_dir = corpus_dir or (self.config.corpus_dir / corpus_id)
        return self._discover_and_create(search_dir, corpus_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_and_create(
        self, corpus_dir: Path, corpus_id: str,
    ) -> List[BootstrapUtterance]:
        """Glob for audio files and build utterance objects."""
        if not corpus_dir.exists():
            logger.warning("Corpus directory does not exist: %s", corpus_dir)
            return []

        audio_files: List[Path] = []
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(sorted(corpus_dir.rglob(f"*{ext}")))

        logger.info(
            "Ingest: found %d audio files in %s", len(audio_files), corpus_dir,
        )

        utterances: List[BootstrapUtterance] = []
        for file_path in audio_files:
            file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()[:12]

            sr, duration, n_channels = self._probe_audio(file_path)
            if sr == 0:
                logger.warning("Skipping unreadable file: %s", file_path)
                continue

            if n_channels > 1:
                logger.info(
                    "Multi-channel audio (%d ch) will be downmixed: %s",
                    n_channels, file_path,
                )

            utt = BootstrapUtterance(
                utterance_id=f"{corpus_id}_{file_hash}_0000",
                corpus_id=corpus_id,
                source_file=str(file_path),
                audio_path=file_path,
                sample_rate=sr,
                duration_sec=duration,
                segment=SegmentInfo(
                    segment_id=f"{file_hash}_0000",
                    source_file=str(file_path),
                    start_sec=0.0,
                    end_sec=duration,
                    duration_sec=duration,
                    sample_rate=sr,
                ),
                stage_completed=BootstrapStage.INGEST,
            )
            utterances.append(utt)

        logger.info("Ingest: %d valid utterances created", len(utterances))
        return utterances

    def _validate_existing(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Validate that audio files exist and have correct metadata."""
        valid: List[BootstrapUtterance] = []
        for utt in utterances:
            if utt.audio_path is None or not Path(utt.audio_path).exists():
                logger.warning(
                    "Audio file missing for %s: %s",
                    utt.utterance_id, utt.audio_path,
                )
                utt.errors.append(f"audio_missing:{utt.audio_path}")
                continue

            if utt.duration_sec <= 0:
                sr, duration, _ = self._probe_audio(Path(utt.audio_path))
                if sr > 0:
                    utt.sample_rate = sr
                    utt.duration_sec = duration

            utt.stage_completed = max(
                utt.stage_completed, BootstrapStage.INGEST,
            )
            valid.append(utt)

        return valid

    @staticmethod
    def _probe_audio(file_path: Path) -> tuple[int, float, int]:
        """Probe an audio file for sample rate, duration, and channel count.

        Tries soundfile first, then falls back to scipy.io.wavfile for wav.

        Returns:
            (sample_rate, duration_sec, n_channels).  Returns (0, 0.0, 0)
            on failure.
        """
        # Try soundfile (handles wav, flac, ogg, etc.)
        try:
            import soundfile as sf

            info = sf.info(str(file_path))
            return info.samplerate, info.duration, info.channels
        except Exception:
            pass

        # Fallback: torchaudio.info
        try:
            import torchaudio

            info = torchaudio.info(str(file_path))
            duration = info.num_frames / info.sample_rate
            return info.sample_rate, duration, info.num_channels
        except Exception:
            pass

        # Final fallback: scipy for wav only
        if file_path.suffix.lower() == ".wav":
            try:
                from scipy.io import wavfile

                sr, data = wavfile.read(str(file_path))
                n_channels = 1 if data.ndim == 1 else data.shape[1]
                duration = len(data) / sr
                return sr, duration, n_channels
            except Exception:
                pass

        logger.error("Cannot probe audio file: %s", file_path)
        return 0, 0.0, 0
