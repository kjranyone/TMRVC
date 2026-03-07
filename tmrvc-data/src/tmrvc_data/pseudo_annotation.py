"""Pseudo-annotation pipeline for raw short-utterance corpora.

Processes unlabeled audio files through a multi-stage pipeline to produce
rich annotations suitable for UCLM training. Designed for bulk processing
of datasets that lack transcriptions, speaker labels, or prosodic markup.

Pipeline stages:
1. VAD cleanup -- remove silence / non-speech regions
2. Speaker clustering / diarization -- assign speaker IDs
3. High-quality ASR -- transcribe speech segments
4. Text normalization -- clean and normalize transcriptions
5. G2P / phoneme generation -- convert text to phoneme IDs
6. Pause / breath / event pseudo-label extraction
7. Style embedding extraction -- prosody + emotion features
8. Confidence-based quality filtering -- reject low-quality segments

Usage::

    from tmrvc_data.pseudo_annotation import (
        PseudoAnnotationConfig,
        PseudoAnnotationPipeline,
    )

    config = PseudoAnnotationConfig(min_asr_confidence=0.7, min_snr_db=10.0)
    pipeline = PseudoAnnotationPipeline(config)
    results = pipeline.process("audio/raw_001.wav")
    pipeline.save_artifacts(results, "data/cache/pseudo")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PseudoAnnotationConfig:
    """Configuration for the pseudo-annotation pipeline.

    Attributes:
        min_asr_confidence: Minimum ASR confidence to keep a segment.
        min_snr_db: Minimum signal-to-noise ratio in dB.
        min_duration_sec: Minimum segment duration after VAD (seconds).
        max_duration_sec: Maximum segment duration after VAD (seconds).
        target_sample_rate: Sample rate for all processing stages.
        vad_aggressiveness: VAD aggressiveness level (0-3, higher = more aggressive).
        diarization_min_speakers: Minimum number of speakers for clustering.
        diarization_max_speakers: Maximum number of speakers for clustering.
        asr_model: ASR model identifier (e.g. ``"openai/whisper-large-v3"``).
        asr_language: Language hint for ASR (ISO 639-1, e.g. ``"ja"``).
        g2p_language: Language for G2P conversion.
        style_device: Device for style embedding extraction.
        allowed_languages: If set, keep only segments matching these languages.
        overlap_threshold: Maximum allowed overlap ratio between segments.
        num_workers: Number of parallel workers for I/O-bound stages.
        config_path: Optional path to a YAML config file (populated by CLI).
    """

    min_asr_confidence: float = 0.7
    min_snr_db: float = 10.0
    min_duration_sec: float = 0.3
    max_duration_sec: float = 30.0
    target_sample_rate: int = 24_000
    vad_aggressiveness: int = 2
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 20
    asr_model: str = "openai/whisper-large-v3"
    asr_language: str | None = None
    g2p_language: str = "ja"
    style_device: str = "cpu"
    allowed_languages: list[str] = field(default_factory=list)
    overlap_threshold: float = 0.3
    num_workers: int = 4
    config_path: str | None = None


@dataclass
class PseudoAnnotationResult:
    """Per-utterance result from the pseudo-annotation pipeline.

    Each field corresponds to output from one or more pipeline stages.
    """

    # -- Segment boundaries (from VAD) --
    start_sec: float = 0.0
    end_sec: float = 0.0

    # -- ASR output --
    text: str = ""
    asr_confidence: float = 0.0

    # -- Speaker clustering --
    speaker_cluster: int = -1

    # -- Style embedding (128-d) --
    style_embedding: np.ndarray | None = None

    # -- Event pseudo-labels --
    pause_events: list[dict] = field(default_factory=list)
    breath_events: list[dict] = field(default_factory=list)

    # -- Phoneme IDs (from G2P) --
    phoneme_ids: list[int] = field(default_factory=list)

    # -- Quality metrics --
    quality_score: float = 0.0
    snr_db: float = 0.0

    # -- Source separation metadata --
    separation_source: str = ""
    separation_confidence: float = 0.0

    # -- Language detection --
    detected_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {}
        d["start_sec"] = round(self.start_sec, 4)
        d["end_sec"] = round(self.end_sec, 4)
        d["text"] = self.text
        d["asr_confidence"] = round(self.asr_confidence, 4)
        d["speaker_cluster"] = self.speaker_cluster
        d["pause_events"] = self.pause_events
        d["breath_events"] = self.breath_events
        d["phoneme_ids"] = self.phoneme_ids
        d["quality_score"] = round(self.quality_score, 4)
        d["snr_db"] = round(self.snr_db, 2)
        d["separation_source"] = self.separation_source
        d["separation_confidence"] = round(self.separation_confidence, 4)
        d["detected_language"] = self.detected_language
        # style_embedding saved separately as .npy
        return d


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


class QualityFilter:
    """Multi-criteria quality filter for pseudo-annotated segments.

    Each filter method returns a boolean mask (True = keep) and can be
    composed via :meth:`filter_all`.
    """

    def __init__(self, config: PseudoAnnotationConfig) -> None:
        self.config = config

    def filter_by_asr_confidence(
        self, segments: list[dict],
    ) -> list[bool]:
        """Keep segments whose ASR confidence meets the threshold.

        Args:
            segments: List of segment dicts with ``asr_confidence`` key.

        Returns:
            Boolean mask aligned with *segments*.
        """
        threshold = self.config.min_asr_confidence
        return [
            s.get("asr_confidence", 0.0) >= threshold for s in segments
        ]

    def filter_by_snr(
        self, segments: list[dict],
    ) -> list[bool]:
        """Keep segments whose estimated SNR meets the threshold.

        Args:
            segments: List of segment dicts with ``snr_db`` key.

        Returns:
            Boolean mask aligned with *segments*.
        """
        threshold = self.config.min_snr_db
        return [
            s.get("snr_db", 0.0) >= threshold for s in segments
        ]

    def filter_by_language(
        self, segments: list[dict],
    ) -> list[bool]:
        """Keep segments whose detected language is in the allowed set.

        If ``config.allowed_languages`` is empty, all segments pass.

        Args:
            segments: List of segment dicts with ``detected_language`` key.

        Returns:
            Boolean mask aligned with *segments*.
        """
        allowed = self.config.allowed_languages
        if not allowed:
            return [True] * len(segments)
        return [
            s.get("detected_language", "") in allowed for s in segments
        ]

    def filter_by_overlap(
        self, segments: list[dict],
    ) -> list[bool]:
        """Remove segments that overlap excessively with neighbours.

        Uses a simple pairwise overlap ratio check.  A segment is rejected
        if it overlaps with any other segment by more than
        ``config.overlap_threshold`` of its own duration.

        Args:
            segments: List of segment dicts with ``start_sec`` / ``end_sec``.

        Returns:
            Boolean mask aligned with *segments*.
        """
        threshold = self.config.overlap_threshold
        keep = [True] * len(segments)

        for i, seg_i in enumerate(segments):
            dur_i = seg_i.get("end_sec", 0.0) - seg_i.get("start_sec", 0.0)
            if dur_i <= 0:
                keep[i] = False
                continue
            for j, seg_j in enumerate(segments):
                if i == j:
                    continue
                overlap_start = max(
                    seg_i.get("start_sec", 0.0),
                    seg_j.get("start_sec", 0.0),
                )
                overlap_end = min(
                    seg_i.get("end_sec", 0.0),
                    seg_j.get("end_sec", 0.0),
                )
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap / dur_i > threshold:
                    keep[i] = False
                    break
        return keep

    def filter_all(
        self, segments: list[dict],
    ) -> list[bool]:
        """Apply all quality filters and return a combined mask.

        Args:
            segments: List of segment dicts.

        Returns:
            Boolean mask -- True for segments that pass all criteria.
        """
        masks = [
            self.filter_by_asr_confidence(segments),
            self.filter_by_snr(segments),
            self.filter_by_language(segments),
            self.filter_by_overlap(segments),
        ]
        return [all(m[i] for m in masks) for i in range(len(segments))]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PseudoAnnotationPipeline:
    """Orchestrates the full pseudo-annotation pipeline.

    Each ``run_*`` method implements one pipeline stage.  The
    :meth:`process` method chains them in order and returns a list of
    :class:`PseudoAnnotationResult` objects.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: PseudoAnnotationConfig) -> None:
        self.config = config
        self._quality_filter = QualityFilter(config)

    # -- Stage 1: VAD ---------------------------------------------------------

    def run_vad(self, audio_path: str | Path) -> list[dict]:
        """Run voice activity detection to identify speech segments.

        Removes silence, noise-only regions, and non-speech audio.  Returns
        a list of segments, each a dict with keys:

        - ``start_sec`` (float): segment start time
        - ``end_sec`` (float): segment end time
        - ``confidence`` (float): VAD confidence score

        Args:
            audio_path: Path to a WAV file.

        Returns:
            List of VAD segment dicts.

        Raises:
            NotImplementedError: Always (stub). Requires ``silero-vad`` or
                ``pyannote.audio``.
        """
        raise NotImplementedError(
            "VAD stage not implemented. Requires silero-vad. "
            "Install with: pip install silero-vad"
        )

    # -- Stage 2: Speaker diarization -----------------------------------------

    def run_diarization(
        self,
        audio_path: str | Path,
        vad_segments: list[dict],
    ) -> list[dict]:
        """Cluster speech segments by speaker identity.

        Adds a ``speaker_id`` (int) to each segment dict.  Speaker IDs are
        local to the file (0-indexed).

        Args:
            audio_path: Path to a WAV file.
            vad_segments: Segments from :meth:`run_vad`.

        Returns:
            Segments with ``speaker_id`` added.

        Raises:
            NotImplementedError: Always (stub). Requires ``pyannote.audio``
                and SpeechBrain.
        """
        raise NotImplementedError(
            "Diarization stage not implemented. Requires pyannote.audio. "
            "Install with: pip install pyannote.audio"
        )

    # -- Stage 3: ASR ----------------------------------------------------------

    def run_asr(
        self,
        audio_path: str | Path,
        segments: list[dict],
    ) -> list[dict]:
        """Transcribe speech segments with a high-quality ASR model.

        Adds the following keys to each segment dict:

        - ``text`` (str): transcribed text
        - ``asr_confidence`` (float): model confidence (0-1)
        - ``detected_language`` (str): ISO 639-1 language code

        Args:
            audio_path: Path to a WAV file.
            segments: Segments from :meth:`run_diarization`.

        Returns:
            Segments with ASR fields added.

        Raises:
            NotImplementedError: Always (stub). Requires ``faster-whisper``
                or ``openai-whisper``.
        """
        raise NotImplementedError(
            "ASR stage not implemented. Requires faster-whisper. "
            "Install with: pip install faster-whisper"
        )

    # -- Stage 4 (implicit): Text normalization is performed inside run_asr
    #    and run_g2p. The G2P frontend in tmrvc_data.g2p handles
    #    normalization as part of its phonemization pipeline.

    # -- Stage 5: G2P ----------------------------------------------------------

    def run_g2p(self, segments: list[dict]) -> list[dict]:
        """Convert transcribed text to phoneme ID sequences.

        Uses the project's unified G2P frontend (``tmrvc_data.g2p``) which
        handles Japanese, English, Chinese, and Korean.

        Adds ``phoneme_ids`` (list[int]) to each segment dict.

        Args:
            segments: Segments with ``text`` from :meth:`run_asr`.

        Returns:
            Segments with ``phoneme_ids`` added.

        Raises:
            NotImplementedError: Always (stub). Requires ``pyopenjtalk``
                (for Japanese) or ``phonemizer`` (for other languages).
        """
        raise NotImplementedError(
            "G2P stage not implemented. Requires pyopenjtalk. "
            "Install with: pip install pyopenjtalk"
        )

    # -- Stage 6: Event detection ----------------------------------------------

    def run_event_detection(
        self,
        audio_path: str | Path,
        segments: list[dict],
    ) -> list[dict]:
        """Extract pause and breath pseudo-labels from audio segments.

        Uses mel-spectrogram energy and F0 analysis (see ``tmrvc_data.events``)
        to detect:

        - **Pause events**: silent gaps within speech (type, start_frame, dur_ms)
        - **Breath events**: audible inhalations (type, start_frame, dur_ms, intensity)

        Adds ``pause_events`` and ``breath_events`` (list[dict]) to each
        segment dict.

        Args:
            audio_path: Path to a WAV file.
            segments: Segments from previous stages.

        Returns:
            Segments with event fields added.

        Raises:
            NotImplementedError: Always (stub). Requires ``librosa`` for
                feature extraction.
        """
        raise NotImplementedError(
            "Event detection stage not implemented. Requires librosa. "
            "Install with: pip install librosa"
        )

    # -- Stage 7: Style extraction ---------------------------------------------

    def run_style_extraction(
        self,
        audio_path: str | Path,
        segments: list[dict],
    ) -> list[dict]:
        """Extract 128-d style/prosody embeddings for each segment.

        Uses the project's :class:`~tmrvc_data.style.StyleEncoder` to
        produce a dense embedding capturing prosody, energy contour, and
        speaking style.

        Adds ``style_embedding`` (np.ndarray of shape ``[128]``) to each
        segment dict.

        Args:
            audio_path: Path to a WAV file.
            segments: Segments from previous stages.

        Returns:
            Segments with ``style_embedding`` added.

        Raises:
            NotImplementedError: Always (stub). Requires ``librosa`` for
                prosody feature extraction.
        """
        raise NotImplementedError(
            "Style extraction stage not implemented. Requires librosa. "
            "Install with: pip install librosa"
        )

    # -- Stage 8: Quality filtering --------------------------------------------

    def run_quality_filter(self, segments: list[dict]) -> list[dict]:
        """Apply confidence-based quality filtering.

        Removes segments that fail any of:

        - ASR confidence below ``config.min_asr_confidence``
        - Estimated SNR below ``config.min_snr_db``
        - Language not in ``config.allowed_languages`` (if set)
        - Excessive overlap with neighbouring segments

        Also computes a composite ``quality_score`` (0-1) for each
        surviving segment.

        Args:
            segments: Segments from previous stages.

        Returns:
            Filtered list of segments with ``quality_score`` added.
        """
        mask = self._quality_filter.filter_all(segments)
        kept = [s for s, m in zip(segments, mask) if m]

        # Compute composite quality score for kept segments
        for seg in kept:
            asr_conf = seg.get("asr_confidence", 0.0)
            snr = seg.get("snr_db", 0.0)
            # Normalize SNR to 0-1 range (assume 40 dB is excellent)
            snr_norm = min(snr / 40.0, 1.0)
            seg["quality_score"] = round(0.6 * asr_conf + 0.4 * snr_norm, 4)

        n_dropped = len(segments) - len(kept)
        if n_dropped > 0:
            logger.info(
                "Quality filter: kept %d / %d segments (dropped %d)",
                len(kept),
                len(segments),
                n_dropped,
            )
        return kept

    # -- Full pipeline ---------------------------------------------------------

    def process(self, audio_path: str | Path) -> list[PseudoAnnotationResult]:
        """Run the full pseudo-annotation pipeline on a single audio file.

        Chains all stages in order:
        VAD -> diarization -> ASR -> G2P -> event detection ->
        style extraction -> quality filtering.

        Args:
            audio_path: Path to a WAV file.

        Returns:
            List of :class:`PseudoAnnotationResult` for each accepted segment.
        """
        audio_path = Path(audio_path)
        logger.info("Processing %s", audio_path)

        # Stage 1: VAD
        segments = self.run_vad(audio_path)
        logger.info("VAD: %d segments", len(segments))

        # Stage 2: Speaker diarization
        segments = self.run_diarization(audio_path, segments)
        logger.info("Diarization: %d segments", len(segments))

        # Stage 3: ASR (includes text normalization)
        segments = self.run_asr(audio_path, segments)
        logger.info("ASR: %d segments transcribed", len(segments))

        # Stage 5: G2P / phoneme generation
        segments = self.run_g2p(segments)
        logger.info("G2P: %d segments phonemized", len(segments))

        # Stage 6: Event detection
        segments = self.run_event_detection(audio_path, segments)
        logger.info("Events: %d segments annotated", len(segments))

        # Stage 7: Style extraction
        segments = self.run_style_extraction(audio_path, segments)
        logger.info("Style: %d segments embedded", len(segments))

        # Stage 8: Quality filtering
        segments = self.run_quality_filter(segments)
        logger.info("Quality filter: %d segments accepted", len(segments))

        # Convert to result objects
        results = []
        for seg in segments:
            result = PseudoAnnotationResult(
                start_sec=seg.get("start_sec", 0.0),
                end_sec=seg.get("end_sec", 0.0),
                text=seg.get("text", ""),
                asr_confidence=seg.get("asr_confidence", 0.0),
                speaker_cluster=seg.get("speaker_id", -1),
                style_embedding=seg.get("style_embedding"),
                pause_events=seg.get("pause_events", []),
                breath_events=seg.get("breath_events", []),
                phoneme_ids=seg.get("phoneme_ids", []),
                quality_score=seg.get("quality_score", 0.0),
                snr_db=seg.get("snr_db", 0.0),
                separation_source=seg.get("separation_source", ""),
                separation_confidence=seg.get("separation_confidence", 0.0),
                detected_language=seg.get("detected_language", ""),
            )
            results.append(result)

        return results

    # -- Artifact I/O ----------------------------------------------------------

    def save_artifacts(
        self,
        results: list[PseudoAnnotationResult],
        output_dir: str | Path,
    ) -> None:
        """Save pipeline results to the feature cache directory layout.

        Creates one sub-directory per segment under *output_dir*::

            output_dir/
              segment_000/
                annotation.json   # text, confidence, events, quality, ...
                style.npy         # 128-d style embedding
              segment_001/
                ...
              manifest.json       # summary manifest for all segments

        Args:
            results: List of :class:`PseudoAnnotationResult`.
            output_dir: Root output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[dict[str, Any]] = []

        for idx, result in enumerate(results):
            seg_dir = output_dir / f"segment_{idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            # Save annotation metadata
            annotation = result.to_dict()
            annotation_path = seg_dir / "annotation.json"
            with open(annotation_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

            # Save style embedding as .npy
            if result.style_embedding is not None:
                style_path = seg_dir / "style.npy"
                np.save(style_path, result.style_embedding)

            manifest_entries.append(
                {
                    "segment_id": f"segment_{idx:04d}",
                    "text": result.text,
                    "speaker_cluster": result.speaker_cluster,
                    "quality_score": round(result.quality_score, 4),
                    "duration_sec": round(
                        result.end_sec - result.start_sec, 4
                    ),
                }
            )

        # Write manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {"segments": manifest_entries, "total": len(manifest_entries)},
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "Saved %d segment artifacts to %s", len(results), output_dir
        )


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


def quality_summary(results: list[PseudoAnnotationResult]) -> dict[str, Any]:
    """Compute audit metrics over a batch of pseudo-annotation results.

    Returns a dictionary with:

    - ``total_segments``: number of segments
    - ``total_duration_sec``: total speech duration
    - ``mean_asr_confidence``: average ASR confidence
    - ``mean_quality_score``: average composite quality score
    - ``mean_snr_db``: average estimated SNR
    - ``speaker_counts``: dict mapping speaker_cluster -> count
    - ``language_counts``: dict mapping detected_language -> count
    - ``event_stats``: breath / pause event counts and mean durations

    Args:
        results: List of :class:`PseudoAnnotationResult`.

    Returns:
        Audit metrics dictionary.
    """
    if not results:
        return {
            "total_segments": 0,
            "total_duration_sec": 0.0,
            "mean_asr_confidence": 0.0,
            "mean_quality_score": 0.0,
            "mean_snr_db": 0.0,
            "speaker_counts": {},
            "language_counts": {},
            "event_stats": {},
        }

    durations = [r.end_sec - r.start_sec for r in results]
    asr_confs = [r.asr_confidence for r in results]
    quality_scores = [r.quality_score for r in results]
    snrs = [r.snr_db for r in results]

    # Speaker distribution
    speaker_counts: dict[int, int] = {}
    for r in results:
        speaker_counts[r.speaker_cluster] = (
            speaker_counts.get(r.speaker_cluster, 0) + 1
        )

    # Language distribution
    language_counts: dict[str, int] = {}
    for r in results:
        lang = r.detected_language or "unknown"
        language_counts[lang] = language_counts.get(lang, 0) + 1

    # Event statistics
    total_pauses = sum(len(r.pause_events) for r in results)
    total_breaths = sum(len(r.breath_events) for r in results)
    pause_durations = [
        e.get("dur_ms", 0.0)
        for r in results
        for e in r.pause_events
    ]
    breath_durations = [
        e.get("dur_ms", 0.0)
        for r in results
        for e in r.breath_events
    ]

    event_stats = {
        "total_pauses": total_pauses,
        "total_breaths": total_breaths,
        "mean_pause_dur_ms": (
            round(float(np.mean(pause_durations)), 1)
            if pause_durations
            else 0.0
        ),
        "mean_breath_dur_ms": (
            round(float(np.mean(breath_durations)), 1)
            if breath_durations
            else 0.0
        ),
    }

    return {
        "total_segments": len(results),
        "total_duration_sec": round(sum(durations), 2),
        "mean_asr_confidence": round(float(np.mean(asr_confs)), 4),
        "mean_quality_score": round(float(np.mean(quality_scores)), 4),
        "mean_snr_db": round(float(np.mean(snrs)), 2),
        "speaker_counts": {
            str(k): v for k, v in sorted(speaker_counts.items())
        },
        "language_counts": dict(sorted(language_counts.items())),
        "event_stats": event_stats,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_config_from_yaml(path: str | Path) -> dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Dictionary of config overrides.

    Raises:
        NotImplementedError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise NotImplementedError(
            "YAML config loading requires PyYAML. "
            "Install with: pip install pyyaml"
        ) from exc

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the pseudo-annotation pipeline.

    Example::

        python -m tmrvc_data.pseudo_annotation \\
            --input-dir data/raw/corpus_x \\
            --output-dir data/cache/pseudo/corpus_x \\
            --min-confidence 0.7 \\
            --min-snr 10
    """
    parser = argparse.ArgumentParser(
        description="Pseudo-annotation pipeline for raw short-utterance corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw WAV files to annotate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for cached annotations.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file with pipeline parameters.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum ASR confidence threshold (default: 0.7).",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=10.0,
        help="Minimum signal-to-noise ratio in dB (default: 10).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Build config
    config_overrides: dict[str, Any] = {}
    if args.config:
        config_overrides = _load_config_from_yaml(args.config)

    config_overrides["min_asr_confidence"] = args.min_confidence
    config_overrides["min_snr_db"] = args.min_snr
    config_overrides["config_path"] = args.config

    # Only pass known fields to the dataclass
    known_fields = {f.name for f in PseudoAnnotationConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in config_overrides.items() if k in known_fields}
    config = PseudoAnnotationConfig(**filtered)

    pipeline = PseudoAnnotationPipeline(config)

    # Discover WAV files
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        logger.error("No WAV files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d WAV files in %s", len(wav_files), input_dir)

    output_dir = Path(args.output_dir)
    all_results: list[PseudoAnnotationResult] = []

    for i, wav_path in enumerate(wav_files):
        logger.info("[%d/%d] %s", i + 1, len(wav_files), wav_path.name)
        try:
            results = pipeline.process(wav_path)
            if results:
                file_output = output_dir / wav_path.stem
                pipeline.save_artifacts(results, file_output)
                all_results.extend(results)
        except NotImplementedError as exc:
            logger.error("Pipeline stage not implemented: %s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", wav_path.name, exc)

    # Print audit summary
    summary = quality_summary(all_results)
    logger.info("=== Pipeline Summary ===")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
