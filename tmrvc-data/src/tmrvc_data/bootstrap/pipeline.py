"""v4 raw-audio bootstrap pipeline.

Converts raw unlabeled audio corpora into v4 train-ready cache.
Implements the canonical 13-stage pipeline from track_data_bootstrap.md.

Pipeline stages:
0. ingest
1. audio normalization
2. VAD segmentation
3. overlap / music / noise rejection
4. diarization or speaker clustering
5. pseudo speaker assignment
6. speaker embedding extraction
7. Whisper transcription
8. text normalization and G2P
9. DSP / SSL physical feature extraction
10. LLM semantic / acting annotation
11. confidence scoring and artifact masking
12. train-ready cache export
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapResult,
    BootstrapStage,
    BootstrapUtterance,
    SegmentInfo,
)
from tmrvc_data.bootstrap.model_loader import BootstrapModelLoader
from tmrvc_data.bootstrap.supervision import (
    QualityGateReport,
    SupervisionTierClassifier,
)

logger = logging.getLogger(__name__)


class BootstrapPipeline:
    """v4 raw-audio bootstrap pipeline.

    Converts raw unlabeled audio files into train-ready cache
    following the v4 dataset contract.

    Usage:
        config = BootstrapConfig(corpus_dir=Path("data/raw_corpus"))
        pipeline = BootstrapPipeline(config)
        result = pipeline.run(corpus_id="my_corpus")
    """

    SUPPORTED_FORMATS = (".wav", ".flac", ".mp3")

    def __init__(self, config: Optional[BootstrapConfig] = None):
        self.config = config or BootstrapConfig()
        self.tier_classifier = SupervisionTierClassifier()
        self._utterances: List[BootstrapUtterance] = []
        self._models: Optional[BootstrapModelLoader] = None

    def _ensure_models(self) -> BootstrapModelLoader:
        """Create the centralized model loader on first use."""
        if self._models is None:
            self._models = BootstrapModelLoader(
                device=self.config.device,
                whisper_model=self.config.whisper_model,
            )
        return self._models

    def cleanup(self) -> None:
        """Free GPU memory by unloading all models."""
        if self._models is not None:
            self._models.cleanup()
            self._models = None

    def run(self, corpus_id: str) -> BootstrapResult:
        """Run the full bootstrap pipeline on a corpus.

        Args:
            corpus_id: Identifier for the corpus being processed.

        Returns:
            BootstrapResult with summary statistics.
        """
        result = BootstrapResult(corpus_id=corpus_id)

        # Stage 0: Ingest
        logger.info("Stage 0: Ingesting corpus '%s'", corpus_id)
        raw_files = list(self._discover_audio_files(corpus_id))
        result.total_files = len(raw_files)
        logger.info("Found %d audio files", result.total_files)

        if not raw_files:
            result.warnings.append(f"No audio files found in corpus '{corpus_id}'")
            return result

        # Stage 1-2: Audio normalization + VAD segmentation
        logger.info("Stages 1-2: Normalizing and segmenting audio")
        utterances = self._stage_normalize_and_segment(raw_files, corpus_id)
        result.total_segments = len(utterances)

        # Stage 3: Rejection
        logger.info("Stage 3: Rejecting low-quality segments")
        utterances = self._stage_reject(utterances, result)

        # Stage 4-5: Diarization + pseudo speaker assignment
        logger.info("Stages 4-5: Diarization and speaker assignment")
        utterances = self._stage_diarize(utterances)

        # Stage 6: Speaker embedding extraction
        logger.info("Stage 6: Extracting speaker embeddings")
        utterances = self._stage_extract_speaker_embeddings(utterances)

        # Stage 7: Whisper transcription
        logger.info("Stage 7: Transcribing with Whisper")
        utterances = self._stage_transcribe(utterances)

        # Stage 8: Text normalization and G2P
        logger.info("Stage 8: Text normalization and G2P")
        utterances = self._stage_text_normalize(utterances)

        # Stage 9: DSP/SSL physical feature extraction
        logger.info("Stage 9: Physical feature extraction (DSP/SSL)")
        utterances = self._stage_extract_physical(utterances)

        # Stage 10: LLM semantic annotation
        logger.info("Stage 10: Semantic annotation (LLM)")
        utterances = self._stage_annotate_semantic(utterances)

        # Stage 11: Confidence scoring
        logger.info("Stage 11: Confidence scoring and tier classification")
        utterances = self._stage_score_confidence(utterances)

        # Stage 12: Cache export
        logger.info("Stage 12: Exporting train-ready cache")
        utterances = self._stage_export_cache(utterances, corpus_id)

        # Compile results
        self._compile_result(utterances, result)
        self._utterances = utterances

        logger.info(
            "Bootstrap complete: %d accepted, %d rejected, tiers A=%d B=%d C=%d D=%d",
            result.accepted_utterances, result.rejected_utterances,
            result.tier_a_count, result.tier_b_count,
            result.tier_c_count, result.tier_d_count,
        )

        return result

    def generate_quality_report(self, corpus_id: str) -> QualityGateReport:
        """Generate quality gate report from pipeline results.

        Must be called after run().
        """
        report = QualityGateReport(corpus_id=corpus_id)

        if not self._utterances:
            report.failed_gates.append("No utterances processed")
            return report

        accepted = [u for u in self._utterances if not u.is_rejected]

        if not accepted:
            report.failed_gates.append("No accepted utterances")
            return report

        # Diarization metrics
        confidences = [u.diarization_confidence for u in accepted]
        report.diarization_purity = float(np.mean(confidences)) if confidences else 0.0

        # Compute speaker cluster consistency
        speaker_groups: dict[str, list] = {}
        for u in accepted:
            speaker_groups.setdefault(u.pseudo_speaker_id, []).append(u)
        if speaker_groups:
            consistencies = []
            for group in speaker_groups.values():
                if len(group) > 1:
                    group_conf = [u.diarization_confidence for u in group]
                    consistencies.append(float(np.std(group_conf)))
            if consistencies:
                report.speaker_cluster_consistency = 1.0 - float(np.mean(consistencies))

        # Transcript quality
        transcript_confs = [u.transcript_confidence for u in accepted]
        report.transcript_wer_proxy = 1.0 - float(np.mean(transcript_confs)) if transcript_confs else 1.0

        # Physical label coverage
        coverages = []
        phys_confs = []
        for u in accepted:
            if u.physical_observed_mask is not None:
                coverages.append(float(np.mean(u.physical_observed_mask)))
            if u.physical_confidence is not None and u.physical_observed_mask is not None:
                observed = u.physical_confidence[u.physical_observed_mask]
                if len(observed) > 0:
                    phys_confs.append(float(np.mean(observed)))

        report.physical_label_coverage = float(np.mean(coverages)) if coverages else 0.0
        report.physical_confidence_mean = float(np.mean(phys_confs)) if phys_confs else 0.0

        # Language coverage
        langs = set(u.language for u in accepted if u.language)
        report.languages_detected = sorted(langs)
        lang_counts: dict[str, int] = {}
        for u in accepted:
            if u.language:
                lang_counts[u.language] = lang_counts.get(u.language, 0) + 1
        report.language_distribution = lang_counts

        # Tier distribution
        tier_counts: dict[str, int] = {}
        for u in accepted:
            tier_counts[u.supervision_tier] = tier_counts.get(u.supervision_tier, 0) + 1
        report.tier_distribution = tier_counts

        # Evaluate gates
        report.evaluate_gates()

        return report

    # ----- Audio loading helper -----

    def _load_audio(self, utt: BootstrapUtterance):
        """Load and resample audio for an utterance.

        Returns:
            (waveform, sample_rate) where waveform is [1, T] float32.
        """
        from tmrvc_data.preprocessing import load_and_resample

        audio_path = str(utt.audio_path) if utt.audio_path else utt.source_file
        waveform, sr = load_and_resample(audio_path, target_sr=self.config.target_sample_rate)

        # Update utterance duration metadata
        utt.duration_sec = waveform.shape[-1] / sr

        return waveform, sr

    # ----- Private stage implementations -----

    def _discover_audio_files(self, corpus_id: str) -> Iterator[Path]:
        """Stage 0: Discover audio files in the corpus directory."""
        corpus_dir = self.config.corpus_dir / corpus_id
        if not corpus_dir.exists():
            logger.warning("Corpus directory does not exist: %s", corpus_dir)
            return

        for ext in self.SUPPORTED_FORMATS:
            yield from sorted(corpus_dir.rglob(f"*{ext}"))

    def _stage_normalize_and_segment(
        self, files: List[Path], corpus_id: str,
    ) -> List[BootstrapUtterance]:
        """Stages 1-2: Audio normalization + VAD segmentation.

        Creates BootstrapUtterance objects for each valid segment.
        Actual audio loading and normalization delegated to tmrvc_data.preprocessing.
        """
        utterances = []
        for file_path in files:
            file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()[:12]

            # Create a single utterance per file for now
            # Real implementation would run VAD and split
            utt = BootstrapUtterance(
                utterance_id=f"{corpus_id}_{file_hash}_0000",
                corpus_id=corpus_id,
                source_file=str(file_path),
                audio_path=file_path,
                segment=SegmentInfo(
                    segment_id=f"{file_hash}_0000",
                    source_file=str(file_path),
                    start_sec=0.0,
                    end_sec=0.0,  # populated by actual VAD
                    duration_sec=0.0,
                ),
                stage_completed=BootstrapStage.VAD_SEGMENTATION,
            )
            utterances.append(utt)

        return utterances

    def _stage_reject(
        self, utterances: List[BootstrapUtterance], result: BootstrapResult,
    ) -> List[BootstrapUtterance]:
        """Stage 3: Overlap / music / noise rejection.

        Rejects segments that would contaminate speaker embedding
        or physical feature extraction.
        """
        accepted = []
        for utt in utterances:
            # Compute duration from segment info or by loading audio
            if utt.duration_sec <= 0.0 and utt.segment is not None and utt.segment.end_sec > utt.segment.start_sec:
                utt.duration_sec = utt.segment.end_sec - utt.segment.start_sec
            if utt.duration_sec <= 0.0:
                try:
                    self._load_audio(utt)
                except Exception as e:
                    logger.warning("Could not load audio for duration check %s: %s", utt.utterance_id, e)

            # Duration check
            if utt.duration_sec < self.config.segment_min_sec:
                utt.is_rejected = True
                utt.rejection_reason = "too_short"
                result.short_rejections += 1
                continue

            # Actual rejection logic would go here (overlap detection, music detection, SNR)
            # For now, accept all non-short segments
            utt.stage_completed = BootstrapStage.REJECTION
            accepted.append(utt)

        result.rejected_utterances = len(utterances) - len(accepted)
        return accepted

    def _stage_diarize(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stages 4-5: Diarization and pseudo speaker assignment.

        Groups utterances by speaker using embedding clustering.
        Assigns stable pseudo speaker IDs.
        """
        models = self._ensure_models()

        # Group utterances by source file for diarization
        by_source: dict[str, List[BootstrapUtterance]] = {}
        for utt in utterances:
            by_source.setdefault(utt.source_file, []).append(utt)

        for source_file, utts in by_source.items():
            diar_segments = models.diarize(source_file)

            if diar_segments:
                # Build a mapping from time ranges to speaker labels
                for utt in utts:
                    seg_start = utt.segment.start_sec if utt.segment else 0.0
                    seg_end = utt.segment.end_sec if utt.segment else 0.0

                    # Find the best-matching diarization segment
                    best_speaker = None
                    best_overlap = 0.0
                    for dseg in diar_segments:
                        overlap_start = max(seg_start, dseg["start"])
                        overlap_end = min(seg_end, dseg["end"]) if seg_end > 0 else dseg["end"]
                        overlap = max(0.0, overlap_end - overlap_start)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_speaker = dseg["speaker"]

                    if best_speaker is not None:
                        utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_{best_speaker}"
                        utt.diarization_confidence = min(1.0, best_overlap / max(0.01, seg_end - seg_start)) if seg_end > seg_start else 0.8
                    else:
                        utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_unknown"
                        utt.diarization_confidence = 0.0

                    utt.stage_completed = BootstrapStage.PSEUDO_SPEAKER
            else:
                # Diarization unavailable or failed — assign all to single pseudo speaker
                logger.warning(
                    "Diarization returned no segments for %s, assigning single speaker",
                    source_file,
                )
                for utt in utts:
                    utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_000"
                    utt.diarization_confidence = 0.0
                    utt.stage_completed = BootstrapStage.PSEUDO_SPEAKER

        return utterances

    def _stage_extract_speaker_embeddings(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stage 6: Speaker embedding extraction."""
        models = self._ensure_models()

        for utt in utterances:
            try:
                waveform, sr = self._load_audio(utt)
                wav_np = waveform.squeeze(0).numpy()
                utt.speaker_embed = models.extract_speaker_embedding(wav_np, sample_rate=sr)
            except Exception as e:
                logger.error("Speaker embedding failed for %s: %s", utt.utterance_id, e)
                utt.speaker_embed = np.zeros(192, dtype=np.float32)
            utt.stage_completed = BootstrapStage.SPEAKER_EMBEDDING

        return utterances

    def _stage_transcribe(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stage 7: Whisper transcription.

        Whisper + LLM own: transcript, punctuation recovery
        """
        models = self._ensure_models()

        for utt in utterances:
            try:
                audio_path = str(utt.audio_path) if utt.audio_path else utt.source_file
                text, confidence, detected_lang = models.transcribe(
                    audio_path, language=self.config.whisper_language,
                )
                utt.text_transcript = text
                utt.transcript_confidence = confidence
                utt.language = detected_lang
            except Exception as e:
                logger.error("Transcription failed for %s: %s", utt.utterance_id, e)
                utt.text_transcript = ""
                utt.transcript_confidence = 0.0
                utt.language = ""
            utt.stage_completed = BootstrapStage.TRANSCRIPTION

        return utterances

    def _stage_text_normalize(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stage 8: Text normalization and G2P."""
        models = self._ensure_models()

        for utt in utterances:
            try:
                lang = utt.language if utt.language in ("ja", "en", "zh", "ko") else "ja"
                if utt.text_transcript:
                    utt.phoneme_ids = models.text_to_phonemes(utt.text_transcript, language=lang)
                else:
                    utt.phoneme_ids = np.array([], dtype=np.int64)
            except Exception as e:
                logger.error("G2P failed for %s: %s", utt.utterance_id, e)
                utt.phoneme_ids = np.array([], dtype=np.int64)
            utt.stage_completed = BootstrapStage.TEXT_NORMALIZATION

        return utterances

    def _stage_extract_physical(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stage 9: DSP/SSL physical feature extraction.

        DSP/SSL own: physical voice control targets, confidence, observed mask, speaker timbre anchor.
        Whisper+LLM must NOT replace physical supervision.

        Extracts 12-D physical controls for v4.
        """
        d_phys = self.config.physical_dim  # 12 for v4
        models = self._ensure_models()

        for utt in utterances:
            try:
                waveform, sr = self._load_audio(utt)
                wav_np = waveform.squeeze(0).numpy()
                targets, observed_mask, confidence = models.extract_voice_state(
                    wav_np, sample_rate=sr,
                )
                utt.physical_targets = targets
                utt.physical_observed_mask = observed_mask
                utt.physical_confidence = confidence
                utt.n_frames = targets.shape[0]
            except Exception as e:
                logger.error("Physical extraction failed for %s: %s", utt.utterance_id, e)
                n_frames = max(1, utt.n_frames) if utt.n_frames > 0 else 1
                utt.physical_targets = np.zeros((n_frames, d_phys), dtype=np.float32)
                utt.physical_observed_mask = np.zeros((n_frames, d_phys), dtype=bool)
                utt.physical_confidence = np.zeros((n_frames, d_phys), dtype=np.float32)
            utt.stage_completed = BootstrapStage.PHYSICAL_EXTRACTION

        return utterances

    def _generate_enriched_transcript(
        self,
        text_transcript: str,
        acting_annotations: dict,
    ) -> str:
        """Generate enriched transcript with inline acting tags.

        Combines the plain transcript with detected vocal events and acting
        annotations to produce an enriched transcript string containing
        inline acting tags.

        Args:
            text_transcript: Plain text transcript from Whisper.
            acting_annotations: Dict with scene_summary, dialogue_intent,
                emotion_description, acting_hint from LLM annotation.

        Returns:
            Enriched transcript string with inline acting tags inserted.
        """
        if not text_transcript:
            return ""

        parts = []

        # Insert emotion/acting directive tag at the start if available
        emotion = acting_annotations.get("emotion_description", "")
        if emotion:
            # Map common emotion descriptions to acting directive tags
            _EMOTION_TAG_MAP = {
                "angry": "[angry]",
                "anger": "[angry]",
                "whisper": "[whisper]",
                "calm": "[calm]",
                "excited": "[excited]",
                "excitement": "[excited]",
                "tender": "[tender]",
                "professional": "[professional]",
                "sad": "[sad]",
                "sadness": "[sad]",
                "happy": "[happy]",
                "happiness": "[happy]",
                "fear": "[fearful]",
                "fearful": "[fearful]",
                "disgust": "[disgusted]",
                "disgusted": "[disgusted]",
                "surprise": "[surprised]",
                "surprised": "[surprised]",
                "bored": "[bored]",
                "boredom": "[bored]",
                "nervous": "[nervous]",
                "confident": "[confident]",
                "sarcastic": "[sarcastic]",
                "playful": "[playful]",
            }
            emotion_lower = emotion.strip().lower()
            if emotion_lower in _EMOTION_TAG_MAP:
                parts.append(_EMOTION_TAG_MAP[emotion_lower])
            elif emotion_lower:
                # Free-form acting instruction for unrecognized emotions
                parts.append(f"[{emotion_lower}]")

        # Insert acting hint as inline tag if available
        acting_hint = acting_annotations.get("acting_hint", "")
        if acting_hint:
            parts.append(f"[{acting_hint.strip().lower()}]")

        # Append the plain transcript
        parts.append(text_transcript)

        return " ".join(parts)

    def _stage_annotate_semantic(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stage 10: LLM semantic / acting annotation.

        Whisper + LLM own: scene summary, dialogue intent, emotion description, acting hint.
        Also generates enriched_transcript with inline acting tags.
        """
        models = self._ensure_models()

        for utt in utterances:
            try:
                if utt.text_transcript:
                    utt.acting_annotations = models.annotate_semantic(
                        utt.text_transcript, language=utt.language,
                    )
                else:
                    utt.acting_annotations = {
                        "scene_summary": "",
                        "dialogue_intent": "",
                        "emotion_description": "",
                        "acting_hint": "",
                    }
            except Exception as e:
                logger.error("Semantic annotation failed for %s: %s", utt.utterance_id, e)
                utt.acting_annotations = {
                    "scene_summary": "",
                    "dialogue_intent": "",
                    "emotion_description": "",
                    "acting_hint": "",
                }

            # Generate enriched transcript from plain transcript + annotations
            utt.enriched_transcript = self._generate_enriched_transcript(
                utt.text_transcript,
                utt.acting_annotations,
            )

            utt.stage_completed = BootstrapStage.SEMANTIC_ANNOTATION

        return utterances

    def _stage_score_confidence(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Stage 11: Confidence scoring and supervision tier classification."""
        for utt in utterances:
            has_semantic = bool(
                utt.acting_annotations.get("scene_summary")
                or utt.acting_annotations.get("emotion_description")
            )

            utt.supervision_tier = self.tier_classifier.classify(
                transcript_confidence=utt.transcript_confidence,
                diarization_confidence=utt.diarization_confidence,
                physical_observed_mask=utt.physical_observed_mask,
                physical_confidence=utt.physical_confidence,
                has_semantic_annotations=has_semantic,
            )

            # Composite quality score
            utt.quality_score = (
                0.3 * utt.transcript_confidence
                + 0.3 * utt.diarization_confidence
                + 0.2 * (float(np.mean(utt.physical_observed_mask)) if utt.physical_observed_mask is not None else 0.0)
                + 0.2 * (1.0 if has_semantic else 0.0)
            )

            utt.stage_completed = BootstrapStage.CONFIDENCE_SCORING

        return utterances

    def _stage_export_cache(
        self, utterances: List[BootstrapUtterance], corpus_id: str,
    ) -> List[BootstrapUtterance]:
        """Stage 12: Export to train-ready cache.

        Writes utterance data to disk in the v4 cache layout.
        Also encodes codec tokens (acoustic + control) for each utterance.
        """
        output_dir = self.config.output_dir / corpus_id
        output_dir.mkdir(parents=True, exist_ok=True)
        models = self._ensure_models()

        for utt in utterances:
            # Encode codec tokens if not already present
            if utt.acoustic_tokens is None or utt.control_tokens is None:
                try:
                    waveform, sr = self._load_audio(utt)
                    wav_np = waveform.squeeze(0).numpy()
                    a_tokens, c_tokens = models.encode_audio(wav_np, sample_rate=sr)
                    utt.acoustic_tokens = a_tokens
                    utt.control_tokens = c_tokens
                    utt.n_frames = a_tokens.shape[-1]
                except Exception as e:
                    logger.error("Codec encoding failed for %s: %s", utt.utterance_id, e)
                    n_frames = max(1, utt.n_frames) if utt.n_frames > 0 else 1
                    utt.acoustic_tokens = np.zeros((8, n_frames), dtype=np.int64)
                    utt.control_tokens = np.zeros((4, n_frames), dtype=np.int64)

            utt_dir = output_dir / utt.pseudo_speaker_id / utt.utterance_id
            utt_dir.mkdir(parents=True, exist_ok=True)

            # Save numpy arrays
            if utt.acoustic_tokens is not None:
                np.save(utt_dir / "acoustic_tokens.npy", utt.acoustic_tokens)
            if utt.control_tokens is not None:
                np.save(utt_dir / "control_tokens.npy", utt.control_tokens)
            if utt.speaker_embed is not None:
                np.save(utt_dir / "spk_embed.npy", utt.speaker_embed)
            if utt.phoneme_ids is not None:
                np.save(utt_dir / "phoneme_ids.npy", utt.phoneme_ids)
            if utt.physical_targets is not None:
                np.save(utt_dir / "physical_targets.npy", utt.physical_targets)
            if utt.physical_observed_mask is not None:
                np.save(utt_dir / "physical_observed_mask.npy", utt.physical_observed_mask)
            if utt.physical_confidence is not None:
                np.save(utt_dir / "physical_confidence.npy", utt.physical_confidence)

            # Save metadata
            meta = {
                "utterance_id": utt.utterance_id,
                "corpus_id": utt.corpus_id,
                "pseudo_speaker_id": utt.pseudo_speaker_id,
                "text_transcript": utt.text_transcript,
                "enriched_transcript": utt.enriched_transcript,
                "language": utt.language,
                "duration_sec": utt.duration_sec,
                "n_frames": utt.n_frames,
                "supervision_tier": utt.supervision_tier,
                "quality_score": utt.quality_score,
                "transcript_confidence": utt.transcript_confidence,
                "diarization_confidence": utt.diarization_confidence,
                "acting_annotations": utt.acting_annotations,
                "schema_version": "v4.0",
            }
            with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            utt.stage_completed = BootstrapStage.CACHE_EXPORT

        return utterances

    def _compile_result(
        self, utterances: List[BootstrapUtterance], result: BootstrapResult,
    ) -> None:
        """Compile final statistics."""
        accepted = [u for u in utterances if not u.is_rejected]
        result.accepted_utterances = len(accepted)

        for utt in accepted:
            if utt.supervision_tier == "tier_a":
                result.tier_a_count += 1
            elif utt.supervision_tier == "tier_b":
                result.tier_b_count += 1
            elif utt.supervision_tier == "tier_c":
                result.tier_c_count += 1
            else:
                result.tier_d_count += 1

        if accepted:
            result.mean_quality_score = float(np.mean([u.quality_score for u in accepted]))
            result.mean_transcript_confidence = float(
                np.mean([u.transcript_confidence for u in accepted])
            )
            result.mean_diarization_confidence = float(
                np.mean([u.diarization_confidence for u in accepted])
            )

            coverages = []
            for u in accepted:
                if u.physical_observed_mask is not None:
                    coverages.append(float(np.mean(u.physical_observed_mask)))
            result.physical_label_coverage = float(np.mean(coverages)) if coverages else 0.0
