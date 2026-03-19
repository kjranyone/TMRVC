"""Stage 4: Speaker diarization.

Wraps the existing PyAnnoteDiarizationProvider to assign per-segment
speaker cluster IDs and overlap flags.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)


class DiarizationStage:
    """Speaker diarization using pyannote.audio.

    Wraps :class:`PyAnnoteDiarizationProvider` from the curation
    providers layer.  Falls back to a simple energy-based heuristic
    when pyannote is unavailable.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()
        self._pipeline = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Assign file-local speaker cluster IDs to each utterance.

        Utterances from the same source file are grouped and diarized
        together so that segments receive consistent local speaker labels.
        """
        if not utterances:
            return utterances

        # Group by source file for batch diarization
        file_groups: Dict[str, List[BootstrapUtterance]] = {}
        for utt in utterances:
            if utt.is_rejected:
                utt.stage_completed = BootstrapStage.DIARIZATION
                continue
            file_groups.setdefault(utt.source_file, []).append(utt)

        for source_file, group in file_groups.items():
            try:
                self._diarize_group(source_file, group)
            except Exception as exc:
                logger.warning(
                    "Diarization failed for %s: %s", source_file, exc,
                )
                for utt in group:
                    utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_unknown"
                    utt.diarization_confidence = 0.0
                    utt.warnings.append(f"diarization_error:{exc}")
                    utt.stage_completed = BootstrapStage.DIARIZATION

        logger.info("Diarization: processed %d utterances", len(utterances))
        return utterances

    # ------------------------------------------------------------------
    # Diarization logic
    # ------------------------------------------------------------------

    def _diarize_group(
        self, source_file: str, utterances: List[BootstrapUtterance],
    ) -> None:
        """Diarize utterances from a single source file."""
        try:
            diarization_result = self._run_pyannote(source_file)
            self._assign_from_pyannote(utterances, diarization_result)
        except (ImportError, Exception) as exc:
            logger.info(
                "pyannote unavailable (%s), using embedding-based fallback",
                exc,
            )
            self._assign_from_embedding(utterances)

    def _run_pyannote(self, audio_path: str) -> Any:
        """Run pyannote speaker-diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for diarization. "
                "Install with: pip install pyannote.audio"
            ) from exc

        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                self.config.diarization_model,
            )

        diarization = self._pipeline(audio_path)
        return diarization

    def _assign_from_pyannote(
        self, utterances: List[BootstrapUtterance], diarization: Any,
    ) -> None:
        """Map pyannote diarization results to utterance segments."""
        # Build a list of (start, end, speaker, confidence) from pyannote output
        speaker_segments: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        for utt in utterances:
            if utt.segment is None:
                utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_unknown"
                utt.diarization_confidence = 0.0
                utt.stage_completed = BootstrapStage.DIARIZATION
                continue

            utt_start = utt.segment.start_sec
            utt_end = utt.segment.end_sec

            # Find the dominant speaker for this segment
            speaker_durations: Dict[str, float] = {}
            for seg in speaker_segments:
                overlap_start = max(utt_start, seg["start"])
                overlap_end = min(utt_end, seg["end"])
                if overlap_start < overlap_end:
                    dur = overlap_end - overlap_start
                    spk = seg["speaker"]
                    speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

            if speaker_durations:
                dominant = max(speaker_durations, key=speaker_durations.get)
                total_dur = utt_end - utt_start
                dominant_dur = speaker_durations[dominant]
                confidence = dominant_dur / total_dur if total_dur > 0 else 0.0

                utt.speaker_cluster_id = hash(dominant) % 10000
                utt.pseudo_speaker_id = f"spk_{dominant}"
                utt.diarization_confidence = round(confidence, 4)
            else:
                utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_unknown"
                utt.diarization_confidence = 0.0

            utt.stage_completed = BootstrapStage.DIARIZATION

    def _assign_from_embedding(
        self, utterances: List[BootstrapUtterance],
    ) -> None:
        """Simple fallback: assign speaker based on segment energy profile.

        When pyannote is unavailable, we use a basic approach: compute
        a simple audio fingerprint per segment and cluster using cosine
        similarity of MFCCs.
        """
        embeddings: List[Optional[np.ndarray]] = []

        for utt in utterances:
            try:
                emb = self._compute_simple_embedding(utt)
                embeddings.append(emb)
            except Exception:
                embeddings.append(None)

        # Simple greedy clustering
        clusters: List[np.ndarray] = []
        cluster_ids: List[int] = []
        similarity_threshold = 0.7

        for i, emb in enumerate(embeddings):
            utt = utterances[i]

            if emb is None:
                utt.pseudo_speaker_id = f"spk_{utt.corpus_id}_unknown"
                utt.diarization_confidence = 0.0
                utt.stage_completed = BootstrapStage.DIARIZATION
                continue

            assigned = False
            for c_idx, centroid in enumerate(clusters):
                sim = self._cosine_sim(emb, centroid)
                if sim > similarity_threshold:
                    # Update centroid (running average)
                    n = cluster_ids.count(c_idx)
                    clusters[c_idx] = (centroid * n + emb) / (n + 1)
                    utt.speaker_cluster_id = c_idx
                    utt.pseudo_speaker_id = f"spk_local_{c_idx:04d}"
                    utt.diarization_confidence = round(float(sim), 4)
                    assigned = True
                    break

            if not assigned:
                c_idx = len(clusters)
                clusters.append(emb.copy())
                cluster_ids.append(c_idx)
                utt.speaker_cluster_id = c_idx
                utt.pseudo_speaker_id = f"spk_local_{c_idx:04d}"
                utt.diarization_confidence = 0.5  # New cluster, moderate conf

            utt.stage_completed = BootstrapStage.DIARIZATION

    @staticmethod
    def _compute_simple_embedding(utt: BootstrapUtterance) -> np.ndarray:
        """Compute a simple MFCC-based embedding for speaker clustering."""
        path = Path(utt.audio_path)

        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32")
        except Exception:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            data = waveform.numpy().squeeze()

        if data.ndim > 1:
            data = np.mean(data, axis=0)

        # Extract segment
        if utt.segment is not None and utt.segment.end_sec > 0:
            start = int(utt.segment.start_sec * sr)
            end = int(utt.segment.end_sec * sr)
            data = data[start:end]

        try:
            import librosa
            mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=512)
            return np.mean(mfcc, axis=1).astype(np.float32)
        except ImportError:
            # Fallback: simple spectral statistics
            n_fft = min(2048, len(data))
            if n_fft < 64:
                return np.zeros(20, dtype=np.float32)
            spectrum = np.abs(np.fft.rfft(data[:n_fft]))
            # Bin into 20 bands
            n_bins = min(20, len(spectrum))
            band_size = len(spectrum) // n_bins
            embedding = np.array([
                np.mean(spectrum[i * band_size:(i + 1) * band_size])
                for i in range(n_bins)
            ], dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding /= norm
            return embedding

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
