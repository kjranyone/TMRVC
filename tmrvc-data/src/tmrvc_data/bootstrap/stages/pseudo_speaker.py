"""Stage 5: Cross-file pseudo speaker assignment.

Uses the existing CrossFileSpeakerClustering provider to merge
file-local speaker clusters into global persistent speaker IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)


class PseudoSpeakerStage:
    """Cross-file speaker clustering for persistent pseudo speaker IDs.

    Wraps :class:`CrossFileSpeakerClustering` from the curation
    providers layer.  Operates on speaker embeddings computed by
    the diarization stage (MFCC-based or ECAPA-TDNN).
    """

    def __init__(
        self,
        config: Optional[BootstrapConfig] = None,
        *,
        similarity_threshold: float = 0.80,
        embedding_dim: int = 192,
    ) -> None:
        self.config = config or BootstrapConfig()
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self._clustering = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Assign global pseudo speaker IDs across all utterances.

        Utterances with existing speaker embeddings are clustered
        using the cross-file clustering algorithm.  Utterances without
        embeddings receive a fallback ID based on their local cluster.
        """
        clustering = self._get_clustering()

        # Collect embeddings for batch processing
        embeddings: List[Tuple[int, np.ndarray, str]] = []  # (idx, embed, local_id)
        for i, utt in enumerate(utterances):
            if utt.is_rejected:
                utt.stage_completed = BootstrapStage.PSEUDO_SPEAKER
                continue

            # Use speaker_embed if available, otherwise use a placeholder
            if utt.speaker_embed is not None and np.any(utt.speaker_embed != 0):
                emb = np.asarray(utt.speaker_embed, dtype=np.float32)
            else:
                # Try to compute a simple embedding from audio
                emb = self._compute_fallback_embedding(utt)

            local_id = utt.pseudo_speaker_id or f"local_{utt.speaker_cluster_id}"
            embeddings.append((i, emb, local_id))

        if not embeddings:
            return utterances

        # Run batch clustering
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
            SpeakerEmbedding,
        )

        speaker_embeddings = [
            SpeakerEmbedding(
                record_id=utterances[idx].utterance_id,
                file_local_speaker_id=local_id,
                embedding=emb,
                duration_sec=utterances[idx].duration_sec,
            )
            for idx, emb, local_id in embeddings
        ]

        results = clustering.cluster_batch(speaker_embeddings)

        # Assign results back to utterances
        for (idx, _, _), (global_id, confidence) in zip(embeddings, results):
            utt = utterances[idx]
            utt.pseudo_speaker_id = global_id
            utt.diarization_confidence = max(
                utt.diarization_confidence, confidence,
            )
            utt.stage_completed = BootstrapStage.PSEUDO_SPEAKER

        n_clusters = len(clustering.get_clusters())
        logger.info(
            "PseudoSpeaker: %d utterances -> %d global clusters",
            len(embeddings), n_clusters,
        )

        return utterances

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_clustering(self):
        """Lazy-init the cross-file speaker clustering provider."""
        if self._clustering is not None:
            return self._clustering

        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
        )

        self._clustering = CrossFileSpeakerClustering(
            similarity_threshold=self.similarity_threshold,
            embedding_dim=self.embedding_dim,
        )
        return self._clustering

    @staticmethod
    def _compute_fallback_embedding(utt: BootstrapUtterance) -> np.ndarray:
        """Compute a simple MFCC-based embedding when no speaker embed exists."""
        if utt.audio_path is None:
            return np.zeros(192, dtype=np.float32)

        try:
            import soundfile as sf
            data, sr = sf.read(str(utt.audio_path), dtype="float32")
        except Exception:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(str(utt.audio_path))
                data = waveform.numpy().squeeze()
            except Exception:
                return np.zeros(192, dtype=np.float32)

        if data.ndim > 1:
            data = np.mean(data, axis=0)

        # Extract segment
        if utt.segment is not None and utt.segment.end_sec > 0:
            start = int(utt.segment.start_sec * sr)
            end = int(utt.segment.end_sec * sr)
            data = data[start:end]

        if len(data) < 100:
            return np.zeros(192, dtype=np.float32)

        try:
            import librosa
            mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40, hop_length=512)
            # Expand to 192-D by computing mean + std + delta stats
            mean_mfcc = np.mean(mfcc, axis=1)  # [40]
            std_mfcc = np.std(mfcc, axis=1)    # [40]
            delta = np.diff(mfcc, axis=1)
            mean_delta = np.mean(delta, axis=1) if delta.shape[1] > 0 else np.zeros(40)  # [40]
            std_delta = np.std(delta, axis=1) if delta.shape[1] > 0 else np.zeros(40)    # [40]

            # Concatenate to get close to 192 dims
            features = np.concatenate([
                mean_mfcc, std_mfcc, mean_delta, std_delta,
                mean_mfcc[:32], std_mfcc[:32],
            ])  # 40*4 + 32*2 = 224, truncate to 192
            features = features[:192].astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features /= norm
            return features
        except ImportError:
            # Ultra-fallback: spectral bins
            n_fft = min(2048, len(data))
            spectrum = np.abs(np.fft.rfft(data[:n_fft]))
            # Rebin to 192
            indices = np.linspace(0, len(spectrum) - 1, 192).astype(int)
            embedding = spectrum[indices].astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding /= norm
            return embedding
