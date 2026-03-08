"""Cross-file speaker clustering provider (Worker 08).

Normalizes file-local diarization speaker IDs into persistent
dataset-global speaker cluster IDs using embedding similarity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import CurationRecord, Provenance
from . import SpeakerClusteringProvider, ProviderOutput, ProviderUnavailableError

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEmbedding:
    """Speaker embedding extracted from a diarized segment."""

    record_id: str
    file_local_speaker_id: str
    embedding: np.ndarray  # shape [D]
    duration_sec: float = 0.0


@dataclass
class GlobalSpeakerCluster:
    """A global speaker cluster spanning multiple files."""

    global_id: str
    centroid: np.ndarray
    member_count: int = 0
    total_duration_sec: float = 0.0
    member_file_local_ids: List[Tuple[str, str]] = field(
        default_factory=list
    )  # [(record_id, local_speaker_id), ...]


class CrossFileSpeakerClustering(SpeakerClusteringProvider):
    """Cross-file speaker clustering using embedding cosine similarity.

    Takes file-local speaker embeddings from the diarization stage and
    assigns persistent global speaker cluster IDs.  This is essential
    for multi-file datasets where the same speaker appears across
    different recordings.

    Recommended embedding backends:
    - ``wespeaker``
    - ``SpeechBrain`` speaker verification embeddings

    Stub mode uses simple centroid-based greedy clustering on whatever
    embeddings are provided (e.g., spectral embeddings from the builtin
    diarization).
    """

    name = "cross_file_speaker_clustering"
    version = "1.0.0"

    artifact_id: str = "builtin/speaker-clustering-v1"
    runtime_backend: str = "numpy"
    calibration_version: str = "uncalibrated"

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.80,
        embedding_dim: int = 192,
        calibration_version: str = "uncalibrated",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.calibration_version = calibration_version

        # Mutable cluster state
        self._clusters: List[GlobalSpeakerCluster] = []
        self._next_cluster_id: int = 0

    def is_available(self) -> bool:
        return True  # numpy-only fallback always works

    def process(self, record: CurationRecord, **kwargs: Any) -> ProviderOutput:
        """Assign a global speaker cluster ID to a single record.

        This is the per-record entry point for integration with the
        orchestrator.  For batch clustering, use ``cluster_batch()``
        instead.

        Expects record.attributes to contain 'speaker_embedding'
        (list of floats) from the diarization stage.
        """
        embedding_raw = record.attributes.get("speaker_embedding")
        if embedding_raw is None:
            return ProviderOutput(
                fields={},
                confidence=0.0,
                warnings=[
                    "No speaker_embedding in record attributes; "
                    "cannot assign global cluster"
                ],
                provenance=self.make_provenance(confidence=0.0),
            )

        embedding = np.array(embedding_raw, dtype=np.float32)
        local_id = record.speaker_cluster or "spk_unknown"

        se = SpeakerEmbedding(
            record_id=record.record_id,
            file_local_speaker_id=local_id,
            embedding=embedding,
            duration_sec=record.duration_sec,
        )

        global_id, conf = self._assign_to_cluster(se)

        return ProviderOutput(
            fields={
                "speaker_cluster": global_id,
                "attributes": {
                    "global_speaker_cluster": global_id,
                    "speaker_clustering_confidence": round(conf, 4),
                    "file_local_speaker_id": local_id,
                },
            },
            confidence=round(conf, 4),
            provenance=self.make_provenance(
                confidence=round(conf, 4),
                metadata={
                    "artifact_id": self.artifact_id,
                    "calibration_version": self.calibration_version,
                    "n_clusters": len(self._clusters),
                    "similarity_threshold": self.similarity_threshold,
                },
            ),
        )

    # ------------------------------------------------------------------
    # Batch clustering API
    # ------------------------------------------------------------------

    def cluster_batch(
        self,
        embeddings: List[SpeakerEmbedding],
    ) -> List[Tuple[str, float]]:
        """Assign global cluster IDs to a batch of speaker embeddings.

        Returns list of (global_id, confidence) tuples, one per input.
        """
        results: List[Tuple[str, float]] = []
        for emb in embeddings:
            gid, conf = self._assign_to_cluster(emb)
            results.append((gid, conf))
        return results

    def get_clusters(self) -> List[GlobalSpeakerCluster]:
        """Return current cluster state (for inspection / serialization)."""
        return list(self._clusters)

    def reset(self) -> None:
        """Clear all cluster state."""
        self._clusters.clear()
        self._next_cluster_id = 0

    # ------------------------------------------------------------------
    # Internal clustering logic
    # ------------------------------------------------------------------

    def _assign_to_cluster(
        self, se: SpeakerEmbedding,
    ) -> Tuple[str, float]:
        """Assign a single embedding to the best matching cluster or create one."""
        emb = se.embedding
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-8:
            # Zero embedding -- assign to a new singleton cluster with low conf
            return self._create_cluster(se), 0.1

        best_sim = -1.0
        best_idx = -1

        for i, cluster in enumerate(self._clusters):
            c_norm = np.linalg.norm(cluster.centroid)
            if c_norm < 1e-8:
                continue
            sim = float(np.dot(cluster.centroid, emb) / (c_norm * emb_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.similarity_threshold and best_idx >= 0:
            # Merge into existing cluster
            cluster = self._clusters[best_idx]
            # Update centroid with running average
            n = cluster.member_count
            cluster.centroid = (cluster.centroid * n + emb) / (n + 1)
            cluster.member_count += 1
            cluster.total_duration_sec += se.duration_sec
            cluster.member_file_local_ids.append(
                (se.record_id, se.file_local_speaker_id)
            )
            return cluster.global_id, round(best_sim, 4)
        else:
            return self._create_cluster(se), 0.5  # new cluster, moderate conf

    def _create_cluster(self, se: SpeakerEmbedding) -> str:
        """Create a new global cluster from a single embedding."""
        gid = f"gspk_{self._next_cluster_id:05d}"
        self._next_cluster_id += 1
        cluster = GlobalSpeakerCluster(
            global_id=gid,
            centroid=se.embedding.copy(),
            member_count=1,
            total_duration_sec=se.duration_sec,
            member_file_local_ids=[(se.record_id, se.file_local_speaker_id)],
        )
        self._clusters.append(cluster)
        return gid
