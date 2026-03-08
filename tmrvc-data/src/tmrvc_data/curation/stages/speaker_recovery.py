"""Stage 3: Speaker Structure Recovery - Diarization and speaker clustering.

Recovers turn-taking information needed for dialogue-sensitive modeling:
speaker turns, overlap flags, speaker clusters, conversation structure.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..models import CurationRecord, Provenance, RecordStatus
from ..providers import ProviderRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SILENCE_THRESHOLD_DB = -40.0
MIN_SEGMENT_SEC = 0.5
EMBEDDING_DIM = 32  # dimension for simple spectral speaker embeddings


def _load_audio_mono(path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
    """Load audio segment as mono float32."""
    info = sf.info(path)
    sr = info.samplerate
    start_frame = int(start * sr)
    n_frames = int((end - start) * sr)
    audio, sr = sf.read(path, start=start_frame, frames=n_frames,
                        dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _frame_energy_db(audio: np.ndarray, sr: int,
                     frame_sec: float = 0.025, hop_sec: float = 0.010) -> np.ndarray:
    """Compute per-frame energy in dB."""
    frame_len = max(1, int(frame_sec * sr))
    hop_len = max(1, int(hop_sec * sr))
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start: start + frame_len]
        e = np.mean(frame ** 2)
        energies[i] = 10.0 * np.log10(e + 1e-12)
    return energies


def _find_speech_segments(
    energies: np.ndarray, hop_sec: float, threshold_db: float
) -> List[Tuple[float, float]]:
    """Find contiguous speech segments from frame energies."""
    segments: List[Tuple[float, float]] = []
    in_speech = False
    seg_start = 0.0

    for i, e in enumerate(energies):
        t = i * hop_sec
        if e > threshold_db:
            if not in_speech:
                seg_start = t
                in_speech = True
        else:
            if in_speech:
                segments.append((seg_start, t))
                in_speech = False

    if in_speech:
        segments.append((seg_start, len(energies) * hop_sec))

    # Merge short gaps (< 0.3s)
    merged: List[Tuple[float, float]] = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) < 0.3:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # Filter short segments
    return [(s, e) for s, e in merged if (e - s) >= MIN_SEGMENT_SEC]


def _extract_spectral_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a simple spectral embedding for speaker characterization.

    Uses mel-frequency-like spectral statistics as a lightweight speaker
    fingerprint. This is a rough proxy - real systems use neural embeddings.
    """
    n_fft = min(2048, len(audio))
    if n_fft < 64 or len(audio) < n_fft:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Compute average spectrum over multiple windows
    hop = n_fft // 2
    n_wins = max(1, (len(audio) - n_fft) // hop)
    spectra = []
    for i in range(min(n_wins, 50)):  # cap at 50 windows
        start = i * hop
        frame = audio[start: start + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        spectra.append(spec)

    avg_spec = np.mean(spectra, axis=0)

    # Reduce to EMBEDDING_DIM via log-spaced binning
    n_bins = len(avg_spec)
    if n_bins <= EMBEDDING_DIM:
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        embedding[:n_bins] = avg_spec[:n_bins]
    else:
        # Log-spaced bin edges
        edges = np.logspace(0, np.log10(n_bins), EMBEDDING_DIM + 1, dtype=int)
        edges = np.clip(edges, 0, n_bins)
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for j in range(EMBEDDING_DIM):
            lo, hi = edges[j], edges[j + 1]
            if hi > lo:
                embedding[j] = np.mean(avg_spec[lo:hi])
            elif lo < n_bins:
                embedding[j] = avg_spec[lo]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def _cluster_segments(
    embeddings: List[np.ndarray], n_segments: int
) -> Tuple[List[str], float]:
    """Assign speaker cluster IDs using cosine similarity.

    Simple greedy clustering: each new segment is compared to existing
    cluster centroids. If similarity exceeds threshold, assign to that
    cluster; otherwise create a new one.
    """
    if n_segments == 0:
        return [], 0.0

    SIMILARITY_THRESHOLD = 0.85
    clusters: List[List[np.ndarray]] = []
    cluster_ids: List[int] = []

    for emb in embeddings:
        best_sim = -1.0
        best_cluster = -1

        for ci, cluster_embs in enumerate(clusters):
            centroid = np.mean(cluster_embs, axis=0)
            c_norm = np.linalg.norm(centroid)
            e_norm = np.linalg.norm(emb)
            if c_norm > 1e-8 and e_norm > 1e-8:
                sim = float(np.dot(centroid, emb) / (c_norm * e_norm))
            else:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_cluster = ci

        if best_sim >= SIMILARITY_THRESHOLD and best_cluster >= 0:
            clusters[best_cluster].append(emb)
            cluster_ids.append(best_cluster)
        else:
            clusters.append([emb])
            cluster_ids.append(len(clusters) - 1)

    # Estimate cluster purity (higher = more confident assignment)
    # Use average intra-cluster similarity
    purity_scores = []
    for cluster_embs in clusters:
        if len(cluster_embs) < 2:
            purity_scores.append(1.0)
            continue
        centroid = np.mean(cluster_embs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-8:
            purity_scores.append(0.5)
            continue
        sims = []
        for e in cluster_embs:
            e_norm = np.linalg.norm(e)
            if e_norm > 1e-8:
                sims.append(float(np.dot(centroid, e) / (c_norm * e_norm)))
        purity_scores.append(np.mean(sims) if sims else 0.5)

    avg_purity = float(np.mean(purity_scores)) if purity_scores else 0.5

    # Generate human-readable cluster IDs
    labels = [f"spk_{cid:03d}" for cid in cluster_ids]
    return labels, round(avg_purity, 4)


def run_speaker_recovery(
    record: CurationRecord,
    registry: Optional[ProviderRegistry] = None,
) -> Optional[CurationRecord]:
    """Process a single record through Stage 3: Speaker Structure Recovery.

    Runs diarization (external provider if available, else builtin energy-based),
    extracts speaker embeddings, assigns cluster IDs, and builds conversation
    structure metadata.

    Args:
        record: The curation record to process.
        registry: Optional provider registry for external diarization models.

    Returns:
        Updated CurationRecord.
    """
    if record.status == RecordStatus.REJECTED:
        return record

    # --- Try external diarization provider ---
    if registry is not None:
        provider = registry.get_primary("diarization")
        if provider is not None and provider.is_available():
            try:
                output = provider.process(record)
                for key, value in output.fields.items():
                    if key == "attributes" and isinstance(value, dict):
                        record.attributes.update(value)
                    else:
                        setattr(record, key, value)
                if output.provenance:
                    record.providers["speaker_recovery"] = output.provenance
                return record
            except Exception as e:
                logger.warning(
                    "Diarization provider failed for %s: %s",
                    record.record_id, e,
                )

    # --- Builtin energy-based segmentation ---
    try:
        audio, sr = _load_audio_mono(
            record.source_path,
            record.segment_start_sec,
            record.segment_end_sec,
        )
    except Exception as e:
        logger.warning("Speaker recovery: cannot load %s: %s", record.record_id, e)
        record.providers["speaker_recovery"] = Provenance(
            stage="speaker_recovery",
            provider="builtin_energy",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata={"error": str(e)},
        )
        return record

    hop_sec = 0.010
    energies = _frame_energy_db(audio, sr, hop_sec=hop_sec)
    segments = _find_speech_segments(energies, hop_sec, SILENCE_THRESHOLD_DB)

    if not segments:
        # No speech segments found - single speaker, whole file
        record.speaker_cluster = "spk_000"
        record.diarization_confidence = 0.5
        record.attributes["speaker_turns"] = []
        record.attributes["overlap_flags"] = []
        record.attributes["cluster_purity_estimate"] = 0.5
        record.attributes["n_speakers_detected"] = 1
        record.providers["speaker_recovery"] = Provenance(
            stage="speaker_recovery",
            provider="builtin_energy",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.5,
            metadata={"n_segments": 0, "method": "no_segments_single_speaker"},
        )
        return record

    # Extract embeddings for each segment
    embeddings = []
    for seg_start, seg_end in segments:
        s_idx = int(seg_start * sr)
        e_idx = int(seg_end * sr)
        seg_audio = audio[s_idx:e_idx]
        if len(seg_audio) > 0:
            emb = _extract_spectral_embedding(seg_audio, sr)
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))

    # Cluster segments
    cluster_labels, purity = _cluster_segments(embeddings, len(segments))
    n_speakers = len(set(cluster_labels))

    # Build speaker turns
    speaker_turns = []
    for i, ((seg_start, seg_end), label) in enumerate(zip(segments, cluster_labels)):
        speaker_turns.append({
            "turn_index": i,
            "speaker": label,
            "start_sec": round(seg_start, 4),
            "end_sec": round(seg_end, 4),
            "duration_sec": round(seg_end - seg_start, 4),
        })

    # Detect overlaps (simplified: check if segments overlap in time)
    overlap_flags = []
    for i in range(len(segments) - 1):
        if segments[i][1] > segments[i + 1][0]:
            overlap_flags.append({
                "turn_a": i,
                "turn_b": i + 1,
                "overlap_sec": round(segments[i][1] - segments[i + 1][0], 4),
            })

    overlap_ratio = 0.0
    if segments:
        total_dur = record.segment_end_sec - record.segment_start_sec
        overlap_dur = sum(o["overlap_sec"] for o in overlap_flags)
        overlap_ratio = overlap_dur / total_dur if total_dur > 0 else 0.0

    # Assign primary speaker (most frequent cluster)
    if cluster_labels:
        from collections import Counter
        primary_speaker = Counter(cluster_labels).most_common(1)[0][0]
        record.speaker_cluster = primary_speaker
    else:
        record.speaker_cluster = "spk_000"

    record.diarization_confidence = purity

    # Build conversation structure
    record.conversation_id = hashlib.md5(
        f"{record.source_path}:{record.segment_start_sec}".encode()
    ).hexdigest()[:12]

    record.attributes["speaker_turns"] = speaker_turns
    record.attributes["overlap_flags"] = overlap_flags
    record.attributes["overlap_ratio"] = round(overlap_ratio, 4)
    record.attributes["cluster_purity_estimate"] = purity
    record.attributes["n_speakers_detected"] = n_speakers
    record.attributes["speaker_embeddings_dim"] = EMBEDDING_DIM

    # Build turn adjacency
    if len(speaker_turns) > 1:
        record.turn_index = 0  # Record represents first turn by default
        record.attributes["turn_adjacency"] = [
            {
                "from_turn": i,
                "to_turn": i + 1,
                "from_speaker": speaker_turns[i]["speaker"],
                "to_speaker": speaker_turns[i + 1]["speaker"],
                "gap_sec": round(
                    speaker_turns[i + 1]["start_sec"] - speaker_turns[i]["end_sec"], 4
                ),
            }
            for i in range(len(speaker_turns) - 1)
        ]

    record.providers["speaker_recovery"] = Provenance(
        stage="speaker_recovery",
        provider="builtin_energy",
        version="1.0.0",
        timestamp=time.time(),
        confidence=purity,
        metadata={
            "n_segments": len(segments),
            "n_speakers": n_speakers,
            "overlap_ratio": round(overlap_ratio, 4),
            "cluster_purity": purity,
            "method": "energy_segmentation_spectral_clustering",
        },
    )

    return record
