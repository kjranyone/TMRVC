"""Stage 12: Train-ready cache export.

Writes utterance data to disk in the v4 cache layout (msgpack/numpy).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from tmrvc_data.bootstrap.contracts import BootstrapConfig, BootstrapStage, BootstrapUtterance

logger = logging.getLogger(__name__)


class CacheExportStage:
    """Export utterances to v4 train-ready cache format."""

    def __init__(self, config: BootstrapConfig) -> None:
        self.config = config

    def process(
        self, utterances: List[BootstrapUtterance], corpus_id: str = "",
    ) -> List[BootstrapUtterance]:
        """Export all accepted utterances to cache."""
        active = [u for u in utterances if not u.is_rejected]
        cid = corpus_id or (active[0].corpus_id if active else "unknown")
        output_dir = self.config.output_dir / cid

        exported = 0
        for utt in active:
            try:
                self._export_utterance(utt, output_dir)
                utt.stage_completed = BootstrapStage.CACHE_EXPORT
                exported += 1
            except Exception as e:
                logger.warning("Cache export failed for %s: %s", utt.utterance_id, e)
                utt.errors.append(f"cache_export_failed: {e}")

        logger.info("Exported %d / %d utterances to %s", exported, len(active), output_dir)
        return utterances

    def _export_utterance(self, utt: BootstrapUtterance, output_dir: Path) -> None:
        """Export a single utterance to cache."""
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

        # Save metadata as JSON
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

        # Also try msgpack export
        try:
            self._export_msgpack(utt, utt_dir)
        except ImportError:
            pass  # msgpack optional

    @staticmethod
    def _export_msgpack(utt: BootstrapUtterance, utt_dir: Path) -> None:
        """Export compact msgpack format."""
        import msgpack

        data = {
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
        }
        with open(utt_dir / "meta.msgpack", "wb") as f:
            msgpack.pack(data, f)
