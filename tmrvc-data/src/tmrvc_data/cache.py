"""Feature cache: save/load individual .npy files with optional mmap."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.types import FeatureSet

logger = logging.getLogger(__name__)


class FeatureCache:
    """Disk-backed feature cache using individual .npy files.

    Layout::

        cache_dir/{dataset}/{split}/{speaker}/{utterance}/
            mel.npy         # [80, T]
            content.npy     # [content_dim, T] where content_dim ∈ {768, 1024}
            f0.npy          # [1, T]
            spk_embed.npy   # [192]
            meta.json       # {utterance_id, speaker_id, n_frames, content_dim, ...}
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _utt_dir(
        self, dataset: str, split: str, speaker_id: str, utterance_id: str
    ) -> Path:
        return self.cache_dir / dataset / split / speaker_id / utterance_id

    def save(
        self,
        features: FeatureSet,
        dataset: str,
        split: str = "train",
        waveform: torch.Tensor | None = None,
    ) -> Path:
        """Save a FeatureSet to disk as individual .npy files.

        Args:
            waveform: Optional ``[1, T_samples]`` waveform for audio-level augmentation.

        Returns:
            Path to the utterance directory.
        """
        utt_dir = self._utt_dir(
            dataset, split, features.speaker_id, features.utterance_id
        )
        utt_dir.mkdir(parents=True, exist_ok=True)

        np.save(utt_dir / "mel.npy", features.mel.numpy())
        np.save(utt_dir / "content.npy", features.content.numpy())
        np.save(utt_dir / "f0.npy", features.f0.numpy())
        np.save(utt_dir / "spk_embed.npy", features.spk_embed.numpy())

        if waveform is not None:
            np.save(utt_dir / "waveform.npy", waveform.numpy())

        meta = {
            "utterance_id": features.utterance_id,
            "speaker_id": features.speaker_id,
            "n_frames": features.n_frames,
            "content_dim": features.content_dim,
        }
        with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return utt_dir

    def load(
        self,
        dataset: str,
        split: str,
        speaker_id: str,
        utterance_id: str,
        mmap: bool = True,
        load_waveform: bool = False,
    ) -> FeatureSet:
        """Load a FeatureSet from disk.

        Args:
            mmap: If True, use ``mmap_mode='r'`` for zero-copy reads.
            load_waveform: If True, also load ``waveform.npy`` (if it exists).
        """
        utt_dir = self._utt_dir(dataset, split, speaker_id, utterance_id)
        mmap_mode = "r" if mmap else None

        mel = np.load(utt_dir / "mel.npy", mmap_mode=mmap_mode)
        content = np.load(utt_dir / "content.npy", mmap_mode=mmap_mode)
        f0 = np.load(utt_dir / "f0.npy", mmap_mode=mmap_mode)
        spk_embed = np.load(utt_dir / "spk_embed.npy", mmap_mode=mmap_mode)

        with open(utt_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        content_dim = meta["content_dim"]

        waveform = None
        if load_waveform:
            wav_path = utt_dir / "waveform.npy"
            if wav_path.exists():
                waveform = torch.from_numpy(
                    np.array(np.load(wav_path, mmap_mode=mmap_mode))
                )

        return FeatureSet(
            mel=torch.from_numpy(np.array(mel)),
            content=torch.from_numpy(np.array(content)),
            f0=torch.from_numpy(np.array(f0)),
            spk_embed=torch.from_numpy(np.array(spk_embed)),
            utterance_id=meta["utterance_id"],
            speaker_id=meta["speaker_id"],
            n_frames=meta["n_frames"],
            content_dim=content_dim,
            waveform=waveform,
        )

    def exists(
        self, dataset: str, split: str, speaker_id: str, utterance_id: str
    ) -> bool:
        """Check if features exist for an utterance."""
        utt_dir = self._utt_dir(dataset, split, speaker_id, utterance_id)
        return (utt_dir / "meta.json").exists()

    def iter_entries(self, dataset: str, split: str = "train") -> list[dict[str, str]]:
        """List all cached entries for a dataset/split.

        Returns:
            List of dicts with keys ``speaker_id``, ``utterance_id``.
        """
        base = self.cache_dir / dataset / split
        if not base.exists():
            return []

        entries = []
        for spk_dir in sorted(base.iterdir()):
            if not spk_dir.is_dir():
                continue
            for utt_dir in sorted(spk_dir.iterdir()):
                if not utt_dir.is_dir():
                    continue
                if (utt_dir / "meta.json").exists():
                    entries.append(
                        {
                            "speaker_id": spk_dir.name,
                            "utterance_id": utt_dir.name,
                        }
                    )
        return entries

    def verify(self, dataset: str, split: str = "train") -> dict[str, int]:
        """Verify cache integrity.

        Returns:
            Dict with counts: ``total``, ``valid``, ``invalid``.
        """
        entries = self.iter_entries(dataset, split)
        valid = 0
        invalid = 0

        for entry in entries:
            utt_dir = self._utt_dir(
                dataset, split, entry["speaker_id"], entry["utterance_id"]
            )
            required_files = [
                "mel.npy",
                "content.npy",
                "f0.npy",
                "spk_embed.npy",
                "meta.json",
            ]
            if all((utt_dir / f).exists() for f in required_files):
                try:
                    with open(utt_dir / "meta.json", encoding="utf-8") as f:
                        meta = json.load(f)
                    mel = np.load(utt_dir / "mel.npy", mmap_mode="r")
                    assert mel.shape[0] == 80
                    assert mel.shape[1] == meta["n_frames"]
                    valid += 1
                except Exception:
                    logger.warning("Invalid cache entry: %s", utt_dir)
                    invalid += 1
            else:
                invalid += 1

        return {"total": len(entries), "valid": valid, "invalid": invalid}
