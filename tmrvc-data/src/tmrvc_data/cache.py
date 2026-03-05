"""Feature cache: save/load individual .npy files for UCLM v2."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import HOP_LENGTH
from tmrvc_core.types import UCLMFeatureSet

logger = logging.getLogger(__name__)


def _align_waveform_to_frame_count(
    waveform: torch.Tensor, n_frames: int, hop_length: int = HOP_LENGTH
) -> torch.Tensor:
    """Align waveform samples to n_frames * hop_length (trim/pad tail)."""
    target_samples = int(n_frames) * int(hop_length)

    if waveform.dim() == 1:
        wave = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        if waveform.shape[0] == 1:
            wave = waveform
        elif waveform.shape[1] == 1:
            wave = waveform.transpose(0, 1)
        else:
            raise ValueError(
                f"Expected mono waveform with one channel, got shape={tuple(waveform.shape)}."
            )
    else:
        raise ValueError(
            f"Expected waveform rank 1 or 2, got rank={waveform.dim()} shape={tuple(waveform.shape)}."
        )

    current_samples = int(wave.shape[-1])
    if current_samples == target_samples:
        return wave
    if current_samples > target_samples:
        return wave[..., :target_samples]

    pad_samples = target_samples - current_samples
    return torch.nn.functional.pad(wave, (0, pad_samples))


class FeatureCache:
    """Disk-backed feature cache for UCLM v2.

    Layout::

        cache_dir/{dataset}/{split}/{speaker}/{utterance}/
            codec_tokens_a.npy       # [8, T]
            codec_tokens_b.npy       # [4, T]
            explicit_state.npy       # [T, 8]
            ssl_state.npy            # [T, 128]
            spk_embed.npy            # [192]
            phoneme_ids.npy          # [L] (optional)
            durations.npy            # [L] (optional)
            meta.json
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _utt_dir(
        self, dataset: str, split: str, speaker_id: str, utterance_id: str
    ) -> Path:
        return self.cache_dir / dataset / split / speaker_id / utterance_id

    def save(
        self,
        features: UCLMFeatureSet,
        dataset: str,
        split: str = "train",
    ) -> Path:
        """Save a UCLMFeatureSet to disk."""
        utt_dir = self._utt_dir(dataset, split, features.speaker_id, features.utterance_id)
        utt_dir.mkdir(parents=True, exist_ok=True)

        np.save(utt_dir / "codec_tokens.npy", features.codec_tokens_a.numpy())
        np.save(utt_dir / "control_tokens.npy", features.codec_tokens_b.numpy())
        np.save(utt_dir / "explicit_state.npy", features.voice_state_explicit.numpy().T) # Save as [T, 8]
        np.save(utt_dir / "ssl_state.npy", features.voice_state_ssl.numpy().T)           # Save as [T, 128]
        np.save(utt_dir / "spk_embed.npy", features.spk_embed.numpy())

        if features.phoneme_ids is not None:
            np.save(utt_dir / "phoneme_ids.npy", features.phoneme_ids.numpy())
        if features.durations is not None:
            np.save(utt_dir / "durations.npy", features.durations.numpy())
        if features.waveform is not None:
            waveform = _align_waveform_to_frame_count(
                features.waveform.detach().cpu(), features.n_frames
            )
            np.save(utt_dir / "waveform.npy", waveform.numpy())

        meta = {
            "utterance_id": features.utterance_id,
            "speaker_id": features.speaker_id,
            "n_frames": features.n_frames,
            "text": features.text,
            "language_id": features.language_id,
        }
        
        meta_tmp = utt_dir / "meta.json.tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        meta_tmp.replace(utt_dir / "meta.json")

        return utt_dir

    def exists(
        self, dataset: str, split: str, speaker_id: str, utterance_id: str
    ) -> bool:
        """Check if features exist for an utterance."""
        utt_dir = self._utt_dir(dataset, split, speaker_id, utterance_id)
        return (utt_dir / "meta.json").exists()

    def iter_entries(self, dataset: str, split: str = "train") -> list[dict[str, str]]:
        base = self.cache_dir / dataset / split
        if not base.exists(): return []
        entries = []
        for spk_dir in sorted(base.iterdir()):
            if not spk_dir.is_dir(): continue
            for utt_dir in sorted(spk_dir.iterdir()):
                if not utt_dir.is_dir(): continue
                if (utt_dir / "meta.json").exists():
                    entries.append({"speaker_id": spk_dir.name, "utterance_id": utt_dir.name})
        return entries

    def verify(self, dataset: str, split: str = "train") -> dict[str, int]:
        entries = self.iter_entries(dataset, split)
        valid, invalid = 0, 0
        for entry in entries:
            utt_dir = self._utt_dir(dataset, split, entry["speaker_id"], entry["utterance_id"])
            if (utt_dir / "codec_tokens.npy").exists() and (utt_dir / "meta.json").exists():
                valid += 1
            else:
                invalid += 1
        return {"total": len(entries), "valid": valid, "invalid": invalid}
