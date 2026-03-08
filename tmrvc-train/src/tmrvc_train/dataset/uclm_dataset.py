from __future__ import annotations

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import json
from typing import Optional, Any

# Valid tts_mode values
_VALID_TTS_MODES = {"auto", "pointer", "legacy_duration"}


class DisentangledUCLMDataset(Dataset):
    """Dataset for Disentangled UCLM multi-task training (TTS & VC).

    Args:
        cache_dir: Root directory of the feature cache.
        max_frames: Maximum number of frames per utterance (for batching).
        include_datasets: Optional whitelist of dataset names to include.
        tts_mode: Controls how text supervision artifacts are loaded.
            - ``"auto"`` (default): load whatever is available (phoneme_ids
              and/or durations).  This preserves the original v2 behaviour.
            - ``"pointer"``: load ``phoneme_ids.npy`` but never require or
              load ``durations.npy``.  This is the recommended mode for
              UCLM v3 pointer-based text progression.
            - ``"legacy_duration"``: require *both* ``phoneme_ids.npy`` **and**
              ``durations.npy`` for a sample to have text supervision.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_frames: int = 400,
        include_datasets: list[str] | None = None,
        tts_mode: str = "auto",
    ):
        if tts_mode not in _VALID_TTS_MODES:
            raise ValueError(
                f"Invalid tts_mode={tts_mode!r}. Must be one of {sorted(_VALID_TTS_MODES)}."
            )
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        self.tts_mode = tts_mode
        self.include_datasets = (
            set(include_datasets) if include_datasets is not None else None
        )
        self.utterances = []
        self.speaker_to_id = {}
        self.id_to_speaker = {}

        # Scan for utterance meta.json files
        for meta_path in sorted(self.cache_dir.rglob("meta.json")):
            try:
                rel = meta_path.relative_to(self.cache_dir)
                dataset_name = rel.parts[0] if len(rel.parts) > 0 else None
                split_name = rel.parts[1] if len(rel.parts) > 1 else None
            except ValueError:
                continue

            if split_name != "train":
                continue
            if (
                self.include_datasets is not None
                and dataset_name not in self.include_datasets
            ):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                utt_dir = meta_path.parent
                required_files = [
                    "codec_tokens.npy",
                    "explicit_state.npy",
                    "ssl_state.npy",
                    "spk_embed.npy",
                ]

                if all((utt_dir / req).exists() for req in required_files):
                    spk_id = meta.get("speaker_id") or utt_dir.parent.name
                    self.utterances.append(
                        {
                            "meta": meta,
                            "path": utt_dir,
                            "dataset": dataset_name,
                            "speaker_id": spk_id,
                        }
                    )

                    if spk_id not in self.speaker_to_id:
                        int_id = len(self.speaker_to_id)
                        self.speaker_to_id[spk_id] = int_id
                        self.id_to_speaker[int_id] = spk_id
            except Exception:
                continue

        ds_info = (
            ",".join(sorted(self.include_datasets))
            if self.include_datasets is not None
            else "ALL"
        )
        print(
            f"[info] DisentangledUCLMDataset[{ds_info}]: loaded {len(self.utterances)} utterances, {len(self.speaker_to_id)} speakers."
        )

    def __len__(self) -> int:
        return len(self.utterances)

    def supervision_report(self) -> dict:
        """Return per-dataset supervision statistics.

        Returns a dict with coverage and UNK ratio.
        """
        from tmrvc_data.g2p import UNK_ID, PHONE2ID

        total = len(self.utterances)
        with_text = 0
        with_legacy_duration = 0
        total_phones = 0
        unk_phones = 0
        
        # JA Accent tracking
        accents = [PHONE2ID.get(a) for a in ["^", "=", "_"] if a in PHONE2ID]
        accent_count = 0

        for utt in self.utterances:
            utt_dir = utt["path"]
            has_phonemes = (utt_dir / "phoneme_ids.npy").exists()
            has_durations = (utt_dir / "durations.npy").exists()
            if has_phonemes:
                with_text += 1
                if has_durations:
                    with_legacy_duration += 1
                try:
                    pids = np.load(utt_dir / "phoneme_ids.npy")
                    total_phones += len(pids)
                    unk_phones += int((pids == UNK_ID).sum())
                    if accents:
                        accent_count += sum(1 for p in pids if p in accents)
                except Exception:
                    pass

        return {
            "total": total,
            "text_supervision_coverage": with_text / max(total, 1),
            "legacy_duration_coverage": with_legacy_duration / max(total, 1),
            "unknown_phone_ratio": round(unk_phones / max(total_phones, 1), 6),
            "pitch_accent_coverage": round(accent_count / max(total_phones, 1), 6),
            "tts_mode": self.tts_mode,
        }

    def expressive_readiness_report(self) -> dict:
        """Return expressive-data readiness statistics."""
        total = len(self.utterances)
        fields = [
            "dialogue_context", "acting_intent", "prosody_targets", 
            "style_embedding", "pause_events"
        ]
        counts = {f: 0 for f in fields}
        multi_take_texts: dict[str, int] = {}

        for utt in self.utterances:
            utt_dir = utt["path"]
            for f in fields:
                ext = ".json" if f == "pause_events" else ".npy"
                if (utt_dir / f"{f}{ext}").exists():
                    counts[f] += 1
            text = utt["meta"].get("text", "")
            if text:
                multi_take_texts[text] = multi_take_texts.get(text, 0) + 1

        multi_take_count = sum(1 for c in multi_take_texts.values() if c > 1)

        return {
            "total": total,
            **{f"with_{f}": v for f, v in counts.items()},
            "multi_take_texts": multi_take_count,
            "expressive_readiness_score": sum(counts.values()) / (len(fields) * total) if total > 0 else 0
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        utt = self.utterances[idx]
        utt_dir = utt["path"]
        meta = utt["meta"]

        codec_tokens = np.load(utt_dir / "codec_tokens.npy")
        explicit_state = np.load(utt_dir / "explicit_state.npy")
        ssl_state = np.load(utt_dir / "ssl_state.npy")
        spk_embed = np.load(utt_dir / "spk_embed.npy")

        T = codec_tokens.shape[-1]
        if T > self.max_frames:
            start = np.random.randint(0, T - self.max_frames)
            codec_tokens = codec_tokens[:, start : start + self.max_frames]
            explicit_state = explicit_state[start : start + self.max_frames, :]
            ssl_state = ssl_state[start : start + self.max_frames, :]

        target_a = torch.from_numpy(codec_tokens).long()

        if (utt_dir / "control_tokens.npy").exists():
            target_b = torch.from_numpy(np.load(utt_dir / "control_tokens.npy")).long()
            if target_b.shape[-1] > target_a.shape[-1]:
                target_b = target_b[:, :target_a.shape[-1]]
        else:
            target_b = torch.zeros((4, target_a.shape[1]), dtype=torch.long)

        f0_condition = None
        if (utt_dir / "f0.npy").exists():
            f0 = np.load(utt_dir / "f0.npy").squeeze()
            f0_mean = meta.get("f0_mean", 150.0)
            f0_norm = np.log2((f0 + 1e-8) / f0_mean)
            f0_norm = np.clip(f0_norm, -2.0, 2.0)
            f0_cond_np = np.stack([f0_norm, np.zeros_like(f0_norm)], axis=-1)
            f0_condition = torch.from_numpy(f0_cond_np).float()
            if f0_condition.shape[0] > target_a.shape[1]:
                f0_condition = f0_condition[:target_a.shape[1], :]

        dialogue_context = None
        acting_intent = None
        prosody_targets = None
        if (utt_dir / "dialogue_context.npy").exists():
            dialogue_context = torch.from_numpy(np.load(utt_dir / "dialogue_context.npy")).float()
        if (utt_dir / "acting_intent.npy").exists():
            acting_intent = torch.from_numpy(np.load(utt_dir / "acting_intent.npy")).float()
        if (utt_dir / "prosody_targets.npy").exists():
            pt = np.load(utt_dir / "prosody_targets.npy")
            if T > self.max_frames and pt.shape[0] > self.max_frames:
                pt = pt[start : start + self.max_frames, :]
            prosody_targets = torch.from_numpy(pt).float()

        phoneme_ids = None
        durations = None
        has_phonemes = (utt_dir / "phoneme_ids.npy").exists()
        has_durations = (utt_dir / "durations.npy").exists()

        if self.tts_mode == "pointer":
            if has_phonemes:
                phoneme_ids = torch.from_numpy(np.load(utt_dir / "phoneme_ids.npy")).long()
        elif self.tts_mode == "legacy_duration":
            if has_phonemes and has_durations:
                phoneme_ids = torch.from_numpy(np.load(utt_dir / "phoneme_ids.npy")).long()
                durations = torch.from_numpy(np.load(utt_dir / "durations.npy")).long()
        else:
            if has_phonemes:
                phoneme_ids = torch.from_numpy(np.load(utt_dir / "phoneme_ids.npy")).long()
                if has_durations:
                    durations = torch.from_numpy(np.load(utt_dir / "durations.npy")).long()

        phoneme_lens = None
        if phoneme_ids is not None:
            phoneme_lens = torch.tensor(len(phoneme_ids)).long()

        spk_str = utt.get("speaker_id") or meta.get("speaker_id") or utt_dir.parent.name
        spk_int = self.speaker_to_id.get(spk_str, 0)

        return {
            "source_a_t": target_a.clone(),
            "target_a": target_a,
            "target_b": target_b,
            "explicit_state": torch.from_numpy(explicit_state).float(),
            "ssl_state": torch.from_numpy(ssl_state).float(),
            "speaker_embed": torch.from_numpy(spk_embed).float(),
            "speaker_id": torch.tensor(spk_int).long(),
            "phoneme_ids": phoneme_ids,
            "phoneme_lens": phoneme_lens,
            "durations": durations,
            "f0_condition": f0_condition,
            "language_id": torch.tensor(meta.get("language_id", 0)).long(),
            "text": meta.get("text", ""),
            "dialogue_context": dialogue_context,
            "acting_intent": acting_intent,
            "prosody_targets": prosody_targets,
        }
