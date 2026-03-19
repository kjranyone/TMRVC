from __future__ import annotations

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import json
from typing import Optional, Any

# Valid tts_mode values
_VALID_TTS_MODES = {"pointer"}


class DisentangledUCLMDataset(Dataset):
    """Dataset for Disentangled UCLM multi-task training (TTS & VC).

    Args:
        cache_dir: Root directory of the feature cache.
        max_frames: Maximum number of frames per utterance (for batching).
        include_datasets: Optional whitelist of dataset names to include.
        tts_mode: Text supervision loading mode. Only ``"pointer"`` is supported.
            Loads ``phoneme_ids.npy`` but never requires ``durations.npy``.
        min_quality_score: Minimum quality_score to include (0.0 = no filter).
        provenance_filter: If set, only include utterances with this provenance_class.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_frames: int = 400,
        include_datasets: list[str] | None = None,
        tts_mode: str = "pointer",
        min_quality_score: float = 0.0,
        provenance_filter: str | None = None,
        require_tts_supervision: bool = False,
    ):
        if tts_mode not in _VALID_TTS_MODES:
            raise ValueError(
                f"Invalid tts_mode={tts_mode!r}. Must be one of {sorted(_VALID_TTS_MODES)}."
            )
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        self.tts_mode = tts_mode
        self.min_quality_score = min_quality_score
        self.provenance_filter = provenance_filter
        self.require_tts_supervision = require_tts_supervision
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
                # durations.npy is NOT in the required list --
                # v3 pointer mode does not use external duration labels.

                if all((utt_dir / req).exists() for req in required_files):
                    # SOTA: Strictly enforce TTS supervision if requested (GEMINI.md Mandate)
                    if self.require_tts_supervision:
                        if not (utt_dir / "phoneme_ids.npy").exists():
                            continue

                    # Quality filtering (Worker 03 contract)
                    if self.min_quality_score > 0.0:
                        qs = meta.get("quality_score", 1.0)
                        if qs < self.min_quality_score:
                            continue
                    if self.provenance_filter is not None:
                        if meta.get("provenance_class", "") != self.provenance_filter:
                            continue

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

        Returns a dict with coverage and UNK ratio.  Distinguishes:
        - text_supervision_coverage (has text in meta)
        - canonical_text_unit_coverage (has phoneme_ids.npy)
        """
        from tmrvc_data.g2p import UNK_ID, PHONE2ID

        total = len(self.utterances)
        with_text = 0
        with_canonical_text_units = 0
        with_voice_state = 0
        with_suprasegmentals = 0
        with_prompt_eligible = 0
        total_phones = 0
        unk_phones = 0

        # JA Accent tracking
        accents = [PHONE2ID.get(a) for a in ["^", "=", "_"] if a in PHONE2ID]
        accent_count = 0

        for utt in self.utterances:
            utt_dir = utt["path"]
            meta = utt.get("meta", {})
            has_phonemes = (utt_dir / "phoneme_ids.npy").exists()
            has_vs_targets = (utt_dir / "voice_state_targets.npy").exists()
            has_supra = (utt_dir / "text_suprasegmentals.npy").exists()

            if meta.get("text"):
                with_text += 1
            if has_phonemes:
                with_canonical_text_units += 1
                try:
                    pids = np.load(utt_dir / "phoneme_ids.npy")
                    total_phones += len(pids)
                    unk_phones += int((pids == UNK_ID).sum())
                    if accents:
                        accent_count += sum(1 for p in pids if p in accents)
                except Exception:
                    pass
            if has_vs_targets:
                with_voice_state += 1
            if has_supra:
                with_suprasegmentals += 1
            if "prompt_eligible" in meta:
                with_prompt_eligible += 1

        return {
            "total": total,
            "text_supervision_coverage": with_text / max(total, 1),
            "canonical_text_unit_coverage": with_canonical_text_units / max(total, 1),
            "unknown_phone_ratio": round(unk_phones / max(total_phones, 1), 6),
            "pitch_accent_coverage": round(accent_count / max(total_phones, 1), 6),
            "voice_state_supervision_coverage": with_voice_state / max(total, 1),
            "suprasegmental_coverage": with_suprasegmentals / max(total, 1),
            "prompt_pairing_coverage": with_prompt_eligible / max(total, 1),
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
        if (utt_dir / "phoneme_ids.npy").exists():
            phoneme_ids = torch.from_numpy(np.load(utt_dir / "phoneme_ids.npy")).long()

        phoneme_lens = None
        if phoneme_ids is not None:
            phoneme_lens = torch.tensor(len(phoneme_ids)).long()

        # Suprasegmentals (Worker 03 contract)
        text_suprasegmentals = None
        if (utt_dir / "text_suprasegmentals.npy").exists():
            text_suprasegmentals = torch.from_numpy(np.load(utt_dir / "text_suprasegmentals.npy")).float()

        # Voice state supervision (Worker 03 contract)
        voice_state_targets = None
        voice_state_observed_mask = None
        voice_state_confidence = None
        if (utt_dir / "voice_state_targets.npy").exists():
            voice_state_targets = torch.from_numpy(np.load(utt_dir / "voice_state_targets.npy")).float()
            if (utt_dir / "voice_state_observed_mask.npy").exists():
                voice_state_observed_mask = torch.from_numpy(np.load(utt_dir / "voice_state_observed_mask.npy")).bool()
            if (utt_dir / "voice_state_confidence.npy").exists():
                voice_state_confidence = torch.from_numpy(np.load(utt_dir / "voice_state_confidence.npy")).float()

        # Bootstrap alignment (Worker 03 contract)
        bootstrap_alignment = None
        if (utt_dir / "bootstrap_alignment.json").exists():
            try:
                with open(utt_dir / "bootstrap_alignment.json", encoding="utf-8") as f:
                    bs_data = json.load(f)
                if "phoneme_indices" in bs_data:
                    bootstrap_alignment = {
                        "phoneme_indices": torch.tensor(bs_data["phoneme_indices"], dtype=torch.long),
                        "frame_count": bs_data.get("frame_count", 0),
                        "phoneme_count": bs_data.get("phoneme_count", 0),
                    }
            except Exception:
                pass

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
            "f0_condition": f0_condition,
            "language_id": torch.tensor(meta.get("language_id", 0)).long(),
            "text": meta.get("text", ""),
            "dialogue_context": dialogue_context,
            "acting_intent": acting_intent,
            "prosody_targets": prosody_targets,
            "text_suprasegmentals": text_suprasegmentals,
            "voice_state_targets": voice_state_targets,
            "voice_state_observed_mask": voice_state_observed_mask,
            "voice_state_confidence": voice_state_confidence,
            "bootstrap_alignment": bootstrap_alignment,
            # Curation fields (Worker 03)
            "curation_record_id": meta.get("curation_record_id"),
            "promotion_bucket": meta.get("promotion_bucket"),
            "curation_pass": meta.get("curation_pass"),
            "quality_score": meta.get("quality_score"),
            # Few-shot prompt fields (Worker 03)
            "prompt_eligible": meta.get("prompt_eligible"),
            "prompt_pair_id": meta.get("prompt_pair_id"),
        }
