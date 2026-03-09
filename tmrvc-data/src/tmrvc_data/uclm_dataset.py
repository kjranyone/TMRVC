"""UCLM Dataset for training the Unified Codec Language Model.

Loads pre-extracted features from cache:
- codec_tokens: [n_codebooks, T] from EnCodec
- voice_state: [T, 8] from VoiceStateEstimator
- phoneme_ids: [L] from G2P (includes Pitch Accent for Japanese)
- durations: [L] from MFA (optional in v3 -- never required for pointer mode)
- spk_embed: [192] from SpeakerEncoder

Worker 03 contract:
- ``durations.npy`` is NEVER required in v3 pointer mode.
- ``voice_state.npy`` is the core frame-level inference artifact.
  ``voice_state_targets.npy``, ``voice_state_observed_mask.npy``, and
  ``voice_state_confidence.npy`` are optional supervision artifacts.
- ``text_suprasegmentals.npy`` is the canonical companion to ``phoneme_ids.npy``
  for accent/tone/phrase-boundary cues.  Shape ``[L, 4]`` (d_suprasegmental=4).
- ``bootstrap_alignment.json`` maps frames to phoneme indices using the
  canonical frame convention (sample_rate=24000, hop_length=240).
- meta.json may contain curation fields: ``curation_record_id``,
  ``promotion_bucket``, ``curation_pass``, ``quality_score``.
- meta.json may contain few-shot prompt fields: ``prompt_eligible``,
  ``prompt_pair_id``.
- Quality filtering via ``min_quality_score`` skips low-quality samples.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Canonical voice_state dimension names (Worker 03 contract).
# Order matches the 8 columns of voice_state.npy / voice_state_targets.npy.
VOICE_STATE_DIMS = (
    "pitch_level",
    "pitch_range",
    "energy_level",
    "speech_rate",
    "spectral_tilt",
    "breathiness",
    "voice_irregularity",
    "pause_tendency",
)

# Canonical text_suprasegmentals dimension names (Worker 03 contract).
# Order matches the 4 columns of text_suprasegmentals.npy.
SUPRASEGMENTAL_DIMS = (
    "accent_upstep",
    "accent_downstep",
    "phrase_break",
    "lexical_tone_id",
)


@dataclass
class UCLMBatch:
    """Batch of UCLM training data."""

    codec_tokens: torch.Tensor  # [B, n_codebooks, T]
    voice_state: torch.Tensor  # [B, T, d_voice_state]
    phoneme_ids: torch.Tensor  # [B, L]
    durations: Optional[torch.Tensor]  # [B, L] -- None in v3 pointer mode
    spk_embed: torch.Tensor  # [B, d_speaker]
    text: list[str]  # [B]
    utterance_ids: list[str]  # [B]
    frame_lengths: torch.Tensor  # [B]
    phoneme_lengths: torch.Tensor  # [B]
    # v3 expressive fields
    dialogue_context: Optional[torch.Tensor] = None
    acting_intent: Optional[torch.Tensor] = None
    prosody_targets: Optional[torch.Tensor] = None
    text_suprasegmentals: Optional[torch.Tensor] = None  # [B, L, 4]
    bootstrap_alignment: Optional[dict] = None  # Contains 'phoneme_indices'
    # v3 voice state supervision (Worker 01/03)
    voice_state_targets: Optional[torch.Tensor] = None       # [B, T, 8]
    voice_state_observed_mask: Optional[torch.Tensor] = None  # [B, T, 8] bool
    voice_state_confidence: Optional[torch.Tensor] = None     # [B, T, 8]
    # Few-shot prompt tokens (Worker 02 Task 14)
    prompt_codec_tokens: Optional[torch.Tensor] = None  # [B, T_prompt, n_codebooks]


class UCLMDataset(Dataset):
    """Dataset for UCLM training.

    Expects cache directory structure:
        data/cache/{dataset}/train/{speaker}/{utterance}/
        +-- codec_tokens.npy     # [n_codebooks, T]
        +-- voice_state.npy      # [T, 8]
        +-- phoneme_ids.npy      # [L]
        +-- durations.npy        # [L] (optional -- never required in v3)
        +-- spk_embed.npy        # [192]
        +-- text_suprasegmentals.npy  # [L, 4] (optional)
        +-- voice_state_targets.npy   # [T, 8] (optional supervision)
        +-- voice_state_observed_mask.npy  # [T, 8] bool (optional)
        +-- voice_state_confidence.npy     # [T, 8] float32 (optional)
        +-- bootstrap_alignment.json  # (optional)
        +-- meta.json

    Args:
        cache_dir: Root cache directory.
        datasets: List of dataset names to include.
        max_frames: Maximum frames per utterance (truncate or skip).
        min_frames: Minimum frames per utterance (skip if shorter).
        include_datasets: If set, only include these datasets.
        exclude_speakers: Speaker IDs to exclude.
        tts_mode: "auto", "pointer", or "legacy_duration".
            - "pointer": load phoneme_ids but NEVER require durations.npy.
            - "legacy_duration": require BOTH phoneme_ids.npy AND durations.npy.
            - "auto": load whatever is available (v2 compat).
        min_quality_score: Minimum quality_score to include (0.0 = no filter).
        provenance_filter: If set, only include utterances with this provenance_class.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        datasets: list[str] | None = None,
        max_frames: int = 400,
        min_frames: int = 20,
        include_datasets: list[str] | None = None,
        exclude_speakers: list[str] | None = None,
        tts_mode: str = "auto",
        min_quality_score: float = 0.0,
        provenance_filter: str | None = None,
        # Legacy alias kept for backward compatibility
        quality_score_threshold: float | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.datasets = datasets or []
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.include_datasets = include_datasets
        self.exclude_speakers = set(exclude_speakers or [])
        self.tts_mode = tts_mode
        # Support both param names for backward compat
        if quality_score_threshold is not None and min_quality_score == 0.0:
            self.min_quality_score = quality_score_threshold
        else:
            self.min_quality_score = min_quality_score
        self.provenance_filter = provenance_filter

        self.utterances: list[dict[str, Any]] = []
        self.speaker_to_indices: dict[str, list[int]] = {}
        self._scan_utterances()

        logger.info(
            "UCLMDataset: %d utterances from %d datasets (tts_mode=%s, min_quality_score=%.2f)",
            len(self.utterances),
            len(self.datasets) if self.datasets else 0,
            self.tts_mode,
            self.min_quality_score,
        )

    def _scan_utterances(self) -> None:
        """Scan cache directory for valid utterances."""
        if not self.cache_dir.exists():
            logger.warning("Cache directory does not exist: %s", self.cache_dir)
            return

        search_dirs = []
        if self.include_datasets:
            search_dirs = [self.cache_dir / d for d in self.include_datasets]
        elif self.datasets:
            search_dirs = [self.cache_dir / d for d in self.datasets]
        else:
            search_dirs = list(self.cache_dir.iterdir())

        for dataset_dir in search_dirs:
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
                continue

            train_dir = dataset_dir / "train"
            if not train_dir.exists():
                continue

            for speaker_dir in train_dir.iterdir():
                speaker_id = speaker_dir.name
                if not speaker_dir.is_dir() or speaker_id in self.exclude_speakers:
                    continue

                for utt_dir in speaker_dir.iterdir():
                    if not utt_dir.is_dir():
                        continue

                    # Check for required core files.
                    codec_path = utt_dir / "codec_tokens.npy"
                    voice_path = utt_dir / "voice_state.npy"
                    meta_path = utt_dir / "meta.json"

                    if not all(p.exists() for p in [codec_path, voice_path, meta_path]):
                        continue

                    # Load metadata
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        continue

                    n_frames = meta.get("n_frames", 0)
                    if n_frames < self.min_frames:
                        continue

                    # Quality filtering (Worker 03 contract)
                    if self.min_quality_score > 0.0:
                        qs = meta.get("quality_score", 1.0)
                        if qs < self.min_quality_score:
                            continue
                    if self.provenance_filter is not None:
                        if meta.get("provenance_class", "") != self.provenance_filter:
                            continue

                    idx = len(self.utterances)
                    self.utterances.append(
                        {
                            "utterance_id": meta.get("utterance_id", utt_dir.name),
                            "speaker_id": speaker_id,
                            "dataset": dataset_dir.name,
                            "path": utt_dir,
                            "n_frames": n_frames,
                            "text": meta.get("text", ""),
                            "meta": meta,
                        }
                    )
                    if speaker_id not in self.speaker_to_indices:
                        self.speaker_to_indices[speaker_id] = []
                    self.speaker_to_indices[speaker_id].append(idx)

    def __len__(self) -> int:
        return len(self.utterances)

    def get_random_prompt(self, speaker_id: str, exclude_idx: int | None = None) -> dict[str, Any] | None:
        """Sample a random 3-10s prompt from the same speaker (Worker 02 Task 14)."""
        indices = self.speaker_to_indices.get(speaker_id, [])
        if exclude_idx is not None:
            indices = [i for i in indices if i != exclude_idx]
        
        if not indices:
            return None
            
        # Try to find a prompt with 3-10s duration (roughly 300-1000 frames)
        # Fallback to any clip if none in range.
        candidates = [i for i in indices if 300 <= self.utterances[i]["n_frames"] <= 1000]
        if not candidates:
            candidates = indices
            
        prompt_idx = np.random.choice(candidates)
        return self.__getitem__(int(prompt_idx), return_prompt=False)

    def supervision_report(self) -> dict:
        """Return dataset-level supervision statistics (Worker 03 requirement).

        Returns a dict compatible with DatasetReport fields.  Distinguishes:
        - text_supervision_coverage (has text in meta)
        - canonical_text_unit_coverage (has phoneme_ids.npy)
        - legacy_duration_coverage (has durations.npy)
        - voice_state_supervision_coverage (has voice_state_targets.npy)
        - prompt_pairing_coverage (has prompt_eligible in meta)
        """
        from tmrvc_data.g2p import UNK_ID, PHONE2ID, ID2PHONE

        total = len(self.utterances)
        with_text = 0
        with_canonical_text_units = 0
        with_legacy_duration = 0
        with_voice_state = 0
        with_dialogue_context = 0
        with_suprasegmentals = 0
        with_bootstrap_alignment = 0
        with_prompt_eligible = 0
        with_curation_record = 0
        total_phones = 0
        unk_phones = 0
        active_ids: set[int] = set()
        vs_observed_sum = 0.0
        vs_observed_count = 0
        vs_confidence_values: list[float] = []

        # JA Accent tracking
        accents = [PHONE2ID.get(a) for a in ["^", "=", "_"] if a in PHONE2ID]
        accent_count = 0
        multi_take_texts: dict[str, int] = {}

        for utt in self.utterances:
            utt_path = utt["path"]
            meta = utt.get("meta", {})
            has_phonemes = (utt_path / "phoneme_ids.npy").exists()
            has_durations = (utt_path / "durations.npy").exists()
            has_vs_targets = (utt_path / "voice_state_targets.npy").exists()
            has_dialogue = (utt_path / "dialogue_context.npy").exists()
            has_supra = (utt_path / "text_suprasegmentals.npy").exists()
            has_bootstrap = (utt_path / "bootstrap_alignment.json").exists()

            if utt.get("text"):
                with_text += 1
                multi_take_texts[utt["text"]] = multi_take_texts.get(utt["text"], 0) + 1
            if has_phonemes:
                with_canonical_text_units += 1
                try:
                    pids = np.load(utt_path / "phoneme_ids.npy")
                    total_phones += len(pids)
                    unk_phones += int((pids == UNK_ID).sum())
                    active_ids.update(int(p) for p in pids)
                    if accents:
                        accent_count += sum(1 for p in pids if p in accents)
                except Exception:
                    pass
            if has_durations:
                with_legacy_duration += 1
            if has_supra:
                with_suprasegmentals += 1
            if has_bootstrap:
                with_bootstrap_alignment += 1
            if has_vs_targets:
                with_voice_state += 1
                try:
                    mask_path = utt_path / "voice_state_observed_mask.npy"
                    if mask_path.exists():
                        mask = np.load(mask_path)
                        vs_observed_sum += float(mask.mean())
                        vs_observed_count += 1
                    conf_path = utt_path / "voice_state_confidence.npy"
                    if conf_path.exists():
                        conf = np.load(conf_path)
                        vs_confidence_values.append(float(conf.mean()))
                except Exception:
                    pass
            if has_dialogue:
                with_dialogue_context += 1

            # Few-shot prompt and curation fields
            if "prompt_eligible" in meta:
                with_prompt_eligible += 1
            if "curation_record_id" in meta:
                with_curation_record += 1

        active_phone_inventory = sorted(
            ID2PHONE.get(i, f"id:{i}") for i in active_ids if i != UNK_ID
        )
        multi_context_count = sum(1 for c in multi_take_texts.values() if c > 1)

        return {
            "num_utterances": total,
            "text_supervision_coverage": with_text / max(total, 1),
            "canonical_text_unit_coverage": with_canonical_text_units / max(total, 1),
            "legacy_duration_coverage": with_legacy_duration / max(total, 1),
            "unknown_phone_ratio": unk_phones / max(total_phones, 1),
            "direct_hit_ratio": (total_phones - unk_phones) / max(total_phones, 1),
            "alias_hit_ratio": 0.0,  # Requires normalization pass to compute
            "active_phone_inventory": active_phone_inventory,
            "pitch_accent_coverage": accent_count / max(total_phones, 1),
            "suprasegmental_coverage": with_suprasegmentals / max(total, 1),
            "bootstrap_alignment_coverage": with_bootstrap_alignment / max(total, 1),
            "voice_state_supervision_coverage": with_voice_state / max(total, 1),
            "voice_state_observed_ratio": (
                vs_observed_sum / vs_observed_count if vs_observed_count > 0 else 0.0
            ),
            "voice_state_confidence_summary": {
                "mean": sum(vs_confidence_values) / len(vs_confidence_values) if vs_confidence_values else 0.0,
                "count": len(vs_confidence_values),
            },
            "dialogue_context_coverage": with_dialogue_context / max(total, 1),
            "same_text_multi_context_coverage": multi_context_count / max(len(multi_take_texts), 1) if multi_take_texts else 0.0,
            "prompt_pairing_coverage": with_prompt_eligible / max(total, 1),
            "curation_record_coverage": with_curation_record / max(total, 1),
            "tts_mode": self.tts_mode,
        }

    def expressive_readiness_report(self) -> dict:
        """Return expressive-data readiness statistics (Worker 03 requirement)."""
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
            text = utt["text"]
            if text:
                multi_take_texts[text] = multi_take_texts.get(text, 0) + 1

        multi_take_count = sum(1 for c in multi_take_texts.values() if c > 1)

        return {
            "total": total,
            **{f"with_{f}": v for f, v in counts.items()},
            "multi_take_texts": multi_take_count,
            "expressive_readiness_score": sum(counts.values()) / (len(fields) * total) if total > 0 else 0
        }

    def __getitem__(self, idx: int, return_prompt: bool = True) -> dict[str, Any]:
        utt = self.utterances[idx]
        utt_path = utt["path"]
        meta = utt.get("meta", {})

        # Load codec tokens
        codec_tokens = np.load(utt_path / "codec_tokens.npy")
        codec_tokens = torch.from_numpy(codec_tokens).long()

        # Load voice state
        voice_state = np.load(utt_path / "voice_state.npy")
        voice_state = torch.from_numpy(voice_state).float()

        # Load phoneme ids and durations (Worker 03: durations never required in v3)
        phoneme_ids_path = utt_path / "phoneme_ids.npy"
        durations_path = utt_path / "durations.npy"

        phoneme_ids = None
        durations = None

        if self.tts_mode == "pointer":
            # v3 pointer mode: load phoneme_ids, NEVER require durations
            if phoneme_ids_path.exists():
                phoneme_ids = torch.from_numpy(np.load(phoneme_ids_path)).long()
        elif self.tts_mode == "legacy_duration":
            # Legacy v2 mode: require BOTH phoneme_ids AND durations
            if phoneme_ids_path.exists() and durations_path.exists():
                phoneme_ids = torch.from_numpy(np.load(phoneme_ids_path)).long()
                durations = torch.from_numpy(np.load(durations_path)).long()
        else:  # auto
            if phoneme_ids_path.exists():
                phoneme_ids = torch.from_numpy(np.load(phoneme_ids_path)).long()
                if durations_path.exists():
                    durations = torch.from_numpy(np.load(durations_path)).long()

        # Load speaker embedding
        spk_embed_path = utt_path / "spk_embed.npy"
        spk_embed = torch.from_numpy(np.load(spk_embed_path)).float() if spk_embed_path.exists() else torch.zeros(192)

        # Load v3 expressive fields
        dialogue_context = None
        if (utt_path / "dialogue_context.npy").exists():
            dialogue_context = torch.from_numpy(np.load(utt_path / "dialogue_context.npy")).float()

        acting_intent = None
        if (utt_path / "acting_intent.npy").exists():
            acting_intent = torch.from_numpy(np.load(utt_path / "acting_intent.npy")).float()

        prosody_targets = None
        if (utt_path / "prosody_targets.npy").exists():
            prosody_targets = torch.from_numpy(np.load(utt_path / "prosody_targets.npy")).float()

        # Voice state supervision artifacts (Worker 03 contract)
        voice_state_targets = None
        voice_state_observed_mask = None
        voice_state_confidence = None
        vs_targets_path = utt_path / "voice_state_targets.npy"
        if vs_targets_path.exists():
            voice_state_targets = torch.from_numpy(np.load(vs_targets_path)).float()
            vs_mask_path = utt_path / "voice_state_observed_mask.npy"
            if vs_mask_path.exists():
                voice_state_observed_mask = torch.from_numpy(np.load(vs_mask_path)).bool()
            vs_conf_path = utt_path / "voice_state_confidence.npy"
            if vs_conf_path.exists():
                voice_state_confidence = torch.from_numpy(np.load(vs_conf_path)).float()

        # Suprasegmentals (Worker 03 contract: [L, 4] aligned with phoneme_ids)
        text_suprasegmentals = None
        suprasegmentals_path = utt_path / "text_suprasegmentals.npy"
        if suprasegmentals_path.exists():
            text_suprasegmentals = torch.from_numpy(np.load(suprasegmentals_path)).float()

        # Bootstrap Alignment (Worker 03 contract)
        bootstrap_alignment = None
        bootstrap_path = utt_path / "bootstrap_alignment.json"
        if bootstrap_path.exists():
            try:
                with open(bootstrap_path, encoding="utf-8") as f:
                    bs_data = json.load(f)
                if "phoneme_indices" in bs_data:
                    bootstrap_alignment = {
                        "phoneme_indices": torch.tensor(bs_data["phoneme_indices"], dtype=torch.long),
                        "frame_count": bs_data.get("frame_count", 0),
                        "phoneme_count": bs_data.get("phoneme_count", 0),
                    }
            except Exception as e:
                logger.warning("Failed to load bootstrap_alignment.json for %s: %s", utt["utterance_id"], e)

        # Expose curation fields from meta (Worker 03 contract)
        curation_record_id = meta.get("curation_record_id")
        promotion_bucket = meta.get("promotion_bucket")
        curation_pass = meta.get("curation_pass")
        quality_score = meta.get("quality_score")

        # Expose few-shot prompt fields from meta (Worker 03 contract)
        prompt_eligible = meta.get("prompt_eligible")
        prompt_pair_id = meta.get("prompt_pair_id")

        res = {
            "codec_tokens": codec_tokens,
            "voice_state": voice_state,
            "phoneme_ids": phoneme_ids,
            "durations": durations,
            "spk_embed": spk_embed,
            "text": utt["text"],
            "utterance_id": utt["utterance_id"],
            "speaker_id": utt["speaker_id"],
            "n_frames": utt["n_frames"],
            "dialogue_context": dialogue_context,
            "acting_intent": acting_intent,
            "prosody_targets": prosody_targets,
            "voice_state_targets": voice_state_targets,
            "voice_state_observed_mask": voice_state_observed_mask,
            "voice_state_confidence": voice_state_confidence,
            "text_suprasegmentals": text_suprasegmentals,
            "bootstrap_alignment": bootstrap_alignment,
            # Curation fields (Worker 03)
            "curation_record_id": curation_record_id,
            "promotion_bucket": promotion_bucket,
            "curation_pass": curation_pass,
            "quality_score": quality_score,
            # Few-shot prompt fields (Worker 03)
            "prompt_eligible": prompt_eligible,
            "prompt_pair_id": prompt_pair_id,
        }

        if return_prompt:
            prompt = self.get_random_prompt(utt["speaker_id"], exclude_idx=idx)
            if prompt is not None:
                res["prompt_codec_tokens"] = prompt["codec_tokens"]

        return res


def collate_uclm_batch(
    batch: list[dict[str, Any]],
    pad_id: int = 0,
) -> UCLMBatch:
    """Collate a batch of UCLM samples."""
    max_codec_frames = max(sample["codec_tokens"].shape[1] for sample in batch)

    # Filter out samples without phoneme_ids for max_phonemes calc
    samples_with_text = [s for s in batch if s["phoneme_ids"] is not None]
    max_phonemes = max(s["phoneme_ids"].shape[0] for s in samples_with_text) if samples_with_text else 1

    B = len(batch)
    n_codebooks = batch[0]["codec_tokens"].shape[0]
    d_voice_state = batch[0]["voice_state"].shape[1]
    d_speaker = batch[0]["spk_embed"].shape[0]

    codec_tokens = torch.zeros(B, n_codebooks, max_codec_frames, dtype=torch.long)
    voice_state = torch.zeros(B, max_codec_frames, d_voice_state, dtype=torch.float)
    phoneme_ids = torch.full((B, max_phonemes), pad_id, dtype=torch.long)
    durations = torch.zeros(B, max_phonemes, dtype=torch.long) if any(s["durations"] is not None for s in batch) else None
    spk_embed = torch.zeros(B, d_speaker, dtype=torch.float)
    frame_lengths = torch.zeros(B, dtype=torch.long)
    phoneme_lengths = torch.zeros(B, dtype=torch.long)

    dialogue_context = None
    if any(s["dialogue_context"] is not None for s in batch):
        d_ctx = next(s["dialogue_context"].shape[-1] for s in batch if s["dialogue_context"] is not None)
        dialogue_context = torch.zeros(B, d_ctx)

    acting_intent = None
    if any(s["acting_intent"] is not None for s in batch):
        d_act = next(s["acting_intent"].shape[-1] for s in batch if s["acting_intent"] is not None)
        acting_intent = torch.zeros(B, d_act)

    prosody_targets = None
    if any(s["prosody_targets"] is not None for s in batch):
        d_pro = next(s["prosody_targets"].shape[-1] for s in batch if s["prosody_targets"] is not None)
        prosody_targets = torch.zeros(B, max_codec_frames, d_pro)

    # Suprasegmentals
    text_suprasegmentals = None
    if any(s.get("text_suprasegmentals") is not None for s in batch):
        d_supra = next(s["text_suprasegmentals"].shape[-1] for s in batch if s.get("text_suprasegmentals") is not None)
        text_suprasegmentals = torch.zeros(B, max_phonemes, d_supra)

    # Bootstrap alignment
    bootstrap_alignment = None
    if any(s.get("bootstrap_alignment") is not None for s in batch):
        bootstrap_alignment = {"phoneme_indices": torch.zeros(B, max_codec_frames, dtype=torch.long)}

    # Voice state targets
    voice_state_targets = None
    voice_state_observed_mask = None
    voice_state_confidence = None
    if any(s.get("voice_state_targets") is not None for s in batch):
        d_vs = next(s["voice_state_targets"].shape[-1] for s in batch if s.get("voice_state_targets") is not None)
        voice_state_targets = torch.zeros(B, max_codec_frames, d_vs)
        voice_state_observed_mask = torch.zeros(B, max_codec_frames, d_vs, dtype=torch.bool)
        voice_state_confidence = torch.zeros(B, max_codec_frames, d_vs)

    # Prompt codec tokens
    prompt_codec_tokens = None
    if any("prompt_codec_tokens" in s for s in batch):
        max_prompt_frames = max(s["prompt_codec_tokens"].shape[1] for s in batch if "prompt_codec_tokens" in s)
        prompt_codec_tokens = torch.zeros(B, n_codebooks, max_prompt_frames, dtype=torch.long)

    text = []
    utterance_ids = []

    for i, sample in enumerate(batch):
        n_codec_frames = sample["codec_tokens"].shape[1]
        codec_tokens[i, :, :n_codec_frames] = sample["codec_tokens"]

        vs = sample["voice_state"]
        if vs.shape[0] != n_codec_frames:
            vs = torch.nn.functional.interpolate(
                vs.unsqueeze(0).transpose(1, 2),
                size=n_codec_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).squeeze(0)
        voice_state[i, :n_codec_frames, :] = vs

        if sample["phoneme_ids"] is not None:
            n_p = sample["phoneme_ids"].shape[0]
            phoneme_ids[i, :n_p] = sample["phoneme_ids"]
            phoneme_lengths[i] = n_p
            if durations is not None and sample["durations"] is not None:
                durations[i, :n_p] = sample["durations"]
            if text_suprasegmentals is not None and sample.get("text_suprasegmentals") is not None:
                ts = sample["text_suprasegmentals"]
                text_suprasegmentals[i, :n_p, :] = ts[:n_p]

        spk_embed[i] = sample["spk_embed"]
        frame_lengths[i] = n_codec_frames

        if dialogue_context is not None and sample["dialogue_context"] is not None:
            dialogue_context[i] = sample["dialogue_context"]
        if acting_intent is not None and sample["acting_intent"] is not None:
            acting_intent[i] = sample["acting_intent"]
        if prosody_targets is not None and sample["prosody_targets"] is not None:
            pt = sample["prosody_targets"]
            prosody_targets[i, :min(pt.shape[0], max_codec_frames)] = pt[:max_codec_frames]

        if bootstrap_alignment is not None and sample.get("bootstrap_alignment") is not None:
            bs = sample["bootstrap_alignment"]["phoneme_indices"]
            bootstrap_alignment["phoneme_indices"][i, :min(bs.shape[0], max_codec_frames)] = bs[:max_codec_frames]

        if voice_state_targets is not None and sample.get("voice_state_targets") is not None:
            vst = sample["voice_state_targets"]
            vsm = sample["voice_state_observed_mask"]
            vsc = sample["voice_state_confidence"]
            limit = min(vst.shape[0], max_codec_frames)
            voice_state_targets[i, :limit] = vst[:limit]
            if vsm is not None:
                voice_state_observed_mask[i, :limit] = vsm[:limit]
            if vsc is not None:
                voice_state_confidence[i, :limit] = vsc[:limit]

        if prompt_codec_tokens is not None and "prompt_codec_tokens" in sample:
            pct = sample["prompt_codec_tokens"]
            limit_p = pct.shape[1]
            prompt_codec_tokens[i, :, :limit_p] = pct

        text.append(sample["text"])
        utterance_ids.append(sample["utterance_id"])

    return UCLMBatch(
        codec_tokens=codec_tokens,
        voice_state=voice_state,
        phoneme_ids=phoneme_ids,
        durations=durations,
        spk_embed=spk_embed,
        text=text,
        utterance_ids=utterance_ids,
        frame_lengths=frame_lengths,
        phoneme_lengths=phoneme_lengths,
        dialogue_context=dialogue_context,
        acting_intent=acting_intent,
        prosody_targets=prosody_targets,
        text_suprasegmentals=text_suprasegmentals,
        bootstrap_alignment=bootstrap_alignment,
        voice_state_targets=voice_state_targets,
        voice_state_observed_mask=voice_state_observed_mask,
        voice_state_confidence=voice_state_confidence,
        prompt_codec_tokens=prompt_codec_tokens,
    )
