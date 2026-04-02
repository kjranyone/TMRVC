"""v4 dataset loader for BootstrapCacheEntry train-ready cache.

Loads utterances from the v4 cache layout including:
- 12-D physical control targets with observed_mask and confidence
- Acting semantic annotations
- Supervision tier classification
- Enriched transcripts (when available)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tmrvc_core.acting_tags import (
    ACTING_TAG_VOCAB,
    ALL_ACTING_TAGS,
    parse_enriched_transcript,
)

logger = logging.getLogger(__name__)


def tokenize_enriched_transcript(
    enriched_text: str,
    phoneme_ids: torch.Tensor | None = None,
    phoneme_vocab_size: int = 200,
) -> torch.Tensor:
    """Convert enriched transcript with inline acting tags to token IDs.

    Parses the enriched_text into segments. Text segments reuse the
    original phoneme_ids (interleaved at the original positions), while
    acting tag segments get their ACTING_TAG_VOCAB IDs.

    Args:
        enriched_text: Enriched transcript with inline tags.
        phoneme_ids: Original phoneme_ids tensor [L]. Used for text segments.
        phoneme_vocab_size: Base phoneme vocabulary size.

    Returns:
        Token ID tensor with phoneme IDs and acting tag IDs interleaved.
    """
    if not enriched_text:
        return phoneme_ids if phoneme_ids is not None else torch.tensor([], dtype=torch.long)

    parts = parse_enriched_transcript(enriched_text)
    token_ids = []
    phoneme_cursor = 0  # Tracks position in original phoneme_ids

    # Pre-compute total character count across all text segments for proportional allocation
    total_text_chars = 0
    for content, part_type in parts:
        if part_type not in ("tag", "freeform"):
            total_text_chars += max(len(content.replace(" ", "")), 1)
    total_text_chars = max(total_text_chars, 1)

    for content, part_type in parts:
        if part_type == "tag" and content in ACTING_TAG_VOCAB:
            # Known acting tag -> use its vocab ID
            token_ids.append(ACTING_TAG_VOCAB[content])
        elif part_type == "freeform":
            # Free-form tag -> use [act_start] content [act_end]
            if "[act_start]" in ACTING_TAG_VOCAB:
                token_ids.append(ACTING_TAG_VOCAB["[act_start]"])
            if "[freeform]" in ACTING_TAG_VOCAB:
                token_ids.append(ACTING_TAG_VOCAB["[freeform]"])
            if "[act_end]" in ACTING_TAG_VOCAB:
                token_ids.append(ACTING_TAG_VOCAB["[act_end]"])
        else:
            # Text segment -> consume from original phoneme_ids
            if phoneme_ids is not None:
                # Estimate how many phonemes this text segment covers.
                # Proportional allocation based on character count relative to total text.
                n_chars = max(len(content.replace(" ", "")), 1)
                total_phonemes = len(phoneme_ids)
                # Proportional allocation based on character count
                n_phonemes = max(1, min(
                    len(phoneme_ids) - phoneme_cursor,
                    int(n_chars * total_phonemes / total_text_chars),
                ))
                # Take the next n_phonemes from the original phoneme_ids
                end_cursor = min(phoneme_cursor + n_phonemes, len(phoneme_ids))
                for pid in phoneme_ids[phoneme_cursor:end_cursor].tolist():
                    token_ids.append(int(pid))
                phoneme_cursor = end_cursor

    # Append any remaining phonemes that weren't consumed
    if phoneme_ids is not None and phoneme_cursor < len(phoneme_ids):
        for pid in phoneme_ids[phoneme_cursor:].tolist():
            token_ids.append(int(pid))

    return torch.tensor(token_ids, dtype=torch.long)


class V4UCLMDataset(Dataset):
    """v4 UCLM training dataset.

    Loads from v4 bootstrap cache with supervision tier support.

    Cache layout:
        v4_cache/{corpus_id}/{pseudo_speaker_id}/{utterance_id}/
            acoustic_tokens.npy    [8, T]
            control_tokens.npy     [4, T]  (optional)
            spk_embed.npy          [192]
            phoneme_ids.npy        [L]
            physical_targets.npy   [T, 12]
            physical_observed_mask.npy [T, 12]
            physical_confidence.npy    [T, 12]
            meta.json
    """

    def __init__(
        self,
        cache_dir: str | Path,
        min_tier: str = "tier_d",        # Include all tiers down to this
        min_quality_score: float = 0.0,
        max_frames: int = 2000,
        min_frames: int = 10,
        use_enriched_transcript: bool = False,
        enriched_transcript_prob: float = 0.5,  # Prob of using enriched vs plain
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.min_tier = min_tier
        self.min_quality_score = min_quality_score
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.use_enriched_transcript = use_enriched_transcript
        self.enriched_transcript_prob = enriched_transcript_prob

        # Tier ordering for filtering
        self._tier_order = {"tier_a": 0, "tier_b": 1, "tier_c": 2, "tier_d": 3}
        self._min_tier_idx = self._tier_order.get(min_tier, 3)

        # Discover utterances
        self.utterances = self._discover_utterances()
        logger.info(
            "V4UCLMDataset: %d utterances from %s (min_tier=%s)",
            len(self.utterances), cache_dir, min_tier,
        )

    def _discover_utterances(self) -> List[dict]:
        """Scan cache directory for valid utterances."""
        utterances = []

        if not self.cache_dir.exists():
            logger.warning("Cache dir does not exist: %s", self.cache_dir)
            return utterances

        for meta_path in sorted(self.cache_dir.rglob("meta.json")):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Tier filter
                tier = meta.get("supervision_tier", "tier_d")
                if self._tier_order.get(tier, 3) > self._min_tier_idx:
                    continue

                # Quality filter
                quality = meta.get("quality_score", 0.0)
                if quality < self.min_quality_score:
                    continue

                # Frame count filter
                n_frames = meta.get("n_frames", 0)
                if n_frames < self.min_frames or n_frames > self.max_frames:
                    continue

                meta["_dir"] = str(meta_path.parent)
                utterances.append(meta)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping %s: %s", meta_path, e)
                continue

        return utterances

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        meta = self.utterances[idx]
        utt_dir = Path(meta["_dir"])

        # Load required arrays
        result = {
            "utterance_id": meta.get("utterance_id", ""),
            "speaker_id": meta.get("speaker_id", meta.get("pseudo_speaker_id", "")),
            "language": meta.get("language", ""),
            "supervision_tier": meta.get("supervision_tier", "tier_d"),
            "quality_score": meta.get("quality_score", 0.0),
        }

        # Acoustic tokens [8, T] — v4 name or v3 fallback
        tokens_path = utt_dir / "acoustic_tokens.npy"
        if not tokens_path.exists():
            tokens_path = utt_dir / "codec_tokens.npy"
        if tokens_path.exists():
            result["codec_tokens_a"] = torch.from_numpy(
                np.load(tokens_path).astype(np.int64)
            )

        # Control tokens [4, T] (optional)
        ctrl_path = utt_dir / "control_tokens.npy"
        if ctrl_path.exists():
            result["codec_tokens_b"] = torch.from_numpy(
                np.load(ctrl_path).astype(np.int64)
            )

        # Speaker embedding [192]
        spk_path = utt_dir / "spk_embed.npy"
        if spk_path.exists():
            result["speaker_embed"] = torch.from_numpy(
                np.load(spk_path).astype(np.float32)
            )

        # Phoneme IDs [L]
        phone_path = utt_dir / "phoneme_ids.npy"
        if phone_path.exists():
            result["phoneme_ids"] = torch.from_numpy(
                np.load(phone_path).astype(np.int64)
            )

        # Physical targets [T, 12] — v4 name or v3 fallback
        phys_path = utt_dir / "physical_targets.npy"
        if not phys_path.exists():
            phys_path = utt_dir / "voice_state_targets.npy"
        if phys_path.exists():
            result["physical_targets"] = torch.from_numpy(
                np.load(phys_path).astype(np.float32)
            )

        # Physical observed mask [T, 12] — v4 name or v3 fallback
        mask_path = utt_dir / "physical_observed_mask.npy"
        if not mask_path.exists():
            mask_path = utt_dir / "voice_state_observed_mask.npy"
        if mask_path.exists():
            result["physical_observed_mask"] = torch.from_numpy(
                np.load(mask_path)
            ).bool()

        # Physical confidence [T, 12] — v4 name or v3 fallback
        conf_path = utt_dir / "physical_confidence.npy"
        if not conf_path.exists():
            conf_path = utt_dir / "voice_state_confidence.npy"
        if conf_path.exists():
            result["physical_confidence"] = torch.from_numpy(
                np.load(conf_path).astype(np.float32)
            )

        # SSL state [T, 128] — WavLM features for acting-latent losses
        ssl_path = utt_dir / "ssl_state.npy"
        if ssl_path.exists():
            result["ssl_state"] = torch.from_numpy(
                np.load(ssl_path).astype(np.float32)
            )

        # Ensure optional fields have defaults
        # Bootstrap alignment [T] — phoneme index per frame
        align_path = utt_dir / "bootstrap_alignment.npy"
        if align_path.exists():
            result["bootstrap_alignment"] = torch.from_numpy(
                np.load(align_path).astype(np.int64)
            )

        result.setdefault("physical_observed_mask", None)
        result.setdefault("physical_confidence", None)
        result.setdefault("codec_tokens_b", None)
        result.setdefault("ssl_state", None)
        result.setdefault("bootstrap_alignment", None)

        # v4: Acting texture latent target [24] or [T, 24] (optional)
        act_latent_path = utt_dir / "acting_texture_latent_target.npy"
        if act_latent_path.exists():
            result["acting_texture_latent_target"] = torch.from_numpy(
                np.load(act_latent_path).astype(np.float32)
            )
        else:
            result["acting_texture_latent_target"] = None

        # v4: Pacing controls [3] or [5] (optional)
        pacing_path = utt_dir / "pacing_controls.npy"
        if pacing_path.exists():
            result["pacing_controls"] = torch.from_numpy(
                np.load(pacing_path).astype(np.float32)
            )
        else:
            result["pacing_controls"] = None

        # Acting annotations
        result["acting_annotations"] = meta.get("acting_annotations", {})

        # Text / enriched transcript (check both v4 and v3 meta key names)
        result["text"] = meta.get("text_transcript", meta.get("text", ""))
        result["enriched_transcript"] = meta.get("enriched_transcript", "")

        # Decide whether to use enriched or plain transcript for this sample
        # Task 3-4: 50% chance use enriched, 50% plain phoneme_ids
        if (self.use_enriched_transcript
                and result["enriched_transcript"]
                and random.random() < self.enriched_transcript_prob):
            result["use_enriched"] = True
            # Tokenize enriched transcript with acting tags interleaved
            enriched_token_ids = tokenize_enriched_transcript(
                result["enriched_transcript"],
                phoneme_ids=result.get("phoneme_ids"),
            )
            result["enriched_phoneme_ids"] = enriched_token_ids
        else:
            result["use_enriched"] = False
            result["enriched_phoneme_ids"] = None

        return result


def get_tier_loss_weights(tier: str) -> dict:
    """Get loss weight multipliers for a supervision tier.

    Tier A: full weight on all losses
    Tier B: reduced physical/acting (partially pseudo)
    Tier C: sparse physical, minimal acting
    Tier D: auxiliary only
    """
    weights = {
        "tier_a": {
            "codec_loss": 1.0,
            "control_loss": 1.0,
            "pointer_loss": 1.0,
            "physical_loss": 1.0,
            "acting_latent_loss": 1.0,
            "disentanglement_loss": 1.0,
            "speaker_loss": 1.0,
            "prosody_loss": 1.0,
            "semantic_loss": 1.0,
        },
        "tier_b": {
            "codec_loss": 1.0,
            "control_loss": 1.0,
            "pointer_loss": 1.0,
            "physical_loss": 0.5,
            "acting_latent_loss": 0.5,
            "disentanglement_loss": 0.5,
            "speaker_loss": 1.0,
            "prosody_loss": 0.8,
            "semantic_loss": 0.5,
        },
        "tier_c": {
            "codec_loss": 1.0,
            "control_loss": 1.0,
            "pointer_loss": 1.0,
            "physical_loss": 0.2,
            "acting_latent_loss": 0.2,
            "disentanglement_loss": 0.2,
            "speaker_loss": 0.8,
            "prosody_loss": 0.5,
            "semantic_loss": 0.1,
        },
        "tier_d": {
            "codec_loss": 0.5,
            "control_loss": 0.5,
            "pointer_loss": 0.5,
            "physical_loss": 0.0,
            "acting_latent_loss": 0.0,
            "disentanglement_loss": 0.0,
            "speaker_loss": 0.5,
            "prosody_loss": 0.2,
            "semantic_loss": 0.0,
        },
    }
    return weights.get(tier, weights["tier_d"])


def v4_collate_fn(batch: list[dict]) -> dict:
    """Collate function for V4UCLMDataset batches.

    Handles variable-length tensors by padding and non-tensor types
    by collecting into lists.
    """
    keys = batch[0].keys()
    collated: dict = {}

    # Keys that contain variable-length tensors requiring padding
    _PAD_KEYS = {
        "codec_tokens_a", "codec_tokens_b",
        "phoneme_ids", "enriched_phoneme_ids",
        "physical_targets", "physical_observed_mask", "physical_confidence",
    }

    for key in keys:
        values = [sample[key] for sample in batch]

        # Skip None-only fields
        if all(v is None for v in values):
            collated[key] = None
            continue

        # Find first non-None value to determine type
        non_none = [v for v in values if v is not None]
        if not non_none or not isinstance(non_none[0], torch.Tensor):
            # Non-tensor fields -> list
            collated[key] = values
            continue

        # Replace None values with zero tensors matching non-None shape
        if any(v is None for v in values):
            template = non_none[0]
            values = [v if v is not None else torch.zeros_like(template) for v in values]

        # Fixed-size tensors (e.g. speaker_embed [192]) -> stack directly
        if key not in _PAD_KEYS:
            try:
                collated[key] = torch.stack(values)
                continue
            except RuntimeError:
                pass  # Fall through to padding

        # Variable-length tensors -> pad to max length in batch
        non_none = [v for v in values if v is not None]
        if not non_none:
            collated[key] = None
            continue

        ndim = non_none[0].ndim
        if ndim == 1:
            # [L] tensors (phoneme_ids, enriched_phoneme_ids)
            max_len = max(v.shape[0] for v in non_none)
            padded = torch.zeros(len(values), max_len, dtype=non_none[0].dtype)
            lengths = []
            for i, v in enumerate(values):
                if v is not None:
                    padded[i, :v.shape[0]] = v
                    lengths.append(v.shape[0])
                else:
                    lengths.append(0)
            collated[key] = padded
            collated[f"{key}_lengths"] = torch.tensor(lengths, dtype=torch.long)
        elif ndim == 2:
            # [C, T] or [T, D] tensors
            # Determine which axis varies: if shape[0] varies across samples, pad dim 0
            shapes_0 = [v.shape[0] for v in non_none]
            shapes_1 = [v.shape[1] for v in non_none]
            if non_none[0].dtype == torch.bool:
                pad_val = False
            elif non_none[0].dtype in (torch.long, torch.int32, torch.int64) and key in ("codec_tokens_a", "codec_tokens_b", "target_a", "target_b"):
                pad_val = -1  # matches ignore_index=-1 in cross_entropy
            else:
                pad_val = 0

            if len(set(shapes_1)) == 1 and len(set(shapes_0)) > 1:
                # [T, D] format — T varies, D fixed. Pad along T (dim 0)
                max_t = max(shapes_0)
                d = shapes_1[0]
                padded = torch.full((len(values), max_t, d), pad_val, dtype=non_none[0].dtype)
                for i, v in enumerate(values):
                    if v is not None:
                        padded[i, :v.shape[0], :] = v
            elif len(set(shapes_0)) == 1 and len(set(shapes_1)) > 1:
                # [C, T] format — C fixed, T varies. Pad along T (dim 1)
                c = shapes_0[0]
                max_t = max(shapes_1)
                padded = torch.full((len(values), c, max_t), pad_val, dtype=non_none[0].dtype)
                for i, v in enumerate(values):
                    if v is not None:
                        padded[i, :, :v.shape[1]] = v
            else:
                # Both dims vary or both fixed — pad both to max
                max_0 = max(shapes_0)
                max_1 = max(shapes_1)
                padded = torch.full((len(values), max_0, max_1), pad_val, dtype=non_none[0].dtype)
                for i, v in enumerate(values):
                    if v is not None:
                        padded[i, :v.shape[0], :v.shape[1]] = v
            collated[key] = padded
        else:
            # Fallback: collect as list
            collated[key] = values

    return collated


def create_v4_dataloader(
    cache_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    min_tier: str = "tier_d",
    min_quality_score: float = 0.0,
    use_enriched_transcript: bool = False,
    enriched_transcript_prob: float = 0.5,
    **kwargs,
) -> DataLoader:
    """Create a v4 UCLM training DataLoader."""
    dataset = V4UCLMDataset(
        cache_dir=cache_dir,
        min_tier=min_tier,
        min_quality_score=min_quality_score,
        use_enriched_transcript=use_enriched_transcript,
        enriched_transcript_prob=enriched_transcript_prob,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=v4_collate_fn,
        **kwargs,
    )
