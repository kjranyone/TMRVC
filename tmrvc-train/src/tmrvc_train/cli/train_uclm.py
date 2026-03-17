"""``tmrvc-train-uclm`` — Train Disentangled UCLM (v3 pointer mode)."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from tmrvc_train.models import DisentangledUCLM
from tmrvc_train.dataset import DisentangledUCLMDataset
from tmrvc_train.trainer import CurriculumScheduler, UCLMTrainer

logger = logging.getLogger(__name__)


def _compute_balanced_sample_weights(
    utterances: list[dict],
) -> list[float]:
    """Compute per-sample weights balancing datasets and speakers within dataset."""
    dataset_counts = Counter(u.get("dataset", "unknown") for u in utterances)
    speaker_counts = Counter(
        (u.get("dataset", "unknown"), u.get("speaker_id", "unknown"))
        for u in utterances
    )
    weights: list[float] = []
    for u in utterances:
        ds = u.get("dataset", "unknown")
        spk = u.get("speaker_id", "unknown")
        w_ds = 1.0 / float(dataset_counts[ds])
        w_spk = 1.0 / float(speaker_counts[(ds, spk)])
        weights.append(w_ds * w_spk)
    return weights


def _build_sampler(
    dataset: DisentangledUCLMDataset,
    sampling_strategy: str,
    seed: int,
) -> WeightedRandomSampler | None:
    if sampling_strategy == "shuffle":
        return None
    if sampling_strategy != "balanced":
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
    if len(dataset) == 0:
        return None

    weights = _compute_balanced_sample_weights(dataset.utterances)
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=g,
    )


def _collect_tts_supervision_by_dataset(
    utterances: list[dict],
) -> dict[str, dict[str, int]]:
    """Collect text and duration supervision statistics per dataset.

    Reports three categories:
    - text_supervised: has phoneme_ids.npy (sufficient for v3 pointer mode)
    - legacy_duration_supervised: has both phoneme_ids.npy and durations.npy
    - tts_supervised: alias for legacy_duration_supervised (backward compat)
    """
    stats: dict[str, dict[str, int]] = {}
    for utt in utterances:
        ds = str(utt.get("dataset", "unknown"))
        rec = stats.setdefault(
            ds,
            {"total": 0, "tts_supervised": 0, "text_supervised": 0, "legacy_duration_supervised": 0},
        )
        rec["total"] += 1
        utt_dir = Path(utt["path"])
        has_phonemes = (utt_dir / "phoneme_ids.npy").exists()
        has_durations = (utt_dir / "durations.npy").exists()
        if has_phonemes:
            rec["text_supervised"] += 1
        if has_phonemes and has_durations:
            rec["tts_supervised"] += 1
            rec["legacy_duration_supervised"] += 1
    return stats


def collate_fn(batch):
    """Unified collate for TTS and VC tasks."""
    max_len = max(item["target_a"].shape[1] for item in batch)
    
    # Handle TTS lengths
    max_phonemes = 0
    if any(item.get("phoneme_ids") is not None for item in batch):
        max_phonemes = max(len(item["phoneme_ids"]) for item in batch if item.get("phoneme_ids") is not None)

    collated = {
        "target_a": [], "target_b": [], "source_a_t": [],
        "explicit_state": [], "ssl_state": [], "speaker_embed": [],
        "speaker_id": [], "f0_condition": [],
        "phoneme_ids": [], "phoneme_lens": [], "durations": [], "language_id": []
    }

    for item in batch:
        T = item["target_a"].shape[1]
        pad = max_len - T
        
        collated["target_a"].append(nn.functional.pad(item["target_a"], (0, pad), value=-1))
        collated["target_b"].append(nn.functional.pad(item["target_b"], (0, pad), value=-1))
        # source_a_t is fed to nn.Embedding, so it must remain a valid token id.
        collated["source_a_t"].append(nn.functional.pad(item["source_a_t"], (0, pad), value=0))
        
        collated["explicit_state"].append(nn.functional.pad(item["explicit_state"].transpose(0, 1), (0, pad)).transpose(0, 1))
        collated["ssl_state"].append(nn.functional.pad(item["ssl_state"].transpose(0, 1), (0, pad)).transpose(0, 1))
        
        collated["speaker_embed"].append(item["speaker_embed"])
        collated["speaker_id"].append(item["speaker_id"])
        
        if item.get("f0_condition") is not None:
            collated["f0_condition"].append(nn.functional.pad(item["f0_condition"].transpose(0, 1), (0, pad)).transpose(0, 1))
        else:
            collated["f0_condition"].append(torch.zeros(max_len, 2))

        # TTS padding
        if max_phonemes > 0 and item.get("phoneme_ids") is not None:
            P = len(item["phoneme_ids"])
            p_pad = max_phonemes - P
            collated["phoneme_ids"].append(nn.functional.pad(item["phoneme_ids"], (0, p_pad), value=0))
            if item.get("durations") is not None:
                collated["durations"].append(nn.functional.pad(item["durations"], (0, p_pad), value=0))
            else:
                collated["durations"].append(None)
            collated["phoneme_lens"].append(item["phoneme_lens"])
            collated["language_id"].append(item["language_id"])
        elif max_phonemes > 0:
            # Empty placeholders if some items lack TTS data
            collated["phoneme_ids"].append(torch.zeros(max_phonemes, dtype=torch.long))
            collated["durations"].append(None)
            collated["phoneme_lens"].append(torch.tensor(0))
            collated["language_id"].append(torch.tensor(0))

    # Convert lists to stacks
    res = {}
    for k, v in collated.items():
        if not v:
            continue
        # durations may contain None entries (v3 pointer mode)
        if any(x is None for x in v):
            if all(x is None for x in v):
                # All None -> omit from batch (pointer mode, no durations)
                continue
            # Mixed: stack non-None with zero placeholders for None entries
            ref = next(x for x in v if x is not None)
            filled = [x if x is not None else torch.zeros_like(ref) for x in v]
            res[k] = torch.stack(filled)
        else:
            res[k] = torch.stack(v)
    return res


def train_uclm(
    cache_dir,
    output_dir,
    batch_size,
    max_steps,
    device,
    lr,
    datasets: str | None = None,
    seed: int = 42,
    sampling_strategy: str = "balanced",
    require_tts_supervision: bool = False,
    tts_mode: str = "pointer",
    pointer_loss_weight: float = 0.5,
    progress_loss_weight: float = 0.2,
    alignment_loss_type: str = "none",
    pointer_target_source: str = "heuristic_bootstrap",
    legacy_duration_loss_weight: float = 0.0,
    voice_state_loss_weight: float = 0.0,
    delta_voice_state_loss_weight: float = 0.0,
    conditioning_dropout_prob: float = 0.15,
    curriculum_stage2_start: int = 5000,
    curriculum_stage3_start: int = 15000,
    prompt_sampling_prob: float = 0.0,
    stage3_replay_mix_ratio: float = 0.2,
    base_checkpoint: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    include_datasets = None
    if datasets:
        include_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    dataset = DisentangledUCLMDataset(
        cache_dir,
        include_datasets=include_datasets,
        require_tts_supervision=require_tts_supervision,
        tts_mode=tts_mode,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No training utterances found in cache_dir={cache_dir} datasets={include_datasets or 'ALL'}"
        )
    
    # ... (skipping stats logs) ...
    
    sampler = _build_sampler(dataset, sampling_strategy=sampling_strategy, seed=seed)
    if sampler is None:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
        )

    num_speakers = len(dataset.speaker_to_id)
    num_speakers = max(num_speakers, 1)
    
    model = DisentangledUCLM(num_speakers=num_speakers).to(device)
    
    if base_checkpoint and base_checkpoint.exists():
        logger.info("Loading base checkpoint: %s", base_checkpoint)
        ckpt = torch.load(base_checkpoint, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    curriculum = CurriculumScheduler(
        stage2_start=curriculum_stage2_start,
        stage3_start=curriculum_stage3_start,
        stage3_replay_mix_ratio=stage3_replay_mix_ratio,
    )
    # Validation data (Worker 06)
    val_dataset = UCLMDataset(cache_dir, datasets=datasets_list, split="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if len(val_dataset) > 0 else None

    trainer = UCLMTrainer(
        model, optimizer, device=device, tts_mode=tts_mode,
        pointer_loss_weight=pointer_loss_weight,
        progress_loss_weight=progress_loss_weight,
        alignment_loss_type=alignment_loss_type,
        pointer_target_source=pointer_target_source,
        legacy_duration_loss_weight=legacy_duration_loss_weight,
        voice_state_loss_weight=voice_state_loss_weight,
        delta_voice_state_loss_weight=delta_voice_state_loss_weight,
        conditioning_dropout_prob=conditioning_dropout_prob,
        curriculum=curriculum,
        prompt_sampling_prob=prompt_sampling_prob,
    )

    pbar = tqdm(total=max_steps, desc="Training UCLM")
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break
            metrics = trainer.train_step(batch)
            pbar.update(1)
            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "mode": "TTS" if metrics["mode"] else "VC"})
            step += 1
            
            if step % 1000 == 0:
                # Run validation (Worker 06)
                if val_loader:
                    val_losses = []
                    for val_batch in val_loader:
                        v_m = trainer.val_step(val_batch)
                        val_losses.append(v_m["loss"])
                    avg_val = sum(val_losses) / len(val_losses)
                    print(f"\n[Step {step}] Validation Loss: {avg_val:.4f}")
                
                torch.save({"model": model.state_dict(), "step": step}, output_dir / f"uclm_step_{step}.pt")

    torch.save({"model": model.state_dict()}, output_dir / "uclm_final.pt")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/uclm"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--max-steps",
        "--train-steps",
        dest="max_steps",
        type=int,
        default=10000,
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names to include from cache (e.g., jvs,vctk).",
    )
    p.add_argument(
        "--sampling-strategy",
        choices=["balanced", "shuffle"],
        default="balanced",
        help="Sampling strategy for batches.",
    )
    p.add_argument(
        "--require-tts-supervision",
        action="store_true",
        help="Fail fast if no utterance has required text supervision.",
    )
    p.add_argument(
        "--tts-mode",
        choices=["legacy_duration", "pointer"],
        default="pointer",
        help="TTS training mode: 'pointer' (v3, MFA-free, default) or 'legacy_duration' (v2, requires durations.npy).",
    )
    p.add_argument(
        "--pointer-loss-weight",
        type=float,
        default=0.5,
        help="Weight for pointer advance loss (default: 0.5).",
    )
    p.add_argument(
        "--progress-loss-weight",
        type=float,
        default=0.2,
        help="Weight for progress regression loss (default: 0.2).",
    )
    p.add_argument(
        "--alignment-loss-type",
        choices=["none", "mas", "ctc"],
        default="none",
        help="Alignment loss type: 'none' (default), 'mas', or 'ctc'.",
    )
    p.add_argument(
        "--pointer-target-source",
        choices=["mas", "ctc", "legacy_duration", "heuristic_bootstrap"],
        default="heuristic_bootstrap",
        help="Source for pointer targets: 'heuristic_bootstrap' (default), 'legacy_duration', 'mas', or 'ctc'.",
    )
    p.add_argument(
        "--legacy-duration-loss-weight",
        type=float,
        default=0.0,
        help="Weight for legacy duration loss (default: 0.0).",
    )
    p.add_argument(
        "--voice-state-loss-weight",
        type=float,
        default=0.0,
        help="Weight for voice state supervision loss (default: 0.0).",
    )
    p.add_argument(
        "--delta-voice-state-loss-weight",
        type=float,
        default=0.0,
        help="Weight for delta voice state supervision loss (default: 0.0).",
    )
    p.add_argument(
        "--conditioning-dropout-prob",
        type=float,
        default=0.15,
        help="Probability of CFG conditioning dropout (default: 0.15).",
    )
    p.add_argument(
        "--curriculum-stage2-start",
        type=int,
        default=5000,
        help="Step at which curriculum stage 2 (alignment & pointer) begins (default: 5000).",
    )
    p.add_argument(
        "--curriculum-stage3-start",
        type=int,
        default=15000,
        help="Step at which curriculum stage 3 (drama & dialogue) begins (default: 15000).",
    )
    p.add_argument(
        "--prompt-sampling-prob",
        type=float,
        default=0.0,
        help="Probability of zeroing speaker_embed during TTS for prompt diversity (default: 0.0).",
    )
    p.add_argument(
        "--stage3-replay-mix-ratio",
        type=float,
        default=0.2,
        help="Fraction of Stage 3 batches replaced with Stage 1/2 stability data for anti-forgetting (default: 0.2).",
    )
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Path to base checkpoint to load before training.",
    )
    args = p.parse_args(argv)
    
    logging.basicConfig(level=logging.INFO)
    train_uclm(**vars(args))

if __name__ == "__main__":
    main()
