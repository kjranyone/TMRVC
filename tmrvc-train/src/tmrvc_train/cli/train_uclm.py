"""``tmrvc-train-uclm`` — Train Disentangled UCLM (v2)."""

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
from tmrvc_train.trainer import UCLMTrainer

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
            collated["durations"].append(nn.functional.pad(item["durations"], (0, p_pad), value=0))
            collated["phoneme_lens"].append(item["phoneme_lens"])
            collated["language_id"].append(item["language_id"])
        elif max_phonemes > 0:
            # Empty placeholders if some items lack TTS data
            collated["phoneme_ids"].append(torch.zeros(max_phonemes, dtype=torch.long))
            collated["durations"].append(torch.zeros(max_phonemes, dtype=torch.long))
            collated["phoneme_lens"].append(torch.tensor(0))
            collated["language_id"].append(torch.tensor(0))

    # Convert lists to stacks
    res = {}
    for k, v in collated.items():
        if v: res[k] = torch.stack(v)
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
):
    output_dir.mkdir(parents=True, exist_ok=True)
    include_datasets = None
    if datasets:
        include_datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    dataset = DisentangledUCLMDataset(
        cache_dir, include_datasets=include_datasets
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No training utterances found in cache_dir={cache_dir} datasets={include_datasets or 'ALL'}"
        )

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

    dataset_counts = Counter(u.get("dataset", "unknown") for u in dataset.utterances)
    logger.info(
        "Training sampler=%s datasets=%s",
        sampling_strategy,
        dict(sorted(dataset_counts.items())),
    )
    
    num_speakers = len(dataset.speaker_to_id)
    # Ensure at least 1 speaker even if dataset is empty
    num_speakers = max(num_speakers, 1)
    
    model = DisentangledUCLM(num_speakers=num_speakers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer = UCLMTrainer(model, optimizer, device=device)

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
    args = p.parse_args(argv)
    
    logging.basicConfig(level=logging.INFO)
    train_uclm(**vars(args))

if __name__ == "__main__":
    main()
