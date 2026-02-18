#!/usr/bin/env python3
"""Generate (noise, clean) pairs for Reflow training.

Runs the trained Teacher ODE solver to transport clean mel spectrograms
to noise space, producing paired data for trajectory straightening.

Usage:
    python scripts/generate_reflow_pairs.py \
        --teacher-ckpt checkpoints/phase2/best.pt \
        --cache-dir data/cache \
        --dataset vctk \
        --output-dir data/reflow_pairs \
        --steps 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.models.teacher_unet import TeacherUNet

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate (noise, clean) reflow pairs from a trained Teacher.",
    )
    parser.add_argument(
        "--teacher-ckpt", required=True, type=Path,
        help="Path to trained teacher checkpoint.",
    )
    parser.add_argument(
        "--cache-dir", required=True, type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (e.g. 'vctk').",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to save reflow pairs.",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of ODE steps for pair generation (default: 20).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for pair generation.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for computation (default: cpu).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = torch.device(args.device)

    # Load teacher
    teacher = TeacherUNet().to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    logger.info("Loaded teacher from %s", args.teacher_ckpt)

    scheduler = FlowMatchingScheduler()

    # Create dataloader
    from tmrvc_data.dataset import create_dataloader

    dataloader = create_dataloader(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_idx = 0
    for batch in dataloader:
        mel_target = batch.mel_target.to(device)
        content = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        x_1_noise, x_0_teacher = scheduler.generate_reflow_pairs(
            teacher, mel_target, steps=args.steps,
            content=content, f0=f0, spk_embed=spk_embed,
        )

        # Save pairs
        for i in range(x_1_noise.shape[0]):
            pair_path = output_dir / f"pair_{pair_idx:06d}.pt"
            torch.save(
                {
                    "x_0_teacher": x_0_teacher[i].cpu(),
                    "x_1_noise": x_1_noise[i].cpu(),
                    "f0": f0[i].cpu(),
                    "spk_embed": spk_embed[i].cpu(),
                },
                pair_path,
            )
            pair_idx += 1

        if pair_idx % 1000 == 0:
            logger.info("Generated %d pairs", pair_idx)

    logger.info("Done. Generated %d reflow pairs in %s", pair_idx, output_dir)


if __name__ == "__main__":
    main()
