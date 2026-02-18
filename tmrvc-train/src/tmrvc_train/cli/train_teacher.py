"""``tmrvc-train-teacher`` — Teacher U-Net training CLI.

Usage::

    tmrvc-train-teacher --cache-dir /data/cache --phase 0 --max-steps 100000
    tmrvc-train-teacher --cache-dir /data/cache --phase 1b --resume checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.models.teacher_unet import TeacherUNet
from tmrvc_train.trainer import TeacherTrainer, TrainerConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-train-teacher",
        description="Train the Teacher U-Net for voice conversion.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (e.g. configs/train_teacher.yaml). CLI flags override.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. 'vctk', 'jvs', 'libritts_r').",
    )
    parser.add_argument(
        "--phase",
        default="0",
        choices=["0", "1a", "1b", "2", "reflow"],
        help="Training phase (default: 0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: phase-dependent).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device (default: cpu).",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0, default: 1.0=all).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0, use 0 on Windows).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def _load_config(config_path: Path | None, phase: str) -> dict:
    """Load YAML config and merge phase-specific overrides."""
    if config_path is None:
        return {}
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Merge phase-specific overrides
    phases = cfg.pop("phases", {})
    phase_cfg = phases.get(phase, {})
    phase_cfg.pop("description", None)
    cfg.update(phase_cfg)
    return cfg


def _default_lr(phase: str) -> float:
    return {
        "0": 2e-4,
        "1a": 1e-4,
        "1b": 5e-5,
        "2": 5e-5,
        "reflow": 5e-5,
    }[phase]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Merge YAML config with CLI overrides
    file_cfg = _load_config(args.config, args.phase)

    lr = args.lr or file_cfg.get("lr") or _default_lr(args.phase)
    max_steps = args.max_steps or file_cfg.get("max_steps", 100_000)
    batch_size = args.batch_size or file_cfg.get("batch_size", 64)
    save_every = args.save_every or file_cfg.get("save_every", 10_000)
    checkpoint_dir = args.checkpoint_dir or Path(file_cfg.get("checkpoint_dir", "checkpoints"))
    warmup_steps = file_cfg.get("warmup_steps", 0)

    device = torch.device(args.device)

    # Create model
    teacher = TeacherUNet().to(device)
    scheduler = FlowMatchingScheduler()

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=0.01)

    # Create dataloader
    from tmrvc_data.dataset import create_dataloader

    dataloader = create_dataloader(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        subset=args.subset,
        num_workers=args.num_workers,
    )

    config = TrainerConfig(
        phase=args.phase,
        lr=lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_every=save_every,
        checkpoint_dir=str(checkpoint_dir),
        lambda_stft=file_cfg.get("lambda_stft", 0.5),
        lambda_spk=file_cfg.get("lambda_spk", 0.3),
        lambda_ir=file_cfg.get("lambda_ir", 0.1),
        use_wandb=args.wandb,
    )

    trainer = TeacherTrainer(teacher, scheduler, optimizer, dataloader, config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(
        "Starting teacher training phase %s (lr=%.2e, max_steps=%d)",
        args.phase, lr, max_steps,
    )

    while trainer.global_step < max_steps:
        trainer.train_epoch()

    trainer.save_checkpoint()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
