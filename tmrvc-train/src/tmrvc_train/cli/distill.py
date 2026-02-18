"""``tmrvc-distill`` — Teacher → Student distillation CLI.

Usage::

    tmrvc-distill --cache-dir /data/cache --teacher-ckpt teacher.pt --phase A
    tmrvc-distill --cache-dir /data/cache --teacher-ckpt teacher.pt --phase B --resume distill.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.distillation import DistillationConfig, DistillationTrainer
from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.converter import ConverterStudent
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.teacher_unet import TeacherUNet
from tmrvc_train.models.vocoder import VocoderStudent

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-distill",
        description="Distill Teacher U-Net into Student models.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (e.g. configs/train_student.yaml). CLI flags override.",
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
        "--teacher-ckpt",
        required=True,
        type=Path,
        help="Path to trained teacher checkpoint.",
    )
    parser.add_argument(
        "--phase",
        default="A",
        choices=["A", "B", "B2", "C"],
        help="Distillation phase (default: A).",
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
        help="Resume from distillation checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device (default: cpu).",
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
        "A": 1e-4,
        "B": 5e-5,
        "B2": 5e-5,
        "C": 2e-5,
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
    max_steps = args.max_steps or file_cfg.get("max_steps", 200_000)
    batch_size = args.batch_size or file_cfg.get("batch_size", 64)
    save_every = args.save_every or file_cfg.get("save_every", 10_000)
    checkpoint_dir = args.checkpoint_dir or Path(file_cfg.get("checkpoint_dir", "checkpoints/distill"))

    device = torch.device(args.device)

    # Load teacher
    teacher = TeacherUNet().to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded teacher from %s", args.teacher_ckpt)

    # Create student models
    content_encoder = ContentEncoderStudent().to(device)
    converter = ConverterStudent().to(device)
    vocoder = VocoderStudent().to(device)
    ir_estimator = IREstimator().to(device)

    scheduler = FlowMatchingScheduler()

    # Optimizer for all student params
    student_params = (
        list(content_encoder.parameters())
        + list(converter.parameters())
        + list(vocoder.parameters())
        + list(ir_estimator.parameters())
    )
    optimizer = torch.optim.AdamW(student_params, lr=lr, weight_decay=0.01)

    # Dataloader
    from tmrvc_data.dataset import create_dataloader

    dataloader = create_dataloader(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        batch_size=batch_size,
    )

    config = DistillationConfig(
        phase=args.phase,
        lr=lr,
        max_steps=max_steps,
        save_every=save_every,
        checkpoint_dir=str(checkpoint_dir),
        lambda_stft=file_cfg.get("lambda_stft", 0.5),
        lambda_spk=file_cfg.get("lambda_spk", 0.3),
        lambda_ir=file_cfg.get("lambda_ir", 0.1),
    )

    trainer = DistillationTrainer(
        teacher, content_encoder, converter, vocoder, ir_estimator,
        scheduler, optimizer, dataloader, config,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(
        "Starting distillation phase %s (lr=%.2e, max_steps=%d)",
        args.phase, lr, max_steps,
    )

    while trainer.global_step < max_steps:
        trainer.train_epoch()

    trainer.save_checkpoint()
    logger.info("Distillation complete.")


if __name__ == "__main__":
    main()
