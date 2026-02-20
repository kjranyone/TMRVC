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
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0, default: 1.0=all).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Fixed frame count per batch (crop/pad). Default: 400. "
        "Prevents XPU kernel recompilation. Set to 0 to disable.",
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


def _parse_speaker_groups(cfg: dict) -> list | None:
    """Parse ``speaker_groups`` from config dict into SpeakerGroupConfig list."""
    raw = cfg.pop("speaker_groups", None)
    if not raw:
        return None
    from tmrvc_data.sampler import SpeakerGroupConfig

    groups = []
    for name, entry in raw.items():
        groups.append(
            SpeakerGroupConfig(
                speakers=entry.get("speakers", []),
                weight=int(entry.get("weight", 1)),
            )
        )
        logger.info("Speaker group '%s': speakers=%s, weight=%d", name, groups[-1].speakers, groups[-1].weight)
    return groups


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

    speaker_groups = _parse_speaker_groups(file_cfg)
    device = torch.device(args.device)

    # Load teacher — infer d_content from checkpoint
    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    d_content = ckpt["model_state_dict"]["content_proj.weight"].shape[1]
    teacher = TeacherUNet(d_content=d_content).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded teacher from %s", args.teacher_ckpt)

    # Create student models
    use_vq = file_cfg.get("use_vq", False)
    content_encoder = ContentEncoderStudent(use_vq=use_vq).to(device)
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

    # Create augmenter (if rir_augment enabled)
    augmenter = None
    if file_cfg.get("rir_augment", False):
        from tmrvc_data.augmentation import Augmenter, AugmentationConfig

        rir_dirs = file_cfg.get("rir_dirs", [])
        augmenter = Augmenter(AugmentationConfig(
            rir_dirs=[Path(d) for d in rir_dirs],
        ))
        logger.info("Augmentation enabled (rir_dirs=%s)", rir_dirs)

    # Dataloader
    from tmrvc_data.dataset import create_dataloader

    max_frames = args.max_frames if args.max_frames is not None else file_cfg.get("max_frames", 400)

    dataloader = create_dataloader(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        subset=args.subset,
        speaker_groups=speaker_groups,
        augmenter=augmenter,
        max_frames=max_frames,
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
        disc_lr=file_cfg.get("disc_lr", 2e-4),
        disc_update_ratio=file_cfg.get("disc_update_ratio", 2),
        lambda_gan=file_cfg.get("lambda_gan", 1.0),
    )

    # Discriminator for Phase B2 (DMD2)
    discriminator = None
    disc_optimizer = None
    if args.phase == "B2":
        from tmrvc_train.models.discriminator import MelDiscriminator

        discriminator = MelDiscriminator().to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=config.disc_lr, weight_decay=0.01,
        )
        if args.resume:
            ckpt_data = torch.load(args.resume, map_location=device, weights_only=False)
            if "discriminator" in ckpt_data:
                discriminator.load_state_dict(ckpt_data["discriminator"])
            if "disc_optimizer" in ckpt_data:
                disc_optimizer.load_state_dict(ckpt_data["disc_optimizer"])
        logger.info("Discriminator loaded for Phase B2 DMD2")

    # Speaker encoder for Phase C SV loss
    speaker_encoder = None
    if args.phase == "C":
        from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA

        speaker_encoder = SpeakerEncoderWithLoRA().to(device)
        if args.resume:
            ckpt_data = torch.load(args.resume, map_location=device, weights_only=False)
            if "speaker_encoder" in ckpt_data:
                speaker_encoder.load_state_dict(ckpt_data["speaker_encoder"])
        logger.info("Speaker encoder loaded for Phase C SV loss")

    trainer = DistillationTrainer(
        teacher, content_encoder, converter, vocoder, ir_estimator,
        scheduler, optimizer, dataloader, config,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        speaker_encoder=speaker_encoder,
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
