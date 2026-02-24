"""``tmrvc-train-style`` — StyleEncoder training CLI (Phase 3a).

Usage::

    tmrvc-train-style --cache-dir data/cache --dataset expresso,jvnv --device xpu
    tmrvc-train-style --cache-dir data/cache --config configs/train_style.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from tmrvc_train.models.style_encoder import StyleEncoder
from tmrvc_train.style_trainer import StyleTrainer, StyleTrainerConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-train-style",
        description="Train StyleEncoder for emotion classification and VAD regression.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config file. CLI flags override.",
    )
    parser.add_argument(
        "--cache-dir", required=True, type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name(s), comma-separated (e.g. 'expresso,jvnv,emov_db,ravdess').",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 5e-4).")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (default: 50000).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 64).")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N steps.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint.")
    parser.add_argument("--device", default="cpu", help="Training device.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def _load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    file_cfg = _load_config(args.config)

    lr = args.lr if args.lr is not None else file_cfg.get("lr", 5e-4)
    max_steps = args.max_steps if args.max_steps is not None else file_cfg.get("max_steps", 50_000)
    batch_size = args.batch_size if args.batch_size is not None else file_cfg.get("batch_size", 64)
    save_every = args.save_every if args.save_every is not None else file_cfg.get("save_every", 5_000)
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else Path(file_cfg.get("checkpoint_dir", "checkpoints/style"))
    warmup_steps = file_cfg.get("warmup_steps", 2000)

    if args.dataset:
        datasets = [d.strip() for d in args.dataset.split(",")]
    else:
        datasets = file_cfg.get("datasets")
        if not datasets:
            parser.error("--dataset is required (or specify 'datasets' in YAML config)")
    logger.info("Datasets: %s", datasets)

    device = torch.device(args.device)

    model = StyleEncoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("StyleEncoder: %.2fM parameters", n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Create emotion dataloader
    from tmrvc_data.emotion_dataset import create_emotion_dataloader

    dataloader = create_emotion_dataloader(
        cache_dir=args.cache_dir,
        datasets=datasets,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    logger.info("Emotion dataset: %d samples", len(dataloader.dataset))

    config = StyleTrainerConfig(
        lr=lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_every=save_every,
        checkpoint_dir=str(checkpoint_dir),
        grad_clip=file_cfg.get("grad_clip", 1.0),
        lambda_emotion=file_cfg.get("lambda_emotion", 1.0),
        lambda_vad=file_cfg.get("lambda_vad", 0.5),
        lambda_prosody=file_cfg.get("lambda_prosody", 0.3),
        use_wandb=args.wandb,
    )

    trainer = StyleTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        config=config,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(
        "Starting style training (lr=%.2e, max_steps=%d, batch_size=%d)",
        lr, max_steps, batch_size,
    )

    try:
        for step, losses in trainer.train_iter():
            pass
    except KeyboardInterrupt:
        logger.info("Interrupted at step %d, saving checkpoint...", trainer.global_step)

    trainer.save_checkpoint()
    logger.info("Style training complete (step %d).", trainer.global_step)


if __name__ == "__main__":
    main()
