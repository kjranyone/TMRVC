#!/usr/bin/env python3
"""Train UCLM (Unified Codec Language Model).

Usage:
    uv run tmrvc-train-uclm --cache-dir data/cache --datasets libritts_r --device cuda

    # Resume from checkpoint
    uv run tmrvc-train-uclm --resume checkpoints/uclm/uclm_step10000.pt

    # Custom config
    uv run tmrvc-train-uclm --config configs/train_uclm.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UCLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory with preprocessed data",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["libritts_r"],
        help="Datasets to use for training",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Maximum frames per utterance",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum frames per utterance",
    )

    # Model
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=12,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10000,
        help="Warmup steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/uclm",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "xpu"],
        help="Device to use",
    )

    # Logging
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Import here to avoid loading modules on --help
    from tmrvc_train.uclm_trainer import UCLMTrainer, UCLMTrainerConfig

    # Create config
    config = UCLMTrainerConfig(
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_accumulation=args.gradient_accumulation,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
    )

    # Create trainer
    trainer = UCLMTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Run training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Training failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
