"""``tmrvc-train-pipeline`` — Reproducible end-to-end training pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml

from tmrvc_train.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-train-pipeline",
        description="Reproducible UCLM v2 training pipeline (preprocess + train)",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., vctk, jvs)",
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for experiment (will create experiment subdir)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_uclm.yaml"),
        help="Training configuration file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel preprocessing workers (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use existing cache)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.experiment_name:
        experiment_id = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{args.dataset}_{timestamp}"

    experiment_dir = args.output_dir / experiment_id

    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config file not found: %s, using defaults", args.config)
        config = {}

    datasets_yaml = Path("configs/datasets.yaml")
    if datasets_yaml.exists():
        with open(datasets_yaml) as f:
            registry = yaml.safe_load(f) or {}
        ds_cfg = (registry.get("datasets") or {}).get(args.dataset) or {}
        config.update(ds_cfg)

    config.setdefault("language", "ja")
    config.setdefault("train_steps", 100000)

    cache_dir = experiment_dir / "cache"

    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)
    logger.info("Experiment ID: %s", experiment_id)
    logger.info("Experiment dir: %s", experiment_dir)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Raw dir: %s", args.raw_dir)
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Workers: %d", args.workers)
    logger.info("Seed: %d", args.seed)
    logger.info("=" * 60)

    pipeline = TrainingPipeline(
        experiment_dir=experiment_dir,
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        cache_dir=cache_dir,
        config=config,
        workers=args.workers,
        seed=args.seed,
        skip_preprocess=args.skip_preprocess,
    )

    success = pipeline.run()

    logger.info("=" * 60)
    if success:
        logger.info("Pipeline completed successfully!")
        logger.info("Experiment dir: %s", experiment_dir)
    else:
        logger.error("Pipeline failed!")
        logger.error("Check logs in: %s", experiment_dir / "logs")
    logger.info("=" * 60)

    import sys

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
