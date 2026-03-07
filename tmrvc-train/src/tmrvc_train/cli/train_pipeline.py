"""``tmrvc-train-pipeline`` — Reproducible end-to-end training pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from tmrvc_train.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    raw_dir: Path
    config: dict


def _cache_has_required_datasets(cache_dir: Path, datasets: list[str]) -> bool:
    """Return True when cache has expected per-dataset train directories."""
    for dataset in datasets:
        if not (cache_dir / dataset / "train").is_dir():
            return False
    return True


def find_latest_cache(
    output_dir: Path,
    experiment_name: str,
    required_datasets: list[str] | None = None,
) -> Path | None:
    """Find latest valid cache directory for an experiment name prefix."""
    candidates = []
    pattern = f"{experiment_name}_*"
    for exp_dir in output_dir.glob(pattern):
        cache_dir = exp_dir / "cache"
        if not cache_dir.is_dir():
            continue
        if required_datasets and not _cache_has_required_datasets(
            cache_dir, required_datasets
        ):
            continue
        candidates.append(cache_dir)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_adapter_type(dataset_config: dict, dataset_name: str) -> str:
    """Resolve adapter type from config with backward-compatible keys."""
    return (
        dataset_config.get("adapter_type")
        or dataset_config.get("type")
        or dataset_name
    )


def _resolve_dataset_raw_dir(
    dataset_name: str, dataset_config: dict, fallback_raw_dir: Path | None
) -> Path:
    """Resolve raw directory for one dataset.

    Priority:
      1. datasets.yaml per-dataset `raw_dir`
      2. CLI `--raw-dir` fallback (for datasets without explicit raw_dir)
    """
    raw_dir_value = dataset_config.get("raw_dir")
    if raw_dir_value:
        return Path(raw_dir_value)
    if fallback_raw_dir is not None:
        return fallback_raw_dir
    raise ValueError(
        f"raw_dir not configured for dataset={dataset_name}. "
        "Set datasets.<name>.raw_dir in configs/datasets.yaml or pass --raw-dir."
    )


def _build_dataset_plans(
    datasets_to_use: list[str],
    base_config: dict,
    registry: dict,
    fallback_raw_dir: Path | None,
    train_batch_size: int | None = None,
    train_device: str | None = None,
) -> list[DatasetPlan]:
    """Build fully-resolved run plans (dataset, raw_dir, config)."""
    plans: list[DatasetPlan] = []
    for dataset in datasets_to_use:
        ds_cfg = (registry.get("datasets") or {}).get(dataset) or {}
        dataset_config = {**base_config, **ds_cfg}
        dataset_config["adapter_type"] = _resolve_adapter_type(dataset_config, dataset)
        if train_batch_size is not None:
            dataset_config["train_batch_size"] = int(train_batch_size)
        if train_device is not None:
            dataset_config["train_device"] = str(train_device)
        raw_dir = _resolve_dataset_raw_dir(dataset, dataset_config, fallback_raw_dir)
        plans.append(DatasetPlan(name=dataset, raw_dir=raw_dir, config=dataset_config))
    return plans


def _dataset_has_train_utterances(
    dataset: str, raw_dir: Path, dataset_config: dict
) -> bool:
    """Fast preflight: check whether at least one train utterance is available."""
    from tmrvc_data.dataset_adapters import get_adapter

    adapter = get_adapter(
        dataset,
        adapter_type=_resolve_adapter_type(dataset_config, dataset),
        language=dataset_config.get("language", "ja"),
        speaker_map_path=dataset_config.get("speaker_map"),
    )
    return next(adapter.iter_utterances(raw_dir, "train"), None) is not None


def _find_datasets_missing_raw(
    plans: list[DatasetPlan],
) -> list[str]:
    """Return datasets that have no discoverable train utterances."""
    missing: list[str] = []
    for plan in plans:
        try:
            has_data = _dataset_has_train_utterances(
                plan.name, plan.raw_dir, plan.config
            )
        except Exception as e:
            logger.warning("Raw-data preflight check failed for %s: %s", plan.name, e)
            has_data = False
        if not has_data:
            missing.append(plan.name)
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-train-pipeline",
        description="Reproducible UCLM v3 training pipeline (preprocess + train)",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (e.g., vctk, jvs). If not specified, uses all enabled datasets.",
    )
    parser.add_argument(
        "--raw-dir",
        required=False,
        default=None,
        type=Path,
        help="Fallback raw dataset directory (used only when per-dataset raw_dir is absent in configs/datasets.yaml).",
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
        "--train-batch-size",
        type=int,
        default=None,
        help="Override UCLM training batch size (keeps default behavior when omitted)",
    )
    parser.add_argument(
        "--train-device",
        default=None,
        help="Override UCLM training device (e.g., cuda, cpu)",
    )
    parser.add_argument(
        "--require-tts-supervision",
        action="store_true",
        help="Fail training if cache has no TTS text supervision (phoneme_ids).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use existing cache)",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run preprocessing only and skip UCLM training.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only UCLM training using existing cache (implies --skip-preprocess).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory to reuse (required for deterministic skip-preprocess resumes)",
    )
    parser.add_argument(
        "--tts-mode",
        choices=["legacy_duration", "pointer"],
        default="pointer",
        help="TTS training mode: 'pointer' (v3, MFA-free, default) or 'legacy_duration' (v2).",
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

    if args.preprocess_only and args.train_only:
        logger.error("--preprocess-only and --train-only cannot be used together.")
        import sys

        sys.exit(1)
    if args.train_only:
        args.skip_preprocess = True
    if args.preprocess_only and args.skip_preprocess:
        logger.error("--preprocess-only cannot be used with --skip-preprocess.")
        import sys

        sys.exit(1)

    datasets_yaml = Path("configs/datasets.yaml")
    registry: dict = {}
    if datasets_yaml.exists():
        with open(datasets_yaml) as f:
            registry = yaml.safe_load(f) or {}

    if args.dataset:
        datasets_to_use = [args.dataset]
    else:
        all_datasets = registry.get("datasets") or {}
        datasets_to_use = [
            name for name, cfg in all_datasets.items() if cfg.get("enabled", False)
        ]
        if not datasets_to_use:
            logger.error("No enabled datasets found in %s", datasets_yaml)
            import sys

            sys.exit(1)

    experiment_name = args.experiment_name or "_".join(sorted(datasets_to_use))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"

    experiment_dir = args.output_dir / experiment_id

    config_path = args.config
    if not config_path.exists() and config_path.name == "train_uclm.yaml":
        fallback_example = config_path.with_name("train_uclm.yaml.example")
        if fallback_example.exists():
            logger.info(
                "Config file not found: %s, using example defaults: %s",
                config_path,
                fallback_example,
            )
            config_path = fallback_example

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config file not found: %s, using defaults", config_path)
        config = {}

    config.setdefault("language", "ja")
    config.setdefault("train_steps", 100000)
    if args.require_tts_supervision:
        config["train_require_tts_supervision"] = True
    config["tts_mode"] = args.tts_mode

    try:
        dataset_plans = _build_dataset_plans(
            datasets_to_use=datasets_to_use,
            base_config=config,
            registry=registry,
            fallback_raw_dir=args.raw_dir,
            train_batch_size=args.train_batch_size,
            train_device=args.train_device,
        )
    except ValueError as e:
        logger.error(str(e))
        import sys

        sys.exit(1)

    if not args.skip_preprocess:
        missing_raw = _find_datasets_missing_raw(dataset_plans)
        if missing_raw:
            logger.error(
                "No train utterances found for datasets: %s",
                ", ".join(missing_raw),
            )
            logger.error(
                "Check datasets.<name>.raw_dir in configs/datasets.yaml, "
                "or pass --raw-dir as fallback, or use --skip-preprocess with a valid --cache-dir."
            )
            import sys

            sys.exit(1)

    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    elif args.skip_preprocess:
        latest_cache = find_latest_cache(
            args.output_dir,
            experiment_name,
            required_datasets=datasets_to_use,
        )
        if latest_cache is not None:
            cache_dir = latest_cache
            logger.info("Auto-selected latest cache for --skip-preprocess: %s", cache_dir)
        else:
            logger.error(
                "No reusable cache found for experiment_name=%s under %s (required datasets: %s)",
                experiment_name,
                args.output_dir,
                ", ".join(datasets_to_use),
            )
            import sys

            sys.exit(1)
    else:
        cache_dir = experiment_dir / "cache"

    if args.skip_preprocess and not _cache_has_required_datasets(cache_dir, datasets_to_use):
        logger.error(
            "Invalid cache for --skip-preprocess: %s (missing one of: %s)",
            cache_dir,
            ", ".join(f"{d}/train" for d in datasets_to_use),
        )
        import sys

        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)
    logger.info("Experiment ID: %s", experiment_id)
    logger.info("Experiment dir: %s", experiment_dir)
    logger.info("Datasets: %s", ", ".join(datasets_to_use))
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Workers: %d", args.workers)
    logger.info("Seed: %d", args.seed)
    for plan in dataset_plans:
        logger.info(
            "Dataset plan: %s raw_dir=%s adapter=%s",
            plan.name,
            plan.raw_dir,
            plan.config.get("adapter_type"),
        )
    logger.info("=" * 60)

    success = True
    if len(dataset_plans) == 1:
        plan = dataset_plans[0]
        pipeline = TrainingPipeline(
            experiment_dir=experiment_dir,
            dataset=plan.name,
            raw_dir=plan.raw_dir,
            cache_dir=cache_dir,
            config=plan.config,
            workers=args.workers,
            seed=args.seed,
            skip_preprocess=args.skip_preprocess,
            run_training=not args.preprocess_only,
            train_datasets=[plan.name],
        )
        success = pipeline.run()
    else:
        if not args.skip_preprocess:
            for plan in dataset_plans:
                preprocess_pipeline = TrainingPipeline(
                    experiment_dir=experiment_dir,
                    dataset=plan.name,
                    raw_dir=plan.raw_dir,
                    cache_dir=cache_dir,
                    config=plan.config,
                    workers=args.workers,
                    seed=args.seed,
                    skip_preprocess=False,
                    run_training=False,
                    train_datasets=[p.name for p in dataset_plans],
                )
                if not preprocess_pipeline.run():
                    success = False
                    break

        if success and not args.preprocess_only:
            # Train once on the selected dataset union.
            training_config = dict(config)
            if args.train_batch_size is not None:
                training_config["train_batch_size"] = int(args.train_batch_size)
            if args.train_device is not None:
                training_config["train_device"] = str(args.train_device)
            training_pipeline = TrainingPipeline(
                experiment_dir=experiment_dir,
                dataset="multi",
                raw_dir=dataset_plans[0].raw_dir,
                cache_dir=cache_dir,
                config=training_config,
                workers=args.workers,
                seed=args.seed,
                skip_preprocess=True,
                run_training=True,
                train_datasets=[p.name for p in dataset_plans],
            )
            success = training_pipeline.run()

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
