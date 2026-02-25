#!/usr/bin/env python3
"""Automated batch evaluation across B0-B4 research ablation variants.

Runs ``eval_research_baseline.evaluate()`` for each selected variant,
saves per-variant results, and generates a merged ablation table.

Usage::

    uv run python scripts/eval_research_ablation.py \
        --variants b0 b1 b2 b3 b4 \
        --checkpoint b0=checkpoints/b0_teacher.pt \
        --checkpoint b1=checkpoints/b1_teacher.pt \
        --checkpoint b2=checkpoints/b2_teacher.pt \
        --checkpoint b3=checkpoints/b3_teacher.pt \
        --checkpoint b4=checkpoints/b4_teacher.pt \
        --cache-dir data/cache \
        --seed 42 \
        --device xpu \
        --output-dir eval/research
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from scripts.eval_research_baseline import (
    EvalResult,
    FeatureCache,
    evaluate,
    load_teacher,
    resolve_test_set,
    seed_everything,
)

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "research"
METRICS = ["mel_mse", "secs", "f0_corr", "utmos"]


def parse_checkpoint_map(values: list[str]) -> dict[str, Path]:
    """Parse repeated `variant=path` arguments into a mapping."""
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --checkpoint '{value}'. Expected format: variant=path."
            )
        variant, path_str = value.split("=", 1)
        variant = variant.strip().lower()
        path = Path(path_str.strip())
        if not variant:
            raise ValueError(f"Invalid --checkpoint '{value}': empty variant.")
        if variant in mapping:
            raise ValueError(f"Duplicate checkpoint mapping for variant '{variant}'.")
        mapping[variant] = path
    return mapping


def run_variant(
    variant: str,
    checkpoint: Path,
    cache: FeatureCache,
    seed: int,
    device: str,
    output_dir: Path,
) -> EvalResult:
    """Run evaluation for a single variant."""
    import torch

    config_path = CONFIGS_DIR / f"{variant}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_checkpoint_path"] = str(checkpoint)

    var_output = output_dir / variant
    var_output.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    seed_everything(seed)

    teacher, step = load_teacher(checkpoint, torch_device)
    logger.info("[%s] Model loaded (step %d)", variant, step)

    test_set = resolve_test_set(cache, config)
    logger.info("[%s] Test set: %d utterances", variant, len(test_set))

    if not test_set:
        logger.warning("[%s] No test utterances — skipping", variant)
        return EvalResult(
            variant=variant,
            checkpoint=str(checkpoint),
            seed=seed,
            sampling_steps=0,
            sway_coefficient=0.0,
            cfg_scale=1.0,
            n_utterances=0,
            elapsed_sec=0.0,
        )

    result = evaluate(
        teacher=teacher,
        cache=cache,
        test_set=test_set,
        config=config,
        device=torch_device,
        output_dir=var_output,
        seed=seed,
    )

    # Save per-variant results
    with open(var_output / "results.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    return result


def format_ablation_table(results: list[EvalResult]) -> str:
    """Format results as a markdown ablation table."""
    lines = [
        "| Variant | N | mel_MSE | SECS | F0 Corr | UTMOS | Time (s) |",
        "|---------|---|---------|------|---------|-------|----------|",
    ]
    for r in results:
        agg = r.aggregate
        if not agg:
            lines.append(f"| {r.variant} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r.variant} "
            f"| {r.n_utterances} "
            f"| {agg['mel_mse']['mean']:.4f}±{agg['mel_mse']['std']:.4f} "
            f"| {agg['secs']['mean']:.4f} "
            f"| {agg['f0_corr']['mean']:.4f} "
            f"| {agg['utmos']['mean']:.2f}±{agg['utmos']['std']:.2f} "
            f"| {r.elapsed_sec:.1f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_research_ablation",
        description="Batch ablation evaluation across B0-B4 variants.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["b0", "b1", "b2", "b3", "b4"],
        help="Variant names to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="VARIANT=PATH",
        help="Per-variant checkpoint mapping. Repeat for each variant.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/research"))
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        checkpoint_map = parse_checkpoint_map(args.checkpoint)
    except ValueError as e:
        parser.error(str(e))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    missing_variants = [v for v in args.variants if v not in checkpoint_map]
    if missing_variants:
        parser.error(
            "Missing --checkpoint mappings for variants: "
            + ", ".join(missing_variants)
        )
    for variant in args.variants:
        ckpt = checkpoint_map[variant]
        if not ckpt.exists():
            parser.error(f"Checkpoint not found for {variant}: {ckpt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(args.cache_dir)

    all_results: list[EvalResult] = []
    t_total = time.monotonic()

    for variant in args.variants:
        logger.info("=" * 60)
        logger.info("Evaluating variant: %s", variant)
        logger.info("=" * 60)
        result = run_variant(
            variant=variant,
            checkpoint=checkpoint_map[variant],
            cache=cache,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
        )
        all_results.append(result)
        logger.info("[%s] Done (%.1fs)", variant, result.elapsed_sec)

    total_time = time.monotonic() - t_total

    # Save merged results
    merged = {
        "seed": args.seed,
        "checkpoints": {
            variant: str(checkpoint_map[variant]) for variant in args.variants
        },
        "total_elapsed_sec": round(total_time, 2),
        "variants": {r.variant: asdict(r) for r in all_results},
    }
    merged_path = args.output_dir / "ablation_results.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Merged results saved to %s", merged_path)

    # Ablation table
    table = format_ablation_table(all_results)
    table_path = args.output_dir / "ablation_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Results\n\n")
        f.write(table)
        f.write("\n")
    logger.info("Ablation table saved to %s", table_path)

    logger.info("")
    logger.info(table)
    logger.info("")
    logger.info("Total time: %.1fs", total_time)


if __name__ == "__main__":
    main()
