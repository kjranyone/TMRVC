#!/usr/bin/env python3
"""Statistical analysis for research ablation results.

Reads per-variant evaluation results from ``eval/research/``, computes
bootstrap confidence intervals and paired significance tests (Wilcoxon
signed-rank), and outputs a publication-ready statistics table.

Usage::

    uv run python scripts/stats_research.py --input eval/research
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

METRICS = ["mel_mse", "secs", "f0_corr", "utmos"]
# mel_mse: lower is better; others: higher is better
HIGHER_IS_BETTER = {"secs": True, "f0_corr": True, "utmos": True, "mel_mse": False}
BASELINE_VARIANT = "b0"


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n: int


@dataclass
class PairedTest:
    """Result of a paired significance test."""
    baseline: str
    variant: str
    metric: str
    baseline_mean: float
    variant_mean: float
    delta: float
    p_value: float
    significant: bool  # p < 0.05


@dataclass
class VariantStats:
    """Per-variant statistics."""
    variant: str
    n_utterances: int
    bootstrap_ci: dict[str, BootstrapCI] = field(default_factory=dict)
    paired_tests: list[PairedTest] = field(default_factory=list)


def load_variant_results(input_dir: Path) -> dict[str, dict]:
    """Load results.json from each variant subdirectory."""
    results = {}

    # Try merged file first
    merged_path = input_dir / "ablation_results.json"
    if merged_path.exists():
        with open(merged_path, encoding="utf-8") as f:
            merged = json.load(f)
        for variant, data in merged.get("variants", {}).items():
            results[variant] = data
        return results

    # Fall back to per-variant directories
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        results_path = d / "results.json"
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                results[d.name] = json.load(f)

    return results


def extract_per_utterance_values(
    result: dict,
    metric: str,
) -> np.ndarray:
    """Extract per-utterance metric values as an array."""
    per_utt = result.get("per_utterance", [])
    return np.array([u[metric] for u in per_utt], dtype=np.float64)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval.

    Args:
        values: Per-utterance metric values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (0.95 = 95% CI).
        seed: Random seed.

    Returns:
        BootstrapCI with mean and CI bounds.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0, std=0.0, n=0)

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()

    alpha = (1.0 - ci) / 2.0
    return BootstrapCI(
        mean=round(float(values.mean()), 6),
        ci_lower=round(float(np.percentile(means, 100 * alpha)), 6),
        ci_upper=round(float(np.percentile(means, 100 * (1.0 - alpha))), 6),
        std=round(float(values.std()), 6),
        n=n,
    )


def wilcoxon_signed_rank(
    baseline_values: np.ndarray,
    variant_values: np.ndarray,
) -> float:
    """Compute Wilcoxon signed-rank test p-value.

    Uses scipy if available, otherwise falls back to a normal
    approximation.
    """
    if len(baseline_values) != len(variant_values):
        raise ValueError("Arrays must have the same length for paired test")

    diffs = variant_values - baseline_values
    # Remove zero differences
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0

    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(nonzero)
        return float(p)
    except ImportError:
        # Normal approximation fallback
        n = len(nonzero)
        ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1.0
        w_plus = float(ranks[nonzero > 0].sum())
        w_minus = float(ranks[nonzero < 0].sum())
        w = min(w_plus, w_minus)
        # Expected value and variance under H0
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if sigma == 0:
            return 1.0
        z = (w - mu) / sigma
        # Two-tailed p-value using normal CDF approximation
        p = 2.0 * (1.0 - _normal_cdf(abs(z)))
        return float(p)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_stats(
    results: dict[str, dict],
    baseline: str = BASELINE_VARIANT,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> list[VariantStats]:
    """Compute bootstrap CIs and paired tests for all variants."""
    all_stats: list[VariantStats] = []

    baseline_values: dict[str, np.ndarray] = {}
    if baseline in results:
        for metric in METRICS:
            baseline_values[metric] = extract_per_utterance_values(results[baseline], metric)

    for variant in sorted(results.keys()):
        data = results[variant]
        vs = VariantStats(
            variant=variant,
            n_utterances=data.get("n_utterances", 0),
        )

        for metric in METRICS:
            values = extract_per_utterance_values(data, metric)
            if len(values) == 0:
                continue
            vs.bootstrap_ci[metric] = bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)

        # Paired tests against baseline
        if variant != baseline and baseline in results:
            for metric in METRICS:
                var_vals = extract_per_utterance_values(data, metric)
                base_vals = baseline_values.get(metric, np.array([]))
                if len(var_vals) == 0 or len(base_vals) == 0:
                    continue
                if len(var_vals) != len(base_vals):
                    logger.warning(
                        "Skipping paired test %s vs %s for %s: different lengths (%d vs %d)",
                        variant, baseline, metric, len(var_vals), len(base_vals),
                    )
                    continue

                p = wilcoxon_signed_rank(base_vals, var_vals)
                vs.paired_tests.append(PairedTest(
                    baseline=baseline,
                    variant=variant,
                    metric=metric,
                    baseline_mean=round(float(base_vals.mean()), 6),
                    variant_mean=round(float(var_vals.mean()), 6),
                    delta=round(float(var_vals.mean() - base_vals.mean()), 6),
                    p_value=round(p, 6),
                    significant=p < 0.05,
                ))

        all_stats.append(vs)

    return all_stats


def format_stats_table(stats: list[VariantStats]) -> str:
    """Format statistics as a publication-ready markdown table."""
    lines = [
        "| Variant | N | mel_MSE [95% CI] | SECS [95% CI] | F0 Corr [95% CI] | UTMOS [95% CI] |",
        "|---------|---|------------------|---------------|------------------|----------------|",
    ]
    for vs in stats:
        row = f"| {vs.variant} | {vs.n_utterances}"
        for metric in METRICS:
            ci = vs.bootstrap_ci.get(metric)
            if ci is None:
                row += " | —"
                continue
            row += f" | {ci.mean:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
        row += " |"
        lines.append(row)
    return "\n".join(lines)


def format_significance_table(stats: list[VariantStats]) -> str:
    """Format paired significance test results."""
    lines = [
        "| Variant | Metric | Baseline | Variant | Delta | p-value | Sig. |",
        "|---------|--------|----------|---------|-------|---------|------|",
    ]
    for vs in stats:
        for pt in vs.paired_tests:
            sig = "***" if pt.p_value < 0.001 else ("**" if pt.p_value < 0.01 else ("*" if pt.significant else ""))
            lines.append(
                f"| {pt.variant} | {pt.metric} "
                f"| {pt.baseline_mean:.4f} | {pt.variant_mean:.4f} "
                f"| {pt.delta:+.4f} | {pt.p_value:.4f} | {sig} |"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stats_research",
        description="Statistical analysis for research ablation results.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input directory with variant results.")
    parser.add_argument("--baseline", default=BASELINE_VARIANT, help="Baseline variant for paired tests.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: same as input).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = load_variant_results(args.input)
    if not results:
        logger.error("No results found in %s", args.input)
        sys.exit(1)

    logger.info("Loaded %d variants: %s", len(results), ", ".join(sorted(results.keys())))

    stats = compute_stats(
        results,
        baseline=args.baseline,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    output_dir = args.output or args.input
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stats JSON
    stats_data = [asdict(vs) for vs in stats]
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    logger.info("Stats saved to %s", stats_path)

    # CI table
    ci_table = format_stats_table(stats)
    ci_path = output_dir / "ci_table.md"
    with open(ci_path, "w", encoding="utf-8") as f:
        f.write("# Bootstrap 95% Confidence Intervals\n\n")
        f.write(ci_table)
        f.write("\n")
    logger.info("CI table saved to %s", ci_path)

    # Significance table
    sig_table = format_significance_table(stats)
    sig_path = output_dir / "significance_table.md"
    with open(sig_path, "w", encoding="utf-8") as f:
        f.write("# Paired Significance Tests (Wilcoxon signed-rank vs %s)\n\n" % args.baseline)
        f.write(sig_table)
        f.write("\n")
    logger.info("Significance table saved to %s", sig_path)

    # Print summary
    logger.info("")
    logger.info(ci_table)
    logger.info("")
    logger.info(sig_table)


if __name__ == "__main__":
    main()
