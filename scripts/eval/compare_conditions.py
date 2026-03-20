#!/usr/bin/env python3
"""Compare 2-4 ConditionEvalReport JSONs and produce a summary.

Usage:
    python scripts/eval/compare_conditions.py \
        results/condition_A_eval.json \
        results/condition_B_eval.json \
        [results/condition_C_eval.json] \
        [--output results/comparison.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_conditions(reports: list[dict]) -> dict:
    """Compare condition reports with condition A as baseline."""

    if not reports:
        return {"error": "No reports provided"}

    # Find baseline (condition A) or use first report
    baseline = None
    for r in reports:
        if r.get("condition") == "A":
            baseline = r
            break
    if baseline is None:
        baseline = reports[0]

    baseline_condition = baseline["condition"]
    axes = list(baseline.get("axes", {}).keys())

    comparison = {
        "baseline": baseline_condition,
        "conditions_compared": [r["condition"] for r in reports],
        "axes": {},
        "per_axis_winner": {},
        "summary": [],
    }

    for axis_name in axes:
        axis_data = {}
        baseline_axis = baseline["axes"].get(axis_name, {})
        baseline_value = baseline_axis.get("value", 0.0)
        higher_is_better = baseline_axis.get("higher_is_better", True)

        for r in reports:
            cond = r["condition"]
            r_axis = r.get("axes", {}).get(axis_name, {})
            r_value = r_axis.get("value", 0.0)
            delta = r_value - baseline_value
            relative_delta = (delta / baseline_value * 100) if baseline_value != 0 else 0.0

            axis_data[cond] = {
                "value": r_value,
                "delta_vs_baseline": delta,
                "relative_delta_pct": relative_delta,
            }

        comparison["axes"][axis_name] = axis_data

        # Determine winner
        best_cond = baseline_condition
        best_value = baseline_value
        for r in reports:
            cond = r["condition"]
            v = r.get("axes", {}).get(axis_name, {}).get("value", 0.0)
            if (higher_is_better and v > best_value) or (not higher_is_better and v < best_value):
                best_value = v
                best_cond = cond

        comparison["per_axis_winner"][axis_name] = best_cond

    # Generate summary lines
    for axis_name, winner in comparison["per_axis_winner"].items():
        if winner == baseline_condition:
            comparison["summary"].append(f"{axis_name}: Baseline ({baseline_condition}) wins")
        else:
            delta = comparison["axes"][axis_name][winner]["relative_delta_pct"]
            comparison["summary"].append(f"{axis_name}: {winner} wins ({delta:+.1f}% vs baseline)")

    return comparison


def format_markdown(comparison: dict) -> str:
    """Format comparison as Markdown table."""
    lines = []
    lines.append(f"# Codec Condition Comparison (baseline: {comparison['baseline']})")
    lines.append("")

    conditions = comparison["conditions_compared"]

    # Header
    header = "| Axis |"
    sep = "|------|"
    for c in conditions:
        header += f" {c} |"
        sep += "------|"
    header += " Winner |"
    sep += "--------|"
    lines.append(header)
    lines.append(sep)

    # Rows
    for axis_name in comparison["axes"]:
        row = f"| {axis_name} |"
        for c in conditions:
            data = comparison["axes"][axis_name].get(c, {})
            v = data.get("value", 0.0)
            delta = data.get("relative_delta_pct", 0.0)
            if c == comparison["baseline"]:
                row += f" {v:.4f} (base) |"
            else:
                row += f" {v:.4f} ({delta:+.1f}%) |"
        winner = comparison["per_axis_winner"].get(axis_name, "?")
        row += f" **{winner}** |"
        lines.append(row)

    lines.append("")
    lines.append("## Summary")
    for s in comparison.get("summary", []):
        lines.append(f"- {s}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare condition evaluation reports")
    parser.add_argument("reports", nargs="+", help="ConditionEvalReport JSON files")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown summary")
    args = parser.parse_args()

    reports = [load_report(p) for p in args.reports]
    comparison = compare_conditions(reports)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to: {args.output}")

    if args.markdown or not args.output:
        print(format_markdown(comparison))


if __name__ == "__main__":
    main()
