"""Tests for eval_research_ablation.py and stats_research.py."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache_entry(tmp_path: Path, dataset: str, speaker: str, utt: str, T: int = 50) -> None:
    utt_dir = tmp_path / dataset / "train" / speaker / utt
    utt_dir.mkdir(parents=True, exist_ok=True)
    np.save(utt_dir / "mel.npy", np.random.randn(80, T).astype(np.float32))
    np.save(utt_dir / "content.npy", np.random.randn(768, T).astype(np.float32))
    np.save(utt_dir / "f0.npy", np.abs(np.random.randn(1, T).astype(np.float32)) * 200 + 100)
    np.save(utt_dir / "spk_embed.npy", np.random.randn(192).astype(np.float32))
    meta = {"utterance_id": utt, "speaker_id": speaker, "n_frames": T, "content_dim": 768}
    with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _make_eval_result(variant: str, n_utt: int = 10, seed: int = 42) -> dict:
    """Create a synthetic EvalResult dict."""
    rng = np.random.RandomState(seed)
    per_utt = []
    for i in range(n_utt):
        per_utt.append({
            "speaker_id": f"spk_{i % 3:03d}",
            "utterance_id": f"utt_{i:03d}",
            "n_frames": 50,
            "mel_mse": round(float(rng.uniform(0.05, 0.25)), 6),
            "secs": round(float(rng.uniform(0.8, 1.0)), 4),
            "f0_corr": round(float(rng.uniform(0.7, 0.99)), 4),
            "utmos": round(float(rng.uniform(2.5, 4.5)), 4),
        })

    aggregate = {}
    for metric in ["mel_mse", "secs", "f0_corr", "utmos"]:
        vals = [u[metric] for u in per_utt]
        aggregate[metric] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
        }

    return {
        "variant": variant,
        "checkpoint": "test.pt",
        "seed": seed,
        "sampling_steps": 2,
        "sway_coefficient": 1.0,
        "cfg_scale": 1.0,
        "n_utterances": n_utt,
        "elapsed_sec": 1.0,
        "per_utterance": per_utt,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# eval_research_ablation.py tests
# ---------------------------------------------------------------------------


class TestFormatAblationTable:
    def test_formats_table(self):
        from scripts.eval_research_ablation import EvalResult, format_ablation_table

        r = EvalResult(
            variant="b0",
            checkpoint="test.pt",
            seed=42,
            sampling_steps=32,
            sway_coefficient=1.0,
            cfg_scale=1.0,
            n_utterances=10,
            elapsed_sec=5.0,
            aggregate={
                "mel_mse": {"mean": 0.15, "std": 0.03, "min": 0.1, "max": 0.2},
                "secs": {"mean": 0.95, "std": 0.02, "min": 0.9, "max": 0.99},
                "f0_corr": {"mean": 0.88, "std": 0.05, "min": 0.8, "max": 0.95},
                "utmos": {"mean": 3.5, "std": 0.3, "min": 3.0, "max": 4.0},
            },
        )
        table = format_ablation_table([r])
        assert "b0" in table
        assert "0.1500" in table
        assert "3.50" in table

    def test_handles_empty_results(self):
        from scripts.eval_research_ablation import EvalResult, format_ablation_table

        r = EvalResult(
            variant="b0",
            checkpoint="test.pt",
            seed=42,
            sampling_steps=0,
            sway_coefficient=0.0,
            cfg_scale=1.0,
            n_utterances=0,
            elapsed_sec=0.0,
        )
        table = format_ablation_table([r])
        assert "—" in table


class TestBuildParser:
    def test_default_variants(self):
        from scripts.eval_research_ablation import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--checkpoint", "b0=test.pt",
            "--cache-dir", "data/cache",
        ])
        assert args.variants == ["b0", "b1", "b2", "b3", "b4"]
        assert args.seed == 42
        assert args.checkpoint == ["b0=test.pt"]

    def test_custom_variants(self):
        from scripts.eval_research_ablation import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--variants", "b0", "b3",
            "--checkpoint", "b0=ckpt0.pt",
            "--checkpoint", "b3=ckpt3.pt",
            "--cache-dir", "data/cache",
            "--device", "xpu",
        ])
        assert args.variants == ["b0", "b3"]
        assert args.device == "xpu"
        assert args.checkpoint == ["b0=ckpt0.pt", "b3=ckpt3.pt"]


class TestCheckpointMapParser:
    def test_parse_checkpoint_map(self):
        from scripts.eval_research_ablation import parse_checkpoint_map

        mapping = parse_checkpoint_map(["b0=a.pt", "b1=b.pt"])
        assert mapping["b0"] == Path("a.pt")
        assert mapping["b1"] == Path("b.pt")

    def test_parse_checkpoint_map_rejects_invalid(self):
        from scripts.eval_research_ablation import parse_checkpoint_map

        with pytest.raises(ValueError):
            parse_checkpoint_map(["invalid_format"])

    def test_parse_checkpoint_map_rejects_duplicate(self):
        from scripts.eval_research_ablation import parse_checkpoint_map

        with pytest.raises(ValueError):
            parse_checkpoint_map(["b0=a.pt", "b0=b.pt"])


# ---------------------------------------------------------------------------
# stats_research.py tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_basic_ci(self):
        from scripts.stats_research import bootstrap_ci

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = bootstrap_ci(values, n_bootstrap=1000, seed=42)
        assert ci.n == 5
        assert abs(ci.mean - 3.0) < 1e-6
        assert ci.ci_lower < ci.mean
        assert ci.ci_upper > ci.mean
        assert ci.ci_lower > 0.0
        assert ci.ci_upper < 6.0

    def test_empty_values(self):
        from scripts.stats_research import bootstrap_ci

        ci = bootstrap_ci(np.array([]))
        assert ci.n == 0
        assert ci.mean == 0.0

    def test_single_value(self):
        from scripts.stats_research import bootstrap_ci

        ci = bootstrap_ci(np.array([5.0]), n_bootstrap=100, seed=42)
        assert ci.n == 1
        assert ci.mean == 5.0
        assert ci.ci_lower == 5.0
        assert ci.ci_upper == 5.0

    def test_reproducibility(self):
        from scripts.stats_research import bootstrap_ci

        values = np.random.randn(50)
        ci1 = bootstrap_ci(values, seed=123)
        ci2 = bootstrap_ci(values, seed=123)
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper


class TestWilcoxonSignedRank:
    def test_identical_arrays(self):
        from scripts.stats_research import wilcoxon_signed_rank

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = wilcoxon_signed_rank(a, a)
        assert p == 1.0

    def test_different_arrays(self):
        from scripts.stats_research import wilcoxon_signed_rank

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        p = wilcoxon_signed_rank(a, b)
        assert 0.0 < p < 1.0

    def test_significant_difference(self):
        from scripts.stats_research import wilcoxon_signed_rank

        rng = np.random.RandomState(42)
        a = rng.randn(30)
        b = a + 2.0  # large shift
        p = wilcoxon_signed_rank(a, b)
        assert p < 0.05

    def test_mismatched_lengths_raises(self):
        from scripts.stats_research import wilcoxon_signed_rank

        with pytest.raises(ValueError):
            wilcoxon_signed_rank(np.array([1.0, 2.0]), np.array([1.0]))


class TestComputeStats:
    def test_computes_ci_for_all_variants(self):
        from scripts.stats_research import compute_stats

        results = {
            "b0": _make_eval_result("b0", seed=1),
            "b1": _make_eval_result("b1", seed=2),
        }
        stats = compute_stats(results, baseline="b0", n_bootstrap=100, seed=42)
        assert len(stats) == 2

        b0_stats = stats[0]
        assert b0_stats.variant == "b0"
        assert "mel_mse" in b0_stats.bootstrap_ci
        assert b0_stats.bootstrap_ci["mel_mse"].n == 10

    def test_paired_tests_against_baseline(self):
        from scripts.stats_research import compute_stats

        results = {
            "b0": _make_eval_result("b0", seed=1),
            "b1": _make_eval_result("b1", seed=2),
        }
        stats = compute_stats(results, baseline="b0", n_bootstrap=100, seed=42)

        b0_stats = [s for s in stats if s.variant == "b0"][0]
        b1_stats = [s for s in stats if s.variant == "b1"][0]

        # Baseline has no paired tests
        assert len(b0_stats.paired_tests) == 0
        # b1 should have paired tests for each metric
        assert len(b1_stats.paired_tests) == len(["mel_mse", "secs", "f0_corr", "utmos"])

    def test_paired_test_has_p_value(self):
        from scripts.stats_research import compute_stats

        results = {
            "b0": _make_eval_result("b0", seed=1),
            "b1": _make_eval_result("b1", seed=2),
        }
        stats = compute_stats(results, baseline="b0", n_bootstrap=100, seed=42)
        b1_stats = [s for s in stats if s.variant == "b1"][0]

        for pt in b1_stats.paired_tests:
            assert 0.0 <= pt.p_value <= 1.0
            assert pt.baseline == "b0"
            assert pt.variant == "b1"

    def test_no_baseline_skips_paired(self):
        from scripts.stats_research import compute_stats

        results = {
            "b1": _make_eval_result("b1", seed=1),
            "b2": _make_eval_result("b2", seed=2),
        }
        stats = compute_stats(results, baseline="b0", n_bootstrap=100, seed=42)
        for s in stats:
            assert len(s.paired_tests) == 0


class TestLoadVariantResults:
    def test_loads_from_merged_file(self, tmp_path):
        from scripts.stats_research import load_variant_results

        merged = {
            "variants": {
                "b0": _make_eval_result("b0"),
                "b1": _make_eval_result("b1"),
            }
        }
        with open(tmp_path / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(merged, f)

        results = load_variant_results(tmp_path)
        assert "b0" in results
        assert "b1" in results

    def test_loads_from_subdirectories(self, tmp_path):
        from scripts.stats_research import load_variant_results

        for v in ["b0", "b1"]:
            d = tmp_path / v
            d.mkdir()
            with open(d / "results.json", "w", encoding="utf-8") as f:
                json.dump(_make_eval_result(v), f)

        results = load_variant_results(tmp_path)
        assert "b0" in results
        assert "b1" in results


class TestFormatTables:
    def test_ci_table_has_headers(self):
        from scripts.stats_research import BootstrapCI, VariantStats, format_stats_table

        stats = [VariantStats(
            variant="b0",
            n_utterances=10,
            bootstrap_ci={
                "mel_mse": BootstrapCI(mean=0.15, ci_lower=0.12, ci_upper=0.18, std=0.03, n=10),
                "secs": BootstrapCI(mean=0.95, ci_lower=0.93, ci_upper=0.97, std=0.02, n=10),
                "f0_corr": BootstrapCI(mean=0.88, ci_lower=0.85, ci_upper=0.91, std=0.03, n=10),
                "utmos": BootstrapCI(mean=3.5, ci_lower=3.3, ci_upper=3.7, std=0.2, n=10),
            },
        )]
        table = format_stats_table(stats)
        assert "Variant" in table
        assert "b0" in table
        assert "0.1500" in table
        assert "95% CI" in table.split("\n")[0] or "[" in table

    def test_significance_table(self):
        from scripts.stats_research import PairedTest, VariantStats, format_significance_table

        stats = [VariantStats(
            variant="b1",
            n_utterances=10,
            paired_tests=[PairedTest(
                baseline="b0",
                variant="b1",
                metric="mel_mse",
                baseline_mean=0.15,
                variant_mean=0.12,
                delta=-0.03,
                p_value=0.02,
                significant=True,
            )],
        )]
        table = format_significance_table(stats)
        assert "b1" in table
        assert "mel_mse" in table
        assert "0.0200" in table
        assert "*" in table


class TestStatsParser:
    def test_default_args(self):
        from scripts.stats_research import build_parser

        parser = build_parser()
        args = parser.parse_args(["--input", "eval/research"])
        assert args.baseline == "b0"
        assert args.n_bootstrap == 10000
        assert args.seed == 42

    def test_custom_args(self):
        from scripts.stats_research import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--input", "eval/research",
            "--baseline", "b1",
            "--n-bootstrap", "5000",
            "--seed", "123",
        ])
        assert args.baseline == "b1"
        assert args.n_bootstrap == 5000


class TestStatsMainIntegration:
    def test_main_produces_output(self, tmp_path):
        from scripts.stats_research import main

        # Create synthetic results
        merged = {
            "variants": {
                "b0": _make_eval_result("b0", seed=1),
                "b1": _make_eval_result("b1", seed=2),
                "b2": _make_eval_result("b2", seed=3),
            }
        }
        with open(tmp_path / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(merged, f)

        output = tmp_path / "output"
        main(["--input", str(tmp_path), "--output", str(output), "--n-bootstrap", "100"])

        assert (output / "stats.json").exists()
        assert (output / "ci_table.md").exists()
        assert (output / "significance_table.md").exists()

        with open(output / "stats.json", encoding="utf-8") as f:
            stats = json.load(f)
        assert len(stats) == 3
        assert stats[0]["variant"] == "b0"
