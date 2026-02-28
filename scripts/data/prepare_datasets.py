#!/usr/bin/env python3
"""Prepare training feature cache from a deterministic dataset registry.

This script avoids ad-hoc dataset discovery by loading dataset paths from
``configs/datasets.yaml`` and running ``tmrvc-preprocess`` for enabled entries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass
class DatasetSpec:
    name: str
    raw_dir: Path
    enabled: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets into feature cache.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Dataset registry YAML path.",
    )
    parser.add_argument(
        "--device",
        default="xpu",
        help="Device passed to tmrvc-preprocess (default: xpu).",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Subset ratio passed to tmrvc-preprocess.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=0,
        help="Max utterances passed to tmrvc-preprocess (0=all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass --skip-existing to tmrvc-preprocess.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional explicit dataset names to run (overrides enabled filter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print actions without executing.",
    )
    return parser.parse_args()


def load_registry(path: Path) -> tuple[Path, str, list[DatasetSpec]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cache_dir = Path(data.get("cache_dir", "data/cache"))
    split = data.get("split", "train")
    raw_specs = data.get("datasets", {})

    specs: list[DatasetSpec] = []
    for name, cfg in raw_specs.items():
        specs.append(
            DatasetSpec(
                name=name,
                raw_dir=Path(cfg.get("raw_dir", "")),
                enabled=bool(cfg.get("enabled", False)),
            )
        )
    return cache_dir, split, specs


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def write_manifest(
    repo_root: Path,
    cache_dir: Path,
    dataset: str,
    split: str,
    raw_dir: Path,
) -> None:
    from tmrvc_data.cache import FeatureCache

    cache = FeatureCache(cache_dir)
    stats = cache.verify(dataset, split)
    manifest_dir = repo_root / "data" / "cache" / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"{dataset}_{split}.json"
    payload = {
        "dataset": dataset,
        "split": split,
        "raw_dir": str(raw_dir),
        "cache_dir": str(cache_dir),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {out}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir, split, specs = load_registry(args.config)
    cache_dir = (repo_root / cache_dir).resolve() if not cache_dir.is_absolute() else cache_dir

    if args.datasets:
        selected = {name.strip() for name in args.datasets}
        specs = [s for s in specs if s.name in selected]
    else:
        specs = [s for s in specs if s.enabled]

    if not specs:
        print("No datasets selected. Check configs/datasets.yaml or --datasets.")
        return

    print(f"[info] cache_dir={cache_dir}")
    print(f"[info] split={split}")
    for spec in specs:
        raw_dir = (repo_root / spec.raw_dir).resolve() if not spec.raw_dir.is_absolute() else spec.raw_dir
        print(f"[dataset] {spec.name}: raw_dir={raw_dir}")
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw dir not found for {spec.name}: {raw_dir}")

        cmd = [
            "uv", "run", "tmrvc-preprocess",
            "--dataset", spec.name,
            "--raw-dir", str(raw_dir),
            "--cache-dir", str(cache_dir),
            "--split", split,
            "--device", args.device,
            "--subset", str(args.subset),
            "--max-utterances", str(args.max_utterances),
            "-v",
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")

        if args.dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            continue

        run_cmd(cmd, repo_root)
        write_manifest(repo_root, cache_dir, spec.name, split, raw_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
