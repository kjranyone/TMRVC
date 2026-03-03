#!/usr/bin/env python3
"""Config generator for TMRVC.

Usage:
    python scripts/config_generator.py --init
    python scripts/config_generator.py --add-dataset
    python scripts/config_generator.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path("configs")
DATASETS_YAML = CONFIGS_DIR / "datasets.yaml"
DATASETS_EXAMPLE = CONFIGS_DIR / "datasets.yaml.example"


def cmd_init() -> None:
    """Initialize config files from .example templates."""
    for name in ["datasets", "constants", "export"]:
        example = CONFIGS_DIR / f"{name}.yaml.example"
        target = CONFIGS_DIR / f"{name}.yaml"
        if not example.exists():
            print(f"ERROR: Template not found: {example}")
            sys.exit(1)
        if target.exists():
            print(f"SKIP: {target} already exists")
        else:
            target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"CREATED: {target}")
    print("\nDone! Edit configs/datasets.yaml to add your datasets.")


def cmd_list() -> None:
    """List registered datasets."""
    if not DATASETS_YAML.exists():
        print(f"ERROR: {DATASETS_YAML} not found. Run --init first.")
        sys.exit(1)

    with open(DATASETS_YAML) as f:
        cfg = yaml.safe_load(f) or {}

    print(f"Config file: {DATASETS_YAML}")
    print()
    datasets = cfg.get("datasets", {})
    if not datasets:
        print("No datasets registered.")
        return

    print(f"{'Name':<25} {'Status':<8} {'Lang':<6} {'Path'}")
    print("-" * 80)
    for name, ds in datasets.items():
        status = "enabled" if ds.get("enabled", False) else "disabled"
        lang = ds.get("language", "?")
        raw_dir = ds.get("raw_dir", "")
        print(f"{name:<25} {status:<8} {lang:<6} {raw_dir}")


def cmd_add_dataset() -> None:
    """Interactively add a dataset."""
    if not DATASETS_YAML.exists():
        print(f"ERROR: {DATASETS_YAML} not found. Run --init first.")
        sys.exit(1)

    with open(DATASETS_YAML) as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("cache_dir", "data/cache")
    cfg.setdefault("split", "train")
    cfg.setdefault("datasets", {})

    print("=== Add New Dataset ===")
    print()

    name = input("Dataset name (e.g., my_voices): ").strip()
    if not name:
        print("ERROR: Name is required.")
        sys.exit(1)

    if name in cfg["datasets"]:
        print(f"WARNING: Dataset '{name}' already exists. Overwrite? [y/N]: ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    print("\nDataset types:")
    print("  1) generic  - Flat folder of wav files (auto speaker clustering)")
    print("  2) vctk     - VCTK corpus")
    print("  3) jvs      - JVS corpus (Japanese)")
    print("  4) tsukuyomi - Tsukuyomi corpus")
    ds_type = input("Select type [1-4]: ").strip()
    type_map = {"1": "generic", "2": "vctk", "3": "jvs", "4": "tsukuyomi"}
    ds_type = type_map.get(ds_type, ds_type)

    raw_dir = input("Raw directory path (e.g., data/raw/my_voices): ").strip()
    if not raw_dir:
        print("ERROR: Path is required.")
        sys.exit(1)

    lang = input("Language code [ja/en]: ").strip() or "ja"
    enabled = input("Enable this dataset? [Y/n]: ").strip().lower() != "n"

    cfg["datasets"][name] = {
        "type": ds_type,
        "enabled": enabled,
        "language": lang,
        "raw_dir": raw_dir,
    }

    with open(DATASETS_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nAdded dataset '{name}' to {DATASETS_YAML}")

    if ds_type == "generic":
        print("\nNOTE: For generic datasets, run speaker clustering first:")
        print(
            f"  python scripts/eval/cluster_speakers.py --input {raw_dir} --device cuda"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TMRVC config generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--init", action="store_true", help="Initialize configs from .example templates"
    )
    group.add_argument(
        "--add-dataset", action="store_true", help="Interactively add a dataset"
    )
    group.add_argument("--list", action="store_true", help="List registered datasets")

    args = parser.parse_args()

    if args.init:
        cmd_init()
    elif args.list:
        cmd_list()
    elif args.add_dataset:
        cmd_add_dataset()


if __name__ == "__main__":
    main()
