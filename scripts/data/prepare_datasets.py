"""Prepare training feature cache from a deterministic dataset registry. (FIXED)"""

import argparse
import subprocess
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    # Absolute path to repo root
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / args.config
    
    print(f"[info] Loading config from: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cache_dir = repo_root / data.get("cache_dir", "data/cache")
    split = data.get("split", "train")
    datasets = data.get("datasets", {})

    processed_any = False
    for name, cfg in datasets.items():
        if not cfg.get("enabled", False):
            print(f"[skip] {name} (disabled)")
            continue

        processed_any = True
        raw_dir = repo_root / Path(cfg.get("raw_dir"))
        print(f"[dataset] {name}: raw_dir={raw_dir}")

        cmd = [
            "uv", "run", "tmrvc-preprocess",
            "--dataset", name,
            "--raw-dir", str(raw_dir),
            "--cache-dir", str(cache_dir),
            "--split", split,
            "--device", args.device,
            "-v",
        ]
        if args.sample_ratio < 1.0: cmd.extend(["--sample-ratio", str(args.sample_ratio)])
        if args.skip_existing:
            cmd.append("--skip-existing")

        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    
    if not processed_any:
        print("No enabled datasets found in config.")

if __name__ == "__main__":
    main()
