#!/usr/bin/env python3
"""TMRVC v4 Development Menu.

Pointer-based causal is the sole mainline.  All v2/v3 legacy paths
(MFA, legacy_duration, voice_state_loss_weight) have been removed.

v4 additions:
- Bootstrap pipeline (raw corpus -> train-ready cache)
- 12-D physical + 24-D acting latent supervision
- Enriched transcript (inline acting tags)
- RL fine-tuning phase (instruction-following)
- Supervision tier-aware loss weighting
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from tmrvc_core.constants import SERVE_PORT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIGS_DIR = Path("configs")
DATASETS_YAML = CONFIGS_DIR / "datasets.yaml"
CHARACTERS_JSON = CONFIGS_DIR / "characters.json"
TRAIN_UCLM_YAML = CONFIGS_DIR / "train_uclm.yaml.example"
EXPERIMENTS_DIR = Path("experiments")
CHECKPOINTS_DIR = Path("checkpoints")
MODELS_DIR = Path("models")
CHARACTERS_DIR = MODELS_DIR / "characters"
V4_CACHE_DIR = Path("data") / "v4_cache"
RL_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "rl"

# ---------------------------------------------------------------------------
# v4 required training config fields.
# voice_state_loss_weight is replaced by physical_12d + acting_latent.
# legacy_duration mode is deleted; pointer is the only mode.
# ---------------------------------------------------------------------------
REQUIRED_TRAINING_FIELDS: dict[str, type | tuple[type, ...]] = {
    "tts_mode": str,
    "pointer_loss_weight": (int, float),
    "progress_loss_weight": (int, float),
    "physical_12d_loss_weight": (int, float),
    "acting_latent_loss_weight": (int, float),
}

OPTIONAL_TRAINING_DEFAULTS: dict[str, object] = {
    "pointer_mode": True,
    "cfg_enabled": True,
    "cfg_drop_rate": 0.1,
    "physical_supervision": True,
    "acting_latent_supervision": True,
    "prosody_flow_matching": True,
    "training_stage": "base",
    "few_shot_prompt_training": True,
    "replay_mix_ratio": 0.1,
    "enriched_transcript_mix_ratio": 0.5,
    "bio_constraint_enabled": True,
    "bio_transition_penalty_weight": 0.1,
    "disentanglement_loss_enabled": True,
    "speaker_consistency_loss_enabled": True,
    "semantic_alignment_loss_enabled": True,
}

# v4 supervision tier loss multipliers (used in display/validation only;
# actual weights live in tmrvc_data.bootstrap.supervision)
TIER_WEIGHTS = {"tier_a": 1.0, "tier_b": 0.7, "tier_c": 0.3, "tier_d": 0.1}


def validate_training_config(cfg: dict) -> list[str]:
    """Return a list of validation errors for training config.

    An empty list means the config is valid.
    """
    errors: list[str] = []
    for key, expected_type in REQUIRED_TRAINING_FIELDS.items():
        if key not in cfg:
            errors.append(f"missing required field: {key}")
        elif not isinstance(cfg[key], expected_type):
            errors.append(f"{key}: expected {expected_type}, got {type(cfg[key])}")
    if cfg.get("tts_mode") != "pointer":
        errors.append(
            f"tts_mode must be 'pointer', got {cfg.get('tts_mode')!r}"
        )
    return errors


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
def clear_screen() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def input_default(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    result = input(f"{prompt}{hint}: ").strip()
    return result if result else default


def select_device() -> str:
    return input_default("デバイス", "cuda")


def get_gpu_info() -> tuple[float, float, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        free = total - reserved
        recommended = max(1, math.floor(free / 4.5))
        print(f"\nGPU {idx}: {torch.cuda.get_device_name(idx)}")
        print(f"  総メモリ: {total:.1f} GB  空き: {free:.1f} GB")
        print(f"  推奨 workers: {recommended}")
        return total, free, recommended
    except Exception:
        return None


def select_workers(device: str) -> int:
    if device != "cuda":
        return 1
    info = get_gpu_info()
    if info is None:
        return 1
    _, _, recommended = info
    workers = input_default("並列度 (workers)", str(recommended))
    return int(workers)


def run_checked(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"\nコマンド失敗: {e}")
        return False


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _api_post(path: str, data: dict) -> bool:
    """Helper to call tmrvc-serve API."""
    import requests

    try:
        url = f"http://localhost:{SERVE_PORT}{path}"
        resp = requests.post(url, json=data, timeout=10)
        if resp.status_code < 300:
            return True
        print(f"API Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Connection failed: {e}")
    return False


def _api_get(path: str) -> dict | None:
    """Helper to GET from tmrvc-serve API."""
    import requests

    try:
        url = f"http://localhost:{SERVE_PORT}{path}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        print(f"API Error: {resp.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Dataset / cache helpers
# ---------------------------------------------------------------------------
def load_datasets() -> dict:
    if not DATASETS_YAML.exists():
        print(f"ERROR: {DATASETS_YAML} not found. Run option 6 (init configs) first.")
        sys.exit(1)
    with open(DATASETS_YAML) as f:
        return yaml.safe_load(f) or {}


def get_enabled_datasets() -> list[str]:
    cfg = load_datasets()
    return [
        name for name, ds in cfg.get("datasets", {}).items() if ds.get("enabled", False)
    ]


def show_enabled_datasets() -> None:
    enabled = get_enabled_datasets()
    if not enabled:
        print("有効なデータセットがありません。")
    else:
        print("使用するデータセット:")
        for name in enabled:
            print(f"  - {name}")


def list_v4_cache_corpora() -> list[Path]:
    """List available v4 cache corpus directories."""
    if not V4_CACHE_DIR.exists():
        return []
    return sorted(
        [p for p in V4_CACHE_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def list_experiment_cache_dirs() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        [p for p in EXPERIMENTS_DIR.glob("*/cache") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def find_latest_cache_for_enabled_datasets(enabled: list[str]) -> Path | None:
    if not EXPERIMENTS_DIR.exists() or not enabled:
        return None
    prefix = "_".join(sorted(enabled))
    candidates = []
    for exp in EXPERIMENTS_DIR.glob(f"{prefix}_*"):
        cache = exp / "cache"
        if not cache.is_dir():
            continue
        if all((cache / ds / "train").is_dir() for ds in enabled):
            candidates.append(cache)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def find_latest_uclm_checkpoint_for_enabled_datasets(
    enabled: list[str],
) -> tuple[Path | None, Path | None]:
    if not enabled or not EXPERIMENTS_DIR.exists():
        return None, None
    prefix = "_".join(sorted(enabled))
    candidates = [
        (
            (exp / "checkpoints" / "uclm_final.pt").stat().st_mtime,
            exp,
            exp / "checkpoints" / "uclm_final.pt",
        )
        for exp in EXPERIMENTS_DIR.glob(f"{prefix}_*")
        if (exp / "checkpoints" / "uclm_final.pt").exists()
    ]
    if not candidates:
        return None, None
    _, exp_dir, ckpt = max(candidates, key=lambda x: x[0])
    return exp_dir, ckpt


def _quality_gate_status(exp_dir: Path) -> str:
    """Read quality gate status from experiment directory."""
    report = exp_dir / "quality_gate_report.json"
    if not report.exists():
        return "missing"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        return data.get("status", "unknown")
    except Exception:
        return "error"


def _find_latest_experiment_for_enabled_datasets(enabled: list[str]) -> Path | None:
    if not EXPERIMENTS_DIR.exists() or not enabled:
        return None
    prefix = "_".join(sorted(enabled))
    candidates = []
    for exp in EXPERIMENTS_DIR.glob(f"{prefix}_*"):
        cache = exp / "cache"
        if not cache.is_dir():
            continue
        if all((cache / ds / "train").is_dir() for ds in enabled):
            candidates.append(exp)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _promote_uclm_checkpoint(ckpt: Path) -> Path:
    """Copy checkpoint to latest location."""
    dst = CHECKPOINTS_DIR / "uclm" / "uclm_latest.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, dst)
    return dst


def _find_codec_checkpoint() -> Path | None:
    p = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"
    return p if p.exists() else None


def _run_serve_health_smoke(uclm_ckpt: Path, codec_ckpt: Path, device: str) -> bool:
    script = (
        f"from tmrvc_serve.app import init_app, app; from fastapi.testclient import TestClient; "
        f"init_app(uclm_checkpoint={str(uclm_ckpt)!r}, codec_checkpoint={str(codec_ckpt)!r}, device={device!r}); "
        f"client = TestClient(app); resp = client.get('/health'); assert resp.status_code == 200"
    )
    return run_checked(["uv", "run", "python", "-c", script])


# ---------------------------------------------------------------------------
# v4 cache inspection helpers
# ---------------------------------------------------------------------------
def _load_tier_summary(corpus_dir: Path) -> dict[str, int]:
    """Scan a v4 cache corpus and count utterances per supervision tier."""
    tier_counts: dict[str, int] = {"tier_a": 0, "tier_b": 0, "tier_c": 0, "tier_d": 0}
    if not corpus_dir.is_dir():
        return tier_counts
    for meta_path in corpus_dir.rglob("meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tier = meta.get("supervision_tier", "tier_d")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        except Exception:
            continue
    return tier_counts


def _print_tier_summary(tier_counts: dict[str, int]) -> None:
    """Pretty-print supervision tier summary."""
    total = sum(tier_counts.values())
    print(f"\n{'Tier':<10} {'Count':>8} {'Ratio':>8} {'Weight':>8}")
    print("-" * 38)
    for tier in ("tier_a", "tier_b", "tier_c", "tier_d"):
        count = tier_counts.get(tier, 0)
        ratio = f"{count / total * 100:.1f}%" if total > 0 else "0.0%"
        weight = TIER_WEIGHTS.get(tier, 0.0)
        label = tier.replace("_", " ").title()
        print(f"{label:<10} {count:>8} {ratio:>8} {weight:>8.1f}")
    print(f"{'Total':<10} {total:>8}")


def _show_enriched_transcript_preview(corpus_dir: Path, n: int = 5) -> None:
    """Show a preview of enriched transcripts from the cache."""
    print(f"\n--- Enriched Transcript Preview (top {n}) ---")
    shown = 0
    for meta_path in sorted(corpus_dir.rglob("meta.json")):
        if shown >= n:
            break
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            enriched = meta.get("enriched_transcript", "")
            plain = meta.get("text_transcript", "")
            tier = meta.get("supervision_tier", "?")
            if enriched:
                print(f"\n  [{tier}] plain:    {plain}")
                print(f"         enriched: {enriched}")
                shown += 1
        except Exception:
            continue
    if shown == 0:
        print("  (no enriched transcripts found)")


# ---------------------------------------------------------------------------
# Menu display
# ---------------------------------------------------------------------------
def print_menu() -> None:
    clear_screen()
    print("=== TMRVC v4 Development Menu ===")
    print()
    print("1) Bootstrap: raw corpus -> train-ready cache")
    print("2) Training: v4 supervised training (12-D + 24-D + enriched transcript)")
    print("3) RL Fine-tuning: instruction-following RL phase")
    print("4) Dataset Management: corpus listing, tier summary, cache regeneration")
    print("5) Curation: v4 pipeline (ingest -> score -> export -> validate)")
    print("6) Finalize: checkpoint promotion with v4 quality gates")
    print("7) Character Management: few-shot enrollment via backend API")
    print("8) Serve: v4 inference server startup")
    print("9) Integrity Check: v4 contract validation across core/train/export/serve/rust")
    print()
    print("h) 依存関係マップ / 推奨フロー表示")
    print("q) 終了")
    print()


def print_dependency_map() -> None:
    print("\n=== v4 コマンド依存関係マップ ===")
    print()
    print("推奨フロー:")
    print("  1 (bootstrap) -> 4 (tier確認) -> 2 (training) -> 3 (RL) -> 6 (finalize) -> 8 (serve)")
    print()
    print("1) Bootstrap")
    print("  依存: data/raw_corpus/<corpus_id>/ に音声ファイル")
    print("  出力: data/v4_cache/<corpus_id>/  (13-stage pipeline)")
    print()
    print("2) Training (v4 supervised)")
    print("  依存: 1 (v4 cache) or enabled datasets")
    print("  出力: experiments/*/checkpoints/uclm_final.pt")
    print("  備考: 12-D physical + 24-D acting latent + enriched transcript")
    print()
    print("3) RL Fine-tuning")
    print("  依存: 2 (converged supervised checkpoint)")
    print("  出力: checkpoints/rl/rl_step_*.pt")
    print("  備考: PPO, 4 reward objectives, early-stop on quality degradation")
    print()
    print("4) Dataset Management")
    print("  依存: 1+ (v4 cache available)")
    print("  備考: tier summary, enriched transcript preview, cache regeneration")
    print()
    print("5) Curation")
    print("  依存: tmrvc-serve running (API-based)")
    print("  備考: ingest -> score -> export -> validate")
    print()
    print("6) Finalize")
    print("  依存: 2 or 3 (quality gate pass)")
    print("  出力: checkpoints/uclm/uclm_latest.pt")
    print()
    print("7) Character Management")
    print("  依存: codec checkpoint")
    print()
    print("8) Serve")
    print("  依存: 6 (uclm_latest.pt + codec_latest.pt)")
    print()
    print("9) Integrity Check")
    print("  依存: なし")
    print("  備考: ruff + pytest across core/train/export/serve/rust")


# ===========================================================================
# 1. Bootstrap
# ===========================================================================
def cmd_bootstrap() -> None:
    while True:
        clear_screen()
        print("=== 1. Bootstrap: raw corpus -> train-ready cache ===")
        print()

        # Show existing caches
        corpora = list_v4_cache_corpora()
        if corpora:
            print("既存 v4 cache:")
            for p in corpora:
                tier_counts = _load_tier_summary(p)
                total = sum(tier_counts.values())
                print(f"  {p.name:<30} {total:>6} utterances  "
                      f"A={tier_counts['tier_a']} B={tier_counts['tier_b']} "
                      f"C={tier_counts['tier_c']} D={tier_counts['tier_d']}")
            print()

        print("a) 新規コーパスから bootstrap 実行")
        print("b) 既存 bootstrap を resume")
        print("c) Quality gate レポート生成")
        print("d) 戻る")
        choice = input("\n選択: ").strip().lower()

        if choice == "d":
            break
        elif choice == "a":
            _cmd_bootstrap_new()
        elif choice == "b":
            _cmd_bootstrap_resume()
        elif choice == "c":
            _cmd_bootstrap_quality_report()


def _cmd_bootstrap_new() -> None:
    print("\n--- 新規 Bootstrap ---")
    corpus_dir = input_default("コーパスディレクトリ", "data/raw_corpus")
    corpus_id = input_default("コーパスID")
    if not corpus_id:
        print("ERROR: コーパスIDを指定してください。")
        input("\nEnterで戻る...")
        return

    source = Path(corpus_dir) / corpus_id
    if not source.is_dir():
        print(f"ERROR: {source} が見つかりません。")
        input("\nEnterで戻る...")
        return

    device = select_device()
    workers = select_workers(device)

    output_dir = input_default("出力先", str(V4_CACHE_DIR))
    batch_size = input_default("バッチサイズ", "16")

    cmd = [
        "uv", "run", "python", "-m", "tmrvc_data.cli.bootstrap",
        "--corpus-dir", corpus_dir,
        "--corpus-id", corpus_id,
        "--output-dir", output_dir,
        "--device", device,
        "--num-workers", str(workers),
        "--batch-size", batch_size,
    ]

    print(f"\nBootstrap 開始: {corpus_id}")
    print(f"  source: {source}")
    print(f"  output: {output_dir}/{corpus_id}")
    run_checked(cmd)
    input("\nEnterで戻る...")


def _cmd_bootstrap_resume() -> None:
    print("\n--- Bootstrap Resume ---")
    corpora = list_v4_cache_corpora()
    if not corpora:
        print("再開可能な bootstrap セッションがありません。")
        input("\nEnterで戻る...")
        return

    print("再開可能なコーパス:")
    for i, p in enumerate(corpora):
        print(f"  {i + 1}) {p.name}")

    idx = input_default("番号選択", "1")
    try:
        corpus_path = corpora[int(idx) - 1]
    except (ValueError, IndexError):
        print("無効な選択。")
        input("\nEnterで戻る...")
        return

    device = select_device()
    cmd = [
        "uv", "run", "python", "-m", "tmrvc_data.cli.bootstrap",
        "--resume", str(corpus_path),
        "--device", device,
    ]
    run_checked(cmd)
    input("\nEnterで戻る...")


def _cmd_bootstrap_quality_report() -> None:
    print("\n--- Bootstrap Quality Gate Report ---")
    corpora = list_v4_cache_corpora()
    if not corpora:
        print("v4 cache が見つかりません。")
        input("\nEnterで戻る...")
        return

    print("コーパス選択:")
    for i, p in enumerate(corpora):
        print(f"  {i + 1}) {p.name}")

    idx = input_default("番号選択", "1")
    try:
        corpus_path = corpora[int(idx) - 1]
    except (ValueError, IndexError):
        print("無効な選択。")
        input("\nEnterで戻る...")
        return

    cmd = [
        "uv", "run", "python", "-m", "tmrvc_data.cli.bootstrap",
        "--quality-report",
        "--corpus-dir", str(corpus_path.parent),
        "--corpus-id", corpus_path.name,
    ]
    run_checked(cmd)
    input("\nEnterで戻る...")


# ===========================================================================
# 2. Training (v4 supervised)
# ===========================================================================
def cmd_training() -> None:
    while True:
        clear_screen()
        print("=== 2. Training: v4 supervised (12-D + 24-D + enriched transcript) ===")
        print()
        print("a) フル学習 (v4 cache -> supervised training)")
        print("b) 既存キャッシュで学習のみ (skip preprocess)")
        print("c) Codec 学習 (最新 cache から)")
        print("d) 学習設定バリデーション")
        print("e) 戻る")
        choice = input("\n選択: ").strip().lower()

        if choice == "e":
            break
        elif choice == "a":
            _cmd_full_training()
        elif choice == "b":
            _cmd_skip_preprocess()
        elif choice == "c":
            _cmd_train_codec()
        elif choice == "d":
            _cmd_validate_training_config()


def _select_v4_cache() -> Path | None:
    """Interactive v4 cache selection."""
    corpora = list_v4_cache_corpora()

    if not corpora:
        # Fall back to experiment cache
        exp_caches = list_experiment_cache_dirs()
        if not exp_caches:
            print("v4 cache も experiment cache も見つかりません。")
            print("先に Bootstrap (メニュー 1) を実行してください。")
            return None
        print("v4 cache が見つかりません。experiment cache を使用:")
        for i, p in enumerate(exp_caches[:5]):
            print(f"  {i + 1}) {p}")
        idx = input_default("番号選択", "1")
        try:
            return exp_caches[int(idx) - 1]
        except (ValueError, IndexError):
            return None

    print("v4 cache コーパス選択:")
    for i, p in enumerate(corpora):
        tier_counts = _load_tier_summary(p)
        total = sum(tier_counts.values())
        print(f"  {i + 1}) {p.name:<25} ({total} utterances, "
              f"A={tier_counts['tier_a']} B={tier_counts['tier_b']} "
              f"C={tier_counts['tier_c']} D={tier_counts['tier_d']})")

    idx = input_default("番号選択", "1")
    try:
        selected = corpora[int(idx) - 1]
    except (ValueError, IndexError):
        return None

    # Show tier summary for selected corpus
    tier_counts = _load_tier_summary(selected)
    _print_tier_summary(tier_counts)
    return selected


def _cmd_full_training() -> None:
    print("\n--- v4 フル学習 (Mimi codec + 全実モデル) ---")

    # Source: raw audio or existing v4 cache
    print("\nデータソース:")
    print("  a) raw audio からフルパイプライン (bootstrap + train)")
    print("  b) 既存 v4 cache で学習のみ")
    source = input_default("選択", "a").lower()

    device = select_device()

    if source == "a":
        # Full pipeline: bootstrap + train
        sample_pct = input_default("サンプル比率 (%)", "100")
        steps = input_default("学習ステップ数", "10000")
        batch_size = input_default("バッチサイズ", "4")
        annotation_model = input_default(
            "LLM (annotation/enriched transcript)",
            "Qwen/Qwen3.5-4B",
        )
        max_frames = input_default("最大フレーム数 (100Hz control rate)", "400")

        cmd = [
            sys.executable, "scripts/train_v4_full.py",
            "--device", device,
            "--steps", steps,
            "--batch-size", batch_size,
            "--sample-pct", sample_pct,
            "--max-frames", max_frames,
            "--log-every", "50",
            "--save-every", "500",
            "--annotation-model", annotation_model,
            "--output-dir", str(CHECKPOINTS_DIR / "v4_full"),
        ]

        print(f"\nv4 フル学習開始:")
        print(f"  source: raw audio ({sample_pct}% sample)")
        print(f"  steps: {steps}")
        print(f"  codec: Mimi (kyutai/mimi, frozen 12.5 Hz)")
        print(f"  LLM: {annotation_model}")
        print(f"  device: {device}")

    else:
        # Train only from existing cache
        cache_dir = _select_v4_cache()
        if cache_dir is None:
            input("\nEnterで戻る...")
            return

        steps = input_default("学習ステップ数", "10000")
        batch_size = input_default("バッチサイズ", "4")

        cmd = [
            sys.executable, "scripts/train_v4_full.py",
            "--device", device,
            "--steps", steps,
            "--batch-size", batch_size,
            "--skip-cache",
            "--log-every", "50",
            "--save-every", "500",
            "--output-dir", str(CHECKPOINTS_DIR / "v4_full"),
        ]

        print(f"\nv4 学習開始 (キャッシュ済みデータ):")
        print(f"  cache: {cache_dir}")
        print(f"  steps: {steps}")

    if run_checked(cmd):
        # Auto-promote checkpoint
        final_ckpt = CHECKPOINTS_DIR / "v4_full" / "v4_full_final.pt"
        if final_ckpt.exists():
            dst = _promote_uclm_checkpoint(final_ckpt)
            print(f"\nUCLM latest 更新: {dst}")

    input("\nEnterで戻る...")


def _cmd_skip_preprocess() -> None:
    print("\n--- 既存キャッシュで学習のみ ---")

    cache_dir = _select_v4_cache()
    if cache_dir is None:
        input("\nEnterで戻る...")
        return

    device = select_device()

    cmd = [
        "uv", "run", "tmrvc-train-pipeline",
        "--output-dir", "experiments",
        "--cache-dir", str(cache_dir),
        "--skip-preprocess",
        "--train-device", device,
        "--tts-mode", "pointer",
        "--v4-physical-supervision",
        "--v4-acting-latent-supervision",
    ]

    print(f"\n既存キャッシュで学習開始: {cache_dir}")
    if run_checked(cmd):
        cmd_finalize(preferred_device=device)

    input("\nEnterで戻る...")


def _cmd_train_codec(preferred_device: str | None = None) -> bool:
    print("\n--- Codec 学習 ---")

    cache_dir = _select_v4_cache()
    if cache_dir is None:
        input("\nEnterで戻る...")
        return False

    device = preferred_device or select_device()

    cmd = [
        "uv", "run", "tmrvc-train-codec",
        "--cache-dir", str(cache_dir),
        "--output-dir", str(CHECKPOINTS_DIR / "codec"),
        "--device", device,
    ]
    if run_checked(cmd):
        src = CHECKPOINTS_DIR / "codec" / "codec_final.pt"
        dst = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Codec latest 更新: {dst}")
        return True
    return False


def _cmd_validate_training_config() -> None:
    print("\n--- 学習設定バリデーション ---")
    example = _read_yaml(TRAIN_UCLM_YAML)
    if not example:
        print(f"WARNING: {TRAIN_UCLM_YAML} が見つかりません。")
        input("\nEnterで戻る...")
        return

    errors = validate_training_config(example)
    if errors:
        print("バリデーションエラー:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("設定は有効です。")

    print("\nv4 必須フィールド:")
    for key, typ in REQUIRED_TRAINING_FIELDS.items():
        val = example.get(key, "(missing)")
        print(f"  {key}: {val}")

    print("\nv4 オプション:")
    for key, default in OPTIONAL_TRAINING_DEFAULTS.items():
        val = example.get(key, default)
        print(f"  {key}: {val}")

    input("\nEnterで戻る...")


# ===========================================================================
# 3. RL Fine-tuning
# ===========================================================================
def cmd_rl_finetune() -> None:
    while True:
        clear_screen()
        print("=== 3. RL Fine-tuning: instruction-following RL phase ===")
        print()

        # Show RL checkpoint status
        _show_rl_status()

        print()
        print("a) RL fine-tuning 開始 (新規)")
        print("b) RL fine-tuning 再開 (resume)")
        print("c) RL status 詳細表示")
        print("d) 戻る")
        choice = input("\n選択: ").strip().lower()

        if choice == "d":
            break
        elif choice == "a":
            _cmd_rl_start()
        elif choice == "b":
            _cmd_rl_resume()
        elif choice == "c":
            _cmd_rl_status_detail()


def _show_rl_status() -> None:
    """Brief RL status display."""
    if not RL_CHECKPOINTS_DIR.exists():
        print("RL 状態: 未開始")
        return

    rl_ckpts = sorted(RL_CHECKPOINTS_DIR.glob("rl_step_*.pt"))
    if not rl_ckpts:
        print("RL 状態: 未開始")
        return

    latest = rl_ckpts[-1]
    print(f"RL 状態: {len(rl_ckpts)} checkpoint(s)")
    print(f"  最新: {latest.name}")

    # Try to read RL metrics
    metrics_path = RL_CHECKPOINTS_DIR / "rl_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"  step: {metrics.get('step', '?')}")
            print(f"  reward_mean: {metrics.get('reward_mean', '?'):.4f}"
                  if isinstance(metrics.get('reward_mean'), (int, float))
                  else f"  reward_mean: {metrics.get('reward_mean', '?')}")
            print(f"  instruction_following: {metrics.get('instruction_following', '?')}")
            print(f"  plain_text_degradation: {metrics.get('plain_text_degradation', '?')}")
        except Exception:
            pass


def _cmd_rl_start() -> None:
    print("\n--- RL Fine-tuning 開始 ---")

    # Find base checkpoint
    enabled = get_enabled_datasets()
    _, base_ckpt = find_latest_uclm_checkpoint_for_enabled_datasets(enabled)

    if base_ckpt is None:
        # Try latest checkpoint directly
        base_ckpt = CHECKPOINTS_DIR / "uclm" / "uclm_latest.pt"
        if not base_ckpt.exists():
            print("ERROR: supervised training の checkpoint が見つかりません。")
            print("先に Training (メニュー 2) を完了してください。")
            input("\nEnterで戻る...")
            return

    print(f"Base checkpoint: {base_ckpt}")

    device = select_device()

    # RL hyperparameters
    lr = input_default("学習率", "1e-5")
    kl_coeff = input_default("KL penalty coefficient", "0.01")
    max_steps = input_default("最大ステップ数", "1000")
    w_instruction = input_default("reward weight: instruction_following", "1.0")
    w_physical = input_default("reward weight: physical_compliance", "0.5")
    w_intelligibility = input_default("reward weight: intelligibility", "0.3")
    w_naturalness = input_default("reward weight: naturalness", "0.2")
    max_degradation = input_default("plain-text 劣化許容 (%)", "5")

    cmd = [
        "uv", "run", "python", "-m", "tmrvc_train.rl_trainer",
        "--base-checkpoint", str(base_ckpt),
        "--output-dir", str(RL_CHECKPOINTS_DIR),
        "--device", device,
        "--lr", lr,
        "--kl-coeff", kl_coeff,
        "--max-steps", max_steps,
        "--w-instruction", w_instruction,
        "--w-physical", w_physical,
        "--w-intelligibility", w_intelligibility,
        "--w-naturalness", w_naturalness,
        "--max-degradation", str(float(max_degradation) / 100),
    ]

    print(f"\nRL fine-tuning 開始:")
    print(f"  base: {base_ckpt}")
    print(f"  output: {RL_CHECKPOINTS_DIR}")
    print(f"  max steps: {max_steps}")
    print(f"  reward weights: instr={w_instruction} phys={w_physical} "
          f"intel={w_intelligibility} nat={w_naturalness}")
    print(f"  max degradation: {max_degradation}%")

    run_checked(cmd)
    input("\nEnterで戻る...")


def _cmd_rl_resume() -> None:
    print("\n--- RL Fine-tuning Resume ---")

    if not RL_CHECKPOINTS_DIR.exists():
        print("ERROR: RL checkpoint ディレクトリが見つかりません。")
        input("\nEnterで戻る...")
        return

    rl_ckpts = sorted(RL_CHECKPOINTS_DIR.glob("rl_step_*.pt"))
    if not rl_ckpts:
        print("ERROR: 再開可能な RL checkpoint がありません。")
        input("\nEnterで戻る...")
        return

    latest = rl_ckpts[-1]
    print(f"最新 RL checkpoint: {latest.name}")

    device = select_device()
    additional_steps = input_default("追加ステップ数", "500")

    cmd = [
        "uv", "run", "python", "-m", "tmrvc_train.rl_trainer",
        "--resume", str(latest),
        "--output-dir", str(RL_CHECKPOINTS_DIR),
        "--device", device,
        "--additional-steps", additional_steps,
    ]

    run_checked(cmd)
    input("\nEnterで戻る...")


def _cmd_rl_status_detail() -> None:
    print("\n--- RL Fine-tuning Status ---")
    _show_rl_status()

    # Show full metrics history if available
    metrics_log = RL_CHECKPOINTS_DIR / "rl_metrics_history.jsonl"
    if metrics_log.exists():
        print("\n最近の reward 履歴:")
        try:
            lines = metrics_log.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-10:]:  # last 10 entries
                entry = json.loads(line)
                step = entry.get("step", "?")
                reward = entry.get("reward_mean", 0)
                instr = entry.get("instruction_following", 0)
                phys = entry.get("physical_compliance", 0)
                print(f"  step={step:>6}  reward={reward:.4f}  "
                      f"instr={instr:.3f}  phys={phys:.3f}")
        except Exception:
            print("  (履歴の読み込みに失敗)")

    input("\nEnterで戻る...")


# ===========================================================================
# 4. Dataset Management
# ===========================================================================
def cmd_dataset_management() -> None:
    while True:
        clear_screen()
        print("=== 4. Dataset Management ===")
        print()
        print("a) コーパス一覧 (datasets.yaml)")
        print("b) v4 cache コーパス一覧 + tier summary")
        print("c) enriched transcript プレビュー")
        print("d) データセット追加 (対話式)")
        print("e) 設定ファイル初期化")
        print("f) cache 再生成")
        print("g) 戻る")
        choice = input("\n選択: ").strip().lower()

        if choice == "g":
            break
        elif choice == "a":
            _cmd_list_datasets()
        elif choice == "b":
            _cmd_v4_cache_summary()
        elif choice == "c":
            _cmd_enriched_preview()
        elif choice == "d":
            _cmd_add_dataset()
        elif choice == "e":
            _cmd_init_configs()
        elif choice == "f":
            _cmd_regenerate_cache()


def _cmd_list_datasets() -> None:
    cfg = load_datasets()
    datasets = cfg.get("datasets", {})
    print(f"\n{'Name':<25} {'Status':<10} {'Lang':<6} {'Path'}")
    print("-" * 80)
    for name, ds in datasets.items():
        print(
            f"{name:<25} {'enabled' if ds.get('enabled') else 'disabled':<10} "
            f"{ds.get('language', '?'):<6} {ds.get('raw_dir', '')}"
        )
    input("\nEnterで戻る...")


def _cmd_v4_cache_summary() -> None:
    print("\n--- v4 Cache Corpus Summary ---")
    corpora = list_v4_cache_corpora()
    if not corpora:
        print("v4 cache が見つかりません。Bootstrap (メニュー 1) を先に実行してください。")
        input("\nEnterで戻る...")
        return

    for p in corpora:
        print(f"\nCorpus: {p.name}")
        tier_counts = _load_tier_summary(p)
        _print_tier_summary(tier_counts)

        # Count speakers
        speakers = set()
        for meta_path in p.rglob("meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                spk = meta.get("pseudo_speaker_id", "")
                if spk:
                    speakers.add(spk)
            except Exception:
                continue
        print(f"  Pseudo speakers: {len(speakers)}")

    input("\nEnterで戻る...")


def _cmd_enriched_preview() -> None:
    print("\n--- Enriched Transcript Preview ---")
    corpora = list_v4_cache_corpora()
    if not corpora:
        print("v4 cache が見つかりません。")
        input("\nEnterで戻る...")
        return

    print("コーパス選択:")
    for i, p in enumerate(corpora):
        print(f"  {i + 1}) {p.name}")

    idx = input_default("番号選択", "1")
    try:
        corpus_path = corpora[int(idx) - 1]
    except (ValueError, IndexError):
        print("無効な選択。")
        input("\nEnterで戻る...")
        return

    n = int(input_default("表示件数", "10"))
    _show_enriched_transcript_preview(corpus_path, n=n)
    input("\nEnterで戻る...")


def _cmd_add_dataset() -> None:
    run_checked(["uv", "run", "python", "scripts/config_generator.py", "--add-dataset"])
    input("\nEnterで戻る...")


def _cmd_init_configs() -> None:
    run_checked(["uv", "run", "python", "scripts/config_generator.py", "--init"])
    input("\nEnterで戻る...")


def _cmd_regenerate_cache() -> None:
    print("\n--- Cache 再生成 ---")
    corpora = list_v4_cache_corpora()
    if not corpora:
        print("v4 cache が見つかりません。")
        input("\nEnterで戻る...")
        return

    print("再生成するコーパス:")
    for i, p in enumerate(corpora):
        print(f"  {i + 1}) {p.name}")

    idx = input_default("番号選択", "1")
    try:
        corpus_path = corpora[int(idx) - 1]
    except (ValueError, IndexError):
        print("無効な選択。")
        input("\nEnterで戻る...")
        return

    if input(f"'{corpus_path.name}' の cache を再生成しますか? (y/n): ").lower() != "y":
        return

    device = select_device()
    cmd = [
        "uv", "run", "python", "-m", "tmrvc_data.cli.bootstrap",
        "--corpus-dir", str(corpus_path.parent.parent / "raw_corpus"),
        "--corpus-id", corpus_path.name,
        "--output-dir", str(V4_CACHE_DIR),
        "--device", device,
        "--overwrite",
    ]
    run_checked(cmd)
    input("\nEnterで戻る...")


# ===========================================================================
# 5. Curation
# ===========================================================================
def cmd_curation() -> None:
    while True:
        clear_screen()
        print("=== 5. Curation: v4 pipeline ===")
        print()
        print("a) 音声ファイル取込 (ingest)")
        print("b) スコアリング & 昇格判定 (run)")
        print("c) 中断再開 (resume)")
        print("d) エクスポート (promoted -> cache)")
        print("e) 検証レポート")
        print("f) サマリー表示 (status)")
        print("g) 戻る")
        choice = input("\n選択: ").strip().lower()

        if choice == "g":
            break
        elif choice == "a":
            _cmd_curate_ingest()
        elif choice == "b":
            _cmd_curate_run()
        elif choice == "c":
            _cmd_curate_resume()
        elif choice == "d":
            _cmd_curate_export()
        elif choice == "e":
            _cmd_curate_validate()
        elif choice == "f":
            _cmd_curate_status()


def _cmd_curate_ingest() -> None:
    input_dir = input_default("取込対象ディレクトリ")
    ext = input_default("拡張子", ".wav")
    print("\nAPI経由でキュレーション取込を開始します...")
    payload = {"input_dir": input_dir, "extension": ext}
    if _api_post("/ui/curation/jobs/ingest", payload):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def _cmd_curate_run() -> None:
    print("\nAPI経由でスコアリングを開始します...")
    if _api_post("/ui/curation/jobs/run", {}):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def _cmd_curate_resume() -> None:
    print("\nAPI経由でレジュームを開始します...")
    if _api_post("/ui/curation/jobs/resume", {}):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def _cmd_curate_export() -> None:
    export_dir = input_default("エクスポート先", "data/curated_export")
    print(f"\nAPI経由で '{export_dir}' へのエクスポートを開始します...")
    if _api_post("/ui/curation/jobs/export", {"export_dir": export_dir}):
        print("エクスポートジョブを開始しました。")
    input("\nEnterで戻る...")


def _cmd_curate_validate() -> None:
    print("\nAPI経由で検証レポート生成を開始します...")
    if _api_post("/ui/curation/jobs/validate", {}):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def _cmd_curate_status() -> None:
    data = _api_get("/ui/curation/summary")
    if data:
        print("\n=== キュレーションサマリー ===")
        print(json.dumps(data, indent=2))
    input("\nEnterで戻る...")


# ===========================================================================
# 6. Finalize
# ===========================================================================
def cmd_finalize(preferred_device: str | None = None) -> None:
    print("\n=== 6. Finalize: checkpoint promotion with v4 quality gates ===")

    enabled = get_enabled_datasets()

    # Try RL checkpoint first, then supervised
    rl_ckpt = None
    if RL_CHECKPOINTS_DIR.exists():
        rl_ckpts = sorted(RL_CHECKPOINTS_DIR.glob("rl_step_*.pt"))
        if rl_ckpts:
            rl_ckpt = rl_ckpts[-1]

    exp_dir, sup_ckpt = find_latest_uclm_checkpoint_for_enabled_datasets(enabled)

    if rl_ckpt and sup_ckpt:
        print(f"Supervised checkpoint: {sup_ckpt}")
        print(f"RL checkpoint:         {rl_ckpt}")
        use_rl = input_default("RL checkpoint を使用しますか? (y/n)", "y").lower() == "y"
        ckpt_to_promote = rl_ckpt if use_rl else sup_ckpt
    elif rl_ckpt:
        ckpt_to_promote = rl_ckpt
        exp_dir = RL_CHECKPOINTS_DIR
    elif sup_ckpt:
        ckpt_to_promote = sup_ckpt
    else:
        print("checkpoint が見つかりません。先に Training を実行してください。")
        input("\nEnterで戻る...")
        return

    # v4 quality gate check
    if exp_dir is not None:
        qg_status = _quality_gate_status(exp_dir)
        print(f"quality_gate: {qg_status}")
        if qg_status not in ("ok", "missing"):
            proceed = input_default(
                "品質ゲート未通過です。強制的に確定しますか? (y/n)", "n"
            ).lower() == "y"
            if not proceed:
                print("確定をスキップします。")
                input("\nEnterで戻る...")
                return

    # Promote
    dst = _promote_uclm_checkpoint(ckpt_to_promote)
    print(f"UCLM latest 更新: {dst}")

    # Health smoke test
    codec_ckpt = _find_codec_checkpoint()
    if codec_ckpt:
        device = preferred_device or select_device()
        print(f"\n整合性スモーク実行: device={device}")
        _run_serve_health_smoke(dst, codec_ckpt, device)

    input("\nEnterで戻る...")


# ===========================================================================
# 7. Character Management
# ===========================================================================
def load_character_profiles() -> dict:
    if not CHARACTERS_JSON.exists():
        return {}
    try:
        with open(CHARACTERS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_character_profiles(profiles: dict) -> None:
    CHARACTERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CHARACTERS_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def cmd_manage_characters() -> None:
    while True:
        clear_screen()
        print("=== 7. Character Management (Few-shot Enrollment) ===")
        profiles = load_character_profiles()
        if not profiles:
            print("\n登録されているキャラクターはありません。")
        else:
            print(f"\n{'ID':<15} {'Name':<20} {'Adaptation':<10} {'Path'}")
            print("-" * 70)
            for cid, p in profiles.items():
                print(
                    f"{cid:<15} {p.get('name', ''):<20} "
                    f"{p.get('adaptation_level', ''):<10} {p.get('speaker_file', '')}"
                )
        print("\na) 新規キャラクター作成 (Enrollment)")
        print("b) キャラクター削除")
        print("c) 戻る")
        choice = input("\n選択: ").strip().lower()
        if choice == "c":
            break
        elif choice == "a":
            _enroll_character()
        elif choice == "b":
            _delete_character()


def _enroll_character() -> None:
    print("\n--- 新規キャラクター作成 ---")
    char_id = input("キャラクターID: ").strip()
    if not char_id:
        return
    profiles = load_character_profiles()
    if char_id in profiles:
        print(f"ERROR: ID '{char_id}' は既に存在します。")
        input("Enterで戻る...")
        return
    name = input("キャラクター名: ").strip() or char_id
    audio_path = input("参照音声パス: ").strip()
    if not audio_path or not Path(audio_path).exists():
        print("ERROR: パスが見つかりません。")
        input("Enterで戻る...")
        return
    level = input_default("適応レベル (light/standard)", "standard")
    CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CHARACTERS_DIR / f"{char_id}.tmrvc_speaker"
    codec_ckpt = _find_codec_checkpoint()
    cmd = [
        "uv", "run", "tmrvc-enroll",
        "--name", name,
        "--level", level,
        "--output", str(output_file),
    ]
    if Path(audio_path).is_dir():
        cmd.extend(["--audio-dir", audio_path])
    else:
        cmd.extend(["--audio", audio_path])
    if level != "light" and codec_ckpt:
        cmd.extend(["--codec-checkpoint", str(codec_ckpt)])
    if run_checked(cmd):
        profiles[char_id] = {
            "name": name,
            "speaker_file": str(output_file),
            "adaptation_level": level,
        }
        save_character_profiles(profiles)
        print(f"\nキャラクター '{char_id}' を作成しました。")
    input("\nEnterで戻る...")


def _delete_character() -> None:
    char_id = input("\n削除するキャラクターID: ").strip()
    if not char_id:
        return
    profiles = load_character_profiles()
    if char_id not in profiles:
        return
    if input(f"本当に '{char_id}' を削除しますか? (y/n): ").lower() == "y":
        p = profiles.pop(char_id)
        save_character_profiles(profiles)
        sp_file = Path(p.get("speaker_file", ""))
        if (
            sp_file.exists()
            and input(f"ファイル '{sp_file.name}' も削除しますか? (y/n): ").lower()
            == "y"
        ):
            sp_file.unlink()
    input("Enterで戻る...")


# ===========================================================================
# 8. Serve
# ===========================================================================
def cmd_run_serve() -> None:
    print("\n--- v4 推論サーバー起動 ---")
    uclm_ckpt = CHECKPOINTS_DIR / "uclm" / "uclm_latest.pt"
    codec_ckpt = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"

    if not uclm_ckpt.exists():
        print(f"WARNING: {uclm_ckpt} が見つかりません。")
    if not codec_ckpt.exists():
        print(f"WARNING: {codec_ckpt} が見つかりません。")

    device = select_device()
    host = input_default("ホスト", "127.0.0.1")
    port = input_default("ポート番号", str(SERVE_PORT))
    api_key = input_default(
        "LLM Backend API Key (context予測用, 空欄可)",
        os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    use_reload = (
        input_default("オートリロードを有効にしますか? (y/n)", "n").lower() == "y"
    )
    use_verbose = input_default("詳細ログを出力しますか? (y/n)", "n").lower() == "y"

    cmd = [
        "uv", "run", "tmrvc-serve",
        "--uclm-checkpoint", str(uclm_ckpt),
        "--codec-checkpoint", str(codec_ckpt),
        "--device", device,
        "--host", host,
        "--port", port,
    ]

    if api_key:
        cmd.extend(["--api-key", api_key])
    if use_reload:
        cmd.append("--reload")
    if use_verbose:
        cmd.append("--verbose")

    print(f"\nサーバーを起動します: http://{host}:{port}")
    print("Ctrl+C で停止します。\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nサーバーを停止しました。")

    input("\nEnterで戻る...")


# ===========================================================================
# 9. Integrity Check
# ===========================================================================
def cmd_integrity_check() -> None:
    print("\n" + "=" * 60)
    print("  v4 Contract Integrity Check")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    # 1. Ruff
    print("\n[1/5] 静的解析 (Ruff)...")
    ok = run_checked(["uv", "run", "ruff", "check", "."])
    results.append(("Ruff", ok))

    # 2. Core tests
    print("\n[2/5] Core contract tests...")
    ok = run_checked([
        "uv", "run", "pytest", "tests/test_constants_contract.py",
        "tests/test_schema_roundtrip.py", "-v",
    ])
    results.append(("Core contracts", ok))

    # 3. Training tests
    print("\n[3/5] Training contract tests...")
    ok = run_checked([
        "uv", "run", "pytest",
        "tests/train/test_cfg_contract.py",
        "tests/train/test_expressive_features.py",
        "tests/train/test_v4_training.py",
        "-v",
    ])
    results.append(("Training contracts", ok))

    # 4. Serve tests
    print("\n[4/5] Serve contract tests...")
    ok = run_checked([
        "uv", "run", "pytest",
        "tests/serve/test_pointer_serving.py",
        "tests/serve/test_trajectory_flow.py",
        "-v",
    ])
    results.append(("Serve contracts", ok))

    # 5. Bootstrap / validation tests
    print("\n[5/5] Bootstrap & validation tests...")
    ok = run_checked([
        "uv", "run", "pytest",
        "tests/test_bootstrap_quality.py",
        "tests/test_v4_validation_gates.py",
        "tests/test_controllability.py",
        "-v",
    ])
    results.append(("Bootstrap/validation", ok))

    # Rust check (optional)
    print("\n[bonus] Rust tests (if available)...")
    rust_ok = run_checked(["cargo", "test", "--manifest-path", "tmrvc-engine-rs/Cargo.toml"])
    results.append(("Rust engine", rust_ok))

    # Summary
    print("\n" + "-" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nすべてのチェックを通過しました。")
    else:
        print("\nエラーが検出されました。ログを確認してください。")

    input("\nEnterで戻る...")


# ===========================================================================
# Main loop
# ===========================================================================
def main() -> None:
    while True:
        print_menu()
        choice = input("選択 [1-9, h=ヘルプ, q=終了]: ").strip()
        if choice == "q":
            break
        if choice == "h":
            print_dependency_map()
            input("\nEnterで続行...")
            continue
        handlers = {
            "1": cmd_bootstrap,
            "2": cmd_training,
            "3": cmd_rl_finetune,
            "4": cmd_dataset_management,
            "5": cmd_curation,
            "6": cmd_finalize,
            "7": cmd_manage_characters,
            "8": cmd_run_serve,
            "9": cmd_integrity_check,
        }
        if choice in handlers:
            handlers[choice]()
        else:
            input(f"無効な選択: {choice}. Enterで続行...")


if __name__ == "__main__":
    # Non-interactive CLI: python dev.py <cmd> [args...]
    if len(sys.argv) > 1 and sys.argv[1] != "--help":
        subcmd = sys.argv[1]

        if subcmd in ("bootstrap", "1"):
            # Idempotent bootstrap — delegates to bootstrap_v4.py
            cmd = [sys.executable, "scripts/bootstrap_v4.py"] + sys.argv[2:]
            sys.exit(0 if run_checked(cmd) else 1)

        elif subcmd in ("train", "2"):
            # Train only — NO bootstrap, uses existing cache
            import argparse
            p = argparse.ArgumentParser(prog="dev.py train")
            p.add_argument("--device", default="cuda")
            p.add_argument("--steps", default="10000")
            p.add_argument("--batch-size", default="4")
            p.add_argument("--output-dir", default=str(CHECKPOINTS_DIR / "v4_full"))
            args = p.parse_args(sys.argv[2:])

            cmd = [
                sys.executable, "scripts/train_v4_full.py",
                "--device", args.device,
                "--steps", args.steps,
                "--batch-size", args.batch_size,
                "--skip-cache",
                "--log-every", "50",
                "--save-every", "500",
                "--output-dir", args.output_dir,
            ]

            print(f"v4 Train (cache must exist)")
            print(f"  device={args.device} steps={args.steps} batch={args.batch_size}")

            ok = run_checked(cmd)
            if ok:
                final = Path(args.output_dir) / "v4_full_final.pt"
                if final.exists():
                    dst = _promote_uclm_checkpoint(final)
                    print(f"UCLM latest: {dst}")
            sys.exit(0 if ok else 1)

        elif subcmd == "status":
            # Show bootstrap status
            cmd = [sys.executable, "scripts/bootstrap_v4.py", "--status"]
            run_checked(cmd)
            sys.exit(0)

        elif subcmd in ("rl", "3"):
            import argparse as _ap
            p = _ap.ArgumentParser(prog="dev.py rl")
            p.add_argument("--base-checkpoint", default=str(CHECKPOINTS_DIR / "uclm" / "uclm_latest.pt"))
            p.add_argument("--output-dir", default=str(RL_CHECKPOINTS_DIR))
            p.add_argument("--device", default="cuda")
            p.add_argument("--lr", default="1e-5")
            p.add_argument("--kl-coeff", default="0.01")
            p.add_argument("--max-steps", default="5000")
            p.add_argument("--max-degradation", default="0.05")
            p.add_argument("--resume", default=None)
            args = p.parse_args(sys.argv[2:])

            cmd = [
                sys.executable, "-m", "tmrvc_train.rl_trainer",
                "--base-checkpoint", args.base_checkpoint,
                "--output-dir", args.output_dir,
                "--device", args.device,
                "--lr", args.lr,
                "--kl-coeff", args.kl_coeff,
                "--max-steps", args.max_steps,
                "--max-degradation", args.max_degradation,
            ]
            if args.resume:
                cmd += ["--resume", args.resume]

            print(f"v4 RL fine-tuning")
            print(f"  base: {args.base_checkpoint}")
            print(f"  output: {args.output_dir}")
            sys.exit(0 if run_checked(cmd) else 1)

        elif subcmd in ("integrity", "9"):
            cmd_integrity_check()
            sys.exit(0)

        else:
            print(f"Unknown subcommand: {subcmd}")
            print("Available: bootstrap, train, rl, status, integrity")
            sys.exit(1)
    else:
        main()
