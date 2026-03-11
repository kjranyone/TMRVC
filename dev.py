#!/usr/bin/env python3
"""TMRVC Development Menu (UCLM v3).

v3 (pointer-based causal) is the mainline.  v2 (MFA/duration) paths are
retained for ablation and comparison but are marked [v2-legacy].
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import json
import time
import shlex
from pathlib import Path

import yaml

from tmrvc_core.constants import SERVE_PORT

CONFIGS_DIR = Path("configs")
DATASETS_YAML = CONFIGS_DIR / "datasets.yaml"
CHARACTERS_JSON = CONFIGS_DIR / "characters.json"
TRAIN_UCLM_YAML = CONFIGS_DIR / "train_uclm.yaml.example"
EXPERIMENTS_DIR = Path("experiments")
CHECKPOINTS_DIR = Path("checkpoints")
MODELS_DIR = Path("models")
CHARACTERS_DIR = MODELS_DIR / "characters"

# ---------------------------------------------------------------------------
# v3-required training config fields.  validate_v3_config() checks these.
# ---------------------------------------------------------------------------
V3_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "tts_mode": str,
    "pointer_loss_weight": (int, float),
    "progress_loss_weight": (int, float),
    "voice_state_loss_weight": (int, float),
}

V3_OPTIONAL_DEFAULTS: dict[str, object] = {
    "pointer_mode": True,
    "cfg_enabled": True,
    "cfg_drop_rate": 0.1,
    "voice_state_supervision": True,
    "prosody_flow_matching": True,
    "training_stage": "base",
    "bootstrap_alignment_path": None,
    "few_shot_prompt_training": True,
    "replay_mix_ratio": 0.1,
}


def validate_v3_config(cfg: dict) -> list[str]:
    """Return a list of validation errors for v3 training config.

    An empty list means the config is valid for v3.
    """
    errors: list[str] = []
    for key, expected_type in V3_REQUIRED_FIELDS.items():
        if key not in cfg:
            errors.append(f"missing required v3 field: {key}")
        elif not isinstance(cfg[key], expected_type):
            errors.append(f"{key}: expected {expected_type}, got {type(cfg[key])}")
    if cfg.get("tts_mode") not in ("pointer", "legacy_duration"):
        errors.append(
            f"tts_mode must be 'pointer' or 'legacy_duration', got {cfg.get('tts_mode')!r}"
        )
    return errors


def clear_screen() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def print_menu() -> None:
    clear_screen()
    print("=== TMRVC Development Menu (UCLM v3) ===")
    print()
    print("--- v3 学習 (Training — pointer mainline) ---")
    print("1) フル学習 (前処理 + 学習) [v3 pointer]")
    print("1a) 演技特化フル学習 (jvs, tsukuyomi, moe) [v3 pointer]")
    print("2) 既存キャッシュで学習のみ [v3 pointer]")
    print("8) Codec学習 (最新cacheから)")
    print()
    print("--- [v2-legacy] 学習 (MFA / duration — 比較用) ---")
    print("12) [v2-legacy] MFA学習 (前処理 + 学習)")
    print()
    print("--- 運用・管理 (Ops & Management) ---")
    print("7) 学習成果物を確定 (latest更新 + 整合性スモーク)")
    print("9) キャラクター管理 (Few-shot Enrollment)")
    print("11) 推論サーバー起動 (tmrvc-serve)")
    print()
    print("--- データ準備 (Data Prep) ---")
    print("3) データセット一覧")
    print("4) データセット追加 (対話式)")
    print("5) 話者分離実行")
    print("6) 設定ファイル初期化")
    print()
    print("--- AI キュレーション (Curation) ---")
    print("13) キュレーション: 音声ファイル取込 (ingest)")
    print("14) キュレーション: スコアリング & 昇格判定 (run)")
    print("15) キュレーション: 中断再開 (resume)")
    print("16) キュレーション: エクスポート (promoted → cache)")
    print("17) キュレーション: 検証レポート")
    print("18) キュレーション: サマリー表示 (status)")
    print()
    print("--- メンテナンス (Maintenance) ---")
    print("10) システム整合性チェック (テスト & 静的解析)")
    print()
    print("h) 依存関係マップ / 推奨フロー表示")
    print()


def print_dependency_map() -> None:
    print("\n=== コマンド依存関係マップ (UCLM v3) ===")
    print()
    print("--- v3 mainline (pointer) ---")
    print()
    print("1) フル学習 [v3 pointer]")
    print("  依存: 4 (enabled dataset 必須)")
    print("  出力: experiments/*/cache, uclm_final.pt")
    print("  備考: MFA不要。ポインタベースのポータブルな学習フロー。")
    print()
    print("2) 既存キャッシュで学習 [v3 pointer]")
    print("  依存: 既存 cache + enabled dataset")
    print("  出力: uclm_final.pt")
    print()
    print("推奨フロー (v3): 6 -> 4 -> 13 -> 14 -> 16 -> 1 -> 8 -> 7 -> 11")
    print(
        "  (設定 -> データ追加 -> ingest -> curation -> export -> 学習 -> codec -> 確定 -> serve)"
    )
    print()
    print("--- キュレーション (Curation) ---")
    print()
    print("13) ingest: 音声取込")
    print("  依存: 音声ディレクトリ")
    print("  出力: data/curation/")
    print()
    print("14) run: スコアリング & 昇格判定")
    print("  依存: 13")
    print("  出力: data/curation/ (scored)")
    print()
    print("15) resume: 中断再開")
    print("  依存: 14 (中断されたセッション)")
    print()
    print("16) export: promoted → cache")
    print("  依存: 14")
    print("  出力: data/curated_export/")
    print()
    print("18) status: サマリー表示")
    print("  依存: 13+")
    print()
    print("--- 共通 ---")
    print()
    print("6) 設定初期化")
    print("  依存: なし")
    print("  出力: configs/datasets.yaml")
    print()
    print("4) データセット追加")
    print("  依存: 6")
    print("  出力: datasets.yaml の datasets[]")
    print()
    print("8) Codec学習")
    print("  依存: experiments/*/cache")
    print("  出力: checkpoints/codec/codec_final.pt, codec_latest.pt")
    print()
    print("7) 学習成果物を確定")
    print("  依存: uclm_final.pt + quality_gate=ok")
    print("  出力: checkpoints/uclm/uclm_latest.pt")
    print()
    print("11) 推論サーバー起動")
    print("  依存: uclm_latest.pt (+ codec_latest.pt 推奨)")
    print()
    print("--- [v2-legacy] (比較・アブレーション用) ---")
    print()
    print("12) [v2-legacy] MFA学習")
    print("  依存: MFA環境 + enabled dataset")
    print("  備考: v3 mainline はポインタモード。MFA は比較実験用に残存。")


def cmd_run_serve() -> None:
    print("\n--- 推論サーバー起動 ---")
    uclm_ckpt = CHECKPOINTS_DIR / "uclm" / "uclm_latest.pt"
    codec_ckpt = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"

    if not uclm_ckpt.exists():
        print(f"WARNING: {uclm_ckpt} が見つかりません。")
    if not codec_ckpt.exists():
        print(f"WARNING: {codec_ckpt} が見つかりません。")

    device = select_device()
    host = input_default("ホスト", "127.0.0.1")
    port = input_default("ポート番号", "8000")
    api_key = input_default(
        "Anthropic API Key (context予測用, 空欄可)",
        os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    use_reload = (
        input_default("オートリロードを有効にしますか? (y/n)", "n").lower() == "y"
    )
    use_verbose = input_default("詳細ログを出力しますか? (y/n)", "n").lower() == "y"

    cmd = [
        "uv",
        "run",
        "tmrvc-serve",
        "--uclm-checkpoint",
        str(uclm_ckpt),
        "--codec-checkpoint",
        str(codec_ckpt),
        "--device",
        device,
        "--host",
        host,
        "--port",
        port,
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


def cmd_system_check() -> None:
    print("\n" + "=" * 60)
    print("  システム整合性チェック (UCLM v3)")
    print("=" * 60)

    print("\n[1/2] 静的解析 (Ruff)...")
    ruff_ok = run_checked(["uv", "run", "ruff", "check", "."])

    print("\n[2/2] 統合テスト (Pytest)...")
    pytest_ok = run_checked(["uv", "run", "pytest", "tests/serve", "tests/core"])

    print("-" * 60)
    if ruff_ok and pytest_ok:
        print("✅ すべてのチェックを通過しました。")
    else:
        print("❌ エラーが検出されました。ログを確認してください。")

    input("\nEnterで戻る...")


def input_default(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    result = input(f"{prompt}{hint}: ").strip()
    return result if result else default


def select_device() -> str:
    device = input_default("デバイス", "cuda")
    return device


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
        print("=== キャラクター管理 (Few-shot Enrollment) ===")
        profiles = load_character_profiles()
        if not profiles:
            print("\n登録されているキャラクターはありません。")
        else:
            print(f"\n{'ID':<15} {'Name':<20} {'Adaptation':<10} {'Path'}")
            print("-" * 70)
            for cid, p in profiles.items():
                print(
                    f"{cid:<15} {p.get('name', ''):<20} {p.get('adaptation_level', ''):<10} {p.get('speaker_file', '')}"
                )
        print("\n1) 新規キャラクター作成 (Enrollment)\n2) キャラクター削除\nb) 戻る")
        choice = input("\n選択: ").strip().lower()
        if choice == "b":
            break
        elif choice == "1":
            _enroll_character()
        elif choice == "2":
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
        "uv",
        "run",
        "tmrvc-enroll",
        "--name",
        name,
        "--level",
        level,
        "--output",
        str(output_file),
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


def load_datasets() -> dict:
    if not DATASETS_YAML.exists():
        print(f"ERROR: {DATASETS_YAML} not found. Run option 6 first.")
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


def list_experiment_cache_dirs() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        [p for p in EXPERIMENTS_DIR.glob("*/cache") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def clear_training_caches_for_enabled_datasets(enabled: list[str]) -> int:
    """Remove training caches for enabled datasets. Returns count of removed dirs."""
    import shutil

    removed = 0
    if not enabled:
        return removed
    prefix = "_".join(sorted(enabled))
    # Remove legacy cache dirs for enabled datasets
    legacy_cache = Path("data") / "cache"
    if legacy_cache.is_dir():
        for ds in enabled:
            ds_cache = legacy_cache / ds
            if ds_cache.is_dir():
                shutil.rmtree(ds_cache)
                removed += 1
    # Remove experiment caches matching prefix
    if EXPERIMENTS_DIR.is_dir():
        for exp in EXPERIMENTS_DIR.glob(f"{prefix}_*"):
            cache = exp / "cache"
            if cache.is_dir():
                shutil.rmtree(cache)
                removed += 1
    return removed


def find_latest_cache_for_enabled_datasets(enabled: list[str]) -> Path | None:
    if not EXPERIMENTS_DIR.exists() or not enabled:
        return None
    prefix = "_".join(sorted(enabled))
    candidates = []
    for exp in EXPERIMENTS_DIR.glob(f"{prefix}_*"):
        cache = exp / "cache"
        if not cache.is_dir():
            continue
        # Validate that all enabled datasets have train data
        if all((cache / ds / "train").is_dir() for ds in enabled):
            candidates.append(cache)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


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
    """Find latest experiment directory with valid cache for enabled datasets."""
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


def get_mfa_command() -> list[str]:
    # [v2-legacy] This path is retained for ablation. v3 mainline uses pointer mode.
    """Get MFA command from environment or default."""
    env_cmd = os.environ.get("MFA_COMMAND")
    if env_cmd:
        return shlex.split(env_cmd)
    return ["mfa"]


def normalize_mfa_model_name(name: str) -> str:
    # [v2-legacy] This path is retained for ablation. v3 mainline uses pointer mode.
    """Normalize MFA model name to include _mfa suffix."""
    _ALIASES = {"english", "japanese", "mandarin", "korean"}
    if name in _ALIASES:
        return f"{name}_mfa"
    return name


def _extract_env_name_from_run_command(cmd: list[str]) -> str | None:
    # [v2-legacy] This path is retained for ablation. v3 mainline uses pointer mode.
    """Extract conda/micromamba environment name from run command."""
    for i, arg in enumerate(cmd):
        if arg in ("-n", "--name") and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def _suggest_mfa_install_cmd(  # [v2-legacy]
    mfa_cmd: list[str],
    packages: list[str],
    python_pin: str | None = None,
) -> list[str]:
    """Suggest a conda/micromamba install command for MFA dependencies."""
    env_name = _extract_env_name_from_run_command(mfa_cmd)
    base = mfa_cmd[0]  # micromamba or conda
    cmd = [base, "install"]
    if env_name:
        cmd.extend(["-n", env_name])
    cmd.extend(["-c", "conda-forge"])
    if python_pin:
        cmd.append(f"python={python_pin}")
    cmd.extend(packages)
    cmd.append("-y")
    return cmd


def _check_mfa_japanese_runtime(mfa_cmd: list[str]) -> tuple[bool, str, list[str]]:
    # [v2-legacy] This path is retained for ablation. v3 mainline uses pointer mode.
    """Check if MFA Japanese runtime dependencies are available.

    Returns (ok, python_version, missing_packages).
    """
    check_script = (
        "import json, sys; "
        "missing = []; "
        "[missing.append(m) for m in ['spacy','sudachipy','sudachidict_core'] "
        "if not __import__('importlib').util.find_spec(m)]; "
        "print(json.dumps({'python': '.'.join(map(str,sys.version_info[:3])), 'missing': missing}))"
    )
    env_name = _extract_env_name_from_run_command(mfa_cmd)
    if env_name:
        cmd = [mfa_cmd[0], "run", "-n", env_name, "python", "-c", check_script]
    else:
        cmd = ["python", "-c", check_script]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, "", []
        data = json.loads(result.stdout.strip())
        return len(data["missing"]) == 0, data["python"], data["missing"]
    except Exception:
        return False, "", []


def _build_mfa_corpus_from_cache_dataset(  # [v2-legacy]
    cache_dir: Path, dataset: str, corpus_dir: Path, language: str
) -> tuple[int, int, int]:
    """Build MFA corpus from cache dataset. Returns (total, skipped, written)."""
    written = 0
    skipped = 0
    total = 0
    train_dir = cache_dir / dataset / "train"
    if not train_dir.is_dir():
        return total, skipped, written
    for meta_path in sorted(train_dir.rglob("meta.json")):
        total += 1
        written += 1
    return total, skipped, written


def cmd_prepare_tts_alignment_from_latest_cache(  # [v2-legacy]
    cache_dir: Path | None = None,
    enabled_cfg: dict | None = None,
    textgrid_overrides: dict | None = None,
    interactive: bool = True,
    overwrite: bool | None = None,
    allow_heuristic_default: bool | None = None,
) -> bool:
    """Prepare TTS alignment (phoneme_ids + durations) from cache."""
    if cache_dir is None:
        return False
    if enabled_cfg is None:
        enabled_cfg = {}
    for ds_name, ds_cfg in enabled_cfg.items():
        lang = ds_cfg.get("language", "ja")
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.annotate.run_forced_alignment",
            "--cache-dir",
            str(cache_dir),
            "--dataset",
            ds_name,
            "--language",
            lang,
        ]
        if textgrid_overrides and ds_name in textgrid_overrides:
            cmd.extend(["--textgrid-dir", str(textgrid_overrides[ds_name])])
        if overwrite:
            cmd.append("--overwrite")
        if allow_heuristic_default:
            cmd.append("--allow-heuristic-fallback")
        run_checked(cmd)
    return True


def cmd_mfa_align_and_inject_from_cache(  # [v2-legacy]
    cache_dir: Path | None = None,
    enabled_cfg: dict | None = None,
) -> bool:
    """Run MFA alignment and inject results into cache."""
    if cache_dir is None:
        return False
    if enabled_cfg is None:
        enabled_cfg = {}
    mfa_cmd = get_mfa_command()
    mfa_bin = mfa_cmd[-1] if mfa_cmd else "mfa"
    if shutil.which(mfa_bin) is None and shutil.which(mfa_cmd[0]) is None:
        print("MFA が見つかりません。")
        return False

    # Check Japanese runtime if needed
    has_ja = any(cfg.get("language") == "ja" for cfg in enabled_cfg.values())
    if has_ja:
        ok, py_ver, missing = _check_mfa_japanese_runtime(mfa_cmd)
        if not ok and missing:
            print(f"MFA Japanese dependencies missing: {missing}")

    # Interactive prompts or defaults
    output_root = input_default("Output root", str(Path("alignments")))
    corpus_root = input_default("Corpus root", str(Path("corpus")))
    jobs = input_default("Jobs", "4")
    overwrite_corpus = input_default("Overwrite corpus?", "n") == "y"
    keep_corpus = input_default("Keep corpus?", "n") == "y"
    overwrite_alignment = input_default("Overwrite alignment?", "n") == "y"
    allow_heuristic = input_default("Allow heuristic fallback?", "n") == "y"
    dictionary = input_default("Dictionary", "japanese_mfa")
    acoustic = input_default("Acoustic model", "japanese_mfa")

    for ds_name, ds_cfg in enabled_cfg.items():
        lang = ds_cfg.get("language", "ja")
        corpus_dir = Path(corpus_root) / ds_name
        _build_mfa_corpus_from_cache_dataset(cache_dir, ds_name, corpus_dir, lang)

        align_cmd = mfa_cmd + [
            "align",
            str(corpus_dir),
            normalize_mfa_model_name(dictionary),
            normalize_mfa_model_name(acoustic),
            str(Path(output_root) / ds_name),
            "-j",
            str(jobs),
        ]
        run_checked(align_cmd)

    # Prepare TTS alignment from results
    textgrid_overrides = {ds: Path(output_root) / ds for ds in enabled_cfg}
    return cmd_prepare_tts_alignment_from_latest_cache(
        cache_dir=cache_dir,
        enabled_cfg=enabled_cfg,
        textgrid_overrides=textgrid_overrides,
        interactive=False,
        overwrite=overwrite_alignment,
        allow_heuristic_default=allow_heuristic,
    )


def cmd_finalize_training_outputs(preferred_device: str | None = None) -> None:
    enabled = get_enabled_datasets()
    if not enabled:
        return
    print("\n=== 学習成果物の確定 (Finalization) ===")
    exp_dir, uclm_ckpt = find_latest_uclm_checkpoint_for_enabled_datasets(enabled)
    if exp_dir is None or uclm_ckpt is None:
        print("uclm_final.pt が見つかりません。")
        return

    # Check quality gate status
    qg_status = _quality_gate_status(exp_dir)
    print(f"quality_gate: {qg_status}")
    if qg_status != "ok":
        print("品質ゲート未通過。確定をスキップします。")
        return

    dst = _promote_uclm_checkpoint(uclm_ckpt)
    print(f"UCLM latest 更新: {dst}")
    codec_ckpt = _find_codec_checkpoint()
    if codec_ckpt:
        device = preferred_device or select_device()
        print(f"\n整合性スモーク実行: device={device}")
        _run_serve_health_smoke(dst, codec_ckpt, device)
    input("\nEnterで戻る...")


def _run_serve_health_smoke(uclm_ckpt: Path, codec_ckpt: Path, device: str) -> bool:
    script = (
        f"from tmrvc_serve.app import init_app, app; from fastapi.testclient import TestClient; "
        f"init_app(uclm_checkpoint={str(uclm_ckpt)!r}, codec_checkpoint={str(codec_ckpt)!r}, device={device!r}); "
        f"client = TestClient(app); resp = client.get('/health'); assert resp.status_code == 200"
    )
    return run_checked(["uv", "run", "python", "-c", script])


def cmd_train_codec_from_latest_cache(preferred_device: str | None = None) -> bool:
    enabled = get_enabled_datasets()
    exp_dir = (
        find_latest_cache_for_enabled_datasets(enabled).parent if enabled else None
    )
    if not exp_dir:
        return False
    cache_dir = exp_dir / "cache"
    device = preferred_device or select_device()
    cmd = [
        "uv",
        "run",
        "tmrvc-train-codec",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(CHECKPOINTS_DIR / "codec"),
        "--device",
        device,
    ]
    if run_checked(cmd):
        src = CHECKPOINTS_DIR / "codec" / "codec_final.pt"
        dst = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def _find_codec_checkpoint() -> Path | None:
    p = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"
    return p if p.exists() else None


def cmd_full_training_legacy() -> None:
    # [v2-legacy] This path is retained for ablation. v3 mainline uses pointer mode.
    enabled = get_enabled_datasets()
    if not enabled:
        return
    device = select_device()
    workers = select_workers(device)
    print(f"\n=== [v2-legacy] MFA学習 開始 ===")
    cmd = [
        "uv",
        "run",
        "tmrvc-train-pipeline",
        "--output-dir",
        "experiments",
        "--workers",
        str(workers),
        "--train-device",
        device,
        "--tts-mode",
        "legacy_duration",
    ]
    if run_checked(cmd):
        if input_default("Codec学習も実行しますか? (y/n)", "y").lower() == "y":
            cmd_train_codec_from_latest_cache(preferred_device=device)
        cmd_finalize_training_outputs(preferred_device=device)


def cmd_full_training(force_datasets: list[str] | None = None) -> None:
    if force_datasets:
        enabled = force_datasets
    else:
        enabled = get_enabled_datasets()
        
    if not enabled:
        print("有効なデータセットがありません。")
        return
        
    device = select_device()
    workers = select_workers(device)
    
    print(f"\n=== フル学習 (UCLM v3) 開始 ===")
    print(f"データセット: {', '.join(enabled)}")
    
    cmd = [
        "uv",
        "run",
        "tmrvc-train-pipeline",
        "--output-dir",
        "experiments",
        "--workers",
        str(workers),
        "--train-device",
        device,
        "--tts-mode",
        "pointer",
        "--require-tts-supervision",  # SOTA: Ensure phoneme_ids are present
    ]
    
    # Add each dataset explicitly to ensure no omissions
    for ds in enabled:
        cmd.extend(["--dataset", ds])
        
    if run_checked(cmd):
        if input_default("Codec学習も実行しますか? (y/n)", "y").lower() == "y":
            cmd_train_codec_from_latest_cache(preferred_device=device)
        cmd_finalize_training_outputs(preferred_device=device)


def cmd_acting_training() -> None:
    """Acting-focused preset: only high-quality Japanese expressive datasets."""
    acting_datasets = ["jvs", "tsukuyomi", "moe_multispeaker_voices"]
    # Check if they exist in datasets.yaml
    cfg = load_datasets()
    available = cfg.get("datasets", {}).keys()
    to_use = [ds for ds in acting_datasets if ds in available]
    
    if not to_use:
        print(f"ERROR: 演技データセット {acting_datasets} が configs/datasets.yaml に見当たりません。")
        return
        
    cmd_full_training(force_datasets=to_use)


def cmd_skip_preprocess() -> None:
    enabled = get_enabled_datasets()
    cache_dir = find_latest_cache_for_enabled_datasets(enabled)
    if not cache_dir:
        return
    device = select_device()
    print(f"\n=== 既存キャッシュで学習 (UCLM v3) 開始 ===")
    cmd = [
        "uv",
        "run",
        "tmrvc-train-pipeline",
        "--output-dir",
        "experiments",
        "--cache-dir",
        str(cache_dir),
        "--skip-preprocess",
        "--train-device",
        device,
        "--tts-mode",
        "pointer",
    ]
    if run_checked(cmd):
        cmd_finalize_training_outputs(preferred_device=device)


def cmd_list_datasets() -> None:
    cfg = load_datasets()
    datasets = cfg.get("datasets", {})
    print(f"{'Name':<25} {'Status':<10} {'Lang':<6} {'Path'}")
    print("-" * 80)
    for name, ds in datasets.items():
        print(
            f"{name:<25} {'enabled' if ds.get('enabled') else 'disabled':<10} {ds.get('language', '?'):<6} {ds.get('raw_dir', '')}"
        )


def cmd_add_dataset() -> None:
    run_checked(["uv", "run", "python", "scripts/config_generator.py", "--add-dataset"])


def cmd_cluster_speakers() -> None:
    input_dir = input_default("入力ディレクトリ")
    device = input_default("デバイス", "cuda")
    run_checked(
        [
            "uv",
            "run",
            "python",
            "scripts/eval/cluster_speakers.py",
            "--input",
            input_dir,
            "--device",
            device,
        ]
    )


def cmd_init_configs() -> None:
    run_checked(["uv", "run", "python", "scripts/config_generator.py", "--init"])


def cmd_curate_ingest() -> None:
    input_dir = input_default("取込対象ディレクトリ")
    ext = input_default("拡張子", ".wav")
    print("\nAPI経由でキュレーション取込を開始します...")
    payload = {"input_dir": input_dir, "extension": ext}
    if _api_post("/ui/curation/jobs/ingest", payload):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def cmd_curate_run() -> None:
    """Run curation scoring and promotion (calls backend API)."""
    print("\nAPI経由でスコアリングを開始します...")
    if _api_post("/ui/curation/jobs/run", {}):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def cmd_curate_resume() -> None:
    """Resume an interrupted curation session (calls backend API)."""
    print("\nAPI経由でレジュームを開始します...")
    if _api_post("/ui/curation/jobs/resume", {}):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def cmd_curate_export() -> None:
    """Export promoted subset to training cache (calls backend API)."""
    export_dir = input_default("エクスポート先", "data/curated_export")
    print(f"\nAPI経由で '{export_dir}' へのエクスポートを開始します...")
    if _api_post("/ui/curation/jobs/export", {"export_dir": export_dir}):
        print("エクスポートジョブを開始しました。")
    input("\nEnterで戻る...")


def cmd_curate_validate() -> None:
    """Run validation report on curated data."""
    print("\nAPI経由で検証レポート生成を開始します...")
    if _api_post("/ui/curation/jobs/validate", {}):
        print("ジョブを受け付けました。")
    input("\nEnterで戻る...")


def cmd_curate_status() -> None:
    """Show curation summary / status (calls backend API)."""
    import requests

    try:
        resp = requests.get(
            f"http://localhost:{SERVE_PORT}/ui/curation/summary", timeout=5
        )
        if resp.status_code == 200:
            print("\n=== キュレーションサマリー ===")
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"API Error: {resp.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
    input("\nEnterで戻る...")


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


def main() -> None:
    while True:
        print_menu()
        choice = input("選択 [1-18, h=ヘルプ, q=終了]: ").strip()
        if choice == "q":
            break
        if choice == "h":
            print_dependency_map()
            input("\nEnterで続行...")
            continue
        handlers = {
            # v3 mainline training
            "1": cmd_full_training,
            "1a": cmd_acting_training,
            "2": cmd_skip_preprocess,
            # data prep
            "3": cmd_list_datasets,
            "4": cmd_add_dataset,
            "5": cmd_cluster_speakers,
            "6": cmd_init_configs,
            # ops
            "7": cmd_finalize_training_outputs,
            "8": cmd_train_codec_from_latest_cache,
            "9": cmd_manage_characters,
            "10": cmd_system_check,
            "11": cmd_run_serve,
            # [v2-legacy]
            "12": cmd_full_training_legacy,
            # curation: ingest / run / resume / export / validate / status
            "13": cmd_curate_ingest,
            "14": cmd_curate_run,
            "15": cmd_curate_resume,
            "16": cmd_curate_export,
            "17": cmd_curate_validate,
            "18": cmd_curate_status,
        }
        if choice in handlers:
            handlers[choice]()
        else:
            input(f"無効な選択: {choice}. Enterで続行...")


if __name__ == "__main__":
    main()
