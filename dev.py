#!/usr/bin/env python3
"""TMRVC Development Menu."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

import yaml

CONFIGS_DIR = Path("configs")
DATASETS_YAML = CONFIGS_DIR / "datasets.yaml"
CHARACTERS_JSON = CONFIGS_DIR / "characters.json"
EXPERIMENTS_DIR = Path("experiments")
CHECKPOINTS_DIR = Path("checkpoints")
MODELS_DIR = Path("models")
CHARACTERS_DIR = MODELS_DIR / "characters"


def clear_screen() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def print_menu() -> None:
    clear_screen()
    print("=== TMRVC Development Menu ===")
    print("1) キャッシュ削除 + フル学習")
    print("2) 既存キャッシュで学習のみ")
    print("3) データセット一覧")
    print("4) データセット追加 (対話式)")
    print("5) 話者分離実行")
    print("6) 設定ファイル初期化")
    print("7) 学習成果物を確定 (latest更新 + 整合性スモーク)")
    print("8) Codec学習 (最新cacheから)")
    print("9) キャラクター管理 (Few-shot Enrollment)")
    print("10) システム整合性チェック (テスト & 静的解析)")
    print("11) 推論サーバー起動 (tmrvc-serve)")
    print()


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
    api_key = input_default("Anthropic API Key (context予測用, 空欄可)", os.environ.get("ANTHROPIC_API_KEY", ""))
    use_reload = input_default("オートリロードを有効にしますか? (y/n)", "n").lower() == "y"
    use_verbose = input_default("詳細ログを出力しますか? (y/n)", "n").lower() == "y"
    
    cmd = [
        "uv", "run", "tmrvc-serve",
        "--uclm-checkpoint", str(uclm_ckpt),
        "--codec-checkpoint", str(codec_ckpt),
        "--device", device,
        "--host", host,
        "--port", port
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
    print("\n" + "="*60)
    print("  システム整合性チェックを実行します")
    print("="*60)
    
    print("\n[1/2] 静的解析 (Ruff)...")
    ruff_ok = run_checked(["uv", "run", "ruff", "check", "."])
    
    print("\n[2/2] 統合テスト (Pytest)...")
    # Run only serve and core tests as a quick smoke test
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
    """Returns (total_gb, free_gb, recommended_workers) or None if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        idx = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        free = total - reserved

        # Conservative: based on free memory, not total
        # 4.5GB per worker to leave margin
        recommended = max(1, math.floor(free / 4.5))

        print(f"\nGPU {idx}: {torch.cuda.get_device_name(idx)}")
        print(f"  総メモリ: {total:.1f} GB")
        print(f"  使用中:   {reserved:.1f} GB")
        print(f"  空き:     {free:.1f} GB")
        print(f"\n推奨 workers: {recommended}")

        return total, free, recommended
    except Exception:
        return None


def select_workers(device: str) -> int:
    if device != "cuda":
        return 1

    info = get_gpu_info()
    if info is None:
        print("CUDA が利用できません。workers=1 で実行します。")
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
                level = p.get("adaptation_level", "unknown")
                path = p.get("speaker_file", "none")
                print(f"{cid:<15} {p.get('name', ''):<20} {level:<10} {path}")

        print("\n1) 新規キャラクター作成 (Enrollment)")
        print("2) キャラクター削除")
        print("b) 戻る")
        
        choice = input("\n選択: ").strip().lower()
        
        if choice == "b":
            break
        elif choice == "1":
            _enroll_character()
        elif choice == "2":
            _delete_character()


def _enroll_character() -> None:
    print("\n--- 新規キャラクター作成 ---")
    char_id = input("キャラクターID (例: my_char): ").strip()
    if not char_id:
        return
    
    profiles = load_character_profiles()
    if char_id in profiles:
        print(f"ERROR: ID '{char_id}' は既に存在します。")
        input("Enterで戻る...")
        return

    name = input("キャラクター名 (表示用): ").strip() or char_id
    audio_path = input("参照音声パス (ファイルまたはディレクトリ): ").strip()
    if not audio_path or not Path(audio_path).exists():
        print("ERROR: 音声パスが見つかりません。")
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
        "--output", str(output_file)
    ]
    
    if Path(audio_path).is_dir():
        cmd.extend(["--audio-dir", audio_path])
    else:
        cmd.extend(["--audio", audio_path])
        
    if level != "light" and codec_ckpt:
        cmd.extend(["--codec-checkpoint", str(codec_ckpt)])

    print(f"\nEnrollmentを実行中: {char_id}...")
    if run_checked(cmd):
        # 成功したらJSONに登録
        profiles[char_id] = {
            "name": name,
            "speaker_file": str(output_file),
            "adaptation_level": level,
            "personality": "",
            "voice_description": "",
            "language": "ja"
        }
        save_character_profiles(profiles)
        print(f"\nキャラクター '{char_id}' を作成し、登録しました。")
    
    input("\nEnterで戻る...")


def _delete_character() -> None:
    char_id = input("\n削除するキャラクターID: ").strip()
    if not char_id:
        return
    
    profiles = load_character_profiles()
    if char_id not in profiles:
        print(f"ERROR: ID '{char_id}' が見つかりません。")
        input("Enterで戻る...")
        return
    
    confirm = input(f"本当にキャラクター '{char_id}' を削除しますか? (y/n): ").lower()
    if confirm == "y":
        p = profiles.pop(char_id)
        save_character_profiles(profiles)
        
        # ファイルも消すか確認
        sp_file = Path(p.get("speaker_file", ""))
        if sp_file.exists():
            rm_file = input(f"実体ファイル '{sp_file.name}' も削除しますか? (y/n): ").lower()
            if rm_file == "y":
                sp_file.unlink()
                print("ファイルも削除しました。")
        
        print(f"キャラクター '{char_id}' の登録を削除しました。")
    
    input("Enterで戻る...")


def load_datasets() -> dict:
    if not DATASETS_YAML.exists():
        print(f"ERROR: {DATASETS_YAML} not found. Run --init first.")
        sys.exit(1)
    with open(DATASETS_YAML) as f:
        return yaml.safe_load(f) or {}


def get_enabled_datasets() -> list[str]:
    cfg = load_datasets()
    datasets = cfg.get("datasets", {})
    return [name for name, ds in datasets.items() if ds.get("enabled", False)]


def show_enabled_datasets() -> None:
    enabled = get_enabled_datasets()
    if not enabled:
        print("有効なデータセットがありません。")
        print("データセットを追加して有効化してください (メニュー4)")
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


def cache_has_required_datasets(cache_dir: Path, datasets: list[str]) -> bool:
    for dataset in datasets:
        if not (cache_dir / dataset / "train").is_dir():
            return False
    return True


def clear_training_caches_for_enabled_datasets(enabled: list[str]) -> int:
    removed = 0
    if not enabled:
        return removed

    # Legacy shared cache: remove only enabled dataset subtrees.
    legacy_cache = Path("data/cache")
    for dataset in enabled:
        p = legacy_cache / dataset
        if p.exists():
            shutil.rmtree(p)
            removed += 1

    experiment_prefix = "_".join(sorted(enabled))
    for cache_dir in list_experiment_cache_dirs():
        if not cache_dir.parent.name.startswith(f"{experiment_prefix}_"):
            continue
        shutil.rmtree(cache_dir)
        removed += 1

    return removed


def find_latest_cache_for_enabled_datasets(enabled: list[str]) -> Path | None:
    if not EXPERIMENTS_DIR.exists() or not enabled:
        return None

    experiment_prefix = "_".join(sorted(enabled))
    candidates = []
    for exp_dir in EXPERIMENTS_DIR.glob(f"{experiment_prefix}_*"):
        cache_dir = exp_dir / "cache"
        if not cache_dir.is_dir():
            continue
        if not cache_has_required_datasets(cache_dir, enabled):
            continue
        candidates.append(cache_dir)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_checked(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nコマンド失敗 (exit={e.returncode}): {' '.join(cmd)}")
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
    candidates: list[tuple[float, Path, Path]] = []
    for exp_dir in EXPERIMENTS_DIR.glob(f"{prefix}_*"):
        if not exp_dir.is_dir():
            continue
        ckpt = exp_dir / "checkpoints" / "uclm_final.pt"
        if not ckpt.exists():
            continue
        candidates.append((ckpt.stat().st_mtime, exp_dir, ckpt))

    if not candidates:
        return None, None

    _, exp_dir, ckpt = max(candidates, key=lambda x: x[0])
    return exp_dir, ckpt


def _quality_gate_status(exp_dir: Path) -> str:
    report_path = exp_dir / "quality_gate_report.json"
    if not report_path.exists():
        return "missing"
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        status = str(report.get("status", "unknown")).lower()
        return status
    except Exception:
        return "invalid"


def _promote_uclm_checkpoint(src_ckpt: Path) -> Path:
    dst = CHECKPOINTS_DIR / "uclm" / "uclm_latest.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst)
    return dst


def _promote_codec_checkpoint(src_ckpt: Path) -> Path:
    dst = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst)
    return dst


def _find_latest_experiment_for_enabled_datasets(enabled: list[str]) -> Path | None:
    if not enabled or not EXPERIMENTS_DIR.exists():
        return None

    prefix = "_".join(sorted(enabled))
    candidates: list[Path] = []
    for exp_dir in EXPERIMENTS_DIR.glob(f"{prefix}_*"):
        if not exp_dir.is_dir():
            continue
        cache_dir = exp_dir / "cache"
        if not cache_dir.is_dir():
            continue
        if not cache_has_required_datasets(cache_dir, enabled):
            continue
        candidates.append(exp_dir)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_codec_checkpoint() -> Path | None:
    preferred = CHECKPOINTS_DIR / "codec" / "codec_latest.pt"
    if preferred.exists():
        return preferred

    candidates: list[Path] = []
    for path in [
        CHECKPOINTS_DIR / "codec" / "codec_final.pt",
        CHECKPOINTS_DIR / "codec" / "latest.pt",
    ]:
        if path.exists():
            candidates.append(path)

    if EXPERIMENTS_DIR.exists():
        for ckpt in EXPERIMENTS_DIR.glob("*/checkpoints/codec_final.pt"):
            if ckpt.exists():
                candidates.append(ckpt)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_serve_health_smoke(
    uclm_ckpt: Path,
    codec_ckpt: Path,
    device: str,
) -> bool:
    script = (
        "from tmrvc_serve.app import init_app, app; "
        "from fastapi.testclient import TestClient; "
        f"init_app(uclm_checkpoint={str(uclm_ckpt)!r}, "
        f"codec_checkpoint={str(codec_ckpt)!r}, "
        f"device={device!r}, api_key=None); "
        "client = TestClient(app); "
        "resp = client.get('/health'); "
        "print('health:', resp.status_code, resp.json()); "
        "data = resp.json(); "
        "assert resp.status_code == 200 and data.get('models_loaded') is True"
    )
    return run_checked(["uv", "run", "python", "-c", script])


def cmd_finalize_training_outputs(preferred_device: str | None = None) -> None:
    enabled = get_enabled_datasets()
    if not enabled:
        print("有効なデータセットがありません。")
        return

    print("\n" + "="*60)
    print("  学習成果物の確定 (Finalization) プロセスを開始します")
    print("="*60)
    print("この操作では以下のステップを自動で実行します:")
    print("1. 最新成果物の特定: 有効なdatasetに対応する最新の uclm_final.pt を探索")
    print("2. 品質チェック: 実験ディレクトリ内の Quality Gate 結果を確認")
    print("3. 正式版への昇格: uclm_latest.pt として運用ディレクトリへコピー")
    print("4. 整合性テスト: 推論サーバー (FastAPI) を擬似起動し、ロード可否を確認")
    print("-" * 60)

    exp_dir, uclm_ckpt = find_latest_uclm_checkpoint_for_enabled_datasets(enabled)
    if exp_dir is None or uclm_ckpt is None:
        print("最新の学習済み UCLM checkpoint (uclm_final.pt) が見つかりません。")
        return

    print(f"\n対象実験: {exp_dir.name}")
    exp_meta = _read_yaml(exp_dir / "experiment.yaml")
    if exp_meta:
        print(f"  status: {exp_meta.get('status', 'unknown')}")
    qg_status = _quality_gate_status(exp_dir)
    print(f"  quality_gate: {qg_status}")
    if qg_status not in {"ok", "missing"}:
        print("品質ゲート結果が正常ではありません。運用反映を中断します。")
        return

    promoted = _promote_uclm_checkpoint(uclm_ckpt)
    print(f"UCLM latest 更新: {promoted}")

    codec_ckpt = _find_codec_checkpoint()
    if codec_ckpt is None:
        print("Codec checkpoint が見つかりません。")
        manual = input_default(
            "Codec checkpoint パス (空欄でスキップ)",
            "",
        )
        if not manual:
            print("serve整合性スモークはスキップします。")
            return
        codec_ckpt = Path(manual)
        if not codec_ckpt.exists():
            print(f"指定された codec checkpoint が存在しません: {codec_ckpt}")
            return

    device = preferred_device or select_device()
    print(f"\n整合性スモーク実行: device={device}")
    print(f"  uclm:  {promoted}")
    print(f"  codec: {codec_ckpt}")
    ok = _run_serve_health_smoke(promoted, codec_ckpt, device)
    if ok:
        print("整合性スモーク: OK")
    else:
        print("整合性スモーク: NG (ログを確認してください)")


def cmd_train_codec_from_latest_cache(preferred_device: str | None = None) -> bool:
    enabled = get_enabled_datasets()
    if not enabled:
        print("有効なデータセットがありません。")
        return False

    exp_dir = _find_latest_experiment_for_enabled_datasets(enabled)
    if exp_dir is None:
        print("有効datasetに一致する最新実験(cache)が見つかりません。")
        return False

    cache_dir = exp_dir / "cache"
    if not cache_dir.is_dir():
        print(f"cacheが見つかりません: {cache_dir}")
        return False

    print(f"\nCodec学習対象実験: {exp_dir.name}")
    print(f"  cache: {cache_dir}")

    device = preferred_device or select_device()
    batch_size = input_default("Codec学習バッチサイズ", "8")
    max_steps = input_default("Codec学習ステップ数", "10000")

    cmd = [
        "uv",
        "run",
        "tmrvc-train-codec",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(CHECKPOINTS_DIR / "codec"),
        "--batch-size",
        str(batch_size),
        "--max-steps",
        str(max_steps),
        "--device",
        str(device),
    ]
    print("\nCodec学習を開始...")
    ok = run_checked(cmd)
    if not ok:
        return False

    codec_final = CHECKPOINTS_DIR / "codec" / "codec_final.pt"
    if not codec_final.exists():
        print(f"Codec学習完了後に checkpoint が見つかりません: {codec_final}")
        return False

    promoted = _promote_codec_checkpoint(codec_final)
    print(f"Codec latest 更新: {promoted}")
    return True


def cmd_full_training() -> None:
    enabled = get_enabled_datasets()
    if not enabled:
        print("有効なデータセットがありません。")
        return

    print("有効データセット対象のキャッシュを削除中...")
    removed = clear_training_caches_for_enabled_datasets(enabled)
    print(f"削除済みキャッシュ: {removed}")

    print()
    show_enabled_datasets()

    device = select_device()
    workers = select_workers(device)
    train_batch_size = input_default("学習バッチサイズ", "16")

    print(f"\nフル学習を開始 (workers={workers})...")
    cmd = [
        "uv",
        "run",
        "tmrvc-train-pipeline",
        "--output-dir",
        "experiments",
        "--workers",
        str(workers),
        "--train-batch-size",
        train_batch_size,
        "--train-device",
        device,
        "--seed",
        "42",
    ]
    ok = run_checked(cmd)
    if ok:
        run_codec = input_default("Codec学習も実行しますか? (y/n)", "y").lower()
        if run_codec in {"y", "yes", "1", "true"}:
            cmd_train_codec_from_latest_cache(preferred_device=device)
        cmd_finalize_training_outputs(preferred_device=device)


def cmd_skip_preprocess() -> None:
    print()
    show_enabled_datasets()

    enabled = get_enabled_datasets()
    if not enabled:
        return

    latest_cache = find_latest_cache_for_enabled_datasets(enabled)
    if latest_cache:
        print(f"\n最新キャッシュを検出: {latest_cache}")
        cache_dir = input_default("使用するキャッシュ", str(latest_cache))
    else:
        print("\n既存キャッシュが見つかりません。")
        cache_dir = input_default("使用するキャッシュ", "")
        if not cache_dir:
            print("キャッシュ未指定のため中断します。")
            return
    if not Path(cache_dir).exists():
        print(f"キャッシュが存在しません: {cache_dir}")
        return
    if not cache_has_required_datasets(Path(cache_dir), enabled):
        print(f"キャッシュが不正です（必要dataset不足）: {cache_dir}")
        return
    device = select_device()
    train_batch_size = input_default("学習バッチサイズ", "16")

    print("\n既存キャッシュで学習を開始...")
    cmd = [
        "uv",
        "run",
        "tmrvc-train-pipeline",
        "--output-dir",
        "experiments",
        "--cache-dir",
        cache_dir,
        "--skip-preprocess",
        "--train-batch-size",
        train_batch_size,
        "--train-device",
        device,
        "--seed",
        "42",
    ]
    ok = run_checked(cmd)
    if ok:
        run_codec = input_default("Codec学習も実行しますか? (y/n)", "y").lower()
        if run_codec in {"y", "yes", "1", "true"}:
            cmd_train_codec_from_latest_cache(preferred_device=device)
        cmd_finalize_training_outputs(preferred_device=device)


def cmd_list_datasets() -> None:
    print(f"Config file: {DATASETS_YAML}\n")

    cfg = load_datasets()
    datasets = cfg.get("datasets", {})

    if not datasets:
        print("No datasets registered.")
        return

    print(f"{'Name':<25} {'Status':<10} {'Lang':<6} {'Path'}")
    print("-" * 80)
    for name, ds in datasets.items():
        status = "enabled" if ds.get("enabled", False) else "disabled"
        lang = ds.get("language", "?")
        raw_dir = ds.get("raw_dir", "")
        print(f"{name:<25} {status:<10} {lang:<6} {raw_dir}")


def cmd_add_dataset() -> None:
    run_checked(["uv", "run", "python", "scripts/config_generator.py", "--add-dataset"])


def cmd_cluster_speakers() -> None:
    print("話者分離が必要なデータセットを確認中...\n")

    cfg = load_datasets()
    datasets = cfg.get("datasets", {})

    generic_datasets = []
    for name, ds in datasets.items():
        if ds.get("type") == "generic" and ds.get("enabled", False):
            raw_dir = Path(ds.get("raw_dir", ""))
            speaker_map = raw_dir / "_speaker_map.json"
            needs_clustering = not speaker_map.exists()
            generic_datasets.append((name, raw_dir, needs_clustering))

    if not generic_datasets:
        print("話者分離が必要なgenericデータセットがありません。")
        print("まずデータセットを追加してください (メニュー4)")
        return

    print("Generic データセット一覧:")
    for i, (name, raw_dir, needs) in enumerate(generic_datasets, 1):
        status = "要分離" if needs else "済み"
        print(f"  {i}) {name} [{status}] {raw_dir}")

    print()
    choice = input_default("実行するデータセット番号 (または Enter で手動入力)", "")

    if choice:
        idx = int(choice) - 1
        if 0 <= idx < len(generic_datasets):
            _, raw_dir, _ = generic_datasets[idx]
            input_dir = str(raw_dir)
        else:
            print("無効な番号")
            return
    else:
        input_dir = input_default("入力ディレクトリ")

    device = input_default("デバイス", "cuda")

    print(f"\n話者分離を実行: {input_dir}")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/eval/cluster_speakers.py",
        "--input",
        input_dir,
        "--device",
        device,
    ]
    run_checked(cmd)


def cmd_init_configs() -> None:
    run_checked(["uv", "run", "python", "scripts/config_generator.py", "--init"])


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    while True:
        print_menu()
        choice = input("選択 [1-11, q=終了]: ").strip()

        if choice == "q" or choice == "quit":
            break

        handlers = {
            "1": cmd_full_training,
            "2": cmd_skip_preprocess,
            "3": cmd_list_datasets,
            "4": cmd_add_dataset,
            "5": cmd_cluster_speakers,
            "6": cmd_init_configs,
            "7": cmd_finalize_training_outputs,
            "8": cmd_train_codec_from_latest_cache,
            "9": cmd_manage_characters,
            "10": cmd_system_check,
            "11": cmd_run_serve,
        }

        handler = handlers.get(choice)
        if handler:
            handler()
            input("\nEnter で続行...")
        else:
            print(f"無効な選択: {choice}")
            input("\nEnter で続行...")


if __name__ == "__main__":
    main()
