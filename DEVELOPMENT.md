# TMRVC Development Guide

## Prerequisites

### Python 環境

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャ)

```bash
# uv のインストール
pip install uv
# または
winget install astral-sh.uv
```

### Rust 環境 (tmrvc-rt)

- Rust toolchain (rustup 経由)
- Visual Studio Build Tools (C++ ワークロード)

```bash
# Rust のインストール
winget install Rustlang.Rustup
# または https://rustup.rs/ から rustup-init.exe をダウンロード

# Visual Studio Build Tools (未インストールの場合)
winget install Microsoft.VisualStudio.2022.BuildTools
# インストーラーで「C++ によるデスクトップ開発」を選択

# 確認
rustc --version
cargo --version
```

### C++ 環境 (tmrvc-engine, tmrvc-plugin) — 将来

- C++17 対応コンパイラ
- CMake >= 3.20
- ONNX Runtime (ort-builder で静的リンク)
- JUCE (VST3 プラグインのみ)

---

## Building

### Python ワークスペース

```bash
# 依存関係のインストール (仮想環境自動作成)
uv sync

# 特定パッケージのみ
uv sync --package tmrvc-core
```

uv workspace 構成:

| Member | Description |
|---|---|
| `tmrvc-core` | 共有定数、mel 計算、型定義 |
| `tmrvc-data` | データセット、前処理、augmentation |
| `tmrvc-train` | モデル定義、学習、蒸留、Few-shot |
| `tmrvc-export` | ONNX エクスポート、量子化、パリティ検証 |
| `tmrvc-gui` | 開発用 GUI (PySide6) |

### Rust スタンドアロンアプリ (tmrvc-rt)

```bash
# Debug ビルド
cargo build -p tmrvc-rt

# Release ビルド (推奨: 推論パフォーマンスに影響)
cargo build -p tmrvc-rt --release

# 実行
cargo run -p tmrvc-rt --release
```

ONNX Runtime は `ort` crate の `load-dynamic` feature で動的リンク。
`onnxruntime.dll` を PATH に配置するか、実行ファイルと同じディレクトリに置くこと。

### C++ エンジン・プラグイン (将来)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Testing

### Python テスト

```bash
# 全テスト実行
uv run pytest tests/python/

# カバレッジ付き
uv run pytest tests/python/ --cov=tmrvc_core --cov=tmrvc_data

# 特定テストのみ
uv run pytest tests/python/test_constants.py -v
```

現在のテスト構成 (211 テスト):
- `test_constants.py` — constants.yaml と Python 定数の整合性
- `test_generate_constants.py` — 定数自動生成スクリプト
- `test_audio.py` — 音声読み込み、リサンプル
- `test_preprocessing.py` — 前処理パイプライン
- `test_features.py` — STFT、mel、F0 抽出
- `test_dataset.py` — データセットクラス
- `test_cache.py` — キャッシュシステム
- `test_modules.py` — CausalConvNeXt 等の共通モジュール
- `test_content_encoder.py` — ContentEncoder Student
- `test_converter.py` — Converter Student / HQ
- `test_vocoder.py` — Vocoder Student
- `test_teacher.py` — Teacher U-Net
- `test_diffusion.py` — 拡散プロセス (OT-CFM, Sway, Reflow)
- `test_discriminator.py` — MelDiscriminator (DMD2)
- `test_losses.py` — 損失関数 (DMD2Loss, SVLoss)
- `test_trainer.py` — Teacher / Distillation / Reflow Trainer, CLI
- `test_export.py` — ONNX エクスポート
- `test_fewshot_finetuner.py` — Few-shot LoRA ファインチューン

### ONNX パリティ検証 (将来)

```bash
uv run python -m tmrvc_export.verify_parity
```

---

## Project Conventions

### 共有定数

`configs/constants.yaml` が Single Source of Truth。

- Python: `tmrvc-core/src/tmrvc_core/constants.py` (自動生成)
- Rust: `tmrvc-engine-rs/src/constants.rs` (自動生成)
- C++: `tmrvc-engine/include/tmrvc/constants.h` (自動生成、将来)

定数を変更する場合は `constants.yaml` を編集し、各言語の定数ファイルを更新すること。

```bash
# Python 定数の自動生成
uv run python scripts/generate_constants.py
```

### Path の注意点

`constants.py` は `Path(__file__).resolve().parents[3]` でリポジトリルートを取得する。
`tmrvc-core/src/tmrvc_core/constants.py` → parents[3] = TMRVC root。

### 依存関係メモ

- `torchaudio` 2.10+ は `torchcodec` が必要 → `soundfile.read()` を使用
- `numba` は `constraint-dependencies = ["numba>=0.60"]` が必要 (Python 3.12)
- uv workspace: 各メンバーが workspace 内の他パッケージに依存する場合は `[tool.uv.sources]` に `workspace = true` を設定

---

## tmrvc-rt Architecture

### スレッドモデル

```
GUI Thread (eframe, ~60fps)
    │ crossbeam_channel (Command)
    ▼
Processor Thread (std::thread)
    │ Arc<SpscRingBuffer> (audio samples)
    ▼
cpal Audio Callback (WASAPI, RT priority)
```

### ディレクトリ構成

```
tmrvc-rt/src/
├── main.rs                # eframe 起動
├── app.rs                 # egui App + スレッド管理 + コマンドチャネル
├── engine/                # GUI 非依存 (VST3 流用可能)
│   ├── constants.rs       # 共有定数
│   ├── ring_buffer.rs     # SPSC lock-free ring buffer
│   ├── ping_pong.rs       # Ping-Pong double-buffered state
│   ├── tensor_pool.rs     # 単一 Vec アロケーション
│   ├── ort_bundle.rs      # 4 ONNX session 管理
│   ├── speaker.rs         # .tmrvc_speaker 読み込み
│   ├── dsp.rs             # STFT, mel, iSTFT, OLA
│   └── processor.rs       # StreamingEngine (per-frame 処理)
├── audio/
│   └── stream.rs          # cpal ストリーム管理
└── ui/
    ├── device_panel.rs    # デバイス選択
    ├── model_panel.rs     # モデル/Speaker 選択
    ├── monitor.rs         # レベルメーター
    └── controls.rs        # Dry/Wet, Gain
```

### VST3 化パス

`engine/` モジュールは GUI に非依存。将来 `nih-plug-egui` でラップすれば
VST3/CLAP プラグイン化は薄いラッパー追加のみで実現可能。

---

## Design Document References

設計変更時は以下のドキュメントの整合性チェックリスト（各ファイル末尾）を確認すること。

| Document | Key Content |
|---|---|
| `docs/design/onnx-contract.md` | モデル I/O shapes, state 寸法, .tmrvc_speaker format |
| `docs/design/streaming-design.md` | レイテンシバジェット, causal STFT, OLA |
| `docs/design/cpp-engine-design.md` | TensorPool layout, SPSC Queue protocol |
| `docs/design/model-architecture.md` | 各モデルの層構成, パラメータ数 |
| `docs/design/training-plan.md` | コーパス構成, 学習フェーズ |

## Structured Training Data Pipeline

Use a registry file instead of searching for datasets manually.

1. Edit `configs/datasets.yaml` and set each dataset `raw_dir` / `enabled`.
2. Run deterministic preprocessing:

```bash
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --device xpu --skip-existing
```

3. Check generated cache manifests:
- `data/cache/_manifests/<dataset>_train.json`

Example for Tsukuyomi only:

```bash
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --datasets tsukuyomi --device xpu --skip-existing
```
