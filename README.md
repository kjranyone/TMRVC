# TMRVC — IR-aware Real-time Voice Conversion

CPU-only で end-to-end 50ms 以下のリアルタイム Voice Conversion を実現するプロジェクト。
Frame-by-frame causal streaming アーキテクチャ（10ms hop 単位）で処理する。

## Features

- **Design Motto**: キャラ演技は本体、few-shotは仕上げ。

- **低レイテンシ**: 10ms hop × causal 処理で ~25ms (nominal)
- **CPU-only**: ONNX Runtime CPU EP のみ。GPU 不要
- **IR-aware**: 入力音声の残響・マイク特性を推定し、話者変換と環境特性を分離
- **Few-shot 対応**: 数秒の参照音声から話者特徴を抽出 (`.tmrvc_speaker`)
- **NAM 統合**: Neural Amp Modeler の `.nam` プロファイルをポストエフェクトとして適用可能
- **マルチ実行環境**: スタンドアロン GUI (Rust/egui) と VST3 プラグイン (nih-plug)

## Architecture

### Feature-based (現行)

```
入力音声 → Causal STFT → Content Encoder → Converter → Vocoder → iSTFT + OLA → 出力音声
                               ↑                ↑
                           F0 (YIN)         Speaker Embed + Acoustic Params
```

### Codec-Latent (次世代)

```
入力音声 → Causal Codec Encoder → Token Model (Mamba) → Codec Decoder → 出力音声
                                       ↑
                              Speaker + Style Conditioning
```

6 ONNX モデルの実行頻度:

| Model | Frequency | Purpose |
|---|---|---|
| `content_encoder` | per-frame (10ms) | 音素・韻律の抽出 |
| `converter` | per-frame (10ms) | 話者変換 (Live mode) |
| `converter_hq` | per-frame (10ms) | 話者変換 (HQ mode, optional) |
| `vocoder` | per-frame (10ms) | STFT mag/phase 予測 |
| `ir_estimator` | 10 frames (~100ms) | 残響・マイク特性推定 |
| `speaker_encoder` | offline | 話者特徴抽出 + LoRA delta 生成 |

## NAM (Neural Amp Modeler) 統合

ボイス変換出力に対して [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore) の `.nam` プロファイルをポストエフェクトとして適用できる。アンプ・キャビネット等のアナログ音響機器シミュレーションをリアルタイムで行う。

```
VC Engine (24kHz) → Upsample (48kHz) → [NAM Processing (48kHz)] → DAW Output
```

- **Pure Rust 実装**: 外部 C++ ライブラリ依存なし
- **対応アーキテクチャ**: WaveNet (~90% のプロファイル), LSTM, CatLSTM
- **サンプルレート自動変換**: DAW rate と NAM model rate が異なる場合は PolyphaseResampler で変換
- **追加レイテンシ 0**: WaveNet/LSTM は完全 causal
- **VST パラメータ**: NAM Enable, NAM Mix (Dry/Wet), プロファイルパス (DAW プロジェクトに永続化)

## Repository Structure

```
TMRVC/
├── configs/              # constants.yaml (全モジュール共有定数)
├── docs/
│   ├── design/           # 正本設計資料 (13 ファイル)
│   ├── training/         # 学習計画 (5 ファイル)
│   ├── reference/        # 参考資料
│   └── research/         # 研究計画
│
├── tmrvc-core/           # [Python] 共有定数・mel 計算・型定義
├── tmrvc-data/           # [Python] データセット・前処理・augmentation
├── tmrvc-gui/            # [Python] 開発用 GUI (PySide6)
├── tmrvc-train/          # [Python] モデル定義・学習 (VC/TTS/Codec/UCLM)
├── tmrvc-export/         # [Python] ONNX エクスポート・量子化
├── tmrvc-serve/          # [Python] FastAPI サーバー (WebSocket TTS)
│
├── tmrvc-engine-rs/      # [Rust]   ストリーミング推論エンジン + NAM 推論
├── tmrvc-rt/             # [Rust]   スタンドアロン RT アプリ (egui + cpal)
├── tmrvc-vst/            # [Rust]   VST3 プラグイン (nih-plug)
├── xtask/                # [Rust]   ビルドタスク (nih-plug bundler)
│
├── tests/                # Python テスト
└── scripts/              # ユーティリティスクリプト
```

## Key Constants

`configs/constants.yaml` が全モジュールの Single Source of Truth:

```
sample_rate: 24000    hop_length: 240 (10ms)    n_mels: 80
n_fft: 1024           window_length: 960         d_content: 256
d_speaker: 192        n_acoustic_params: 32      d_converter_hidden: 384
lora_delta_size: 15872
```

## Quick Start

### System Requirements

**Linux (Ubuntu/Debian):**
```bash
# VST3 ビルド・スタンドアロン実行用
sudo apt install libasound2-dev pkg-config
```

**Windows / macOS:** 追加のシステムライブラリ不要

### Python (学習・前処理)

```bash
# uv がインストール済みであること
# CUDA環境
uv sync --extra-index-url https://download.pytorch.org/whl/cu128

# XPU環境 (Intel Arc)
uv sync --extra-index-url https://download.pytorch.org/whl/xpu

uv run pytest tests/
```

### Rust (エンジン・VST・スタンドアロン)

```bash
# テスト実行
cargo test

# スタンドアロンアプリ
cargo run -p tmrvc-rt --release

# VST3 プラグインビルド
cargo xtask bundle tmrvc-vst --release
```

詳細は [AGENTS.md](AGENTS.md) を参照。

## CLI Commands

### Python (tmrvc-xxx)

| Command | Package | Description |
|---------|---------|-------------|
| `tmrvc-preprocess` | tmrvc-data | データ前処理・特徴量抽出 |
| `tmrvc-train-codec` | tmrvc-train | Codec モデル学習 |
| `tmrvc-train-token` | tmrvc-train | Token モデル学習 (Mamba) |
| `tmrvc-train-uclm` | tmrvc-train | UCLM モデル学習 |
| `tmrvc-train-tts` | tmrvc-train | TTS モデル学習 |
| `tmrvc-train-style` | tmrvc-train | スタイル埋め込みモデル学習 |
| `tmrvc-export` | tmrvc-export | ONNX エクスポート・量子化 |
| `tmrvc-enroll` | tmrvc-export | `.tmrvc_speaker` 作成 (階層的適応) |
| `tmrvc-create-character` | tmrvc-export | `.tmrvc_character` 作成 (TTS用) |
| `tmrvc-serve` | tmrvc-serve | FastAPI サーバー (WebSocket TTS) |
| `tmrvc-gui` | tmrvc-gui | PySide6 開発用 GUI |

```bash
# 使用例 (--device は環境に合わせて cuda/xpu を指定)
uv run tmrvc-preprocess --config configs/datasets.yaml --device cuda
uv run tmrvc-train-codec --config configs/train_codec.yaml --device cuda
uv run tmrvc-train-token --config configs/train_token.yaml --device cuda
uv run tmrvc-export --checkpoint checkpoints/best.pt --output-dir models/fp32
uv run tmrvc-enroll --audio voice.wav --output speaker.tmrvc_speaker --level standard
uv run tmrvc-serve --port 8000
uv run tmrvc-gui
```

### Rust

| Package | Type | Description |
|---------|------|-------------|
| `tmrvc-engine-rs` | library | ストリーミング推論エンジン (ONNX + NAM) |
| `tmrvc-rt` | binary | スタンドアロン GUI (egui + cpal) |
| `tmrvc-vst` | VST3 | DAW 用プラグイン (nih-plug) |

## Design Documents

### docs/design/ (13 ファイル)

| File | Content |
|---|---|
| [`architecture.md`](docs/design/architecture.md) | 全体アーキテクチャ、モジュール構成 |
| [`streaming-design.md`](docs/design/streaming-design.md) | レイテンシバジェット、Audio Thread パイプライン |
| [`onnx-contract.md`](docs/design/onnx-contract.md) | 6 モデルの I/O テンソル仕様 |
| [`model-architecture.md`](docs/design/model-architecture.md) | 各モデルの詳細設計 |
| [`cpp-engine-design.md`](docs/design/cpp-engine-design.md) | Rust エンジン設計 (タイトルは旧来のまま) |
| [`codec-latent-design.md`](docs/design/codec-latent-design.md) | Codec-Latent パラダイム設計 |
| [`unified-codec-lm.md`](docs/design/unified-codec-lm.md) | UCLM アーキテクチャ |

### docs/training/ (5 ファイル)

| File | Content |
|---|---|
| [`README.md`](docs/training/README.md) | 学習パイプライン統合ガイド |
| [`vc-training-plan.md`](docs/training/vc-training-plan.md) | VC Teacher + 蒸留の学習計画 |
| [`tts-training-plan.md`](docs/training/tts-training-plan.md) | TTS パイプラインの学習計画 |
| [`style-training-plan.md`](docs/training/style-training-plan.md) | StyleEncoder の学習計画 |

## License

MIT

## Structured Data Prep

Configure dataset locations in `configs/datasets.yaml`, then run:

```bash
# --device は環境に合わせて cuda/xpu を指定
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --device cuda --skip-existing
```
