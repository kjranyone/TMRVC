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

```
入力音声 → Causal STFT → Content Encoder → Converter → Vocoder → iSTFT + OLA → 出力音声
                              ↑                ↑
                          F0 (YIN)         Speaker Embed + IR Params
```

5 ONNX モデルの実行頻度:

| Model | Frequency | Purpose |
|---|---|---|
| `content_encoder` | per-frame (10ms) | 音素・韻律の抽出 |
| `converter` | per-frame (10ms) | 話者変換 (1-step denoiser) |
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
│   ├── design/           # 正本設計資料 (7 ファイル)
│   └── reference/        # 参考資料
│
├── tmrvc-core/           # [Python] 共有定数・mel 計算・型定義
├── tmrvc-data/           # [Python] データセット・前処理・augmentation
├── tmrvc-gui/            # [Python] 開発用 GUI (PySide6)
│
├── tmrvc-engine-rs/      # [Rust]   ストリーミング推論エンジン + NAM 推論
├── tmrvc-rt/             # [Rust]   スタンドアロン RT アプリ (egui + cpal + ort)
├── tmrvc-vst/            # [Rust]   VST3 プラグイン (nih-plug)
│
├── tmrvc-train/          # [Python] モデル定義・学習・蒸留
├── tmrvc-export/         # [Python] ONNX エクスポート・量子化
├── tmrvc-serve/          # [Python] FastAPI サーバー (WebSocket TTS)
│
├── tests/                # Python テスト
└── scripts/              # generate_constants.py 等
```

## Key Constants

`configs/constants.yaml` が全モジュールの Single Source of Truth:

```
sample_rate: 24000    hop_length: 240 (10ms)    n_mels: 80
n_fft: 1024           window_length: 960         d_content: 256
d_speaker: 192        n_ir_params: 24            d_converter_hidden: 384
```

## Quick Start

### Python (学習・前処理)

```bash
# uv がインストール済みであること
uv sync
uv run pytest tests/python/
```

### Rust (エンジン・VST・スタンドアロン)

```bash
# テスト実行
cargo test -p tmrvc-engine-rs

# スタンドアロンアプリ
cargo run -p tmrvc-rt --release

# VST3 プラグインビルド
cargo build -p tmrvc-vst --release
```

詳細は [DEVELOPMENT.md](DEVELOPMENT.md) を参照。

## CLI Commands

### Python (tmrvc-xxx)

| Command | Package | Description |
|---------|---------|-------------|
| `tmrvc-train-teacher` | tmrvc-train | Teacher モデル学習 (diffusion U-Net) |
| `tmrvc-train-tts` | tmrvc-train | TTS モデル学習 |
| `tmrvc-train-style` | tmrvc-train | スタイル埋め込みモデル学習 |
| `tmrvc-distill` | tmrvc-train | Teacher → Student 蒸留 |
| `tmrvc-finetune` | tmrvc-train | Few-shot fine-tuning |
| `tmrvc-export` | tmrvc-export | ONNX エクスポート・量子化 |
| `tmrvc-create-character` | tmrvc-export | `.tmrvc_speaker` / `.tmrvc_style` ファイル作成 |
| `tmrvc-serve` | tmrvc-serve | FastAPI サーバー (WebSocket TTS) |
| `tmrvc-gui` | tmrvc-gui | PySide6 開発用 GUI |

```bash
# 使用例
uv run tmrvc-train-teacher --cache-dir data/cache --phase 0 --device xpu
uv run tmrvc-distill --teacher-ckpt checkpoints/teacher.pt --phase A --device xpu
uv run tmrvc-export --checkpoint checkpoints/distill/best.pt --output-dir models/fp32
uv run tmrvc-create-character --audio voice.wav --output character.tmrvc_speaker
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

| File | Content |
|---|---|
| [`architecture.md`](docs/design/architecture.md) | 全体アーキテクチャ、モジュール構成、主要設計判断 |
| [`streaming-design.md`](docs/design/streaming-design.md) | レイテンシバジェット、Audio Thread パイプライン、Causal STFT |
| [`onnx-contract.md`](docs/design/onnx-contract.md) | 5 モデルの I/O テンソル仕様、State tensor、`.tmrvc_speaker` フォーマット |
| [`model-architecture.md`](docs/design/model-architecture.md) | 各モデルの詳細設計 |
| [`cpp-engine-design.md`](docs/design/cpp-engine-design.md) | C++ エンジン・TensorPool・VST3 統合 |
| [`gui-design.md`](docs/design/gui-design.md) | GUI 設計 |
| [`training/README.md`](docs/training/README.md) | 学習パイプライン統合ガイド (VC/TTS/Style) |

## License

MIT

## Structured Data Prep

Configure dataset locations in `configs/datasets.yaml`, then run:

```bash
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --device xpu --skip-existing
```
