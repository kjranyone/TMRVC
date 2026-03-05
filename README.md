# TMRVC — Unified Codec Language Model for Real-time TTS & VC

TMRVC は、Unified Codec Language Model (UCLM) v2 を核とした、リアルタイム TTS（音声合成）および高精度 VC（音声変換）を実現する統合音声生成プロジェクトです。CPU のみで end-to-end 50ms 以下のストリーミング推論を実現します。

## Features

- **Unified Core**: TTS と VC を単一のトランスフォーマー・アーキテクチャで統合。
- **Dual-Stream Token Spec v2**: 音響トークン (`A_t`) と制御トークン (`B_t`) を同時に生成し、豊かな表現力を実現。
- **8-dim Physical Voice State**: 息漏れ、緊張度などの物理的なパラメータによる直接的な演技制御。
- **Low Latency**: 10ms hop 単位の因果的処理により、極低遅延なストリーミングを実現 (~25ms nominal)。
- **CPU-only Inference**: ONNX Runtime を用いた効率的な推論。GPU 不要。
- **LoRA Personalization**: 数秒の参照音声から LoRA を用いた高速な話者適応。

## Architecture (UCLM v2)

```
[Input]
  TTS Mode: Text → Phonemes → TextEncoder ──┐
                                            │
  VC Mode:  Audio → CodecEncoder → VCEncoder ┼─→ [UCLM Transformer] ─→ [Dual Heads] ─→ [Codec Decoder] ─→ Audio
                                            │      (A_t, B_t)
[Control]                                   │
  Voice State: 8-dim Physical + SSL Context ─┘
  Speaker:     Global Embed + LoRA
```

## Repository Structure

```
TMRVC/
├── configs/              # Shared constants (UCLM v2 Spec)
├── docs/design/          # Technical specifications
├── tmrvc-core/           # Common types & constants
├── tmrvc-data/           # Dataset & Preprocessing
├── tmrvc-train/          # UCLM v2 Training & Model definitions
├── tmrvc-export/         # ONNX Export & Quantization
├── tmrvc-serve/          # Unified FastAPI Server (WebSocket)
├── tmrvc-gui/            # PySide6 Development GUI
├── tmrvc-engine-rs/      # Rust Streaming Engine
└── tmrvc-vst/            # VST3 Plugin
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tmrvc-train-uclm` | UCLM モデルのマルチタスク学習 (TTS+VC) |
| `tmrvc-train-codec` | Emotion-Aware Codec の学習 |
| `tmrvc-serve` | 統合推論サーバーの起動 |
| `tmrvc-export` | ONNX へのエクスポート・量子化 |
| `tmrvc-gui` | 開発用 GUI の起動 |

## Quick Start (Python)

```bash
# uv を用いた環境構築
uv sync --extra-index-url https://download.pytorch.org/whl/cu128

# 推論サーバーの起動
uv run tmrvc-serve --uclm-checkpoint checkpoints/uclm.pt --codec-checkpoint checkpoints/codec.pt

# 開発用 GUI (TMRVC Research Studio) の起動
uv run tmrvc-gui
```

## Training

TMRVC は対話式メニューで簡単に学習を開始できます:

```bash
# 環境セットアップ
uv sync

# 設定ファイルの初期化 (初回のみ)
uv run python scripts/config_generator.py --init

# 対話式メニューの起動
uv run dev.py
```

メニューから以下の操作が可能です:
- データセットの追加・管理
- 話者分離 (生のwavファイルから自動分類)
- フル学習 / 既存キャッシュでの学習

詳細は [TRAIN_GUIDE.md](TRAIN_GUIDE.md) を参照してください。

## License

MIT
