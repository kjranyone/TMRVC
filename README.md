# TMRVC — Unified Codec Language Model for Real-time TTS & VC (UCLM v3)

TMRVC は、単一の codec language model で TTS と VC を統合するリアルタイム音声生成システムです。`UCLM v3` アーキテクチャに基づき、`10 ms` 因果生成、`A_t / B_t` の dual-stream token 生成、`MFA 非依存` の内部アライメント学習を実現しています。

## 現行アーキテクチャ (UCLM v3)

- `10 ms causal core`: 未来参照なしで毎フレーム生成する
- `Unified TTS / VC`: テキスト条件と音声条件を同一 backbone で扱う
- `Internal alignment`: TTS は外部 forced alignment (MFA) ではなくポインタベースで text progression を学習する
- `Dual-stream token contract`: acoustic tokens `A_t` と control tokens `B_t` を同期生成する
- `Physical-first control`: 8 次元の voice state と prosody latent を優先し、抽象ラベル依存を避ける

## システム概要

```text
TTS:
  text -> normalizer / g2p / grapheme backend -> text units
       -> TextEncoder -> UCLM Core -> A_t / B_t -> Codec Decoder -> audio
                         |-> pointer head (internal alignment)

VC:
  source audio -> Codec Encoder + causal semantic encoder
               -> UCLM Core -> A_t / B_t -> Codec Decoder -> audio

Shared conditions:
  speaker embedding
  explicit voice state (8-dim)
  ssl state / prosody latent
  pacing controls (pace / hold / boundary bias)
```

## セットアップ

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
```

英語系 G2P を使う場合は `phonemizer` のバックエンドが必要です。

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng
```

## データセット原則

- 1 dataset = 1 language で運用する
- mainline の TTS 学習に `MFA` や `durations.npy` は不要です
- 必要なのは発話テキストと、そこから生成される text units のみです

## 学習フロー

`dev.py` を使用して、対話的に学習を進めることができます。

```bash
uv run python dev.py
```

### 推奨フロー

| 番号 | 役割 | 主な出力 |
|---|---|---|
| `6` | 設定初期化 | `configs/datasets.yaml` |
| `4` | データセット追加 | dataset 定義更新 |
| `1` | フル学習 | cache + `uclm_final.pt` |
| `8` | Codec 学習 | `codec_final.pt` |
| `7` | 成果物確定 | `uclm_latest.pt`, `codec_latest.pt` |
| `11` | 推論サーバー起動 | FastAPI server |

## CLI

| Command | Description |
|---|---|
| `tmrvc-preprocess` | 特徴量キャッシュ生成 |
| `tmrvc-train-pipeline` | 前処理から UCLM 学習までの統合実行 |
| `tmrvc-train-uclm` | UCLM v3 モデル学習 (pointer mode) |
| `tmrvc-train-codec` | Emotion-Aware Codec 学習 |
| `tmrvc-serve` | 統合推論サーバー起動 |
| `tmrvc-export` | ONNX エクスポート |
| `tmrvc-gui` | 開発・デモ用 GUI (UCLM v3 対応) |

## ドキュメント

- 全体入口: `docs/README.md`
- 学習ガイド: `TRAIN_GUIDE.md`
- 設計資料: `docs/design/architecture.md`
- 実装計画: `plan/README.md`

## License

MIT
