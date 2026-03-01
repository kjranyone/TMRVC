# TMRVC UCLM v2 Training Guide

UCLM v2モデル（Unified Codec Language Model）の学習パイプライン全体を解説します。
TMRVCは、単一のトランスフォーマー・アーキテクチャでTTSとVCの両方を実現します。

## 概要

```
Raw Audio → [1. Preprocess] → Cache (UCLM FeatureSet) → [2. Train] → Checkpoints
```

## 1. データ前処理 (Data Preparation)

UCLM v2では、1回の前処理で学習に必要なすべての特徴量（音響トークン、制御トークン、物理音声状態、SSL特徴量、音素アラインメント）を一括抽出します。

### 1.1 `tmrvc-preprocess` による一括抽出

```bash
# configs/datasets.yaml で対象データセットを有効化し、raw_dirを指定
uv run python scripts/data/prepare_datasets.py --device cuda --skip-existing

# 高速プロトタイピング: データセットの1%（約300件）のみをランダム抽出してテスト
uv run python scripts/data/prepare_datasets.py --sample-ratio 0.01 --device cuda
```

- **`--sample-ratio`**: 0.0〜1.0 の範囲で指定。大規模データセットのロジック整合性を数分で確認したい場合に有用。
- **`--skip-existing`**: 既にキャッシュが存在する発話をスキップ。サンプリング後の全件抽出時に併用を推奨。

### 1.2 キャッシュ構造 (UCLM v2 Spec)

`data/cache/{dataset}/train/{speaker_id}/{utterance_id}/` 配下に以下のファイルが生成されます。

| ファイル名 | 形状 | 内容 |
|:---|:---|:---|
| `codec_tokens.npy` | [8, T] | Acoustic tokens (A_t) |
| `control_tokens.npy` | [4, T] | Control tokens (B_t) |
| `explicit_state.npy` | [T, 8] | 8次元物理音声パラメータ |
| `ssl_state.npy` | [T, 128] | WavLM SSL コンテキスト |
| `spk_embed.npy` | [192] | 話者埋め込み |
| `phoneme_ids.npy` | [L] | 音素ID（TTS学習用） |
| `durations.npy` | [L] | 音素単位のデュレーション（TTS学習用） |
| `waveform.npy` | [1, T*240] | 24kHz 正規化済み波形 |
| `meta.json` | - | テキスト、話者ID、統計情報 |

---

## 2. モデル学習 (Training)

学習は大きく分けて2つのステージ（CodecとUCLM本体）で行われます。

### 2.1 Stage 1: Emotion-Aware Codec学習

音声のトークン化と復元を行う基盤モデルを学習します。

```bash
tmrvc-train-codec \
    --cache-dir data/cache \
    --output-dir checkpoints/codec \
    --batch-size 8 \
    --max-steps 50000 \
    --device cuda
```

- **Loss**: Multi-scale STFT loss + Control Stream Cross-Entropy
- **出力**: `codec_final.pt`

### 2.2 Stage 2: Unified UCLM学習

TTSとVCを同時にこなす統合トランスフォーマーを学習します。

```bash
tmrvc-train-uclm \
    --cache-dir data/cache \
    --output-dir checkpoints/uclm \
    --batch-size 16 \
    --max-steps 100000 \
    --device cuda
```

- **学習内容**:
    - **VC Task**: `A_src_t → content → A_t, B_t`
    - **TTS Task**: `Phonemes → content → A_t, B_t`
- **Loss**: CE (A_t, B_t) + VQ Bottleneck + Adversarial Disentanglement (GRL) + Duration Prediction (MSE)

---

## 3. モデルの検証と利用

### 3.1 統合エンジンでのテスト

学習したチェックポイントを `UCLMEngine` にロードして動作確認します。

```bash
uv run python scripts/demo/tts_demo.py \
    --uclm-checkpoint checkpoints/uclm/uclm_final.pt \
    --codec-checkpoint checkpoints/codec/codec_final.pt \
    --text "これはUCLM v2の統合テストです。"
```

### 3.2 ONNX エクスポート

RustエンジンやVSTプラグインで利用するために ONNX 形式へ変換します。

```bash
tmrvc-export \
    --uclm-checkpoint checkpoints/uclm/uclm_final.pt \
    --codec-checkpoint checkpoints/codec/codec_final.pt \
    --output-dir models/onnx
```

---

## 4. トラブルシューティング

- **ImportError (定数不足)**: `tmrvc_core/constants.py` が最新の `configs/constants.yaml` と同期しているか確認してください。
- **Shape Mismatch**: キャッシュ生成時の `hop_length` (240) とモデルのストライド設定が一致しているか確認してください。
- **OOM**: `--batch-size` または `--max-frames` (デフォルト400) を下げて調整してください。
