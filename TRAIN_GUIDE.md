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

### 1.3 TTS用アライメントの実行 (MFA統合)

VC学習には不要ですが、TTS学習を行う場合は正確な音素単位のアライメントが必須です。

```bash
# 1. Montreal Forced Aligner で TextGrid を生成
# (事前に MFA のインストールと acoustic model のダウンロードが必要)
mfa align data/raw/vctk/wav48 english_us_arpa english_us_arpa data/alignments/vctk

# 2. 生成された TextGrid をキャッシュに注入
uv run python scripts/annotate/run_forced_alignment.py \
    --cache-dir data/cache \
    --dataset vctk \
    --language en \
    --textgrid-dir data/alignments/vctk
```

- **MFA統合の重要性**: ヒューリスティックな均等割り（`--allow-heuristic`）は品質を著しく低下させるため、論文実装レベルの学習には MFA の使用を強く推奨します。
- **BOS/EOS**: 注入時に自動的に `<bos>`, `<eos>` トークンが前後に追加されます。

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

## 4. 並列化とパフォーマンス最適化

### 4.1 並列前処理 (Parallel Preprocessing)

単一プロセスではVRAM利用率が低いため（~7%）、複数ワーカーで並列処理することで大幅に高速化できます。

#### 方法1: 2ワーカー並列（推奨）

```bash
# 話者を2グループに分割
ls data/raw/wav48_silence_trimmed/ | grep "^p[0-9]" | sort > /tmp/all_speakers.txt
total=$(wc -l < /tmp/all_speakers.txt)
half=$((total / 2))
head -$half /tmp/all_speakers.txt > /tmp/speakers_group1.txt
tail -n +$((half + 1)) /tmp/all_speakers.txt > /tmp/speakers_group2.txt

# 2ワーカー並列実行
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/preprocess_speakers.py \
    --speaker-list /tmp/speakers_group1.txt \
    --worker-id 1 \
    --device cuda > logs/preprocess_worker1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/preprocess_speakers.py \
    --speaker-list /tmp/speakers_group2.txt \
    --worker-id 2 \
    --device cuda > logs/preprocess_worker2.log 2>&1 &

# 進捗確認
tail -f logs/preprocess_worker1.log logs/preprocess_worker2.log
```

#### パフォーマンス比較

| 構成 | 処理速度 | VRAM使用率 | ETA (VCTK 88k) |
|------|---------|-----------|----------------|
| 単一ワーカー | ~1.4 it/s | 7% | ~8.5h |
| 2ワーカー並列 | ~4 it/s | 69% | ~3h |
| **改善率** | **2.9倍** | **+62%** | **5.5h短縮** |

### 4.2 Whisper Turboモデル

`large-v3-turbo`を使用することで、文字起こし精度を保ちつつ8倍高速化します。

```bash
# tmrvc-data/src/tmrvc_data/cli/preprocess.py
whisper = WhisperModel("large-v3-turbo", device=device, compute_type="float16")
```

- **速度**: large-v3の8倍
- **精度**: ほぼ同等
- **VRAM**: わずかに削減

### 4.3 VRAM監視

```bash
# 1秒ごとにVRAM使用状況を表示
watch -n 1 nvidia-smi

# ログに記録
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > logs/gpu_usage.csv
```

### 4.4 Scientific Rigor準拠

並列化しても以下の検証は維持されます：

- **Frame Alignment**: 各ワーカーでassert検証
- **数学的整合性**: 全ワーカーが同じパイプラインを使用
- **SSL補間**: 50Hz→100Hzの線形補間を正しく実行

---

## 5. トラブルシューティング

- **ImportError (定数不足)**: `tmrvc_core/constants.py` が最新の `configs/constants.yaml` と同期しているか確認してください。
- **Shape Mismatch**: キャッシュ生成時の `hop_length` (240) とモデルのストライド設定が一致しているか確認してください。
- **OOM**: `--batch-size` または `--max-frames` (デフォルト400) を下げて調整してください。
- **Index Out of Bounds**: 古いキャッシュを削除して、新しいフレームアライメント（`pad_length=784`）で再生成してください。
