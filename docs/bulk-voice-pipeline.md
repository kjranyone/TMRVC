# Bulk Voice Pipeline — 3万WAV学習ガイド

複数話者・未分類・書き起こしなしの大量WAVファイルをVC+TTS学習データに変換するパイプライン。

## 前提条件

- TMRVC リポジトリがクローン済み
- `uv` がインストール済み
- `uv sync` で基本依存関係がインストール済み
- XPU (Intel Arc) または CPU が利用可能

## 処理フロー

```
data/raw/galge_voices/*.wav  (3万ファイル, フラット)
    │
    ├─ Phase 0: Survey         → _bulk_report.json
    ├─ Phase 1: Clustering     → _speaker_map.json
    ├─ Phase 2: Filter+Copy    → data/raw/galge_clean/{spk_0001,...}/*.wav
    ├─ Phase 3: Feature Extract → data/cache/galge/train/{spk}/*.npy
    ├─ Phase 4: ASR (任意)     → cache meta.json に text 注入
    └─ Phase 5: Training       → checkpoints/
```

## Phase 0: Survey（品質概況）

データの品質を確認する。コピーは行わない。

```bash
uv run python scripts/prepare_bulk_voice.py \
    --input data/raw/galge_voices \
    --report
```

出力: `data/raw/galge_voices/_bulk_report.json`

確認ポイント:
- 全ファイル数、有効ファイル数、除外理由の内訳
- 除外閾値の調整: `--min-duration 0.5` `--max-duration 60` で調整可能

## Phase 1: Speaker Clustering

ECAPA-TDNN で話者 embedding を抽出し、HDBSCAN でクラスタリング。

```bash
# 依存関係インストール
uv sync --package tmrvc-data --extra cluster

# 全ステップ実行（embed → cluster → report）
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices \
    --device xpu
```

### ステップ分割実行（推奨：3万ファイルの場合）

embedding 抽出は ~1.5時間かかるため、ステップごとに実行すると安全:

```bash
# Step 1: Embedding 抽出（中間保存あり、Ctrl+C で中断可能、再開時は自動レジューム）
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices \
    --step embed \
    --device xpu \
    --save-every 1000

# Step 2: クラスタリング（数秒で完了）
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices \
    --step cluster \
    --min-cluster-size 20 \
    --min-samples 5

# Step 3: 結果確認
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices \
    --step report
```

出力:
- `data/raw/galge_voices/_speaker_embeds.npz` — 中間 embedding
- `data/raw/galge_voices/_speaker_map.json` — 最終クラスタリング結果

### クラスタリングパラメータの調整

結果が粗すぎる/細かすぎる場合は `--step cluster` を再実行:

```bash
# 小さいクラスタも許容（話者数が増える）
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices \
    --step cluster \
    --min-cluster-size 10 \
    --min-samples 3

# 結果確認
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices \
    --step report
```

## Phase 2: Filter + Restructure

品質フィルタ適用 + 話者フォルダ構造に再構成。

```bash
uv run python scripts/prepare_bulk_voice.py \
    --input data/raw/galge_voices \
    --speaker-map data/raw/galge_voices/_speaker_map.json \
    --output data/raw/galge_clean
```

出力構造:
```
data/raw/galge_clean/
├── spk_0001/
│   ├── ev001_a_01.wav
│   ├── ev001_a_02.wav
│   └── ...
├── spk_0002/
│   └── ...
└── ...
```

`spk_noise` にマッピングされたファイルは自動で除外される。

## Phase 3: Feature Extraction

`datasets.yaml` に galge エントリを有効化:

```bash
# configs/datasets.yaml を編集
#   galge:
#     type: generic
#     enabled: true      ← false を true に変更
#     language: ja
#     raw_dir: data/raw/galge_clean
```

特徴量抽出を実行:

```bash
uv run python scripts/prepare_datasets.py \
    --datasets galge \
    --device xpu \
    --skip-existing
```

または直接 `tmrvc-preprocess` を呼ぶ:

```bash
uv run tmrvc-preprocess \
    --dataset galge \
    --raw-dir data/raw/galge_clean \
    --cache-dir data/cache \
    --device xpu \
    --skip-existing
```

出力: `data/cache/galge/train/{speaker_id}/{utterance_id}/` に mel/content/f0/spk_embed の各 .npy

所要時間: ~8-10時間 (3万ファイル, XPU)

## Phase 4: ASR Transcription（TTS用、VC-only なら不要）

### Step 4a: Whisper で書き起こし

```bash
# 依存関係インストール
uv sync --package tmrvc-data --extra asr

# 書き起こし実行
uv run python scripts/prepare_bulk_voice.py \
    --input data/raw/galge_clean \
    --output data/raw/galge_clean \
    --transcribe \
    --language ja \
    --device cpu \
    --whisper-model large-v3
```

各話者フォルダに `transcripts.txt` (`stem|text` 形式) が生成される。

所要時間: ~12-15時間 (3万ファイル, CPU, large-v3)

### Step 4b: Cache に注入

```bash
uv run python scripts/inject_whisper_transcripts.py \
    --cache-dir data/cache \
    --dataset galge \
    --transcript-dir data/raw/galge_clean \
    --language ja
```

cache の各 `meta.json` に `text` と `language_id` フィールドが追加される。

## Phase 5: Training

```bash
# VC 学習（Teacher）
uv run tmrvc-train-teacher \
    --cache-dir data/cache \
    --phase 0 \
    --device xpu

# 蒸留
uv run tmrvc-distill \
    --cache-dir data/cache \
    --teacher-ckpt checkpoints/teacher_step8000.pt \
    --phase A \
    --device xpu
```

## 検証コマンド

```bash
# クラスタリング結果の確認
uv run python scripts/cluster_speakers.py \
    --input data/raw/galge_voices --step report

# キャッシュ整合性チェック
uv run tmrvc-verify-cache --cache-dir data/cache --dataset galge

# テスト実行
uv run pytest tests/python/ -k "adapter or cluster or inject"

# 学習の動作確認（少数ステップ）
uv run tmrvc-train-teacher \
    --cache-dir data/cache --phase 0 --device xpu --max-steps 100
```

## 代替フロー: speaker_map で直接前処理（Phase 2 のコピーを省略）

Phase 2 のコピーを省略し、フラットディレクトリから直接 feature extraction する場合:

```yaml
# configs/datasets.yaml
galge:
  type: generic
  enabled: true
  language: ja
  raw_dir: data/raw/galge_voices
  speaker_map: data/raw/galge_voices/_speaker_map.json
```

```bash
uv run python scripts/prepare_datasets.py --datasets galge --device xpu
```

GenericAdapter が `speaker_map` を読み取り、`spk_noise` を自動スキップして処理する。
ディスク容量を節約できるが、品質フィルタ（Phase 2 の duration/RMS チェック）は適用されない点に注意。

## 所要時間の目安

| Phase | 推定時間 | デバイス | 備考 |
|-------|----------|----------|------|
| Survey | ~5分 | CPU | |
| Embedding 抽出 | ~1.5時間 | XPU | レジューム可能 |
| HDBSCAN | ~数秒 | CPU | 再実行可能 |
| Filter+Copy | ~5分 | CPU | |
| Feature Extraction | ~8-10時間 | XPU | --skip-existing で再開可能 |
| ASR (large-v3) | ~12-15時間 | CPU | TTS用、VC-only なら不要 |
| **合計 (VC のみ)** | **~10-12時間** | | |
| **合計 (VC + TTS)** | **~22-28時間** | | |

## トラブルシューティング

### Embedding 抽出が途中で止まった

`_speaker_embeds.npz` に中間保存されているので、同じコマンドを再実行すれば自動レジューム:

```bash
uv run python scripts/cluster_speakers.py --input data/raw/galge_voices --step embed --device xpu
```

### クラスタ数が少なすぎる/多すぎる

`--min-cluster-size` を調整して `--step cluster` を再実行（embedding の再抽出は不要）。

### XPU で DEVICE_LOST エラー

CPU にフォールバックしない。バッチサイズを下げるか、デバイスの回復を待つ。
embedding 抽出は 1 ファイルずつ処理するため、通常は発生しない。

### Feature Extraction が遅い

`--skip-existing` を付けて再開可能。ContentVec/torchcrepe の初回ロードに数分かかる。
