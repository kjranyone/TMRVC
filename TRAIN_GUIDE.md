# TMRVC Training Guide

この文書は、現行 mainline である `UCLM v3` の学習フローをまとめたものです。前提は次のとおりです。

- TTS / VC は単一の UCLM backbone で学習する
- TTS の主経路は `pointer-based internal alignment` (MFA 不要)
- dual-stream token contract (`A_t / B_t`) は維持する

## 1. 学習成果物

運用に必要な主成果物は次の 2 つです。

- `checkpoints/uclm/uclm_latest.pt`
- `checkpoints/codec/codec_latest.pt`

`UCLM` は token 予測器、`codec` は波形と token の相互変換器です。片方だけでは実運用できません。

## 2. データ要件

### 2.1 データセット単位の原則

- 1 dataset = 1 language
- dataset ごとに `raw_dir` を分ける
- 話者別フォルダでも単一フォルダでもよい
- テキストがある方が TTS 学習は安定する

### 2.2 TTS 監督

mainline の TTS 監督は次を使います。

- `text`: 発話テキスト
- `phoneme_ids.npy` または同等の text units
- 音声側の `codec_tokens.npy`, `control_tokens.npy`, `voice_state.npy`, `ssl_state.npy`

使わないもの:

- `MFA` 前提の forced alignment
- `durations.npy` 必須設計
- TextGrid を品質の中心に置く運用

`phoneme_ids.npy` は言語別 G2P または grapheme backend から作ります。英語系は `phonemizer + espeak-ng`、日本語は `pyopenjtalk` を使います。

## 3. cache スキーマ

標準 cache は以下を持ちます。

| ファイル | 必須 | 役割 |
|---|---|---|
| `codec_tokens.npy` | yes | acoustic tokens `[8, T]` |
| `control_tokens.npy` | yes | control tokens `[4, T]` |
| `voice_state.npy` | optional | canonical physical-control supervision `[T, 8]` |
| `ssl_state.npy` | yes | frame-level latent state `[T, D]` |
| `spk_embed.npy` | yes | speaker embedding |
| `meta.json` | yes | text, language, frame stats |
| `phoneme_ids.npy` | tts | text unit ids |

互換 alias:

- `explicit_state.npy` (`voice_state.npy` の alias)

## 4. dev.py ベースの標準フロー

### 4.1 初回セットアップ

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
sudo apt-get update && sudo apt-get install -y espeak-ng
uv run python dev.py
```

### 4.2 推奨順序

1. `6` 設定初期化
2. `4` データセット追加 (raw_dir, language 設定)
3. `1` フル学習 (前処理 + 学習)
4. `8` Codec 学習
5. `7` 学習成果物を確定
6. `11` 推論サーバー起動

`1` が推奨フローです。前処理→学習を一括実行します。

### 4.3 再学習

前処理済み cache を再利用する場合:

1. `2` 既存キャッシュで学習のみ
2. `7` 成果物を確定
3. `11` 推論サーバー起動

### 4.4 キュレーション経由のデータ準備

raw音声から学習データを準備する場合:

1. `13` キュレーション: 音声ファイル取込 (ingest)
2. `14` キュレーション: スコアリング & 昇格判定 (run)
3. `16` キュレーション: エクスポート (promoted → cache)
4. `1` フル学習

## 5. CLI ベースの実行

### 5.1 前処理のみ

```bash
uv run tmrvc-preprocess \
  --raw-dir /path/to/raw \
  --cache-dir experiments/cache \
  --dataset mydata \
  --language ja \
  --workers 2
```

### 5.2 統合パイプライン (推奨)

```bash
uv run tmrvc-train-pipeline \
  --output-dir experiments \
  --workers 2 \
  --train-device cuda \
  --seed 42
```

### 5.3 既存 cache で学習のみ

```bash
uv run tmrvc-train-pipeline \
  --output-dir experiments \
  --cache-dir experiments/<exp_id>/cache \
  --skip-preprocess \
  --train-device cuda \
  --seed 42
```

### 5.4 codec 学習

```bash
uv run tmrvc-train-codec \
  --cache-dir experiments/<exp_id>/cache \
  --output-dir checkpoints/codec \
  --device cuda
```

### 5.5 UCLM 単体学習 (cache 既存)

```bash
uv run tmrvc-train-uclm \
  --cache-dir experiments/<exp_id>/cache \
  --output-dir checkpoints/uclm \
  --batch-size 16 \
  --max-steps 10000 \
  --device cuda
```

## 6. 利用可能な CLI コマンド一覧

| コマンド | パッケージ | 説明 |
|---|---|---|
| `tmrvc-preprocess` | tmrvc-data | 音声の前処理・特徴量抽出 |
| `tmrvc-extract-features` | tmrvc-data | 特徴量抽出のみ |
| `tmrvc-verify-cache` | tmrvc-data | cache 整合性検証 |
| `tmrvc-prepare-uclm` | tmrvc-data | UCLM 用データ準備 |
| `tmrvc-curate` / `tmrvc-curation` | tmrvc-data | キュレーション CLI |
| `tmrvc-train-pipeline` | tmrvc-train | 統合学習パイプライン (推奨) |
| `tmrvc-train-uclm` | tmrvc-train | UCLM 単体学習 |
| `tmrvc-train-codec` | tmrvc-train | Codec 学習 |
| `tmrvc-finetune-encodec` | tmrvc-train | EnCodec fine-tuning |
| `tmrvc-serve` | tmrvc-serve | 推論サーバー起動 |

## 7. Quality Gate

学習前に cache 品質を検査します。主な検査対象は次のとおりです。

- 必須 artifact の欠損率
- token range 異常
- dataset ごとの最小 utterance / speaker 数
- waveform length と `n_frames * hop_length` の整合
- TTS text coverage

legacy alignment coverage は mainline fail 条件ではなく、別指標として扱います。

## 8. G2P と言語依存

### 8.1 日本語

- backend: `pyopenjtalk`
- dataset `language: ja`

### 8.2 英語

- backend: `phonemizer`
- system dependency: `espeak-ng`
- dataset `language: en`

### 8.3 多言語運用

多言語コーパスであっても、`datasets.yaml` の 1 entry は単一言語に分割してください。language ごとに backend と text normalization が異なるため、混在運用は mainline 品質を崩します。

## 9. 次に参照すべき文書

- 設計入口: `docs/design/architecture.md`
- モデル仕様: `docs/design/unified-codec-lm.md`
- データ準備: `docs/design/dataset-preparation-flow.md`
- 実装計画: `plan/README.md`
