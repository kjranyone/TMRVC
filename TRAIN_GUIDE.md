# TMRVC Training Guide

この文書は、現行 mainline である `UCLM v3` の学習フローをまとめたものです。前提は次のとおりです。

- TTS / VC は単一の UCLM backbone で学習する
- TTS の主経路は `pointer-based internal alignment`
- `MFA` や `durations.npy` は mainline では必須にしない
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
- 音声側の `codec_tokens.npy`, `control_tokens.npy`, `explicit_state.npy`, `ssl_state.npy`

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
| `explicit_state.npy` | yes | 8-dim physical voice state `[T, 8]` |
| `ssl_state.npy` | yes | frame-level latent state `[T, D]` |
| `spk_embed.npy` | yes | speaker embedding |
| `meta.json` | yes | text, language, frame stats |
| `phoneme_ids.npy` | tts | text unit ids |

互換用 legacy artifact:

- `durations.npy`
- `TextGrid`

これらは比較実験や旧経路の検証では読めますが、mainline 学習の必須条件ではありません。

## 4. dev.py ベースの標準フロー

### 4.1 初回セットアップ

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
sudo apt-get update && sudo apt-get install -y espeak-ng
uv run python dev.py
```

### 4.2 推奨順序

1. `6` 設定初期化
2. `4` データセット追加
3. `14` v3 ポインタ: フル学習 (MFA不要)
4. `8` Codec 学習
5. `7` 学習成果物を確定
6. `11` 推論サーバー起動

`14` が v3 mainline の推奨フローです。MFA を必要とせず、前処理→学習を一括実行します。

### 4.3 再学習

前処理済み cache を再利用する場合:

1. `15` v3 ポインタ: 既存キャッシュで学習
2. `7` 成果物を確定
3. `11` 推論サーバー起動

### 4.4 v2 Legacy フロー

MFA ベースの duration 学習が必要な場合は `1` (v2 legacy フル学習) を使用してください。

## 5. CLI ベースの実行

### 5.1 v3 ポインタモード統合パイプライン (推奨)

```bash
uv run tmrvc-train-pipeline \
  --output-dir experiments \
  --workers 2 \
  --seed 42 \
  --tts-mode pointer
```

### 5.2 既存 cache で v3 ポインタ学習のみ

```bash
uv run tmrvc-train-pipeline \
  --output-dir experiments \
  --cache-dir experiments/<exp_id>/cache \
  --skip-preprocess \
  --train-device cuda \
  --seed 42 \
  --tts-mode pointer
```

### 5.3 v2 Legacy (MFA duration モード)

```bash
uv run tmrvc-train-pipeline \
  --output-dir experiments \
  --workers 2 \
  --seed 42 \
  --tts-mode legacy_duration
```

### 5.4 codec 学習

```bash
uv run tmrvc-train-codec \
  --cache-dir experiments/<exp_id>/cache \
  --output-dir checkpoints/codec \
  --device cuda
```

## 6. Quality Gate

学習前に cache 品質を検査します。主な検査対象は次のとおりです。

- 必須 artifact の欠損率
- token range 異常
- dataset ごとの最小 utterance / speaker 数
- waveform length と `n_frames * hop_length` の整合
- TTS text coverage

legacy alignment coverage は mainline fail 条件ではなく、別指標として扱います。

## 7. G2P と言語依存

### 7.1 日本語

- backend: `pyopenjtalk`
- dataset `language: ja`

### 7.2 英語

- backend: `phonemizer`
- system dependency: `espeak-ng`
- dataset `language: en`

### 7.3 多言語運用

多言語コーパスであっても、`datasets.yaml` の 1 entry は単一言語に分割してください。language ごとに backend と text normalization が異なるため、混在運用は mainline 品質を崩します。

## 8. legacy 経路

`dev.py` の `12` と `13` は legacy alignment utilities です。

- 比較実験
- 旧 checkpoint 再現
- デバッグ

には使えますが、mainline 仕様ではありません。新規学習での標準フローとしては扱いません。

## 9. 次に参照すべき文書

- 設計入口: `docs/design/architecture.md`
- モデル仕様: `docs/design/unified-codec-lm.md`
- データ準備: `docs/design/dataset-preparation-flow.md`
- 実装計画: `plan/README.md`
