# データ取得・前処理ガイド

このガイドは、TMRVC mainline 学習に必要なデータ取得と整備の原則をまとめる。焦点は `再現可能な dataset registration` と `MFA 非依存の text supervision` である。

## 1. 前提

- Python `3.12+`
- `uv`
- dataset ごとに単一言語
- raw 音声は `data/raw/<dataset>/...`

## 2. 取得後に確認すること

1. ライセンス
2. サンプルレートと音声長
3. 話者構造
4. transcript の有無
5. dataset language

## 3. dataset 登録

`configs/datasets.yaml` に dataset を登録する。最低限必要なのは次の項目である。

```yaml
datasets:
  my_dataset:
    type: generic
    enabled: true
    language: ja
    raw_dir: data/raw/my_dataset
```

## 4. text supervision

mainline では次を使う。

- transcript
- G2P / grapheme backend による text-unit 化

使わないもの:

- forced alignment の必須化
- TextGrid 必須化
- `durations.npy` 必須化

## 5. 実行

### dev.py

```bash
uv run python dev.py
```

推奨順:

1. `6` 設定初期化
2. `4` データセット追加
3. `1` フル学習

### CLI

```bash
uv run tmrvc-train-pipeline --output-dir experiments --workers 2
```

## 6. 出力

mainline cache の代表 artifact:

- `codec_tokens.npy`
- `control_tokens.npy`
- `explicit_state.npy`
- `ssl_state.npy`
- `spk_embed.npy`
- `meta.json`
- `phoneme_ids.npy` when TTS supervision exists

## 7. 参考

- `docs/design/dataset-preparation-flow.md`
- `TRAIN_GUIDE.md`
- `docs/bulk-voice-pipeline.md`
