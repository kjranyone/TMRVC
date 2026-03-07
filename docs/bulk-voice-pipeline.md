# Bulk Voice Pipeline

大量の未整理 WAV 群を TMRVC mainline 学習データへ変換する運用ガイド。対象は `speaker 未整理 / transcript 未整備 / mixed quality` な音声資産である。

## 1. 目的

- 話者クラスタリング
- 品質フィルタ
- transcript / text supervision 整備
- mainline cache 生成

`MFA` は前提にしない。必要なのは、音声を dataset 単位で単一言語に分割し、text units を作れる状態まで持っていくことである。

## 2. 推奨フロー

1. `report`: raw 音声の長さ・壊れファイル・サンプルレートを調査
2. `speaker clustering`: 話者ごとに整理
3. `language split`: dataset を単一言語へ分割
4. `transcript preparation`: 既存 transcript または ASR で text を作る
5. `tmrvc-preprocess` または `dev.py -> 1` で cache 化
6. quality gate で text coverage と token 整合性を確認

## 3. 成果物

- speaker-separated raw dataset
- `configs/datasets.yaml` entry
- mainline cache
  - `codec_tokens.npy`
  - `control_tokens.npy`
  - `explicit_state.npy`
  - `ssl_state.npy`
  - `spk_embed.npy`
  - `meta.json`
  - `phoneme_ids.npy` when TTS supervision is available

## 4. 注意点

- dataset 内で多言語を混ぜない
- grapheme fallback は進行できても品質警告として扱う
- legacy alignment artifact は比較用途に限定する

詳細は `docs/design/dataset-preparation-flow.md` と `TRAIN_GUIDE.md` を参照。
