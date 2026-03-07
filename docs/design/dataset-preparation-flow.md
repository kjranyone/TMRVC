# データセット作成フロー設計

この文書は、TMRVC mainline における dataset 登録、前処理、cache 保存の標準仕様を定義する。目的は、`TTS/VC 共通で再利用できる 10 ms 同期 cache` を構築し、text supervision を `MFA 非依存` で扱うことである。

## 1. データセット登録ルール

### 1.1 dataset 単位

- 1 dataset = 1 language
- `configs/datasets.yaml` に登録する
- `enabled: true` の dataset が統合パイプライン対象になる

### 1.2 raw データ配置

#### 話者ごとに分かれている場合

```text
data/raw/my_dataset/
├── speaker_001/
│   └── *.wav
└── speaker_002/
    └── *.wav
```

#### 未分類の場合

```text
data/raw/my_dataset/
└── *.wav
```

必要なら `dev.py` の話者分離メニューで speaker clustering を行う。

## 2. 前処理の責務

`tmrvc-preprocess` / `tmrvc-train-pipeline` は 1 発話から以下を同期抽出する。

1. `codec_tokens.npy` `[8, T]`
2. `control_tokens.npy` `[4, T]`
3. `explicit_state.npy` `[T, 8]`
4. `ssl_state.npy` `[T, D]`
5. `spk_embed.npy` `[E]`
6. `meta.json`
7. `phoneme_ids.npy` または同等の text units

mainline の text supervision は transcript と language backend から作る。forced alignment は要求しない。

## 3. text supervision

### 3.1 ソース

- 既存 transcript
- または ASR による transcript 生成

### 3.2 正規化

- dataset language ごとに normalization を行う
- 日本語は `pyopenjtalk`
- 英語系は `phonemizer` を基本とする

### 3.3 出力

- `meta.json` に normalized text を保持する
- `phoneme_ids.npy` に text unit ids を保存する

`durations.npy` は mainline では生成要件に含めない。

## 4. cache スキーマ

保存先:

```text
{cache_dir}/{dataset}/train/{speaker_id}/{utterance_id}/
```

| ファイル名 | 必須 | 形状 | 説明 |
|---|---|---|---|
| `codec_tokens.npy` | yes | `[8, T]` | acoustic tokens |
| `control_tokens.npy` | yes | `[4, T]` | control tokens |
| `explicit_state.npy` | yes | `[T, 8]` | 物理パラメータ |
| `ssl_state.npy` | yes | `[T, D]` | frame latent |
| `spk_embed.npy` | yes | `[E]` | speaker embedding |
| `meta.json` | yes | - | text, language, n_frames, stats |
| `phoneme_ids.npy` | tts | `[L]` | text unit ids |

互換用 optional artifact:

| ファイル名 | 用途 |
|---|---|
| `durations.npy` | legacy duration ablation |
| `*.TextGrid` | legacy debug / comparison |

## 5. 品質基準

### 5.1 必須

- `n_frames` と frame artifacts の長さが一致する
- waveform 長が `hop_length` に対して整合する
- token 値が語彙範囲内にある
- dataset ごとの speaker / utterance 数が最低条件を満たす

### 5.2 TTS 用

- text が空でない
- `phoneme_ids.npy` が存在する TTS サンプル比率を測定する
- G2P backend 不足時の grapheme fallback は警告として扱う

### 5.3 legacy 指標

- `durations.npy` coverage
- TextGrid coverage

これらは mainline fail 条件ではなく、互換検証用メトリクスである。

## 6. 禁止事項

- dataset 内で言語を混在させる
- `durations.npy` 欠損だけで mainline TTS データを無効扱いする
- forced alignment を cache 作成の必須工程にする
