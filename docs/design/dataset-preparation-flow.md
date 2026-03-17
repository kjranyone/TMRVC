# データセット作成フロー設計

この文書は、TMRVC mainline における dataset 登録、前処理、cache 保存の標準仕様の正本である。
active backlog が必要な場合のみ `plan/README.md` と `plan/repo_remaining_inventory_2026_03.md` を参照する。
目的は、`TTS/VC 共通で再利用できる 10 ms 同期 cache` を構築し、text supervision を
`MFA 非依存` で扱うことである。

## 1. データセット登録ルール

### 1.1 dataset 単位

- `configs/datasets.yaml` に登録する
- `enabled: true` の dataset が統合パイプライン対象になる
- monolingual dataset を default とする
- multilingual / code-switch dataset も mainline で許可する。ただし以下が必須:
  - utterance-level `language_id`
  - token-span or segment-level `language_spans` (code-switch がある場合)
  - report の per-language / code-switch stratification

禁止されるのは「混在言語そのもの」ではなく、言語メタデータなしの undocumented mixed-language cache である。

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

必要なら話者分離や speaker clustering を前段で行う。

## 2. 前処理の責務

`tmrvc-preprocess` / `tmrvc-train-pipeline` は 1 発話から以下を同期抽出する。

### 2.1 Required mainline artifacts

1. `codec_tokens.npy` `[8, T]`
2. `control_tokens.npy` `[4, T]`
3. `ssl_state.npy` `[T, D_ssl]`
4. `spk_embed.npy` `[E]`
5. `meta.json`
6. `phoneme_ids.npy` または同等の canonical text units
7. optional `bootstrap_alignment.json`

### 2.2 Optional physical-control supervision bundle

この bundle は `explicit runtime control` と同じ意味空間を持つ supervised artifact であり、
存在しない場合も mainline cache は valid である。

1. `voice_state.npy` `[T, 8]`
2. `voice_state_observed_mask.npy` `[T, 8]`
3. `voice_state_confidence.npy` `[T, 8]` or `[T, 1]`
4. `voice_state_meta.json`
5. optional `explicit_state.npy` compatibility alias

`voice_state.npy` は canonical supervision artifact 名であり、
「常時存在する base preprocessing output」と「たまにある pseudo-label artifact」を兼用しない。
存在する場合は supervision bundle の一部として扱う。

mainline の text supervision は transcript と language backend から作る。forced alignment は要求しない。

## 3. text supervision

### 3.1 ソース

- 既存 transcript
- または ASR による transcript 生成

### 3.2 正規化

- dataset language ごとに normalization を行う
- 日本語は pitch accent 情報を保持できる backend を要求する
- 英語系は `phonemizer` を基本とする
- G2P が弱い場合は normalized text を保持し、fallback mode を metadata に明示する

### 3.3 出力

- `meta.json` に normalized text を保持する
- `phoneme_ids.npy` に canonical text unit ids を保存する
- multilingual / code-switch utterance は `meta.json` に `language_id` と optional `language_spans` を保持する

`durations.npy` は mainline では生成要件に含めない。

## 4. cache スキーマ

保存先:

```text
{cache_dir}/{dataset}/train/{speaker_id}/{utterance_id}/
```

### 4.1 Core artifacts

| ファイル名 | 必須 | 形状 | 説明 |
|---|---|---|---|
| `codec_tokens.npy` | yes | `[8, T]` | acoustic tokens |
| `control_tokens.npy` | yes | `[4, T]` | control tokens |
| `ssl_state.npy` | yes | `[T, D]` | frame latent |
| `spk_embed.npy` | yes | `[E]` | speaker embedding |
| `meta.json` | yes | - | text, language, n_frames, stats, fallback metadata |
| `phoneme_ids.npy` | tts | `[L]` | canonical text unit ids |
| `bootstrap_alignment.json` | optional | - | phoneme-indexed transitional alignment labels |

### 4.2 Optional physical-control supervision artifacts

| ファイル名 | 必須 | 形状 | 説明 |
|---|---|---|---|
| `voice_state.npy` | optional | `[T, 8]` | physical-control targets |
| `voice_state_observed_mask.npy` | optional with `voice_state.npy` | `[T, 8]` | 観測済み次元 mask |
| `voice_state_confidence.npy` | optional with `voice_state.npy` | `[T, 8]` or `[T, 1]` | pseudo-label confidence |
| `voice_state_meta.json` | optional with `voice_state.npy` | - | estimator, calibration, provenance |
| `explicit_state.npy` | compatibility only | `[T, 8]` | legacy alias |

`voice_state.npy` を export するなら、少なくとも mask と provenance を伴うこと。
unknown 次元を dense zero として保存してはならない。

### 4.3 Legacy-only optional artifacts

| ファイル名 | 用途 |
|---|---|
| `durations.npy` | legacy duration ablation |
| `*.TextGrid` | legacy debug / comparison |

## 5. `meta.json` 最低フィールド

- `dataset_id`
- `speaker_id`
- `utterance_id`
- `language_id`
- optional `language_spans`
- `normalized_text`
- optional `g2p_fallback_mode`
- `n_frames`
- `num_samples`
- `sample_rate`
- `quality_score` when curated export provides it
- optional `voice_state_supervision_available`
- optional `voice_state_supervision_density`
- optional `voice_state_supervision_source`

## 6. 品質基準

### 6.1 必須

- `n_frames` と frame artifacts の長さが一致する
- waveform 長が `hop_length` に対して整合する
- token 値が語彙範囲内にある
- dataset ごとの speaker / utterance 数が最低条件を満たす
- frame convention は `sample_rate = 24000`, `hop_length = 240`, `T = ceil(num_samples / 240)` に固定する
- `bootstrap_alignment.json` を出力する場合、`start_frame` は inclusive、`end_frame` は exclusive とする

### 6.2 TTS 用

- text が空でない
- `phoneme_ids.npy` が存在する TTS サンプル比率を測定する
- G2P backend 不足時の grapheme / byte fallback は warning として報告し、silent success にしない
- multilingual / code-switch dataset は per-language と mixed-span の両方で coverage を報告する

### 6.3 Physical-control supervision 用

- `voice_state.npy` がある場合、`voice_state_observed_mask.npy` と provenance を伴う
- `voice_state_confidence.npy` がある場合、shape と calibration version を report する
- coverage / observed ratio / confidence summary を dataset report に含める
- missing or low-confidence 次元を neutral state と解釈しない

### 6.4 legacy 指標

- `durations.npy` coverage
- TextGrid coverage

これらは mainline fail 条件ではなく、互換検証用メトリクスである。

## 7. 禁止事項

- undocumented mixed-language dataset を mainline train-ready と見なすこと
- `durations.npy` 欠損だけで mainline TTS データを無効扱いすること
- forced alignment を cache 作成の必須工程にすること
- `voice_state` 未観測次元を 0 で埋めて教師信号扱いすること
- `bootstrap_alignment.json` の frame index を独自規約で保存すること
