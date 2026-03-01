# データセット作成フロー設計 (UCLM v2 統合版)

## 概要

本ドキュメントは、生音声ファイルから UCLM v2 (Unified Codec Language Model) の学習に必要な全特徴量を一括抽出する標準パイプラインの設計を定義する。TMRVC では「書き捨てスクリプト」を排し、設定ファイル駆動の再現可能なデータ準備を徹底する。

## 1. データの配置と管理

ユーザーデータは以下のいずれかのルールで配置し、`configs/datasets.yaml` に登録する。

### ルール A: 話者分類済みデータ
```
data/my_dataset/
├── speaker_001/
│   └── *.wav
├── speaker_002/
│   └── *.wav
```
- 最上位のディレクトリ名が自動的に `speaker_id` として認識される。

### ルール B: 未分類データ（自動クラスタリング）
1. `data/unclassified/` に全ての wav を置く。
2. `scripts/eval/cluster_speakers.py` を実行して `speaker_map.json` を生成。
3. `datasets.yaml` の `speaker_map` フィールドにそのパスを記述。

## 2. 統合抽出パイプライン (tmrvc-preprocess)

`tmrvc-preprocess` CLI は、1つの音声から以下の 6 要素を同時に抽出し、10ms 精度で同期させて保存する。

1.  **Acoustic Stream (`A_t`)**: `codec_tokens.npy` [8, T]
2.  **Control Stream (`B_t`)**: `control_tokens.npy` [4, T]
3.  **Physical Voice State**: `explicit_state.npy` [T, 8]
4.  **SSL Latent**: `ssl_state.npy` [T, 128] (WavLM Context)
5.  **Speaker Embed**: `spk_embed.npy` [192]
6.  **TTS Alignment**: `phoneme_ids.npy`, `durations.npy` (自動文字起こし & Forced Alignment)

## 3. 実行手順 (Standard Protocol)

個別のスクリプトを直接叩くことは禁止。常に統合管理スクリプトを経由する。

### 手順 1: レジストリ登録
`configs/datasets.yaml` に対象データセットを追加・有効化（`enabled: true`）する。

### 手順 2: パイプライン実行
```bash
# 全有効データセットに対して一括処理を実行
uv run python scripts/data/prepare_datasets.py --device cuda --skip-existing
```

## 4. 特徴量保存スキーマ

各発話はキャッシュディレクトリ内の以下の構造に保存される：
`{cache_dir}/{dataset}/train/{speaker_id}/{utterance_id}/`

| ファイル名 | 形状 | 説明 |
|:---|:---|:---|
| `codec_tokens.npy` | [8, T] | UCLM v2 Acoustic Tokens |
| `control_tokens.npy` | [4, T] | UCLM v2 Control Tokens (op, type, dur, int) |
| `explicit_state.npy` | [T, 8] | Physical parameters (breathiness, tension, etc.) |
| `ssl_state.npy` | [T, 128] | WavLM large layer 7 features |
| `spk_embed.npy` | [192] | ECAPA-TDNN speaker embedding |
| `phoneme_ids.npy` | [L] | Phoneme indices (TTS Mode) |
| `durations.npy` | [L] | Phoneme durations in frames (TTS Mode) |
| `meta.json` | - | Transcription, sample rate, and stats |

## 5. 品質保証 (Manifest)

処理完了後、`data/cache/_manifests/{dataset}_train.json` が自動生成される。このファイルに記載された `n_utterances` や `total_duration_sec` を確認し、データの欠落がないか検証すること。
