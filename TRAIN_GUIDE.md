# TMRVC v4 Training Guide

UCLMv4 の学習フローをまとめた文書。全操作は冪等に設計されている。

## 原則

1. **冪等性**: どのコマンドも何度実行しても同じ結果を返す。途中で止まっても再実行で続きから動く
2. **増分処理**: データ追加は append-only。cache 構築は未処理分だけ実行される
3. **単一台帳**: `data/manifest.jsonl` が全 utterance の source of truth
4. **checkpoint 再開**: 学習は任意の step から再開可能。optimizer state も復元される
5. **アノテーション不要**: raw wav を投入すれば ASR・G2P・話者・声質が自動推定される

## 成果物

| ファイル | 役割 |
|---|---|
| `checkpoints/v4_full/v4_step_*.pt` | UCLM 本体 + acting encoder/predictor + optimizer |
| `checkpoints/codec/codec_latest.pt` | EnCodec (condition A baseline) |
| `data/manifest.jsonl` | データ台帳 (corpus, speaker, wav_path, cached) |
| `data/cache/v4/train/` | 学習 cache (codec tokens, voice state, phonemes 等) |

## Phase 0: 環境

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
sudo apt-get update && sudo apt-get install -y espeak-ng
```

## Phase 1: データ登録

`manage_data.py add` で raw 音声を台帳に登録する。ファイルは移動されない。

```bash
# 話者がディレクトリで分かれている場合
.venv/bin/python scripts/manage_data.py add <name> <path> --speaker-from-dir

# 1 話者 or 話者不明の場合
.venv/bin/python scripts/manage_data.py add <name> <path> --speaker single

# flac など wav 以外
.venv/bin/python scripts/manage_data.py add <name> <path> --speaker-from-dir --ext flac
```

**冪等性**: 同じ wav は二重登録されない。`--force` で再登録。

```bash
# 例: 初回セットアップ
.venv/bin/python scripts/manage_data.py add jvs data/raw/jvs --speaker-from-dir
.venv/bin/python scripts/manage_data.py add moe data/raw/moe_voices --speaker-from-dir
.venv/bin/python scripts/manage_data.py add tsukuyomi data/raw/tsukuyomi --speaker single
.venv/bin/python scripts/manage_data.py add vctk data/raw/vctk --speaker-from-dir --ext flac

# 例: 後から追加
.venv/bin/python scripts/manage_data.py add my_voice ~/recordings --speaker single
.venv/bin/python scripts/manage_data.py add drama ~/drama_corpus --speaker-from-dir
```

確認:

```bash
.venv/bin/python scripts/manage_data.py status
```

## Phase 2: Cache 構築

`manage_data.py build` で manifest の未処理 entry を cache に変換する。

```bash
# 全量
.venv/bin/python scripts/manage_data.py build

# 上限指定 (時間を区切りたいとき)
.venv/bin/python scripts/manage_data.py build --max 10000
```

**冪等性**: 処理済み entry はスキップされる。中断しても再実行で続きから。

内部で実行されるパイプライン:

| ステップ | モデル | 出力 |
|---|---|---|
| ASR | Whisper large-v3 | transcript, language, confidence |
| G2P | pyopenjtalk / phonemizer | `phoneme_ids.npy` |
| Voice State | DSP 12-D estimator | `voice_state.npy` [T, 12] |
| Speaker | ECAPA-TDNN | `spk_embed.npy` [192] |
| Codec | EnCodec 24kHz | `codec_tokens.npy` [8, T] |

### Cache スキーマ (v4)

| ファイル | shape | 役割 |
|---|---|---|
| `codec_tokens.npy` | `[8, T]` | acoustic tokens (RVQ 8 codebooks × 1024 vocab) |
| `voice_state.npy` | `[T, 12]` | 12-D physical voice state |
| `voice_state_targets.npy` | `[T, 12]` | supervision target |
| `voice_state_observed_mask.npy` | `[T, 12]` | 観測マスク (dim 8-11 は低信頼) |
| `voice_state_confidence.npy` | `[T, 12]` | フレーム別信頼度 |
| `spk_embed.npy` | `[192]` | speaker embedding |
| `phoneme_ids.npy` | `[L]` | 音素 ID 列 |
| `text_suprasegmentals.npy` | `[L, 4]` | 韻律素性 |
| `meta.json` | — | text, language, duration, tier, corpus |

## Phase 3: 学習

```bash
# 新規学習
.venv/bin/python scripts/train_v4_full.py \
  --steps 100000 \
  --batch-size 4 \
  --lr 3e-4 \
  --skip-cache \
  --save-every 5000

# checkpoint から再開
.venv/bin/python scripts/train_v4_full.py \
  --resume-from <step> \
  --steps 100000 \
  --batch-size 4 \
  --lr 3e-4 \
  --skip-cache \
  --save-every 5000
```

**冪等性**: `--resume-from` は checkpoint から model + optimizer state を復元する。d_model 等のハイパーパラメータは checkpoint から自動推定される。

### 学習フェーズ (curriculum)

| Phase | Steps | 内容 |
|---|---|---|
| Stage 1 | 0–2,000 | Base LM, codec token 予測 |
| Stage 2 | 2,000–15,000 | Pointer alignment + text progression |
| Stage 3 | 15,000– | Drama/dialogue, CFG, acting latent |

### Loss 構成 (9 項)

| Loss | 対象 |
|---|---|
| loss_a | Stream A codec token prediction |
| loss_b | Stream B control token prediction |
| loss_adv | Pointer advance/hold decision |
| loss_progress | Pointer progress delta |
| loss_boundary_confidence | 音素境界信頼度 |
| loss_voice_state | 12-D physical supervision |
| loss_bio_* | 生物学的制約 (covariance, transition, implausibility) |
| loss_acting_kl | Acting latent KL 正則化 |
| loss_disentangle | Physical / latent 直交性 |

## Phase 4: 検証

```bash
# サンプル生成
.venv/bin/python scripts/generate_v4_sample.py

# メスガキ検証
.venv/bin/python scripts/verify_mesugaki.py --checkpoint checkpoints/v4_full/v4_step_50000.pt
```

## Phase 5: サーブ

```bash
.venv/bin/python -m tmrvc_serve --checkpoint checkpoints/v4_full/v4_full_final.pt
```

## データ追加ワークフロー (学習途中でも可能)

```bash
# 1. データ登録
.venv/bin/python scripts/manage_data.py add new_corpus /path/to/wavs --speaker-from-dir

# 2. cache 構築 (増分)
.venv/bin/python scripts/manage_data.py build

# 3. 学習再開 (新しい cache を含む)
.venv/bin/python scripts/train_v4_full.py --resume-from <last_step> --steps 200000 --skip-cache
```

全ステップが冪等。途中で止まっても同じコマンドを再実行すればよい。

## データ管理コマンド一覧

```bash
# 登録
manage_data.py add <name> <path> [--speaker-from-dir] [--speaker single] [--ext wav] [--force]

# cache 構築
manage_data.py build [--max N] [--device auto]

# 状態確認
manage_data.py status

# コーパス除外
manage_data.py remove <name>
```

## ディレクトリ構成

```
data/
├── manifest.jsonl              # 全 utterance の台帳 (source of truth)
├── raw/                        # 元音声 (読み取り専用)
│   ├── jvs/                    #   JVS corpus (ja)
│   ├── moe_voices/             #   MOE voices (ja, 5 speakers)
│   ├── tsukuyomi/              #   つくよみちゃん (ja)
│   ├── vctk/                   #   VCTK corpus (en, 110 speakers)
│   └── <your_corpus>/          #   任意のコーパスを追加
├── cache/
│   └── v4/
│       └── train/
│           └── <speaker_id>/
│               └── <utt_id>/
│                   ├── codec_tokens.npy
│                   ├── voice_state.npy
│                   ├── spk_embed.npy
│                   ├── phoneme_ids.npy
│                   └── meta.json
├── curated_export/             # キュレーション出力
└── curation/                   # キュレーション DB
```

## G2P

| 言語 | Backend | 依存 |
|---|---|---|
| ja | pyopenjtalk | pip install |
| en | phonemizer + espeak-ng | apt install espeak-ng |

言語は Whisper が自動判定する。G2P backend は言語に応じて自動選択される。

## dev.py との関係

`dev.py` は対話メニュー型のエントリポイントとして残っている。

| dev.py menu | 新体制での推奨 |
|---|---|
| 1. Bootstrap | `manage_data.py add` + `build` |
| 2. Training | `train_v4_full.py --resume-from` |
| 3. RL fine-tuning | dev.py menu 3 をそのまま使う |
| 4. Dataset management | `manage_data.py add/status/remove` |
| 5. Curation | dev.py menu 5 をそのまま使う |
| 6. Finalize | dev.py menu 6 をそのまま使う |
| 7. Characters | dev.py menu 7 をそのまま使う |
| 8. Serve | dev.py menu 8 をそのまま使う |
| 9. Integrity check | dev.py menu 9 をそのまま使う |

v4 の学習ループ (add → build → train) は CLI 直接実行を推奨する。dev.py はキャラクター管理・キュレーション・RL 等の補助操作に使う。

## FAQ

**Q: アノテーション無しの wav を入れてよいか？**
A: よい。Whisper ASR → G2P → DSP voice state → speaker embed が全自動で実行される。

**Q: 1 ファイルが長い (30 秒超) 場合は？**
A: bootstrap pipeline は VAD で無音区間を除外する。長すぎるファイルは学習時に truncate される。

**Q: 途中で新しいデータを足したい**
A: `manage_data.py add` → `build` → `train --resume-from`。既存 cache は壊れない。

**Q: 学習が途中で死んだ**
A: 同じコマンドに `--resume-from <last_saved_step>` を付けて再実行。

**Q: cache を作り直したい**
A: `data/cache/v4/` を削除して `manage_data.py build` を再実行。manifest の `cached` フラグは自動更新される。
