# データ取得・前処理ガイド

本ドキュメントは TMRVC の学習に必要な全データセットの取得・前処理手順をまとめた再現性ガイドである。

---

## 前提条件

| 項目 | 要件 |
|---|---|
| Python | 3.12+ |
| パッケージマネージャ | uv (workspace 構成) |
| ストレージ (raw) | ~100 GB (全データセット) |
| ストレージ (cache) | ~20-30 GB (前処理済み特徴量) |
| GPU | Intel Arc (XPU) or CUDA (特徴量抽出の高速化) |
| OS | Windows 11 / Linux |

```bash
# uv workspace の同期 (初回のみ)
uv sync

# XPU 利用可否の確認
uv run python -c "import torch; print(torch.xpu.is_available())"
```

---

## ディレクトリ構成

```
data/
├── raw/                    # 生音声 (gitignore 対象)
│   ├── VCTK-Corpus/        # VC Phase 0-2
│   ├── jvs_corpus/         # VC Phase 0-2
│   ├── tsukuyomi/          # VC Phase 0-2
│   ├── libritts_r/         # VC Phase 1a+
│   ├── jsut/               # TTS Phase 2
│   ├── ljspeech/           # TTS Phase 2
│   ├── jvnv/               # Expressive Phase 3
│   ├── expresso/           # Expressive Phase 3
│   ├── EmoV-DB/            # Expressive Phase 3
│   ├── ravdess/            # Expressive Phase 3
│   └── rir/                # RIR augmentation Phase 2
│       ├── air/
│       └── but_reverb/
├── cache/                  # 前処理済み (gitignore、再生成可能)
│   ├── _manifests/
│   ├── vctk/train/
│   ├── jvs/train/
│   └── ...
└── sample_voice/           # デモ用
```

> **重要**: `data/raw/` は絶対に git に追加しないこと (VCTK だけで ~11 GB)。

---

## Phase 別データ要件

| Phase | 目的 | 必須データ | 任意データ |
|---|---|---|---|
| **0** (Teacher) | VC 基盤 | VCTK, JVS, tsukuyomi | — |
| **1a** (Teacher) | WavLM 移行 | VCTK, JVS, tsukuyomi | LibriTTS-R |
| **1b-2** (Teacher) | STFT/IR-robust | 同上 + RIR | — |
| **蒸留** | Student | (Phase 2 と同じ cache) | — |
| **TTS Phase 2** | 基本 TTS | JSUT, LJSpeech | VCTK, JVS |
| **TTS Phase 3** | 表現的 TTS | Expresso, JVNV | EmoV-DB, RAVDESS |

---

## 1. VC 学習データセット

### 1.1 VCTK Corpus (英語, 109 話者)

| 項目 | 内容 |
|---|---|
| サイズ | ~11 GB (ZIP) → ~15 GB (展開後) |
| 時間 | ~44h |
| SR | 48 kHz |
| ライセンス | CC BY 4.0 |
| URL | https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip |

```bash
# 自動ダウンロード
uv run python scripts/download_vctk.py --output-dir data/raw

# 展開先: data/raw/VCTK-Corpus-0.92/
# datasets.yaml の raw_dir: data/raw/VCTK-Corpus
```

**ディレクトリ構造:**
```
data/raw/VCTK-Corpus-0.92/
├── wav48_silence_trimmed/
│   ├── p225/
│   │   ├── p225_001_mic1.flac
│   │   └── ...
│   └── p376/
└── txt/
    ├── p225/
    └── ...
```

### 1.2 JVS Corpus (日本語, 100 話者)

| 項目 | 内容 |
|---|---|
| サイズ | ~3 GB |
| 時間 | ~30h |
| SR | 24 kHz |
| ライセンス | CC BY-SA 4.0 |
| 取得方法 | 手動ダウンロード |

**手順:**
1. https://sites.google.com/site/shinaborumiethlab/page3/jvs_corpus にアクセス
2. 利用規約を確認してダウンロード
3. 展開先: `data/raw/jvs_corpus/`

**ディレクトリ構造:**
```
data/raw/jvs_corpus/
├── jvs001/
│   ├── parallel100/
│   │   └── wav24kHz16bit/
│   │       ├── VOICEACTRESS100_001.wav
│   │       └── ...
│   └── nonpara30/
│       └── wav24kHz16bit/
└── jvs100/
```

### 1.3 Tsukuyomi (日本語, 単一話者)

| 項目 | 内容 |
|---|---|
| サイズ | ~300 MB |
| 時間 | ~0.3h |
| SR | 96 kHz |
| 取得方法 | リポジトリ内に配置済み |

```
data/raw/tsukuyomi/corpus1/
```

### 1.4 LibriTTS-R (英語, ~2,456 話者)

| 項目 | 内容 |
|---|---|
| サイズ | ~65 GB |
| 時間 | ~585h |
| SR | 24 kHz |
| ライセンス | CC BY 4.0 |
| URL | https://www.openslr.org/141/ |
| 必要フェーズ | Phase 1a 以降 (任意) |

**手順:**
1. https://www.openslr.org/141/ から以下をダウンロード:
   - `train-clean-100.tar.gz` (~5.5 GB) — 100h
   - `train-clean-360.tar.gz` (~20 GB) — 360h
   - `train-other-500.tar.gz` (~30 GB) — 500h
2. 全て `data/raw/libritts_r/` に展開

```bash
# 例: train-clean-100 のダウンロードと展開
wget https://www.openslr.org/resources/141/train_clean_100.tar.gz
tar xzf train_clean_100.tar.gz -C data/raw/libritts_r/
```

**ディレクトリ構造:**
```
data/raw/libritts_r/
├── train-clean-100/
│   └── {speaker_id}/
│       └── {chapter_id}/
│           └── {speaker}_{chapter}_{utt}.wav
├── train-clean-360/
└── train-other-500/
```

---

## 2. TTS 学習データセット

### 2.1 JSUT (日本語, 単一話者)

| 項目 | 内容 |
|---|---|
| サイズ | ~2 GB |
| 時間 | ~10h |
| SR | 48 kHz |
| ライセンス | 非商用・再配布不可 |
| URL | https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip |

```bash
# 自動ダウンロード
uv run python scripts/download_datasets.py --dataset jsut --output-dir data/raw

# 展開先: data/raw/jsut/
```

### 2.2 LJSpeech (英語, 単一話者)

| 項目 | 内容 |
|---|---|
| サイズ | ~2.6 GB |
| 時間 | ~24h |
| SR | 22.05 kHz |
| ライセンス | Public Domain |
| URL | https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 |

```bash
# 自動ダウンロード
uv run python scripts/download_datasets.py --dataset ljspeech --output-dir data/raw

# 展開先: data/raw/ljspeech/
```

---

## 3. 感情・表現データセット

### 3.1 Expresso (英語, 26 スタイル)

| 項目 | 内容 |
|---|---|
| サイズ | ~15 GB |
| 時間 | ~40h |
| 話者 | 4 |
| ライセンス | CC BY-NC 4.0 |
| 提供元 | Meta Research |

**手順:**
1. https://speechbot.github.io/expresso/ にアクセス
2. Meta のライセンスに同意してダウンロード
3. 展開先: `data/raw/expresso/`

**スタイル一覧** (→ 12 カテゴリにマッピング):
- default → neutral
- happy → happy
- sad → sad
- angry → angry
- enunciated → neutral
- laughing → happy
- whisper → whisper

### 3.2 JVNV (日本語, 6 感情)

| 項目 | 内容 |
|---|---|
| サイズ | ~1.5 GB |
| 時間 | ~4h |
| 話者 | 4 |
| ライセンス | CC BY-SA 4.0 |

**手順:**
1. https://sites.google.com/site/shinaborulab/research/jvnv にアクセス
2. 手動ダウンロード (自動DL非対応)
3. 展開先: `data/raw/jvnv/`

```bash
# ダウンロード案内の表示
uv run python scripts/download_datasets.py --dataset jvnv --output-dir data/raw
```

**ディレクトリ構造:**
```
data/raw/jvnv/jvnv_ver1/
└── JVNV001/
    ├── anger/
    │   └── JVNV001_anger_001.wav
    ├── happiness/
    ├── sadness/
    ├── fear/
    ├── surprise/
    └── disgust/
```

### 3.3 EmoV-DB (英語, 5 感情)

| 項目 | 内容 |
|---|---|
| サイズ | ~3 GB |
| 時間 | ~7h |
| 話者 | 4+ |
| ライセンス | Apache 2.0 |

**手順:**
1. https://openslr.org/115/ からダウンロード
2. 展開先: `data/raw/EmoV-DB/`

**感情**: amused, angry, disgusted, neutral, sleepy

### 3.4 RAVDESS (英語, 8 感情)

| 項目 | 内容 |
|---|---|
| サイズ | ~1 GB (speech のみ) |
| 時間 | ~2h |
| 話者 | 24 |
| ライセンス | CC BY-NC-SA 4.0 |

**手順:**
1. https://zenodo.org/records/1188976 からダウンロード
2. 展開先: `data/raw/ravdess/`

**感情**: neutral, calm, happy, sad, angry, fearful, surprised, disgusted
(各感情に normal/strong の 2 強度)

---

## 4. RIR (Room Impulse Response) データ

Phase 2 (IR-robust 学習) で使用する残響データ。

### 4.1 AIR Database

| 項目 | 内容 |
|---|---|
| サイズ | ~200 MB |
| URL | https://www.openslr.org/20/ |

```bash
wget https://www.openslr.org/resources/20/air.zip
unzip air.zip -d data/raw/rir/air/
```

### 4.2 BUT ReverbDB

| 項目 | 内容 |
|---|---|
| サイズ | ~1 GB |
| URL | https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database |

手動ダウンロード → `data/raw/rir/but_reverb/` に配置。

---

## 5. 前処理パイプライン

### 5.1 基本前処理 (特徴量抽出)

`configs/datasets.yaml` でデータセットを有効化し、`raw_dir` を設定:

```yaml
# configs/datasets.yaml
datasets:
  vctk:
    enabled: true
    raw_dir: data/raw/VCTK-Corpus
  jvs:
    enabled: true
    raw_dir: data/raw/jvs_corpus
```

```bash
# 全有効データセットの前処理
uv run python scripts/prepare_datasets.py \
  --config configs/datasets.yaml \
  --device xpu

# 特定データセットのみ
uv run python scripts/prepare_datasets.py \
  --config configs/datasets.yaml \
  --datasets vctk jvs \
  --device xpu

# 個別実行
uv run tmrvc-preprocess \
  --dataset vctk \
  --raw-dir data/raw/VCTK-Corpus \
  --cache-dir data/cache \
  --device xpu -v
```

**前処理で抽出される特徴量:**

| ファイル | 形状 | 内容 |
|---|---|---|
| `mel.npy` | `[80, T]` | Log-mel スペクトログラム |
| `content.npy` | `[768, T]` or `[1024, T]` | ContentVec / WavLM 特徴量 |
| `f0.npy` | `[1, T]` | 基本周波数 (Hz) |
| `spk_embed.npy` | `[192]` | 話者埋め込み (ECAPA-TDNN) |
| `meta.json` | — | メタデータ (speaker_id, n_frames, content_dim 等) |

### 5.2 Phase 1a 移行 (ContentVec 768d → WavLM 1024d)

```bash
bash scripts/preprocess_phase1a.sh --device xpu

# 古いキャッシュを削除してから再生成
bash scripts/preprocess_phase1a.sh --device xpu --clean-old
```

### 5.3 強制アラインメント (TTS 用)

TTS 学習にはテキストと音素レベルのデュレーション情報が必要。

#### MFA のインストール

```bash
# conda 環境に MFA をインストール (推奨)
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa

# 音響モデルのダウンロード
mfa model download acoustic japanese_mfa
mfa model download acoustic english_mfa

# 辞書のダウンロード
mfa model download dictionary japanese_mfa
mfa model download dictionary english_mfa

# 動作確認
mfa version
```

#### MFA による アラインメント実行

```bash
# JSUT (日本語) — TextGrid 生成
mfa align data/raw/jsut/ japanese_mfa japanese_mfa data/alignments/jsut/

# LJSpeech (英語) — TextGrid 生成
mfa align data/raw/ljspeech/wavs/ english_mfa english_mfa data/alignments/ljspeech/
```

#### phoneme_ids / durations の保存

```bash
# TextGrid がある場合 (MFA 実行済み)
uv run python scripts/run_forced_alignment.py \
  --cache-dir data/cache \
  --dataset jsut \
  --language ja \
  --textgrid-dir data/alignments/jsut

# TextGrid がない場合 (G2P ヒューリスティック、均等分割)
uv run python scripts/run_forced_alignment.py \
  --cache-dir data/cache \
  --dataset jsut \
  --language ja
```

**出力**: 各発話ディレクトリに `phoneme_ids.npy` と `durations.npy` を追加。

### 5.4 感情データセット前処理

```bash
# Expresso
uv run python scripts/preprocess_emotion.py \
  --dataset expresso \
  --raw-dir data/raw/expresso \
  --cache-dir data/cache

# JVNV
uv run python scripts/preprocess_emotion.py \
  --dataset jvnv \
  --raw-dir data/raw/jvnv \
  --cache-dir data/cache

# EmoV-DB
uv run python scripts/preprocess_emotion.py \
  --dataset emov_db \
  --raw-dir data/raw/EmoV-DB \
  --cache-dir data/cache

# RAVDESS
uv run python scripts/preprocess_emotion.py \
  --dataset ravdess \
  --raw-dir data/raw/ravdess \
  --cache-dir data/cache
```

**追加出力**: 各発話ディレクトリに `emotion.json` を生成:
```json
{
  "emotion_id": 0,
  "emotion": "happy",
  "vad": [0.8, 0.6, 0.5],
  "prosody": [0.0, 0.0, 0.0]
}
```

---

## 6. キャッシュ構造

前処理完了後のキャッシュ:

```
data/cache/
├── _manifests/
│   ├── vctk_train.json       # 検証メタデータ
│   └── jvs_train.json
├── vctk/train/
│   └── vctk_p225/
│       └── vctk_p225_001/
│           ├── mel.npy         # [80, T]
│           ├── content.npy     # [768, T] or [1024, T]
│           ├── f0.npy          # [1, T]
│           ├── spk_embed.npy   # [192]
│           ├── meta.json
│           ├── phoneme_ids.npy # [L] (TTS 用、アラインメント後)
│           └── durations.npy   # [L] (TTS 用、アラインメント後)
├── jvs/train/
├── jsut/train/
│   └── jsut_speaker/
├── expresso/train/
│   └── ex01/
│       └── ex01_happy_00001/
│           ├── mel.npy
│           ├── emotion.json    # 感情メタデータ
│           └── meta.json
└── ...
```

---

## 7. ストレージ見積もり

### Raw データ

| データセット | サイズ | 用途 |
|---|---|---|
| VCTK (ZIP) | ~11 GB | VC |
| VCTK (展開後) | ~15 GB | VC |
| JVS | ~3 GB | VC |
| tsukuyomi | ~0.3 GB | VC |
| LibriTTS-R | ~65 GB | VC (Phase 1a+) |
| JSUT | ~2 GB | TTS |
| LJSpeech | ~2.6 GB | TTS |
| Expresso | ~15 GB | Expressive |
| JVNV | ~1.5 GB | Expressive |
| EmoV-DB | ~3 GB | Expressive |
| RAVDESS | ~1 GB | Expressive |
| RIR (AIR + BUT) | ~1.2 GB | Augmentation |
| **合計** | **~120 GB** | |

### Cache データ

| Phase | 対象 | Cache サイズ目安 |
|---|---|---|
| Phase 0 (ContentVec 768d) | VCTK + JVS + tsukuyomi | ~3-4 GB |
| Phase 1a (WavLM 1024d) | VCTK + JVS + tsukuyomi | ~5-6 GB |
| Phase 1a + LibriTTS-R | 全 VC データ | ~15-20 GB |
| TTS | JSUT + LJSpeech | ~1-2 GB |
| Expressive | 感情データセット群 | ~3-5 GB |
| **合計** | | **~25-33 GB** |

---

## 8. 検証

### キャッシュ整合性チェック

```bash
uv run python -c "
from tmrvc_data.cache import FeatureCache
cache = FeatureCache('data/cache')
for ds in ['vctk', 'jvs', 'tsukuyomi']:
    result = cache.verify(ds, 'train')
    print(f'{ds}: {result[\"valid\"]}/{result[\"total\"]} valid')
"
```

### マニフェスト確認

前処理スクリプト (`prepare_datasets.py`) は自動的に `data/cache/_manifests/` にマニフェストを書き出す。

```bash
cat data/cache/_manifests/vctk_train.json
```

---

## 9. 一括実行リファレンス

### VC 学習用 (最小構成: Phase 0)

```bash
# 1. データ取得
uv run python scripts/download_vctk.py --output-dir data/raw
# JVS: 手動ダウンロード → data/raw/jvs_corpus/

# 2. datasets.yaml を編集 (vctk, jvs を enabled: true に)

# 3. 前処理
uv run python scripts/prepare_datasets.py \
  --config configs/datasets.yaml --device xpu

# 4. 学習開始
uv run tmrvc-train-teacher --cache-dir data/cache --phase 0 --device xpu
```

### TTS 学習用 (Phase 2)

```bash
# 1. TTS データ取得
uv run python scripts/download_datasets.py --all --output-dir data/raw

# 2. 前処理
uv run tmrvc-preprocess --dataset jsut --raw-dir data/raw/jsut \
  --cache-dir data/cache --device xpu
uv run tmrvc-preprocess --dataset ljspeech --raw-dir data/raw/ljspeech \
  --cache-dir data/cache --device xpu

# 3. 強制アラインメント (MFA or G2P ヒューリスティック)
uv run python scripts/run_forced_alignment.py \
  --cache-dir data/cache --dataset jsut --language ja
uv run python scripts/run_forced_alignment.py \
  --cache-dir data/cache --dataset ljspeech --language en

# 4. TTS 学習
uv run tmrvc-train-tts --cache-dir data/cache --device xpu
```

### 感情 TTS 学習用 (Phase 3)

```bash
# 1. 感情データセット取得 (手動ダウンロード)
# Expresso → data/raw/expresso/
# JVNV → data/raw/jvnv/
# EmoV-DB → data/raw/EmoV-DB/
# RAVDESS → data/raw/ravdess/

# 2. 感情前処理
for ds in expresso jvnv emov_db ravdess; do
  uv run python scripts/preprocess_emotion.py \
    --dataset $ds --raw-dir data/raw/$ds --cache-dir data/cache
done

# 3. StyleEncoder 学習
uv run tmrvc-train-style --cache-dir data/cache --device xpu
```

---

## 10. トラブルシューティング

| 問題 | 原因 | 対策 |
|---|---|---|
| `UnicodeDecodeError: 'cp932'` | Windows のデフォルトエンコーディング | `PYTHONIOENCODING=utf-8` を設定 |
| ContentVec / WavLM の OOM | GPU メモリ不足 | `--device cpu` で実行 (低速だが安定) |
| `FileNotFoundError: raw dir not found` | `datasets.yaml` のパスが不正 | `raw_dir` を絶対パスまたは正しい相対パスに修正 |
| VCTK ダウンロードが途中で止まる | サーバー側の問題 | 再実行 (途中DLファイルは自動検出) |
| MFA が見つからない | PATH にない | `conda activate mfa` を実行 |
| `torchcrepe` の import エラー | 依存関係不足 | `uv sync` で再インストール |
| XPU で `DEVICE_LOST` | VRAM 不足 | バッチサイズを下げる / CPU にフォールバックしない |

---

## 11. ライセンス一覧

| データセット | ライセンス | 商用利用 |
|---|---|---|
| VCTK | CC BY 4.0 | 可 |
| JVS | CC BY-SA 4.0 | 可 (同一ライセンス) |
| LibriTTS-R | CC BY 4.0 | 可 |
| JSUT | 非商用・再配布不可 | 不可 |
| LJSpeech | Public Domain | 可 |
| Expresso | CC BY-NC 4.0 | 不可 |
| JVNV | CC BY-SA 4.0 | 可 (同一ライセンス) |
| EmoV-DB | Apache 2.0 | 可 |
| RAVDESS | CC BY-NC-SA 4.0 | 不可 |
