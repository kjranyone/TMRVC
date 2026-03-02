# TMRVC UCLM v2 Training Guide

UCLM v2モデル（Unified Codec Language Model）の学習パイプライン全体を解説します。
TMRVCは、単一のトランスフォーマー・アーキテクチャでTTSとVCの両方を実現します。

## 0. 推奨される学習データと構造

TMRVC は、特別なアノテーションや複雑なフォルダ分けがされていない「生の wav ファイル群」からでも学習を開始できます。

### 0.1 最小構成のデータセット
もっともシンプルな開始方法は、一つのフォルダにすべての wav ファイルを投入することです。

```
data/raw/my_dataset/
  ├── sample1.wav
  ├── sample2.wav
  └── ...
```

この状態から、パイプラインが以下の情報を自動的に補完します：
- **話者ID**: デフォルトではフォルダ名から推定しますが、フォルダ分けがない場合はファイル全体を同一話者、または `cluster_speakers.py` を用いた自動話者分離が可能です。
- **テキスト**: `tmrvc-preprocess` 内の Whisper ASR が自動的に文字起こしを行います。
- **音響特徴量**: モデルが自動的に SSL 特徴量や Codec トークンを抽出します。

### 0.2 推奨コーパスと活用法

高品質な UCLM v2 モデルを構築するために、以下の標準コーパスの併用を強く推奨します。自前データが少ない場合でも、これらと混ぜて学習することで汎化性能が向上します。

| コーパス名 | 言語 | 内容 | 推奨用途 |
| :--- | :--- | :--- | :--- |
| **JVS Corpus** | 日本語 | 100話者の高品質音声 | 日本語の基本発音・表現の学習 |
| **VCTK** | 英語 | 110話者の多様なアクセント | 英語の基本発音・話者性の学習 |
| **Tsukuyomi** | 日本語 | 1話者の高品質音声 | 日本語の安定した音質・韻律の学習 |
| **LibriTTS-R** | 英語 | 大規模朗読音声 | 表現豊かな読み上げの学習 |

### 0.3 データセットの登録 (`configs/datasets.yaml`)

自前データを学習に含めるには、以下のように設定ファイルに記述します。

```yaml
datasets:
  # 自前のフォルダ分けなしwavデータ
  my_private_data:
    type: generic
    enabled: true
    language: ja
    raw_dir: data/raw/my_wav_folder

  # 標準コーパス (併用を推奨)
  vctk:
    type: vctk
    enabled: true
    raw_dir: data/raw/wav48_silence_trimmed
```

---

## 1. データ前処理 (Data Preparation)

TMRVCの前処理は、**「音響抽出 → テキスト正規化 → 強制アライメント」**の3フェーズで構成されます。各スクリプトは `--skip-existing` フラグをサポートしており、中断・再開が可能な冪等（Idempotent）な設計となっています。

### Phase A: 音響・制御特徴量の一括抽出
全ての wav ファイルから Codec トークン、WavLM 特徴量、話者埋め込みを一括抽出します。
```bash
uv run python scripts/data/prepare_datasets.py --device cuda --skip-existing
```

### Phase B: 正解テキストの注入 (Metadata Normalization)
Whisper ASR の誤変換を避け、論文品質の整合性を確保するため、公式の台本テキストをキャッシュに上書きします。
```bash
# VCTK, JVS, つくよみちゃんの公式台本をキャッシュに自動適用
uv run python scripts/annotate/inject_text_to_cache.py
```

### Phase C: 強制アライメント (TTS Alignment)
注入された正解テキストと音声波形を同期させ、音素単位のデュレーションを確定させます。
```bash
# 言語ごとに MFA アライメントを実行
# (例: つくよみちゃんの場合)
uv run python scripts/annotate/run_forced_alignment.py \
    --cache-dir data/cache --dataset tsukuyomi --language ja \
    --textgrid-dir data/alignments/tsukuyomi
```

### 1.2 パフォーマンス・チューニングと並列度の決定

(中略: 前述の VRAM 消費量テーブルと計算式)

### 1.3 TTS用アライメントの詳細 (MFA)

(中略: MFA の実行手順)

---

## 2. モデル学習 (Training)

(以下略)
