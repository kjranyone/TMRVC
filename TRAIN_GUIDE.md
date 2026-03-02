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

前処理パイプラインは VRAM を主に消費します。以下のリソース消費表に基づき、ご自身の GPU メモリ容量（VRAM）に収まる範囲で並列ワーカー数を決定してください。

#### モデル別 VRAM 消費量 (目安)

| コンポーネント | モデル | VRAM 消費 (FP16) | 役割 |
| :--- | :--- | :--- | :--- |
| **Whisper ASR** | large-v3-turbo | **~1.6 GB** | テキスト書き起こし |
| **SSL Estimator** | WavLM Large | **~1.3 GB** | SSL音声状態の抽出 |
| **Codec** | Emotion-Aware | **~0.4 GB** | A_t, B_t トークン抽出 |
| **Speaker Encoder**| CAM++ | **~0.2 GB** | 話者埋め込みの抽出 |
| **その他/Overhead**| PyTorch/System | **~0.5 GB** | テンソル演算・キャッシュ |
| **合計 (1ワーカー)** | - | **約 4.0 GB** | - |

#### 並列度の計算式
$$並列ワーカー数 = \lfloor \frac{GPU VRAM (GB) - 1.0}{4.0} \rfloor$$
*(システム用マージン 1GB を差し引いて計算)*

*   **VRAM 12GB の場合**: マージンを引いて **2並列** が推奨です。
*   **VRAM 22GB の場合**: マージンを引いて **5並列** が最適です。
*   **VRAM 24GB (RTX 3090/4090) の場合**: **5〜6並列** が可能です。

#### 並列実行の実行方法
`scripts/parallel_preprocess.sh` を編集し、計算したワーカー数を設定して実行してください。

```bash
# ワーカー数を環境に合わせて編集
./scripts/parallel_preprocess.sh
```

### 1.3 TTS用アライメントの詳細 (MFA)

VC学習には不要ですが、TTS学習を行う場合は正確な音素単位のアライメントが必須です。

```bash
# 1. Montreal Forced Aligner で TextGrid を生成
mfa align data/raw/vctk/wav48 english_us_arpa english_us_arpa data/alignments/vctk

# 2. 生成された TextGrid をキャッシュに注入
uv run python scripts/annotate/run_forced_alignment.py \
    --cache-dir data/cache \
    --dataset vctk \
    --language en \
    --textgrid-dir data/alignments/vctk
```

- **MFA統合の重要性**: ヒューリスティックな均等割り（`--allow-heuristic`）は品質を著しく低下させるため、論文実装レベルの学習には MFA の使用を強く推奨します。
- **BOS/EOS**: 注入時に自動的に `<bos>`, `<eos>` トークンが前後に追加されます。

---

## 2. モデル学習 (Training)

学習は大きく分けて2つのステージ（CodecとUCLM本体）で行われます。

### 2.1 Stage 1: Emotion-Aware Codec学習

音声のトークン化と復元を行う基盤モデルを学習します。

#### VRAMチューニング (Stage 1)
- **消費目安**: Base 2GB + (0.4GB × BatchSize)
- **22GB環境の例**: `--batch-size 48` (約21.2GB消費) が最大効率です。

```bash
tmrvc-train-codec \
    --cache-dir data/cache \
    --output-dir checkpoints/codec \
    --batch-size 48 \
    --max-steps 50000 \
    --device cuda
```

- **Loss**: Multi-scale STFT loss + Control Stream Cross-Entropy
- **出力**: `codec_final.pt`

### 2.2 Stage 2: Unified UCLM学習

TTSとVCを同時にこなす統合トランスフォーマーを学習します。

#### VRAMチューニング (Stage 2)
- **消費目安**: Base 6GB + (0.8GB × BatchSize)
- **22GB環境の例**: `--batch-size 16〜20` が安定稼働のラインです。

```bash
tmrvc-train-uclm \
    --cache-dir data/cache \
    --output-dir checkpoints/uclm \
    --batch-size 16 \
    --max-steps 100000 \
    --device cuda
```

- **学習内容**:
    - **VC Task**: `A_src_t → content → A_t, B_t`
    - **TTS Task**: `Phonemes → content → A_t, B_t`
- **Loss**: CE (A_t, B_t) + VQ Bottleneck + Adversarial Disentanglement (GRL) + Duration Prediction (MSE)

---

## 3. モデルの検証と利用

### 3.1 統合エンジンでのテスト

学習したチェックポイントを `UCLMEngine` にロードして動作確認します。

```bash
uv run python scripts/demo/tts_demo.py \
    --uclm-checkpoint checkpoints/uclm/uclm_final.pt \
    --codec-checkpoint checkpoints/codec/codec_final.pt \
    --text "これはUCLM v2の統合テストです。"
```

### 3.2 ONNX エクスポート

RustエンジンやVSTプラグインで利用するために ONNX 形式へ変換します。

```bash
tmrvc-export \
    --uclm-checkpoint checkpoints/uclm/uclm_final.pt \
    --codec-checkpoint checkpoints/codec/codec_final.pt \
    --output-dir models/onnx
```

---

## 4. トラブルシューティング

- **ImportError (定数不足)**: `tmrvc_core/constants.py` が最新の `configs/constants.yaml` と同期しているか確認してください。
- **Shape Mismatch**: キャッシュ生成時の `hop_length` (240) とモデルのストライド設定が一致しているか確認してください。
- **OOM**: `--batch-size` または `--max-frames` (デフォルト400) を下げて調整してください。
- **Index Out of Bounds**: 古いキャッシュを削除して、新しいフレームアライメント（`pad_length=784`）で再生成してください。
