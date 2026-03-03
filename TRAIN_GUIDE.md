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
- **テキスト**: `tmrvc-train-pipeline` 内の Whisper ASR が自動的に文字起こしを行います。
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

## 1. 統合パイプライン (tmrvc-train-pipeline)

TMRVC v2 では、**再現性のある学習**を保証するために、単一のCLIコマンド `tmrvc-train-pipeline` で前処理から学習までを一括実行します。

### 1.1 基本的な使い方

```bash
tmrvc-train-pipeline \
  --dataset vctk \
  --raw-dir data/raw \
  --output-dir experiments \
  --workers 2 \
  --seed 42
```

この1コマンドで以下が実行されます：
1. **前処理**: GPU並列、発話単位分散、冪等性保証
2. **学習**: UCLMトレーニング、チェックポイント自動保存
3. **メタデータ保存**: Git hash, 乱数シード, 設定ファイル

### 1.2 パフォーマンス・チューニングと並列度の決定

前処理パイプラインは VRAM を主に消費します。以下のリソース消費表に基づき、ご自身の GPU メモリ容量（VRAM）に収まる範囲で `--workers` 数を決定してください。

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

#### 並列実行例

```bash
# RTX 4090 (24GB) の場合 - 5並列
tmrvc-train-pipeline \
  --dataset vctk \
  --raw-dir data/raw \
  --output-dir experiments \
  --workers 5 \
  --seed 42

# RTX 3060 (12GB) の場合 - 2並列
tmrvc-train-pipeline \
  --dataset jvs \
  --raw-dir data/raw \
  --output-dir experiments \
  --workers 2 \
  --seed 42
```

### 1.3 再現性保証

`tmrvc-train-pipeline` は論文レベルの再現性を保証します：

#### 1. メタデータ記録
実験ディレクトリに `experiment.yaml` が自動生成されます：

```yaml
experiment_id: vctk_20260303_123456
dataset: vctk
created_at: "2026-03-03T12:34:56"
git_hash: abc123def456
git_branch: main
python_version: "3.12.0"
config:
  language: en
  train_steps: 100000
seed: 42
workers: 2
status: completed
```

#### 2. 乱数シード固定
全ての乱数シードが固定されます：
- `random.seed(42)`
- `np.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`

#### 3. エラー記録と再実行
失敗した発話は `errors.json` に記録されます：

```json
[
  {
    "utterance_id": "vctk_p225_001",
    "error_type": "RuntimeError",
    "error_message": "CUDA out of memory",
    "stage": "preprocessing",
    "timestamp": "2026-03-03T12:45:00"
  }
]
```

### 1.4 出力ディレクトリ構造

```
experiments/
└── vctk_20260303_123456/          # 実験ID (自動生成)
    ├── experiment.yaml             # メタデータ (git hash, seed, config)
    ├── errors.json                 # 失敗発話リスト (再実行用)
    ├── train_config.yaml           # 学習設定
    ├── cache/                      # 前処理キャッシュ
    │   └── vctk/train/
    ├── checkpoints/                # 学習チェックポイント
    └── logs/                       # ログファイル
```

---

## 2. 高度な使い方

### 2.1 既存キャッシュを使用して学習のみ実行

前処理が完了している場合、`--skip-preprocess` で学習のみ実行できます：

```bash
tmrvc-train-pipeline \
  --dataset vctk \
  --raw-dir data/raw \
  --output-dir experiments \
  --skip-preprocess \
  --seed 42
```

### 2.2 カスタム実験名を指定

```bash
tmrvc-train-pipeline \
  --dataset vctk \
  --raw-dir data/raw \
  --output-dir experiments \
  --experiment-name vctk_baseline_v1 \
  --workers 3
```

### 2.3 設定ファイルのカスタマイズ

`configs/train_uclm.yaml` で学習パラメータをカスタマイズできます：

```yaml
language: en
train_steps: 100000
batch_size: 16
learning_rate: 1e-4
adapter_type: vctk
```

使用例：

```bash
tmrvc-train-pipeline \
  --dataset vctk \
  --raw-dir data/raw \
  --output-dir experiments \
  --config configs/train_uclm.yaml
```

---

## 3. 個別コンポーネントの実行 (上級者向け)

統合パイプラインではなく、個別のコンポーネントを実行することも可能です。

### 3.1 Stage 1: Emotion-Aware Codec学習

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

### 3.2 Stage 2: Unified UCLM学習

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

### 3.3 TTS用アライメントの詳細 (MFA)

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

## 4. モデルの検証と利用

### 4.1 統合エンジンでのテスト

学習したチェックポイントを `UCLMEngine` にロードして動作確認します。

```bash
uv run python scripts/demo/tts_demo.py \
    --uclm-checkpoint checkpoints/uclm/uclm_final.pt \
    --codec-checkpoint checkpoints/codec/codec_final.pt \
    --text "これはUCLM v2の統合テストです。"
```

### 4.2 ONNX エクスポート

RustエンジンやVSTプラグインで利用するために ONNX 形式へ変換します。

```bash
tmrvc-export \
    --uclm-checkpoint checkpoints/uclm/uclm_final.pt \
    --codec-checkpoint checkpoints/codec/codec_final.pt \
    --output-dir models/onnx
```

---

## 5. トラブルシューティング

- **ImportError (定数不足)**: `tmrvc_core/constants.py` が最新の `configs/constants.yaml` と同期しているか確認してください。
- **Shape Mismatch**: キャッシュ生成時の `hop_length` (240) とモデルのストライド設定が一致しているか確認してください。
- **OOM**: `--batch-size` または `--max-frames` (デフォルト400) を下げて調整してください。
- **Index Out of Bounds**: 古いキャッシュを削除して、新しいフレームアライメント（`pad_length=784`）で再生成してください。
- **パイプラインの中断**: `tmrvc-train-pipeline` は冪等性を保証しているため、中断しても再実行すればキャッシュ済み発話をスキップして継続できます。

---

## 6. 推奨ワークフロー

### 初回実行

```bash
# 1. データセットを登録
vim configs/datasets.yaml

# 2. パイプライン実行
tmrvc-train-pipeline \
  --dataset vctk \
  --raw-dir data/raw \
  --output-dir experiments \
  --workers 5 \
  --seed 42 \
  --config configs/train_uclm.yaml

# 3. 結果確認
ls experiments/vctk_*/
cat experiments/vctk_*/experiment.yaml
```

### 実験管理

```bash
# 実験一覧
ls experiments/

# 特定実験の詳細
cat experiments/vctk_20260303_123456/experiment.yaml

# エラー確認
cat experiments/vctk_20260303_123456/errors.json

# ログ確認
tail -f experiments/vctk_20260303_123456/logs/*.log
```
