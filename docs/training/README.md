# TMRVC 学習パイプライン統合ガイド

> 全学習パスの依存関係と実行順序を1ファイルで俯瞰するためのガイド。
> 個別の詳細は各計画書を参照。

---

## 全体依存グラフ

```
データ取得 (docs/data-acquisition-guide.md)
    |
    +-- VC データ (VCTK, JVS, LibriTTS-R, つくよみちゃん)
    |     |
    |     +---> tmrvc-preprocess (特徴量抽出)
    |     |       \---> data/cache/{dataset}/
    |     |
    |     +---> [Phase 0-2] tmrvc-train-teacher
    |     |       \---> checkpoints/teacher_*.pt
    |     |
    |     +---> [Phase A-C] tmrvc-distill
    |     |       \---> checkpoints/distill/*.pt
    |     |
    |     \---> tmrvc-export (ONNX)
    |             \---> models/fp32/*.onnx
    |
    +-- TTS データ (JSUT, LJSpeech + 上記 VC データ)
    |     |
    |     +---> MFA 強制アラインメント
    |     |       \---> data/cache/{dataset}/{utt}/duration.npy
    |     |
    |     \---> [Phase 2-3] tmrvc-train-tts
    |             +-- 前提: Distillation 済み Student モデル (frozen)
    |             \---> checkpoints/tts/tts_*.pt
    |
    +-- Style データ (Expresso, JVNV, EmoV-DB, RAVDESS)
    |     |
    |     \---> [Phase 3a] tmrvc-train-style
    |             \---> checkpoints/tts/style_*.pt
    |
    \-- Few-shot (ユーザー音声 3-60秒)
          |
          \---> tmrvc-finetune (LoRA)
                +-- 前提: distill/*.pt
                \---> speakers/*.tmrvc_speaker
```

---

## CLI リファレンス (実行順)

### Step 1: データ準備

データ取得の詳細は [docs/data-acquisition-guide.md](../data-acquisition-guide.md) を参照。

```bash
# 前処理 (全データセット一括)
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --device xpu

# または個別に
uv run tmrvc-preprocess --dataset vctk --raw-dir data/raw/VCTK-Corpus \
  --cache-dir data/cache --content-teacher wavlm --device xpu
```

### Step 2: VC Teacher 学習

詳細: [vc-training-plan.md](vc-training-plan.md)

```bash
uv run tmrvc-train-teacher \
  --config configs/train_teacher.yaml \
  --cache-dir data/cache \
  --phase 0 \
  --device xpu
```

| Phase | 内容 | 目標 |
|---|---|---|
| Phase 0 | アーキテクチャ検証 (ContentVec, 100K steps) | SECS > 0.75 |
| Phase 1a | WavLM + OT-CFM + マルチデータセット (500K steps) | SECS >= 0.88 |
| Phase 1b | Perceptual loss 追加 (200K steps) | SECS >= 0.90, UTMOS >= 4.0 |
| Phase 2 | IR-robust 化 + RIR augmentation (200K steps) | 残響 SECS >= 0.84 |

### Step 3: 蒸留

詳細: [vc-training-plan.md](vc-training-plan.md) 8

```bash
uv run tmrvc-distill \
  --cache-dir data/cache \
  --teacher-ckpt checkpoints/teacher_step*.pt \
  --phase A \
  --device xpu
```

| Phase | 手法 | 説明 |
|---|---|---|
| Phase A | ODE trajectory matching | v-prediction MSE |
| Phase B | DMD (distribution matching) | Regression loss |
| Phase B2 | DMD2 (GAN discriminator) | MelDiscriminator |
| Phase C | Metric Optimization | SV loss + STFT loss |

### Step 4: ONNX エクスポート

```bash
uv run tmrvc-export \
  --checkpoint checkpoints/distill/best.pt \
  --output-dir models/fp32 \
  --verify
```

### Step 5: TTS 学習 (optional, VC だけなら不要)

詳細: [tts-training-plan.md](tts-training-plan.md)

**前提:** Step 3 の蒸留済み Student モデルが必要 (Converter/Vocoder を凍結して流用)。

```bash
# MFA 強制アラインメント (デュレーション情報の抽出)
uv run python scripts/run_forced_alignment.py \
  --config configs/datasets.yaml --device xpu

# TTS 学習
uv run tmrvc-train-tts \
  --config configs/train_tts.yaml \
  --cache-dir data/cache \
  --device xpu
```

### Step 6: Style 学習 (optional, 表現的 TTS のみ)

詳細: [style-training-plan.md](style-training-plan.md)

**前提:** Step 5 の TTS 学習済みモデルが必要。

```bash
uv run tmrvc-train-style \
  --cache-dir data/cache \
  --dataset expresso,jvnv \
  --device xpu
```

### Step 7: Few-shot (エンドユーザー向け)

詳細: [../user-manual.md](../user-manual.md) 3

```bash
uv run tmrvc-finetune \
  --audio-dir data/sample_voice/ \
  --checkpoint checkpoints/distill/best.pt \
  --device xpu
```

---

## 学習パス早見表

| 目的 | 必要な Step | 推定時間 (B570) | 推定コスト (Cloud) |
|---|---|---|---|
| **VC のみ** | Step 1-4 | ~5-7 日 | ~$150-300 |
| **VC + TTS** | Step 1-5 | ~8-10 日 | ~$200-400 |
| **VC + TTS + Style** | Step 1-6 | ~12-17 日 | ~$300-500 |
| **Few-shot のみ** (事前学習済み利用) | Step 7 | ~5 分 | — |

---

## 関連資料

### 学習計画 (docs/training/)

| ファイル | 内容 |
|---|---|
| [vc-training-plan.md](vc-training-plan.md) | VC Teacher 学習 + 蒸留 (Phase 0-2, A-C) |
| [tts-training-plan.md](tts-training-plan.md) | TTS Phase 2-5 ロードマップ + アーキテクチャ仕様 |
| [style-training-plan.md](style-training-plan.md) | Style/Emotion/LLM 統合・VTuber 設計 |
| [tts-front-end-review-2026-02-25.md](tts-front-end-review-2026-02-25.md) | TTS Front-end レビュー記録 |

### 正本設計資料 (docs/design/)

| ファイル | 内容 |
|---|---|
| `architecture.md` | 全体アーキテクチャ |
| `streaming-design.md` | リアルタイムストリーミング |
| `onnx-contract.md` | ONNX I/O 仕様 |
| `model-architecture.md` | モデル詳細 |
| `cpp-engine-design.md` | C++ エンジン |
| `acoustic-condition-pathway.md` | IR/環境条件付け |
| `gui-design.md` | GUI ワーカー設計 |

### 研究資料 (docs/research/)

| ファイル | 内容 |
|---|---|
| `research-novelty-plan.md` | 研究計画 |
| `research-results-summary.md` | 評価テンプレート |
| `asmr-tts-roadmap.md` | ASMR ロードマップ |
| `paper_level_design.md` | 論文設計 |

---

## デバイスポリシー

- XPU (Intel Arc) が利用可能な場合は `--device xpu` を使用
- Windows では `--num-workers 0` を推奨
- XPU での `DEVICE_LOST` エラーは CPU フォールバックせず、バッチサイズ削減で対処
- `--max-frames 400` がデフォルト (XPU カーネル再コンパイル回避)
