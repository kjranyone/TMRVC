# TMRVC — Codec-Latent Real-time Voice Conversion

## Project Overview

CPU-only で end-to-end 50ms 以下のリアルタイム Voice Conversion を実現する VST3 プラグイン。
Codec-Latent パラダイム: 因果ニューラル音声コーデック + トークン予測モデル (Mamba)。

- **内部サンプルレート:** 24kHz（DAW の 44.1/48kHz とはポリフェーズリサンプルで接続）
- **推論:** 3 ONNX モデルを ONNX Runtime で実行
- **学習パラダイム:** Codec-Latent (Causal neural codec + Mamba token model)
- **学習データ:** VCTK + JVS (T1) → LibriTTS-R (T2) → Emilia (T3) 段階的追加

## Architecture

```
DAW Audio In → VST3 Plugin → StreamingEngine (Rust) → ONNX Models ×3 → DAW Audio Out
                                    ↑
                              nih-plug 非依存
```

3 ONNX モデルの実行頻度:
- `codec_encoder` — per-frame (20ms ごと)
- `token_model` (Mamba) — per-frame (20ms ごと)
- `codec_decoder` — per-frame (20ms ごと)
- `speaker_encoder` — offline (enrollment 時のみ)

## Monorepo Structure

```
TMRVC/
├── docs/design/           # 設計資料 (13 ファイル)
├── docs/training/         # 学習計画 (5 ファイル)
├── docs/reference/        # 参考資料 (旧設計, 先行研究)
├── tmrvc-core/            # Python: 共有定数・mel 計算・型定義
├── tmrvc-data/            # Python: データセット・前処理・augmentation
├── tmrvc-train/           # Python: モデル定義・学習 (VC/TTS/Codec/UCLM)
├── tmrvc-export/          # Python: ONNX エクスポート・量子化・パリティ検証
├── tmrvc-gui/             # Python: Research Studio GUI (PySide6)
├── tmrvc-serve/           # Python: FastAPI TTS server
├── tmrvc-engine-rs/       # Rust: ストリーミング推論エンジン (nih-plug 非依存)
├── tmrvc-vst/             # Rust: VST3 プラグイン (nih-plug)
├── tmrvc-rt/              # Rust: Real-time GUI (egui)
├── xtask/                 # Rust: Build tasks
├── tests/                 # Python / Rust 統合テスト
├── configs/               # constants.yaml (shared source of truth)
└── scripts/               # ユーティリティスクリプト
```

Python と Rust は **ONNX ファイル** と **constants.yaml → 自動生成ヘッダ** でのみ接続。

### scripts/ ディレクトリ構成

ユーティリティスクリプトはカテゴリごとにサブディレクトリを切る:

```
scripts/
├── data/                    # データ準備・ダウンロード
│   ├── prepare_datasets.py  # datasets.yaml から一括前処理
│   ├── prepare_dataset.py   # 単一データセット準備
│   ├── parallel_prepare.py  # 並列実行版
│   ├── prepare_bulk_voice.py
│   └── download_datasets.py
│
├── annotate/                # アノテーション・ラベリング
│   ├── add_codec_to_cache.py
│   ├── add_phoneme_annotations.py
│   ├── auto_annotate_*.py
│   ├── inject_*.py
│   ├── transcribe_with_whisper.py
│   └── run_forced_alignment.py
│
├── eval/                    # 評価・統計
│   ├── eval_research_*.py
│   ├── evaluate_tts_frontends.py
│   └── stats_research.py
│
├── codegen/                 # コード生成
│   └── generate_constants.py  # YAML → Python/Rust 定数
│
└── demo/                    # デモ用
    └── tts_demo.py
```

**ルール:**
- 新規スクリプトは適切なサブディレクトリに配置
- ルート直下にスクリプトを置かない
- 一時的なスクリプトは作成しない (どうしても必要なら `tmp/` を作成し gitignore)

### tests/ ディレクトリ構成

テストは対象モジュールごとにサブディレクトリを切る:

```
tests/
├── conftest.py              # 共有フィクスチャ (ルート)
│
├── core/                    # tmrvc-core のテスト
│   ├── test_constants.py
│   └── test_audio.py
│
├── data/                    # tmrvc-data のテスト
│   ├── test_preprocessing.py
│   ├── test_cache.py
│   ├── test_dataset.py
│   └── ...
│
├── train/                   # tmrvc-train のテスト
│   ├── test_teacher.py
│   ├── test_diffusion.py
│   └── ...
│
├── train_uclm/              # UCLM 関連のテスト
│   ├── test_uclm_model.py
│   └── ...
│
├── export/                  # tmrvc-export のテスト
│   ├── test_export.py
│   └── ...
│
├── serve/                   # tmrvc-serve のテスト
├── gui/                     # tmrvc-gui のテスト
└── scripts/                 # スクリプトのテスト
```

**ルール:**
- テストファイルは対応するモジュールのサブディレクトリに配置
- ファイル名は `test_{対象}.py` 形式
- 共有フィクスチャは `conftest.py` に定義

## Training Data & Artifacts

`data/`, `checkpoints/`, `models/`, `logs/` はすべて **gitignore 対象**。
絶対に git に追加しないこと。

### ディレクトリレイアウト

```
TMRVC/                        # リポジトリルート
├── data/                     # ★ gitignore — 全学習データ
│   ├── raw/                  # 生音声 (ダウンロード・手動配置)
│   │   ├── VCTK-Corpus/      #   T1: 英語 109 話者 48kHz ~44h
│   │   ├── jvs_corpus/       #   T1: 日本語 100 話者 24kHz ~30h
│   │   ├── tsukuyomi/        #   T1: 萌え声 1 話者 96kHz ~0.3h
│   │   ├── libritts_r/       #   T2: 英語 2,456 話者 24kHz ~585h
│   │   └── rir/              #   RIR データ (AIR, BUT ReverbDB 等)
│   ├── cache/                # 前処理済み特徴量 (FeatureCache)
│   │   ├── _manifests/       #   検証メタデータ
│   │   ├── vctk/train/{speaker_id}/{utt_id}/
│   │   │   ├── mel.npy       #     [80, T] log-mel
│   │   │   ├── content.npy   #     [768, T] ContentVec (Phase 0) / [1024, T] WavLM (Phase 1+)
│   │   │   ├── f0.npy        #     [1, T] Hz
│   │   │   ├── spk_embed.npy #     [192]
│   │   │   └── meta.json
│   │   ├── jvs/train/...
│   │   ├── tsukuyomi/train/...
│   │   └── libritts_r/train/...
│   ├── fewshot_test/         # Few-shot テスト用音声 (数ファイル)
│   └── sample_voice/         # デモ・手動テスト用音声
│
├── checkpoints/              # ★ gitignore — PyTorch チェックポイント
│   ├── teacher_step*.pt      #   Teacher 学習チェックポイント
│   ├── codec/                #   Codec 学習チェックポイント
│   ├── uclm/                 #   UCLM 学習チェックポイント
│   └── distill/              #   蒸留チェックポイント (Phase A/B/B2/C)
│
├── models/                   # ★ gitignore — ONNX モデル + speaker ファイル
│   ├── fp32/                 #   FP32 ONNX (6 モデル)
│   ├── int8/                 #   INT8 量子化版 (将来)
│   ├── *.tmrvc_speaker       #   話者プロファイル
│   └── *.tmrvc_style         #   スタイルプロファイル (TTS用)
│
├── scratch/                  # ★ gitignore — 書き捨てファイル
│   ├── eval/                 #   評価サンプル出力 (wav等)
│   ├── logs/                 #   学習・実行ログ
│   ├── outputs/              #   テスト出力
│   └── tmp/                  #   一時ファイル
│
└── configs/
    ├── constants.yaml        # 共有定数 (source of truth)
    ├── datasets.yaml         # データセットレジストリ (パス設定)
    ├── train_*.yaml          # 各種学習設定
    └── export.yaml           # エクスポート設定
```

### データパイプライン

```
1. 生音声取得         data/raw/{dataset}/     手動ダウンロード or スクリプト
       ↓
2. 前処理 (特徴量抽出) data/cache/{dataset}/   tmrvc-preprocess / scripts/prepare_datasets.py
       ↓
3. 学習               checkpoints/            tmrvc-train-* コマンド
       ↓
4. ONNX エクスポート  models/fp32/            tmrvc-export
       ↓
5. 推論               models/fp32/ を参照     tmrvc-engine-rs / tmrvc-rt / tmrvc-gui
```

### CLI コマンド一覧

```bash
# 初回セットアップ (環境に合わせて選択)
# CUDA環境
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
# XPU環境 (Intel Arc)
uv sync --extra-index-url https://download.pytorch.org/whl/xpu

# データ前処理
uv run tmrvc-preprocess --config configs/datasets.yaml --device cuda
uv run tmrvc-extract-features --input data/raw/custom --output data/cache/custom

# 学習 (Codec-Latent パラダイム)
uv run tmrvc-train-codec --config configs/train_codec.yaml --device cuda
uv run tmrvc-train-token --config configs/train_token.yaml --device cuda
uv run tmrvc-train-uclm --config configs/train_uclm.yaml --device cuda

# 学習 (TTS パイプライン)
uv run tmrvc-train-tts --config configs/train_tts.yaml --device cuda
uv run tmrvc-train-style --config configs/train_style.yaml --device cuda

# エクスポート
uv run tmrvc-export --checkpoint checkpoints/best.pt --output-dir models/fp32 --verify

# 話者ファイル作成 (階層的適応 v3)
uv run tmrvc-enroll --audio ref.wav --output models/speaker.tmrvc_speaker --level light
uv run tmrvc-enroll --audio-dir data/sample_voice/ --output models/speaker.tmrvc_speaker --level standard --codec-checkpoint checkpoints/codec/best.pt
uv run tmrvc-enroll --audio-dir data/sample_voice/ --output models/speaker.tmrvc_speaker --level full --token-model checkpoints/token.pt --finetune-steps 200 --device cuda

# GUI / サーバー
uv run tmrvc-gui              # Research Studio GUI
uv run tmrvc-serve            # FastAPI TTS server

# キャッシュ検証
uv run tmrvc-verify-cache --cache-dir data/cache
```

### Rust 側の環境変数

Rust エンジン (tmrvc-engine-rs, tmrvc-rt, tmrvc-vst) は以下の環境変数でモデル・話者パスを上書き可能:

| 変数 | デフォルト | 用途 |
|---|---|---|
| `TMRVC_MODEL_DIR` | `models/fp32` (ワークスペース相対) | ONNX モデルディレクトリ |
| `TMRVC_SPEAKER_PATH` | `models/test_speaker.tmrvc_speaker` | 話者ファイル |
| `TMRVC_STYLE_PATH` | — | スタイルファイル (.tmrvc_style) |
| `TMRVC_ONNX_DIR` | `models/fp32` | tmrvc-rt GUI 用 |

### 重要なルール

1. **`data/raw/` のファイルは絶対に git に追加しない** (vctk.zip だけで 11GB)
2. **`data/cache/` は再生成可能** — `tmrvc-preprocess` で復元
3. **`checkpoints/` は学習の成果物** — 削除すると学習のやり直しが必要
4. **`models/fp32/` は `tmrvc-export` で再生成可能** — チェックポイントがあれば復元可能
5. **`configs/datasets.yaml` の `raw_dir` はマシン固有** — 他者環境では書き換えが必要
6. **RIR データは `rir_dirs` で指定** — `data/raw/rir/` に配置
7. **書き捨てファイルは `scratch/` に出力** — 評価サンプル・ログ・テスト出力は全てここに
8. **ルート直下に wav/log 等を出力しない** — 必ず `scratch/` サブディレクトリを使用

## Training Pipelines

### 1. Voice Conversion (VC) パイプライン

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Teacher     │ ──▶ │ Distillation │ ──▶ │ ONNX Export │ ──▶ │ VST3 Plugin│
│ (Diffusion) │     │ (Phase A-D)  │     │             │     │ (Realtime) │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
   ~17M params        ~7.7M params         68MB total         <50ms latency
```

詳細は `docs/training/vc-training-plan.md` を参照。

### 2. Text-to-Speech (TTS) パイプライン

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Text Input  │ ──▶ │ TTS Frontend │ ──▶ │ Diffusion   │ ──▶ Audio
│ + Phoneme   │     │ (Duration,F0)│     │ Vocoder     │
└─────────────┘     └──────────────┘     └─────────────┘
```

詳細は `docs/training/tts-training-plan.md` を参照。

### 3. Codec-Latent パラダイム (次世代)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Audio Input │ ──▶ │ Causal Codec │ ──▶ │ Token Model │ ──▶ Audio Output
│             │     │ (Encoder)    │     │ (Mamba)     │     (Decoder)
└─────────────┘     └──────────────┘     └─────────────┘
```

詳細は `docs/design/codec-latent-design.md`, `docs/design/unified-codec-lm.md` を参照。

### 4. 学習データの種類

| データセット | 言語 | 話者数 | サイズ | 用途 |
|-------------|------|--------|--------|------|
| VCTK | 英語 | 109 | 44h | VC基本学習 |
| JVS | 日本語 | 100 | 30h | VC日本語対応 |
| Tsukuyomi | 日本語 | 1 | 0.3h | 萌え声 |
| LibriTTS-R | 英語 | 2,456 | 585h | 大規模学習 |
| jvnv | 日本語 | 4 | 4h | 感情学習 |
| Custom | 任意 | 任意 | 任意 | Few-shot / TTS |

## Design Documents

### docs/design/ (13 ファイル)

| File | Content |
|---|---|
| `architecture.md` | 全体アーキテクチャ、モジュール構成、主要設計判断 |
| `streaming-design.md` | レイテンシバジェット、Audio Thread パイプライン、Latency-Quality Spectrum |
| `onnx-contract.md` | 6 モデルの I/O テンソル仕様、State tensor、`.tmrvc_speaker` フォーマット |
| `model-architecture.md` | Content Encoder / Converter / Vocoder / IR Estimator / Teacher の詳細 |
| `cpp-engine-design.md` | Rust エンジンクラス設計、TensorPool メモリレイアウト (タイトルは旧来のまま) |
| `acoustic-condition-pathway.md` | IR + Voice Source 統合条件付け (32dim) |
| `codec-latent-design.md` | Codec-Latent パラダイムの詳細設計 |
| `unified-codec-lm.md` | UCLM (Unified Codec Language Model) アーキテクチャ |
| `unified-generator-tts.md` | 統合 Generator TTS 設計 |
| `expressive-tts-design.md` | 感情表現 TTS 設計 |
| `gui-design.md` | Research Studio GUI アプリケーション設計 |
| `dataset-preparation-flow.md` | データセット準備パイプライン |
| `uclm-implementation-roadmap.md` | UCLM 実装ロードマップ |

### docs/training/ (5 ファイル)

| File | Content |
|---|---|
| `vc-training-plan.md` | VC Teacher + 蒸留の学習計画 |
| `tts-training-plan.md` | TTS パイプラインの学習計画 |
| `style-training-plan.md` | StyleEncoder、感情表現の学習計画 |
| `tts-front-end-review-2026-02-25.md` | TTS Frontend の技術レビュー |
| `README.md` | 学習パイプライン統合ガイド |

設計変更時は各ファイル末尾の整合性チェックリストを確認すること。

## Key Constants (configs/constants.yaml)

```yaml
# Audio
sample_rate: 24000
hop_length: 240          # 10ms
n_fft: 1024
window_length: 960       # 40ms
n_mels: 80

# Model dimensions
d_content: 256
d_speaker: 192
n_acoustic_params: 32    # 24 IR + 8 voice source
d_converter_hidden: 384
d_vocoder_features: 513  # n_fft // 2 + 1

# LoRA
lora_rank: 4
lora_delta_size: 15872   # 4 layers × 3968

# State
converter_state_frames: 52
converter_hq_state_frames: 46
```

## Tech Stack

**Python 側:**
- PyTorch >= 2.2, torchaudio, transformers, speechbrain
- uv workspace (pyproject.toml)
- PySide6 (GUI), FastAPI (serve)

**Rust 側:**
- Rust 2021 edition, Cargo workspace
- ONNX Runtime (`ort` crate, CPU EP)
- nih-plug (VST3), egui (tmrvc-rt GUI)

## Critical Constraints

1. **Audio Thread は RT-safe**: malloc/free/mutex/file I/O/例外 一切禁止
2. **Live models は causal** (look-ahead = 0): 未来の情報を参照しない
3. **HQ mode は semi-causal** (look-ahead = 6): 60ms lookahead 許容
4. **TensorPool は単一 contiguous allocation** (~300KB): 動的確保なし
5. **State tensor は Ping-Pong double buffering**: in-place 更新を回避
6. **Acoustic pathway は Few-shot 時に凍結**: 環境特性・声質特性と話者特性を分離

## Building

```bash
# Python (uv workspace)
uv sync --extra-index-url https://download.pytorch.org/whl/cu128  # CUDA
# または
uv sync --extra-index-url https://download.pytorch.org/whl/xpu    # XPU (Intel Arc)

# Rust (Cargo workspace)
cargo build --release
```

## Testing

```bash
# Python テスト
uv run pytest tests/

# Rust テスト
cargo test

# ONNX パリティ検証
uv run python -m tmrvc_export.verify_parity
```

## Runtime Device Policy

- **CUDA環境**: `--device cuda` を使用
- **XPU環境 (Intel Arc)**: `--device xpu` を使用

### デバイス確認

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"  # CUDA
uv run python -c "import torch; print(torch.xpu.is_available())"   # XPU
```

### XPU 特有の注意点

XPU は JIT カーネルコンパイルを行うため、テンソル形状が変わるたびに再コンパイルが発生し **10〜18 倍の速度低下** を引き起こす。

- `--max-frames 400` がデフォルト
- `--max-frames 0` で無効化可能（バケットバッチングにフォールバック）

### フォールバック禁止ポリシー

- デバイスでエラーが出た場合、CPU にフォールバックしない
- バッチサイズを下げる、デバイスの回復を待つ等で対処する

### Windows 特有の注意点

- `--num-workers 0` を使用（明示的にチューニングしない限り）
