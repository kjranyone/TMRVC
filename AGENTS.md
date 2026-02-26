# TMRVC — IR-aware Real-time Voice Conversion

## Project Overview

CPU-only で end-to-end 50ms 以下のリアルタイム Voice Conversion を実現する VST3 プラグイン。
Frame-by-frame causal streaming アーキテクチャ（10ms hop 単位）で処理する。

- **内部サンプルレート:** 24kHz（DAW の 44.1/48kHz とはポリフェーズリサンプルで接続）
- **推論:** 5 ONNX モデル（実行頻度が異なるため分割）を ONNX Runtime C API で実行
- **学習:** Teacher (diffusion U-Net, ~17M) → Student (causal CNN, ~7.7M) に蒸留
- **学習データ:** VCTK + JVS (T1) → LibriTTS-R (T2) → Emilia (T3) 段階的追加

## Architecture

```
DAW Audio In → VST3 Plugin → StreamingEngine (C++) → ONNX Models ×5 → DAW Audio Out
                                   ↑
                              JUCE 非依存
```

5 ONNX モデルの実行頻度:
- `content_encoder` / `converter` / `vocoder` — per-frame (10ms ごと)
- `ir_estimator` — 10 フレームに 1 回 (~100ms)
- `speaker_encoder` — offline (enrollment 時のみ)

## Monorepo Structure

```
TMRVC/
├── docs/design/          # 正本設計資料 (5 ファイル)
├── docs/reference/       # 参考資料 (旧設計, 先行研究)
├── tmrvc-core/           # Python: 共有定数・mel 計算・型定義
├── tmrvc-data/           # Python: データセット・前処理・augmentation
├── tmrvc-train/          # Python: モデル定義・学習・蒸留・Few-shot
├── tmrvc-export/         # Python: ONNX エクスポート・量子化・パリティ検証
├── tmrvc-engine/         # C++: ストリーミング推論エンジン (JUCE 非依存)
├── tmrvc-plugin/         # C++: JUCE VST3 プラグイン
├── tests/                # Python / C++ 統合テスト
├── configs/              # constants.yaml (shared source of truth)
└── scripts/              # generate_constants.py 等
```

Python と C++ は **ONNX ファイル** と **constants.yaml → 自動生成ヘッダ** でのみ接続。

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
│   │       ├── air/
│   │       └── but_reverb/
│   ├── cache/                # 前処理済み特徴量 (FeatureCache)
│   │   ├── _manifests/       #   検証メタデータ
│   │   ├── vctk/train/{speaker_id}/{utt_id}/
│   │   │   ├── mel.npy       #     [80, T] log-mel
│   │   │   ├── content.npy   #     [768, T] ContentVec
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
│   └── distill/              #   蒸留チェックポイント (Phase A/B/B2/C)
│       ├── phaseA_step*.pt
│       └── ...
│
├── models/                   # ★ gitignore — ONNX モデル + speaker ファイル
│   ├── fp32/                 #   FP32 ONNX (content_encoder / converter / vocoder / ir_estimator / converter_hq)
│   ├── int8/                 #   INT8 量子化版 (将来)
│   ├── test_speaker.tmrvc_speaker
│   └── demo_fewshot.tmrvc_speaker
│
├── logs/                     # ★ gitignore — 学習ログ
│   └── train_teacher_*.log
│
└── configs/
    ├── constants.yaml        # 共有定数 (source of truth)
    ├── datasets.yaml         # データセットレジストリ (パス設定)
    ├── train_teacher.yaml    # Teacher 学習設定
    └── train_student.yaml    # 蒸留設定
```

### データパイプライン

```
1. 生音声取得         data/raw/{dataset}/     手動ダウンロード or スクリプト
       ↓
2. 前処理 (特徴量抽出) data/cache/{dataset}/   scripts/prepare_datasets.py
       ↓
3. Teacher 学習       checkpoints/            tmrvc-train-teacher
       ↓
4. 蒸留               checkpoints/distill/    tmrvc-distill
       ↓
5. ONNX エクスポート  models/fp32/            tmrvc-export
       ↓
6. 推論               models/fp32/ を参照     tmrvc-engine-rs / tmrvc-gui
```

### データセットレジストリ (`configs/datasets.yaml`)

前処理の入出力パスを一元管理する。`raw_dir` を実際のローカルパスに書き換えて使う。
パスはリポジトリルートからの相対パス、または絶対パスで指定可能。

```bash
# 前処理実行 (enabled: true のデータセットのみ処理)
# --device は環境に合わせて cuda/xpu を指定
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --device cuda
```

### CLI コマンドとパス指定

```bash
# 初回セットアップ (環境に合わせて選択)
# CUDA環境
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
# XPU環境 (Intel Arc)
uv sync --extra-index-url https://download.pytorch.org/whl/xpu

# Teacher 学習 (--device は環境に合わせて cuda/xpu を指定)
uv run tmrvc-train-teacher --cache-dir data/cache --phase 0 --device cuda

# 蒸留
uv run tmrvc-distill --cache-dir data/cache \
  --teacher-ckpt checkpoints/teacher_step8000.pt --phase A --device cuda

# ONNX エクスポート
uv run tmrvc-export --checkpoint checkpoints/distill/best.pt \
  --output-dir models/fp32 --verify

# Few-shot
uv run tmrvc-finetune --cache-dir data/cache \
  --checkpoint checkpoints/distill/best.pt \
  --audio-dir data/sample_voice/ --device cuda
```

### Rust 側の環境変数

Rust エンジン (tmrvc-engine-rs, tmrvc-rt) は以下の環境変数でモデル・話者パスを上書き可能:

| 変数 | デフォルト | 用途 |
|---|---|---|
| `TMRVC_MODEL_DIR` | `models/fp32` (ワークスペース相対) | ONNX モデルディレクトリ |
| `TMRVC_SPEAKER_PATH` | `models/test_speaker.tmrvc_speaker` | 話者ファイル |
| `TMRVC_STYLE_PATH` | — | スタイルファイル (.tmrvc_style) |
| `TMRVC_ONNX_DIR` | `models/fp32` | tmrvc-rt GUI 用 |

### 重要なルール

1. **`data/raw/` のファイルは絶対に git に追加しない** (vctk.zip だけで 11GB)
2. **`data/cache/` は再生成可能** — `scripts/prepare_datasets.py` で復元
3. **`checkpoints/` は学習の成果物** — 削除すると学習のやり直しが必要
4. **`models/fp32/` は `tmrvc-export` で再生成可能** — チェックポイントがあれば復元可能
5. **`configs/datasets.yaml` の `raw_dir` はマシン固有** — 他者環境では書き換えが必要
6. **RIR データは `rir_dirs` で指定** — `data/raw/rir/` に配置し、学習 YAML の augmentation config で参照

## Design Documents (docs/design/)

| File | Content |
|---|---|
| `architecture.md` | 全体アーキテクチャ、モジュール構成、主要設計判断 7 件の根拠 |
| `streaming-design.md` | レイテンシバジェット、Audio Thread パイプライン、Ring Buffer、OLA、Causal STFT |
| `onnx-contract.md` | 5 モデルの I/O テンソル仕様、State tensor、`.tmrvc_speaker` フォーマット、数値パリティ基準 |
| `model-architecture.md` | Content Encoder / Converter / Vocoder / IR Estimator / Speaker Encoder / Teacher の詳細 |
| `cpp-engine-design.md` | C++ エンジンクラス設計、TensorPool メモリレイアウト、SPSC Queue、VST3 統合 |
| `training-plan.md` | Teacher 学習計画: コーパス構成、4フェーズ学習、コスト見積もり、蒸留接続 |
| `acoustic-condition-pathway.md` | IR→Acoustic Pathway 拡張: 24dim環境 + 8dim声質 = 32dim統合条件付け |

設計変更時はこれらの整合性チェックリスト（各ファイル末尾）を確認すること。

## Key Constants (configs/constants.yaml)

```
sample_rate: 24000    hop_length: 240 (10ms)    n_mels: 80
n_fft: 1024           window_length: 960         d_content: 256
d_speaker: 192        n_acoustic_params: 32      d_converter_hidden: 384
```

## Tech Stack

**Python 側:**
- PyTorch >= 2.2, torchaudio, transformers (WavLM/HuBERT), speechbrain (ECAPA-TDNN)
- uv workspace (pyproject.toml)

**C++ 側:**
- C++17, CMake >= 3.20
- ONNX Runtime (C API, ort-builder で静的リンク, CPU EP のみ)
- JUCE (VST3 プラグインのみ、エンジンは非依存)

## Critical Constraints

1. **Audio Thread は RT-safe**: malloc/free/mutex/file I/O/例外 一切禁止
2. **全モデルは causal** (look-ahead = 0): 未来の情報を参照しない
3. **TensorPool は単一 contiguous allocation** (~281KB): 動的確保なし
4. **State tensor は Ping-Pong double buffering**: in-place 更新を回避
5. **Worker Thread との通信は lock-free SPSC Queue のみ**
6. **Acoustic pathway は Few-shot 時に凍結**: 環境特性・声質特性と話者特性を分離

## Building

```bash
# Python (uv workspace)
# CUDA環境
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
# XPU環境 (Intel Arc)
uv sync --extra-index-url https://download.pytorch.org/whl/xpu

# C++ (CMake)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Testing

```bash
# Python テスト
uv run pytest tests/python/

# C++ テスト
cd build && ctest

# ONNX パリティ検証 (Python vs C++)
uv run python -m tmrvc_export.verify_parity
```

## Runtime Device Policy

- **CUDA環境**: `--device cuda` を使用
- **XPU環境 (Intel Arc)**: `--device xpu` を使用
- 環境に合わせて初回セットアップ時に適切なtorchインデックスを選択すること

### デバイス確認

```bash
# CUDA
uv run python -c "import torch; print(torch.cuda.is_available())"

# XPU (Intel Arc)
uv run python -c "import torch; print(torch.xpu.is_available())"
```

### XPU 特有の注意点

XPU (Intel Arc) は JIT カーネルコンパイルを行うため、テンソル形状が変わるたびに
再コンパイルが発生し **10〜18 倍の速度低下** を引き起こす。

- `--max-frames 400` がデフォルト（Teacher 学習・蒸留の両方）
  - VCTK 中央値 301 フレーム / 95th pct 569 に合わせた設定
  - 全バッチを固定 400 フレームに crop/pad し、カーネル形状を 1 種類に固定
  - パディング領域のノイズはゼロ化（F5-TTS/VoiceFlow 方式 = Approach A）
  - 400 フレーム超の発話はランダムクロップ
- `--max-frames 0` で無効化可能（バケットバッチングにフォールバック: [250,500,750,1000]）
- Few-shot (`tmrvc-finetune`) は固定 200 フレームで動作するため影響なし

### フォールバック禁止ポリシー

- デバイスでエラーが出た場合、CPU にフォールバックしない
- バッチサイズを下げる、デバイスの回復を待つ等で対処する
- 後方互換性のためのフォールバックコードも書かない（スパゲティコードの原因）
- データが古い場合はデータ自体を修正する（コード側で `.get()` デフォルト値を入れない）

### Windows 特有の注意点

- `--num-workers 0` を使用（明示的にチューニングしない限り）
