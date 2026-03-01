# TMRVC — Unified Codec Language Model (UCLM) v2

## Project Overview

TMRVC は、Unified Codec Language Model (UCLM) v2 アーキテクチャを採用した、リアルタイム音声合成 (TTS) および音声変換 (VC) システムです。

- **コア・アーキテクチャ**: UCLM v2 (Dual-Stream Token Spec)
- **推論エンジン**: `UCLMEngine` (FastAPI / Rust Streaming)
- **特徴**: 10ms 単位の極低遅延処理、8次元物理パラメータによる直接的な演技制御。

## Architecture

UCLM v2 は、音声の「内容」「話者」「演技（Voice State）」を完全に分離し、統合されたトランスフォーマー・モデルで再構成します。

### 推論コンポーネント (ONNX)
- `uclm`: トークン予測のメインコア。
- `codec`: 音声のエンコード/デコードを行う。
- `speaker_encoder`: 数秒の音声から話者埋め込みを抽出。

## Monorepo Structure

```
TMRVC/
├── configs/              # constants.yaml (UCLM v2 Spec)
├── docs/design/          # 正本設計資料
├── tmrvc-core/           # 共有定数・型定義
├── tmrvc-data/           # データセット準備・前処理
├── tmrvc-train/          # UCLM v2 モデル定義・学習
├── tmrvc-export/         # ONNX エクスポート・量子化
├── tmrvc-serve/          # 統合推論サーバー (UCLMEngine)
├── tmrvc-gui/            # 開発・デモ用 GUI (UCLM v2 対応済)
├── tmrvc-engine-rs/      # Rust 推論コア
└── tmrvc-vst/            # VST3 プラグイン
```

## Key Constants (UCLM v2)

`configs/constants.yaml` に定義される主要定数:

- `hop_length`: 240 (10ms @ 24kHz)
- `d_model`: 512
- `d_voice_state`: 8 (物理パラメータ)
- `n_codebooks`: 8 (音響トークン)
- `control_slots`: 4 (制御トークン)

## CLI Commands

| Command | Description |
|---------|-------------|
| `tmrvc-preprocess` | 学習データのトークナイズ・特徴量抽出 |
| `tmrvc-train-uclm` | UCLM v2 モデルのマルチタスク学習 |
| `tmrvc-train-codec` | Emotion-Aware Codec の学習 |
| `tmrvc-serve` | 統合推論サーバーの起動 |
| `tmrvc-export` | UCLM/Codec の ONNX エクスポート |
| `tmrvc-gui` | Research Studio GUI の起動 |

## Critical Constraints

1. **Dual-Stream Token Sync**: Acoustic (`A_t`) と Control (`B_t`) は常に同期して予測されなければならない。
2. **10ms Causal Core**: 未来の情報を参照する非因果的（Non-causal）な処理は一切禁止。
3. **Physical-First Control**: 演技制御は 8次元の物理パラメータ（息漏れ等）を優先し、抽象的なラベルに頼らない。
4. **Scientific Rigor & Zero Compromise**: 論文実装という観点から、緻密さと数学的整合性を最優先する。場当たり的なコード置換、テンソル形状の不一致を誤魔化すための不自然なパディング、あるいは失敗したテストの放置は「悪（Evil）」と定義し、厳禁とする。すべての変更は全スタック層（Core, Train, Serve, GUI, Export）において論理的に整合し、常に厳格な数学的パリティテストによって実証されなければならない。

## Resolved Issues (2026-03-01)

### Issue A: Frame Alignment (FIXED - Root Cause)
- **Problem**: `mel=99 frames`, `codec=100 frames` - 1フレームズレ
- **Root Cause**: `MelSpectrogram.pad_length = window_length - hop_length` (720) was incorrect
- **Solution**: `pad_length = 784` for exact `T = ceil(N / hop_length)` alignment
- **Files**:
  - `tmrvc-core/src/tmrvc_core/audio.py`: Fixed padding calculation
  - `tmrvc-data/src/tmrvc_data/cli/preprocess.py`: Replaced zero-padding with assert validation
  - `tests/data/test_frame_alignment.py`: NEW - 14 tests for frame alignment parity
- **Verification**: All 14 frame alignment tests pass

### Issue B: Constants Management (FIXED)
- **Problem**: `constants.py` hardcoded, diverged from YAML
- **Solution**: YAML is single source of truth, auto-generate Python/Rust
- **Files**:
  - `configs/constants.yaml`: Complete constant definitions
  - `scripts/codegen/generate_constants.py`: Auto-generation
  - `tmrvc-core/_generated_constants.py`: Auto-generated (DO NOT EDIT)
  - `tmrvc-core/constants.py`: Re-exports + minimal compat aliases
  - `tmrvc-engine-rs/constants.rs`: Auto-generated
- **Verification**: Python ↔ Rust constant values match exactly

### Issue E: Single Source of Truth for f0_mean (FIXED)
- **Problem**: f0_mean stored in both binary section and metadata JSON
- **Solution**: Binary section is the single source of truth
- **Files**:
  - `tmrvc-export/speaker_file.py`: Removed f0_mean from metadata
  - `tmrvc-engine-rs/speaker.rs`: Binary f0_mean takes precedence
- **Verification**: Python/Rust roundtrip tests pass

## Remaining Issues

### Issue C: Streaming Numerical Parity
- **Status**: 未検証
- **Problem**: CausalConv1d batch vs streaming processing numerical drift
- **Location**: `tmrvc-engine-rs/src/ort_bundle.rs`
- **Action Required**: Write parity test comparing batch vs frame-by-frame inference

### Issue D: GUI/CLI Implementation
- **Status**: 部分実装
- **Problem**: `tmrvc-train-codec` collate_fn not verified