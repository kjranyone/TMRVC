# TMRVC — IR-aware Real-time Voice Conversion

## Project Overview

CPU-only で end-to-end 50ms 以下のリアルタイム Voice Conversion を実現する VST3 プラグイン。
Frame-by-frame causal streaming アーキテクチャ（10ms hop 単位）で処理する。

- **内部サンプルレート:** 24kHz（DAW の 44.1/48kHz とはポリフェーズリサンプルで接続）
- **推論:** 5 ONNX モデル（実行頻度が異なるため分割）を ONNX Runtime C API で実行
- **学習:** Teacher (diffusion U-Net, ~80M) → Student (causal CNN, ~7.7M) に蒸留
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

## Design Documents (docs/design/)

| File | Content |
|---|---|
| `architecture.md` | 全体アーキテクチャ、モジュール構成、主要設計判断 7 件の根拠 |
| `streaming-design.md` | レイテンシバジェット、Audio Thread パイプライン、Ring Buffer、OLA、Causal STFT |
| `onnx-contract.md` | 5 モデルの I/O テンソル仕様、State tensor、`.tmrvc_speaker` フォーマット、数値パリティ基準 |
| `model-architecture.md` | Content Encoder / Converter / Vocoder / IR Estimator / Speaker Encoder / Teacher の詳細 |
| `cpp-engine-design.md` | C++ エンジンクラス設計、TensorPool メモリレイアウト、SPSC Queue、VST3 統合 |
| `training-plan.md` | Teacher 学習計画: コーパス構成、4フェーズ学習、コスト見積もり、蒸留接続 |

設計変更時はこれらの整合性チェックリスト（各ファイル末尾）を確認すること。

## Key Constants (configs/constants.yaml)

```
sample_rate: 24000    hop_length: 240 (10ms)    n_mels: 80
n_fft: 1024           window_length: 960         d_content: 256
d_speaker: 192        n_ir_params: 24            d_converter_hidden: 384
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
6. **IR pathway は Few-shot 時に凍結**: 環境特性と話者特性を分離

## Building

```bash
# Python (uv workspace)
uv sync

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
