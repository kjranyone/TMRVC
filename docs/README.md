# TMRVC Documentation

## プロジェクト概要

TMRVC は **2 つの音声生成機能** を統合したプロジェクト:

| 機能 | 特徴 | 用途 |
|------|------|------|
| **Voice Conversion (VC)** | 10ms streaming, CPU-only, causal | リアルタイム変換 (VST3 / Standalone) |
| **Text-to-Speech (TTS)** | Server API, WebSocket streaming | バッチ合成・アプリ連携 |

```
                              TMRVC Project
                                              
                    ┌─────────────────────────────────┐
                    │         Shared Foundation        │
                    │  tmrvc-core | tmrvc-data         │
                    │  tmrvc-export (ONNX)             │
                    └───────────────┬─────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                      │
         ▼                                                      ▼
┌─────────────────────────┐                        ┌─────────────────────────┐
│    Voice Conversion     │                        │     Text-to-Speech      │
│    (10ms streaming)     │                        │     (Server API)        │
├─────────────────────────┤                        ├─────────────────────────┤
│ Training:               │                        │ Training:               │
│  • tmrvc-train-teacher  │                        │  • tmrvc-train-tts      │
│  • tmrvc-distill        │                        │  • tmrvc-train-style    │
│  • tmrvc-finetune       │                        │                         │
├─────────────────────────┤                        ├─────────────────────────┤
│ Inference:              │                        │ Inference:              │
│  • tmrvc-engine-rs ─────┼──── 共有 ONNX ─────────┼─► tmrvc-serve           │
│    (Rust library)       │                        │    (FastAPI)            │
│  • tmrvc-rt (egui GUI)  │                        │                         │
│  • tmrvc-vst (DAW VST3) │                        │                         │
└─────────────────────────┘                        └─────────────────────────┘
         │                                                      │
         ▼                                                      ▼
   Real-time Audio                                       HTTP/WebSocket
   (VST3 / Standalone)                                   (JSON/WAV)


                    ┌─────────────────────────────────┐
                    │     tmrvc-gui (Research Studio)  │
                    │                                 │
                    │  ┌─────────────────────────┐    │
                    │  │ Data Prep               │    │
                    │  │ (datasets.yaml → cache) │    │
                    │  └───────────┬─────────────┘    │
                    │              ▼                  │
                    │  ┌─────────────────────────┐    │
                    │  │ Training                │    │
                    │  │ Teacher → Distill → FT  │    │
                    │  └───────────┬─────────────┘    │
                    │              ▼                  │
                    │  ┌─────────────────────────┐    │
                    │  │ Export                  │    │
                    │  │ (ONNX / Speaker files)  │    │
                    │  └───────────┬─────────────┘    │
                    │              ▼                  │
                    │  ┌───────────┴───────────┐      │
                    │  │ VC Demo │ TTS │ Eval  │      │
                    │  └───────────────────────┘      │
                    └─────────────────────────────────┘
```

## モジュール一覧

### 共通基盤

| Module | Language | Description |
|--------|----------|-------------|
| `tmrvc-core` | Python | 共有定数 (constants.yaml)、mel 計算、型定義 |
| `tmrvc-data` | Python | データセット、前処理、augmentation、RIR |
| `tmrvc-export` | Python | ONNX エクスポート、量子化、`.tmrvc_speaker` 作成 |

### 学習 (tmrvc-train)

| CLI | 用途 | 出力 |
|-----|------|------|
| `tmrvc-train-teacher` | VC Teacher (diffusion U-Net) 学習 | `checkpoints/teacher_*.pt` |
| `tmrvc-distill` | Teacher → Student 蒸留 | `checkpoints/distill/*.pt` |
| `tmrvc-finetune` | Few-shot fine-tuning | `*.tmrvc_speaker` |
| `tmrvc-train-tts` | TTS モデル学習 | `checkpoints/tts_*.pt` |
| `tmrvc-train-style` | スタイル埋め込み学習 | `checkpoints/style_*.pt` |

### 推論

#### Voice Conversion (Real-time)

| Module | Language | Type | Description |
|--------|----------|------|-------------|
| `tmrvc-engine-rs` | Rust | library | ONNX 推論エンジン + NAM |
| `tmrvc-rt` | Rust | binary | egui + cpal スタンドアロン GUI |
| `tmrvc-vst` | Rust | VST3 | DAW プラグイン (nih-plug) |

#### Text-to-Speech (Server)

| Module | Language | Type | Description |
|--------|----------|------|-------------|
| `tmrvc-serve` | Python | FastAPI | WebSocket TTS サーバー |

### 開発用 GUI

| Module | Language | Description |
|--------|----------|-------------|
| `tmrvc-gui` | Python | PySide6 Research Studio (全機能統合) |

Pages:
- Data Prep → Teacher Train → Distillation → Enrollment
- ONNX Export → Evaluation → Realtime Demo → TTS → Style Editor

## docs/design/ — 現行設計 (正本)

Frame-by-frame causal streaming アーキテクチャに基づく設計資料群。

| File | Description |
|------|-------------|
| `architecture.md` | システム全体アーキテクチャ、モジュール構成、データフロー、主要設計判断 |
| `streaming-design.md` | ストリーミング設計、レイテンシバジェット、Audio Thread パイプライン、スレッドモデル |
| `onnx-contract.md` | 5 ONNX モデルの I/O 仕様、State tensor、共有定数、数値パリティ基準 |
| `model-architecture.md` | Content Encoder / Converter / Vocoder / IR Estimator / Speaker Encoder / Teacher の詳細設計 |
| `cpp-engine-design.md` | Rust エンジン (tmrvc-engine-rs) と VST3 プラグイン (tmrvc-vst) の設計 |
| `training-plan.md` | Teacher 学習計画: コーパス構成、段階的学習フェーズ、コスト見積もり、蒸留への接続 |
| `research-novelty-plan.md` | 論文新規性トラック: Scene State Latent / Breath-Pause Event Head / latency-conditioned distillation |
| `gui-design.md` | tmrvc-gui (Research Studio) の設計 |

## docs/reference/ — 参考資料

現行設計の意思決定に影響を与えた資料。

| File | Description |
|------|-------------|
| `system_design.md` | 旧設計 (chunk-based, 15M Student, OT-CFM DiT Teacher)。品質目標 (SECS>=0.92)、3段階蒸留の参考 |
| `concept.md` | IR-aware 先行研究マップ。BUDDy, Gencho 等の IR 推定手法、RIR データセット |
