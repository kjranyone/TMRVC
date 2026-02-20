# TMRVC System Architecture

Kojiro Tanaka — architecture design
Created: 2026-02-16 (Asia/Tokyo)

> **Goal:** End-to-end 50ms 以下 (DAW バッファ込み)、CPU-only リアルタイム Voice Conversion。
> Frame-by-frame causal streaming アーキテクチャ。

---

## 1. システム概要図

```
┌────────────────────────────────────────────────────────────────────────┐
│                           DAW (Host)                                   │
│                                                                        │
│   Audio In ──▶ ┌─────────────────────────────────┐ ──▶ Audio Out      │
│                │     TMRVCProcessor (VST3)       │                    │
│                │  ┌───────────────────────────┐  │                    │
│                │  │    StreamingEngine         │  │                    │
│                │  │  ┌─────────────────────┐  │  │                    │
│                │  │  │   ONNX Models (×5)  │  │  │                    │
│                │  │  │  ┌───────────────┐  │  │  │                    │
│                │  │  │  │content_encoder│  │  │  │                    │
│                │  │  │  │ir_estimator   │  │  │  │                    │
│                │  │  │  │speaker_encoder│  │  │  │                    │
│                │  │  │  │converter      │  │  │  │                    │
│                │  │  │  │vocoder        │  │  │  │                    │
│                │  │  │  └───────────────┘  │  │  │                    │
│                │  │  └─────────────────────┘  │  │                    │
│                │  └───────────────────────────┘  │                    │
│                └─────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────────────┘
```

信号フロー:
```
DAW Input ──▶ VST3 Plugin ──▶ StreamingEngine ──▶ ONNX Models ──▶ DAW Output
  (48kHz)     (processBlock)   (C++ library)      (ORT C API)     (48kHz)
```

---

## 2. Monorepo ディレクトリ構成

```
TMRVC/
├── docs/                          # 設計資料 (本ファイル群)
│
├── tmrvc-core/                    # Python: 共有定数・ユーティリティ
│   ├── pyproject.toml
│   └── src/tmrvc_core/
│       ├── constants.py           # sample_rate, n_fft, hop_length, etc.
│       ├── audio.py               # mel, STFT, resampling
│       └── types.py               # shared type definitions
│
├── tmrvc-data/                    # Python: データ前処理パイプライン
│   ├── pyproject.toml
│   └── src/tmrvc_data/
│       ├── dataset.py             # Dataset / DataLoader
│       ├── augmentation.py        # RIR convolution, EQ, noise
│       ├── features.py            # F0, content embedding extraction
│       └── speaker.py             # speaker embedding extraction
│
├── tmrvc-train/                   # Python: モデル定義・学習ループ
│   ├── pyproject.toml
│   └── src/tmrvc_train/
│       ├── models/
│       │   ├── content_encoder.py
│       │   ├── converter.py       # Teacher (U-Net) + Student (causal CNN)
│       │   ├── vocoder.py         # iSTFT vocoder
│       │   ├── ir_estimator.py
│       │   ├── speaker_encoder.py
│       │   └── lora.py            # LoRA adapter
│       ├── losses/
│       │   ├── stft_loss.py       # Multi-res STFT loss
│       │   ├── distill_loss.py    # DMD / trajectory matching
│       │   └── speaker_loss.py    # ECAPA cosine similarity
│       ├── train_teacher.py
│       ├── train_student.py       # Distillation
│       └── train_fewshot.py       # Few-shot adaptation
│
├── tmrvc-export/                  # Python: ONNX エクスポート・検証
│   ├── pyproject.toml
│   └── src/tmrvc_export/
│       ├── export_onnx.py         # PyTorch → ONNX (5 models)
│       ├── quantize.py            # INT8 dynamic quantization
│       ├── verify_parity.py       # Python vs C++ numerical parity
│       └── benchmark.py           # ONNX Runtime benchmark
│
├── tmrvc-engine/                  # C++: ストリーミング推論エンジン (JUCE 非依存)
│   ├── CMakeLists.txt
│   ├── include/tmrvc/
│   │   ├── streaming_engine.h
│   │   ├── ort_session_bundle.h
│   │   ├── tensor_pool.h
│   │   ├── fixed_ring_buffer.h
│   │   ├── polyphase_resampler.h
│   │   ├── spsc_queue.h
│   │   ├── speaker_manager.h
│   │   ├── cross_fader.h
│   │   └── constants.h            # auto-generated from YAML
│   └── src/
│       ├── streaming_engine.cpp
│       ├── ort_session_bundle.cpp
│       ├── tensor_pool.cpp
│       ├── fixed_ring_buffer.cpp
│       ├── polyphase_resampler.cpp
│       ├── spsc_queue.cpp
│       ├── speaker_manager.cpp
│       └── cross_fader.cpp
│
├── tmrvc-plugin/                  # C++: JUCE VST3 プラグイン
│   ├── CMakeLists.txt
│   └── src/
│       ├── TMRVCProcessor.cpp     # PluginProcessor
│       ├── TMRVCProcessor.h
│       ├── TMRVCEditor.cpp        # PluginEditor (GUI)
│       └── TMRVCEditor.h
│
├── tmrvc-gui/                     # Python: Research Studio GUI (PySide6)
│   ├── pyproject.toml
│   └── src/tmrvc_gui/
│       ├── __main__.py            # Entry point (uv run tmrvc-gui)
│       ├── app.py                 # QApplication setup
│       ├── main_window.py         # Sidebar + QStackedWidget + StatusBar
│       ├── pages/                 # 7 workflow screens
│       ├── workers/               # Background jobs + AudioEngine
│       ├── widgets/               # Reusable UI components
│       ├── models/                # ProjectState, ConfigManager
│       └── resources/             # style.qss
│
├── tests/                         # 統合テスト
│   ├── python/
│   └── cpp/
│
├── configs/                       # 学習・エクスポート設定
│   ├── constants.yaml             # 共有定数 (YAML source of truth)
│   ├── train_teacher.yaml
│   ├── train_student.yaml
│   └── export.yaml
│
├── scripts/                       # ユーティリティスクリプト
│   ├── generate_constants.py      # YAML → Python + C++ header
│   └── run_benchmark.py
│
├── pyproject.toml                 # Workspace root (uv workspace)
└── CMakeLists.txt                 # Top-level CMake
```

---

## 3. モジュール責務表

| モジュール | 言語 | 責務 | 依存先 |
|---|---|---|---|
| **tmrvc-core** | Python | 共有定数 (sample_rate, n_fft, hop_length, dims)、mel 計算、型定義 | なし |
| **tmrvc-data** | Python | データセット管理、前処理、augmentation (RIR, EQ, noise)、特徴量抽出 (F0, content, speaker) | tmrvc-core |
| **tmrvc-train** | Python | Teacher/Student モデル定義、学習ループ、損失関数、蒸留、Few-shot adaptation | tmrvc-core, tmrvc-data |
| **tmrvc-export** | Python | PyTorch → ONNX エクスポート、INT8 量子化、Python↔C++ 数値パリティ検証 | tmrvc-core, tmrvc-train |
| **tmrvc-engine** | C++ | ストリーミング推論エンジン。ONNX Runtime でモデル実行、Ring Buffer、Resampler、Speaker 管理。**JUCE 非依存** | ONNX Runtime (C API) |
| **tmrvc-plugin** | C++ | JUCE VST3 ラッパー。DAW 統合 (processBlock, latency reporting, state persistence)、GUI | tmrvc-engine, JUCE |
| **tmrvc-gui** | Python | Research Studio GUI。全ワークフロー統合 (データ準備→学習→蒸留→評価→話者登録→リアルタイムVC→エクスポート)。PySide6 + sounddevice + pyqtgraph | tmrvc-core, tmrvc-data, tmrvc-train, tmrvc-export |

### モジュール間の依存グラフ

```
tmrvc-core ◀─── tmrvc-data ◀─── tmrvc-train
     ▲               ▲               │
     │               │               ▼
     └──────── tmrvc-export ◀────────┘
                     ▲
                     │
               tmrvc-gui (Python GUI, PySide6)
                     │──▶ tmrvc-data
                     │──▶ tmrvc-train
                     └──▶ tmrvc-export

tmrvc-engine ◀─── tmrvc-plugin
  (C++)            (C++ / JUCE)
       ▲
       │
  ONNX Runtime (C API, static link)
```

Python 側と C++ 側は ONNX ファイルと `constants.yaml` (→ 自動生成ヘッダ) でのみ接続される。
`tmrvc-gui` は Python 側の統合フロントエンドで、ONNX Runtime Python API 経由でリアルタイム VC も実行可能。

---

## 4. データフロー図

### 4.1 学習時フロー (Teacher → Student 蒸留 → Few-shot)

```
                        ┌─────── Training Pipeline ───────┐
                        │                                  │
  Audio Dataset         │   Phase 1: Teacher Training      │
  (VCTK, LibriTTS-R,   │   ─────────────────────────      │
   JVS + RIR augment)  │                                  │
        │               │   HuBERT ──▶ content (768d)     │
        ▼               │   RMVPE  ──▶ f0                 │
  ┌──────────┐          │   ECAPA  ──▶ spk_embed (192d)   │
  │ tmrvc-   │          │   IR est ──▶ ir_params (24d)    │
  │   data   │──────────▶                                  │
  └──────────┘          │   U-Net (v-prediction, non-causal)
                        │   + multi-res STFT loss          │
                        │              │                   │
                        │              ▼                   │
                        │   Teacher checkpoint             │
                        │              │                   │
                        │   Phase 2: Student Distillation  │
                        │   ────────────────────────────   │
                        │                                  │
                        │   Teacher (multi-step) generates │
                        │   reference mel/features         │
                        │              │                   │
                        │              ▼                   │
                        │   Student (causal CNN, 1-step):  │
                        │     content_encoder (mel→256d)   │
                        │     converter (1-step denoiser)  │
                        │     vocoder (iSTFT)              │
                        │              │                   │
                        │   Losses:                        │
                        │     Phase A: ODE trajectory      │
                        │     Phase B: DMD                 │
                        │              │                   │
                        │              ▼                   │
                        │   Student checkpoint             │
                        │              │                   │
                        │   Phase 3: Few-shot Adaptation   │
                        │   ───────────────────────────    │
                        │                                  │
                        │   Target speaker audio (3-20 utt)│
                        │   → ECAPA → spk_embed (192d)    │
                        │   → LoRA delta (cross-attn K/V)  │
                        │                                  │
                        │   Update: spk_embed + LoRA only  │
                        │   Freeze: content_enc, vocoder,  │
                        │           IR pathway             │
                        │              │                   │
                        │              ▼                   │
                        │   .tmrvc_speaker file            │
                        │   (spk_embed + LoRA delta)       │
                        └──────────────────────────────────┘
```

### 4.2 推論時フロー (Streaming VC)

```
DAW Audio In (48kHz)
      │
      ▼
┌─ StreamingEngine ──────────────────────────────────────────────┐
│                                                                 │
│  Downsample (48kHz → 24kHz, polyphase)                         │
│      │                                                          │
│      ▼                                                          │
│  Input Ring Buffer                                              │
│      │ accumulate ≥ hop_length (240 samples = 10ms)             │
│      ▼                                                          │
│  Causal STFT + Mel (per frame)                                  │
│      │                                                          │
│      ├──▶ content_encoder.onnx  ──▶ content[1,256,1]           │
│      │      (per-frame, ~0.3ms)     + state update              │
│      │                                                          │
│      ├──▶ ir_estimator.onnx     ──▶ ir_params[1,24]  ← cached │
│      │      (every ~10 frames, ~0.5ms amortized)                │
│      │                                                          │
│      │   speaker_encoder output: pre-loaded from                │
│      │      .tmrvc_speaker file (spk_embed + LoRA delta         │
│      │      + voice_source_preset)                              │
│      │                                                          │
│      ├──▶ Voice Source Blend (if preset loaded):               │
│      │      acoustic_params[24..31] = lerp(estimated, preset, α)│
│      │      (RT-safe: stack copy + 8 mul-add, zero alloc)       │
│      │                                                          │
│      ▼                                                          │
│  converter.onnx (1-step)                                        │
│      content + spk_embed + acoustic_params → pred_features      │
│      (per-frame, ~2ms)                                          │
│      │                                                          │
│      ▼                                                          │
│  vocoder.onnx                                                   │
│      pred_features → STFT mag + phase                           │
│      (per-frame, ~1ms)                                          │
│      │                                                          │
│      ▼                                                          │
│  iSTFT + Overlap-Add                                            │
│      │                                                          │
│      ▼                                                          │
│  Output Ring Buffer                                             │
│      │                                                          │
│  Upsample (24kHz → 48kHz, polyphase)                           │
│      │                                                          │
│  Dry/Wet Mix + Gain                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
DAW Audio Out (48kHz)
```

---

## 5. 主要アーキテクチャ判断と根拠

### 5.1 5 分割 ONNX モデル

| 判断 | 5つの独立した ONNX モデルとしてエクスポート |
|---|---|
| **根拠** | 各モデルの実行頻度が異なるため、個別に管理することで計算量を最小化できる |

実行頻度の違い:

| モデル | 実行頻度 | 理由 |
|---|---|---|
| speaker_encoder | Offline (enrollment 時のみ) | 話者登録は一度だけ。推論時はキャッシュ参照 |
| ir_estimator | ~100ms ごと (every ~10 frames) | 室内音響は緩やかに変化。amortized 実行で十分 |
| content_encoder | Per-frame (10ms ごと) | 音声内容はフレームごとに変化 |
| converter | Per-frame (10ms ごと) | content→target features 変換 |
| vocoder | Per-frame (10ms ごと) | features→waveform 変換 |

モノリシックモデルでは、実行頻度の異なる処理を分離できず無駄な計算が発生する。

### 5.2 tmrvc-engine は JUCE 非依存

| 判断 | StreamingEngine を JUCE に一切依存しない C++ ライブラリとして設計 |
|---|---|
| **根拠** | テスタビリティ、再利用性、ライセンス分離 |

- **テスタビリティ**: DAW なしで単体テスト・ベンチマーク可能
- **再利用性**: VST3 以外 (standalone CLI, mobile, WebAssembly) にも展開可能
- **ライセンス分離**: JUCE (GPLv3 / commercial) の影響をエンジンコアに波及させない
- **ビルド簡素化**: JUCE の複雑なビルドシステムをエンジンから切り離せる

### 5.3 ONNX Runtime 静的リンク (ort-builder, C API only)

| 判断 | ONNX Runtime を ort-builder で必要な EP のみ含む静的ライブラリとしてビルドし、C API のみ使用 |
|---|---|
| **根拠** | バイナリサイズ削減、配布簡素化、ABI 安定性 |

- **ort-builder**: CPU EP のみを含む最小構成でビルド → バイナリサイズ 5-10MB (フル版 50MB+ から大幅削減)
- **C API only**: C++ API は header-only だが ABI 不安定。C API はバージョン間で安定
- **静的リンク**: DLL 配布不要、ユーザー環境の依存関係問題を排除
- **OrtValue direct creation**: `CreateTensorWithDataAsOrtValue` で zero-copy inference

### 5.4 Frame-by-frame causal streaming (chunk-based ではなく)

| 判断 | 1 hop (10ms) 単位の frame-by-frame 処理。50ms chunk-based ではない |
|---|---|
| **根拠** | レイテンシ最小化、パイプラインの柔軟性 |

- **chunk-based** (旧設計 system_design.md): 50ms 分 (5 frames) を溜めて一括処理 → algorithmic latency = 50ms
- **frame-by-frame** (本設計): 10ms 分 (1 frame) が溜まるたびに即座に処理 → algorithmic latency = 10ms
- 10ms の処理時間で推論が完了すれば、DAW buffer latency + 10ms で出力可能
- 先行研究 LLVC は frame-by-frame で <20ms@16kHz を実証済み

### 5.5 iSTFT-based vocoder (Vocos 系)

| 判断 | WaveForm domain ではなく STFT domain で予測し iSTFT で波形復元 |
|---|---|
| **根拠** | フレーム単位処理との親和性、計算量削減 |

- **従来の waveform vocoder** (HiFi-GAN): hop_length 分のサンプルを直接生成 → 計算量大
- **iSTFT vocoder** (Vocos): STFT magnitude + phase を予測 → iSTFT で波形復元
  - フレーム単位の予測と自然に整合
  - overlap-add でフレーム間の連続性を保証
  - 計算量は waveform vocoder の 1/3-1/5
  - Vocos は HiFi-GAN と同等品質を報告 (Kim et al., 2024)

### 5.6 Speaker enrollment = embed + LoRA delta (事前計算、推論時はキャッシュ参照)

| 判断 | 話者登録時に spk_embed と LoRA delta を事前計算し `.tmrvc_speaker` ファイルに保存。推論時はロードのみ |
|---|---|
| **根拠** | 推論時の計算量ゼロ、Hot-swap 可能 |

- speaker_encoder (ECAPA-TDNN, ~5-10M params) は推論パイプラインから完全に除外
- `.tmrvc_speaker` ファイル (~100-500KB) にはバイナリ形式で格納:
  - `spk_embed[192]` (float32)
  - `lora_delta` (LoRA weights for cross-attn K/V)
  - metadata JSON: `voice_source_preset[8]` (optional) — 声質パラメータのプリセット値
- Worker thread でロード → double-buffered slot に書き込み → atomic swap
- Audio thread は常にキャッシュされた embed/LoRA/voice_source_preset を参照するだけ

### 5.7 内部サンプルレート: 24kHz

| 判断 | モデル内部は 24kHz で処理。DAW の 44.1/48kHz とはリサンプルで接続 |
|---|---|
| **根拠** | 計算量削減、先行研究との整合、品質とのトレードオフ |

- 24kHz で 12kHz まで表現可能 (音声帯域には十分)
- 48kHz 処理と比べて hop あたりのサンプル数が半分 → STFT/iSTFT の計算量半減
- 48kHz↔24kHz は整数比 (2:1) のため polyphase resampler が効率的
- LLVC, StreamVC 等の先行研究も 16-24kHz 内部処理を採用

---

## 6. 設計資料間のクロスリファレンス

| 資料 | ファイル | 主な関連 |
|---|---|---|
| 本資料 (アーキテクチャ) | `docs/design/architecture.md` | 全体の統合ビュー |
| ストリーミング設計 | `docs/design/streaming-design.md` | §4.2 推論時フローの詳細化 |
| ONNX I/O 仕様 | `docs/design/onnx-contract.md` | §5.1 の 5 モデル I/O 定義、`.tmrvc_speaker` metadata |
| モデルアーキテクチャ | `docs/design/model-architecture.md` | §4.1 学習時フローの詳細化 |
| C++ エンジン設計 | `docs/design/cpp-engine-design.md` | §5.2-5.3 の実装詳細 |
| Acoustic Condition Pathway | `docs/design/acoustic-condition-pathway.md` | IR + Voice Source 統合条件付け、プリセットブレンド |
| GUI 設計 | `docs/design/gui-design.md` | Research Studio GUI アプリケーション設計 |
| Teacher 学習計画 | `docs/design/training-plan.md` | §4.1 学習時フローのコーパス・スケジュール詳細 |
| 先行研究・コンセプト | `docs/reference/concept.md` | IR-aware 設計の根拠 |
| 参考: 旧システム設計 | `docs/reference/system_design.md` | 品質目標・蒸留手法・データセット計画の参考 |

---

## 7. 追加設計判断 (2026-02-17)

### 7.1 Latency-Quality Spectrum を正式採用

| 判断 | 推論品質とレイテンシを単一ノブで連続可変にする (`q in [0,1]`) |
|---|---|
| **根拠** | 用途ごとに最適点が異なるため。ライブ用途は低遅延、制作用途はイントネーション/活舌の品質を優先する。 |

採用方針:

- `q` に応じて `lookahead_hops`, `F0 window`, `IR update interval`, `model profile` を連動
- レイテンシ報告は `20ms + lookahead_hops * 10ms`
- プロファイル遷移は 100ms クロスフェードで無破綻切替
- 過負荷時は adaptive に `q` を自動降格

参照:

- `docs/design/streaming-design.md` の `## 12. Latency-Quality Spectrum Control`
- `docs/design/gui-design.md` の `## 12. Latency-Quality Spectrum UI`