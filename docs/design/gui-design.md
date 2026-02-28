# TMRVC GUI 画面デザイン仕様 (Codec-Latent Pipeline)

Kojiro Tanaka — GUI Design Specification
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-28 — Codec-Latent パラダイムに一本化

> **対象アプリケーション:**
> - **TMRVC Research Studio** (PySide6, 研究用ページ構成)
> - **TMRVC Realtime VC** (egui/Rust, エンドユーザ向けシングルページ)

---

## 1. デザインシステム

### 1.1 カラーパレット (Catppuccin Mocha ベース)

| Role | Token | Hex | 用途 |
|------|-------|-----|------|
| **Base** | `--bg-primary` | `#1e1e2e` | メイン背景 |
| **Surface 0** | `--bg-sidebar` | `#181825` | サイドバー / ヘッダ |
| **Surface 1** | `--bg-card` | `#313244` | カード / 入力フィールド |
| **Surface 2** | `--bg-hover` | `#45475a` | ホバー / ボーダー |
| **Text** | `--fg-text` | `#cdd6f4` | 本文テキスト |
| **Subtext** | `--fg-sub` | `#a6adc8` | 補助テキスト |
| **Accent** | `--accent` | `#89b4fa` | プライマリボタン |
| **Success** | `--success` | `#a6e3a1` | 成功 |
| **Error** | `--error` | `#f38ba8` | エラー |
| **Terminal** | `--bg-terminal` | `#11111b` | ログビューア |

### 1.2 タイポグラフィ

| Element | Font | Size |
|---------|------|------|
| Sidebar item | System | 13px |
| Body text | System | 12px |
| Log / Terminal | Consolas | 11px |

---

## 2. Research Studio (PySide6) — ページ構成

### 2.1 全体レイアウト

```
┌─────────────────────────────────────────────────────────────────────┐
│  TMRVC Research Studio                                       [─][□][×]│
├──────────────┬──────────────────────────────────────────────────────┤
│  Sidebar     │  Page Content (QStackedWidget)                       │
│  180px       │                                                      │
│  #181825     │  ┌──────────────────────────────────────────────┐    │
│              │  │  GroupBox "Configuration"                    │    │
│  ┌─────────┐ │  │  ...                                         │    │
│  │ Data    │ │  └──────────────────────────────────────────────┘    │
│  │ Prep    │ │                                                      │
│  ├─────────┤ │                                                      │
│  │ Codec   │ │                                                      │
│  │ Train   │ │                                                      │
│  ├─────────┤ │                                                      │
│  │ Token   │ │                                                      │
│  │ Train   │ │                                                      │
│  ├─────────┤ │                                                      │
│  │ Eval    │ │                                                      │
│  ├─────────┤ │                                                      │
│  │ Speaker │ │                                                      │
│  │ Enroll  │ │                                                      │
│  ├─────────┤ │                                                      │
│  │Realtime │ │                                                      │
│  │ Demo    │ │                                                      │
│  ├─────────┤ │                                                      │
│  │ ONNX    │ │                                                      │
│  │ Export  │ │                                                      │
│  └─────────┘ │                                                      │
├──────────────┴──────────────────────────────────────────────────────┤
│  Ready                            │ Model: (none) │ Latency: --    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 ページ一覧

| Page | 用途 |
|------|------|
| **Data Prep** | コーパス管理と前処理 |
| **Codec Training** | Streaming Codec 学習 |
| **Token Model Training** | Mamba Token Model 学習 |
| **Evaluation** | 品質評価 (SECS, UTMOS 等) |
| **Speaker Enrollment** | 話者プロファイル作成 |
| **Realtime Demo** | リアルタイム VC デモ |
| **ONNX Export** | ONNX エクスポート |
| **TTS** | TTS 設定 |
| **Script** | スクリプト実行 |
| **Style Editor** | スタイル編集 |
| **TTS Server** | TTS サーバー制御 |

---

## 3. 画面詳細

### 3.1 Page: Codec Training

**目的:** Streaming Neural Audio Codec の学習制御

```
┌─ Codec Configuration ──────────────────────────────────────────────┐
│  Cache Dir:    [data/cache                        ] [Browse]       │
│  Output Dir:   [checkpoints/codec                 ] [Browse]       │
│  Batch Size:   [16  ▲▼]   LR: [3e-4 ▲▼]   Steps: [100000 ▲▼]    │
│  Device:       [cuda ▼]                                           │
└─────────────────────────────────────────────────────────────────────┘

┌─ Codec Parameters ──────────────────────────────────────────────────┐
│  Frame Size (samples): [480 ▲▼]                                   │
│  RVQ Codebooks:        [4 ▲▼]                                     │
│  Codebook Size:        [1024 ▲▼]                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─ Loss Weights ──────────────────────────────────────────────────────┐
│  λ Reconstruction: [1.0 ▲▼]   λ Adversarial: [1.0 ▲▼]             │
│  λ STFT:           [1.0 ▲▼]   λ Commitment:  [0.25 ▲▼]            │
└─────────────────────────────────────────────────────────────────────┘

┌─ Training Metrics ──────────────────────────────────────────────────┐
│  [Plot: Generator Loss]  [Plot: Discriminator Loss]  [Plot: STFT] │
└─────────────────────────────────────────────────────────────────────┘

                    [Start Training]  [Stop]

┌─ Training Log ──────────────────────────────────────────────────────┐
│  12:34:56  Step 1000: g_loss=0.1234, d_loss=0.0567, stft=0.0234   │
│  ...                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Page: Token Model Training

**目的:** Mamba Token Model の学習制御

```
┌─ Training Configuration ────────────────────────────────────────────┐
│  Codec Checkpoint: [checkpoints/codec/codec_final.pt  ] [Browse]  │
│  Cache Dir:        [data/cache                         ] [Browse]  │
│  Output Dir:       [checkpoints/token_model            ] [Browse]  │
│  Batch Size: [32 ▲▼]   LR: [1e-4 ▲▼]   Steps: [200000 ▲▼]        │
│  Device: [cuda ▼]                                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─ Model Architecture ────────────────────────────────────────────────┐
│  Model Type:          [mamba ▼]                                    │
│  Hidden Dimension:    [256 ▲▼]                                     │
│  Layers:              [6 ▲▼]                                       │
│  Context Length:      [10 ▲▼] frames (200ms)                       │
│  Sampling Temperature:[1.0 ▲▼]                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─ Training Metrics ──────────────────────────────────────────────────┐
│  [Plot: Token Loss]  [Plot: Accuracy]  [Plot: Perplexity]         │
└─────────────────────────────────────────────────────────────────────┘

                    [Start Training]  [Stop]

┌─ Training Log ──────────────────────────────────────────────────────┐
│  12:34:56  Step 1000: loss=2.3456, acc=0.1234, ppl=10.45          │
│  ...                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Realtime VC (egui/Rust)

### 4.1 全体レイアウト

```
┌─────────────────────────────────────────────────────────────────┐
│  TMRVC Realtime VC                                       [─][□][×]│
├─────────────────────────────────────────────────────────────────────┤
│  ┌─ Audio Device ─────────────────────────────────────────────────┐│
│  │  Input:  [Microphone (Realtek)     ▼]                          ││
│  │  Output: [Speakers (Realtek)       ▼]                          ││
│  │  Buffer: [256 ▼]                                               ││
│  └────────────────────────────────────────────────────────────────┘│
│  ┌─ Models ───────────────────────────────────────────────────────┐│
│  │  ONNX Dir:     [models/fp32                    ] [Browse]      ││
│  │  Speaker File: [Speaker_A.tmrvc_speaker        ] [Browse]      ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  [ Start VC ]  [ Stop ]                                             │
│                                                                     │
│  ┌─ Monitor ──────────────────────────────────────────────────────┐│
│  │  Input:  [████████████░░░░░░░░] -12.3 dB                      ││
│  │  Output: [██████████░░░░░░░░░░] -18.5 dB                      ││
│  │  Inference: 20 ms  |  Latency: ~45 ms  |  Overruns: 0         ││
│  └────────────────────────────────────────────────────────────────┘│
│  ┌─ Controls ─────────────────────────────────────────────────────┐│
│  │  Dry/Wet:     [═══════════●═══] 0.85                           ││
│  │  Output Gain: [═════●═════════] -3.0 dB                        ││
│  └────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Status: Running                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 モニター表示

| 項目 | 表示内容 |
|------|---------|
| **Input Level** | 入力音声レベルメーター (-∞ ~ 0 dB) |
| **Output Level** | 出力音声レベルメーター |
| **Inference** | 1フレームあたりの推論時間 (目標: ~20ms) |
| **Latency** | End-to-end レイテンシ (目標: ~45ms) |
| **Overruns** | 処理遅延回数 |

---

## 5. ワーカースレッド

| Worker | 用途 | 対応ページ |
|--------|------|-----------|
| `DataWorker` | 前処理実行 | Data Prep |
| `TrainWorker` | Codec/Token 学習 | Codec Train, Token Train |
| `EvalWorker` | 品質評価 | Evaluation |
| `ExportWorker` | ONNX エクスポート | ONNX Export |
| `TTSWorker` | TTS 生成 | TTS |

---

## 6. 設計整合性チェックリスト

- [ ] 全ページが Codec-Latent パラダイムに対応
- [ ] 訓練ログがリアルタイム更新 (30fps)
- [ ] ワーカーは QThread で実行 (UI ブロック回避)
- [ ] 進捗バーが正確に更新
- [ ] エラー時にステータスバーに赤色表示
- [ ] Realtime Demo のレイテンシ表示が ~45ms を反映
