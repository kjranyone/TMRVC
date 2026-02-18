# TMRVC GUI 画面デザイン仕様

Kojiro Tanaka — GUI Design Specification
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-18

> **対象アプリケーション:**
> - **TMRVC Research Studio** (PySide6, 研究用 7 ページ構成)
> - **TMRVC Realtime VC** (egui/Rust, エンドユーザ向けシングルページ)

---

## 1. デザインシステム

### 1.1 カラーパレット (Catppuccin Mocha ベース)

| Role | Token | Hex | 用途 |
|------|-------|-----|------|
| **Base** | `--bg-primary` | `#1e1e2e` | メイン背景 |
| **Surface 0** | `--bg-sidebar` | `#181825` | サイドバー / ヘッダ / テーブルヘッダ |
| **Surface 1** | `--bg-card` | `#313244` | カード / 入力フィールド / ボタン |
| **Surface 2** | `--bg-hover` | `#45475a` | ホバー / ボーダー |
| **Overlay** | `--bg-overlay` | `#585b70` | 押下状態 / disabled テキスト |
| **Text** | `--fg-text` | `#cdd6f4` | 本文テキスト |
| **Subtext** | `--fg-sub` | `#a6adc8` | 補助テキスト / ラベル |
| **Accent** | `--accent` | `#89b4fa` | プライマリボタン / 選択 / アクセント |
| **Accent Hover** | `--accent-hover` | `#74c7ec` | アクセントホバー |
| **Success** | `--success` | `#a6e3a1` | 成功 / OK / PASS |
| **Warning** | `--warning` | `#f9e2af` | 警告 |
| **Error** | `--error` | `#f38ba8` | エラー / FAIL / 要注意 |
| **Terminal** | `--bg-terminal` | `#11111b` | ログビューア / ターミナル |

### 1.2 タイポグラフィ

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Sidebar item | System (Segoe UI / Noto Sans) | 13px | Regular |
| GroupBox title | System | 13px | Bold, color `#89b4fa` |
| Body text / labels | System | 12px | Regular |
| Button text | System | 12px | Regular (Primary: Bold) |
| Log / Terminal | Consolas / Courier New | 11px | Regular |
| Status bar | System | 11px | Regular |

### 1.3 スペーシング & サイズ

| Element | Value |
|---------|-------|
| Page content spacing | 8px (QVBoxLayout) |
| GroupBox internal padding | 16px top, 12px sides/bottom |
| GroupBox margin-top | 12px |
| GroupBox border-radius | 6px |
| Button padding | 6px vertical, 16px horizontal |
| Button border-radius | 4px |
| Input padding | 4px vertical, 8px horizontal |
| Input border-radius | 4px |
| Progress bar height | 18px |
| Progress bar border-radius | 3px |
| Slider groove height | 6px |
| Slider handle | 14×14px, radius 7px |
| Scrollbar width | 10px |

### 1.4 レベルメーター配色

| Range | Color | Hex | Meaning |
|-------|-------|-----|---------|
| < -20 dB | Green | `#2ecc71` | 正常 |
| -20 ~ -6 dB | Yellow | `#f1c40f` | 注意 |
| > -6 dB | Red | `#e74c3c` | クリッピング |

### 1.5 メトリクスプロットパレット (pyqtgraph)

| Index | Color | Hex | 代表用途 |
|-------|-------|-----|----------|
| 0 | Blue | `#1f77b4` | Loss |
| 1 | Orange | `#ff7f0e` | SECS |
| 2 | Green | `#2ca02c` | UTMOS |
| 3 | Red | `#d62728` | Distill loss |
| 4 | Purple | `#9467bd` | LR schedule |

---

## 2. アプリケーション構造

### 2.1 Research Studio (PySide6) — 全体レイアウト

```
┌─────────────────────────────────────────────────────────────────────┐
│  TMRVC Research Studio                                       [─][□][×]│
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                      │
│  Sidebar     │  Page Content (QStackedWidget)                       │
│  180px fixed │  spacing 8px                                         │
│  #181825     │  #1e1e2e background                                  │
│              │                                                      │
│  ┌─────────┐ │  ┌──────────────────────────────────────────────┐    │
│  │  Data   │ │  │  GroupBox "Section Title"           #89b4fa  │    │
│  │  Prep   │ │  │  border: 1px solid #313244                   │    │
│  ├─────────┤ │  │  ┌────────────────────────────────────────┐  │    │
│  │ Teacher │ │  │  │  Widgets...                            │  │    │
│  │ Train   │ │  │  └────────────────────────────────────────┘  │    │
│  ├─────────┤ │  └──────────────────────────────────────────────┘    │
│  │ Distill │ │                                                      │
│  ├─────────┤ │  ┌──────────────────────────────────────────────┐    │
│  │  Eval   │ │  │  GroupBox "Section 2"                        │    │
│  ├─────────┤ │  │  ...                                         │    │
│  │ Speaker │ │  └──────────────────────────────────────────────┘    │
│  │ Enroll  │ │                                                      │
│  ├─────────┤ │                                                      │
│  │Realtime │ │                                                      │
│  │ Demo    │ │                                                      │
│  ├─────────┤ │                                                      │
│  │  ONNX   │ │                                                      │
│  │ Export  │ │                                                      │
│  └─────────┘ │                                                      │
│              │                                                      │
├──────────────┴──────────────────────────────────────────────────────┤
│  Ready                            │ Model: (none) │ Latency: --    │
└─────────────────────────────────────────────────────────────────────┘
```

**ウィンドウ:** 1280×800 デフォルト, リサイズ可能
**サイドバー:** `QListWidget#sidebar`, 固定幅 180px

- 選択行: 左 3px `#89b4fa` ボーダー, 背景 `#313244`
- ホバー: 背景 `#28283d`
- テキスト: 通常 `#a6adc8`, 選択 `#cdd6f4`

**ステータスバー:** `QStatusBar`, 背景 `#181825`

- 左: ステータスメッセージ (stretch)
- 右固定: "Model: ..." / "Latency: ..."

---

### 2.2 Realtime VC (egui/Rust) — 全体レイアウト

```
┌─────────────────────────────────────────────────────────────────┐
│  TMRVC Realtime VC                                       [─][□][×]│
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─ Audio Device ─────────────────────────────────────────────────┐│
│  │  Input:  [Microphone (Realtek)     ▼]                          ││
│  │  Output: [Speakers (Realtek)       ▼]                          ││
│  │  Buffer: [256 ▼]                                               ││
│  └────────────────────────────────────────────────────────────────┘│
│  ┌─ Models ───────────────────────────────────────────────────────┐│
│  │  ONNX Dir:     [models/fp32                    ] [Browse]      ││
│  │  Speaker File: [Speaker_A.tmrvc_speaker        ] [Browse]      ││
│  └────────────────────────────────────────────────────────────────┘│
│  ┌─ ボイスプロファイル作成 ───────────────────────────────────────┐│
│  │  音声ファイル: [ref1.wav, ref2.wav     ] [Add WAV] [Clear]    ││
│  │  保存先:       [SpeakerB.tmrvc_speaker ] [Save As]            ││
│  │  Mode: (●) Embedding only  ( ) Fine-tune (LoRA)               ││
│  │  [Create Profile]   ◌ 完了: SpeakerB.tmrvc_speaker            ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  [ Start VC ]  [ Stop ]                                             │
│                                                                     │
│  ┌─ Monitor ──────────────────────────────────────────────────────┐│
│  │  Input:  [████████████░░░░░░░░] -12.3 dB                      ││
│  │  Output: [██████████░░░░░░░░░░] -18.5 dB                      ││
│  │  Inference: 3.2 ms  (P50: 2.8 ms  P95: 4.1 ms)               ││
│  │  Latency: ~20 ms (Live mode)  |  Frames: 1234  Overruns: 0   ││
│  └────────────────────────────────────────────────────────────────┘│
│  ┌─ Controls ─────────────────────────────────────────────────────┐│
│  │  Dry/Wet:     [═══════════●═══] 0.85                           ││
│  │  Output Gain: [═════●═════════] -3.0 dB                        ││
│  │  Quality (q): [●═════════════] 0.00 (Live 20ms)                ││
│  └────────────────────────────────────────────────────────────────┘│
│─────────────────────────────────────────────────────────────────────│
│  Status: Running                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**更新頻度:** Running 時 / Profile 作成時は ~30fps (`request_repaint_after(33ms)`)

---

## 3. 画面詳細 — Research Studio (PySide6)

---

### 3.1 Page 1: Data Prep

**目的:** コーパス管理と前処理パイプライン実行

```
┌─ Corpus ────────────────────────────────────────────────────────────┐
│ ┌───────────────┬──────────┬────────┬─────────────────┬──────────┐ │
│ │ Dataset       │ Speakers │ Hours  │ Path            │ Status   │ │
│ ├───────────────┼──────────┼────────┼─────────────────┼──────────┤ │
│ │ VCTK          │ 110      │ 44h    │ /data/vctk      │ ● Ready  │ │
│ │ JVS           │ 100      │ 30h    │ /data/jvs       │ ● Ready  │ │
│ │ LibriTTS-R    │ 2456     │ 585h   │ /data/libritts   │ ○ --     │ │
│ │ Emilia        │ ~50K     │ 101Kh  │ /data/emilia     │ ○ --     │ │
│ └───────────────┴──────────┴────────┴─────────────────┴──────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─ Preprocessing Pipeline ────────────────────────────────────────────┐
│                                                                      │
│  Steps:  [✓] Resample 24kHz   [✓] Normalize   [✓] VAD Trim         │
│          [✓] Segment           [✓] Extract Features                 │
│                                                                      │
│  Cache Dir:  [/data/cache/features              ] [Browse...]        │
│  Workers:    [4  ▲▼]                                                 │
│                                                                      │
│  [▶ Run Preprocessing]  [Cancel]                                     │
│                                                                      │
│  [████████████████████████░░░░░░░░]  1234 / 5678 utterances (21.7%) │
│                                                                      │
│  ┌─ Log ──────────────────────────────────────────────────────────┐ │
│  │ 12:34:56.789  Processing VCTK corpus...                        │ │
│  │ 12:34:57.012  Utterance 100/5678                               │ │
│  │ 12:35:01.234  Utterance 200/5678                               │ │
│  │ ...                                                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**ウィジェット仕様:**

| Widget | Type | 設定 |
|--------|------|------|
| Corpus table | `QTableWidget` | 4行×5列, 読取専用, ヘッダ `#181825` |
| Step checkboxes | `QCheckBox` × 5 | 全初期チェック済 |
| Cache dir | `QLineEdit` + `QPushButton` | ディレクトリ選択ダイアログ |
| Workers | `QSpinBox` | range 1-64, default 4 |
| Run button | `QPushButton[primary]` | `#89b4fa` 背景, bold, `#1e1e2e` テキスト |
| Cancel button | `QPushButton` | 実行中のみ有効 |
| Progress bar | `QProgressBar` | 0-total, chunk `#89b4fa` |
| Log viewer | `LogViewer` | Consolas 11px, `#11111b` 背景, max 5000行, auto-scroll |

**Worker:** `DataWorker(BaseWorker)` — corpus_paths, steps, n_workers, cache_dir

---

### 3.2 Page 2: Teacher Training

**目的:** Teacher U-Net (Diffusion) の学習制御とメトリクス監視

```
┌─ Training Configuration ────────────────────────────────────────────┐
│                                                                      │
│  Phase:      [Phase 0: VCTK+JVS baseline               ▼]          │
│  Datasets:   [✓] VCTK  [✓] JVS  [ ] LibriTTS-R  [ ] Emilia        │
│                                                                      │
│  Batch Size: [16  ▲▼]    LR: [1.0e-4  ▲▼]    Steps: [50000 ▲▼]    │
│                                                                      │
│  Cache Dir:       [/data/cache/features              ] [Browse...]   │
│  Checkpoint Dir:  [checkpoints/teacher               ] [Browse...]   │
│  Resume From:     [(none)                            ] [Browse...]   │
│                                                                      │
│  Execution:  (●) Local   ( ) SSH Remote                              │
│  ┌─ SSH Config (disabled) ──────────────────────────────────────┐   │
│  │  Host: [            ]  User: [         ]  Key: [    ][Browse]│   │
│  │  Remote Dir: [                                    ]          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  [▶ Start]  [⏸ Pause]  [■ Stop]                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Metrics (2×2 grid) ────────────────────────────────────────────────┐
│                                                                      │
│  ┌──────── Loss ─────────┐  ┌──────── SECS ─────────┐              │
│  │  (MetricPlot #1f77b4) │  │  (MetricPlot #ff7f0e) │              │
│  │  x: Step, y: Loss     │  │  x: Step, y: SECS     │              │
│  │  height ≥ 150px       │  │  height ≥ 150px       │              │
│  └───────────────────────┘  └───────────────────────┘              │
│  ┌──────── UTMOS ────────┐  ┌──── LR Schedule ──────┐              │
│  │  (MetricPlot #2ca02c) │  │  (MetricPlot #9467bd) │              │
│  │  x: Step, y: UTMOS    │  │  x: Step, y: LR       │              │
│  └───────────────────────┘  └───────────────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Log ────────────────────────────────────────────────────────────────┐
│ 12:00:00.000  Teacher params: 80,234,567                             │
│ 12:00:01.234  Step 100/50000  loss=0.4523  secs=0.712                │
│ ...                                                                  │
└──────────────────────────────────────────────────────────────────────┘
```

**Phase 選択肢:** `QComboBox`

| Value | Label |
|-------|-------|
| `"0"` | Phase 0: Validation |
| `"1a"` | Phase 1a: VCTK+JVS |
| `"1b"` | Phase 1b: +LibriTTS-R |
| `"2"` | Phase 2: +Emilia |

**Worker:** `TrainWorker(BaseWorker)` — `TeacherTrainer.train_iter()` generator 使用

---

### 3.3 Page 3: Distillation

**目的:** Teacher → Student 蒸留の実行と A/B 試聴

```
┌─ Distillation Configuration ────────────────────────────────────────┐
│                                                                      │
│  Teacher Checkpoint:  [teacher_phase1_step400K.pt       ▼]          │
│  Phase:  (●) A: ODE Trajectory    ( ) B: DMD                       │
│  Student:  [CausalCNN-7.7M                               ▼]        │
│  Batch Size: [8  ▲▼]    Steps: [100000  ▲▼]                        │
│                                                                      │
│  [▶ Start Distillation]                                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Metrics ────────────────────────────────────────────────────────────┐
│  ┌──── Distillation Loss ───────┐  ┌──── Feature MSE ──────────┐   │
│  │  (MetricPlot, 1:1 side)      │  │  (MetricPlot, 1:1 side)   │   │
│  │  x: Step, y: Loss            │  │  x: Step, y: MSE          │   │
│  └──────────────────────────────┘  └────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

┌─ A/B Listening ──────────────────────────────────────────────────────┐
│                                                                      │
│  [▶ Source]        [▶ Teacher]        [▶ Student]                    │
│                                                                      │
│  ┌── Source ──────┐ ┌── Teacher ─────┐ ┌── Student ─────┐          │
│  │ (WaveformView) │ │ (WaveformView) │ │ (WaveformView) │          │
│  │ pen: #4fc3f7   │ │ pen: #ff7f0e   │ │ pen: #a6e3a1   │          │
│  │ 80px height    │ │ 80px height    │ │ 80px height    │          │
│  └────────────────┘ └────────────────┘ └────────────────┘          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 Page 4: Evaluation

**目的:** 客観メトリクス比較と A/B ブラインドテスト

```
┌─ Evaluation Setup ──────────────────────────────────────────────────┐
│                                                                      │
│  Model A:  [student_v1_fp32                               ▼]       │
│  Model B:  [student_v1_int8                               ▼]       │
│  Eval Set: [VCTK-test                                     ▼]       │
│                                                                      │
│  [▶ Run Evaluation]                                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Objective Metrics ─────────────────────────────────────────────────┐
│  ┌──────────────┬──────────┬──────────┬──────────┐                  │
│  │ Metric       │ Model A  │ Model B  │ Target   │                  │
│  ├──────────────┼──────────┼──────────┼──────────┤                  │
│  │ SECS ↑       │  0.812   │  0.798   │ > 0.80   │                  │
│  │ UTMOS ↑      │  3.72    │  3.68    │ > 3.5    │                  │
│  │ MCD (dB) ↓   │  5.23    │  5.41    │ < 6.0    │                  │
│  │ F0 RMSE ↓    │  12.3    │  13.1    │ < 15.0   │                  │
│  │ Latency (ms) │  2.8     │  1.5     │ < 10.0   │                  │
│  └──────────────┴──────────┴──────────┴──────────┘                  │
│                                                                      │
│  ↑ = higher is better, ↓ = lower is better                          │
│  セル背景: Target 達成 → #a6e3a1 10% , 未達 → #f38ba8 10%          │
└──────────────────────────────────────────────────────────────────────┘

┌─ A/B Blind Listening Test ──────────────────────────────────────────┐
│                                                                      │
│  Sample 3 / 20                                                       │
│                                                                      │
│  [▶ Source]     [▶ A]     [▶ B]                                     │
│                                                                      │
│  [Prefer A]    [Same]    [Prefer B]                                  │
│                                                                      │
│  Results:  A = 5   Same = 1   B = 4                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Worker:** `EvalWorker(BaseWorker)` — `result(dict)` signal で結果テーブルを更新

---

### 3.5 Page 5: Speaker Enrollment

**目的:** 話者プロファイル生成 (embedding / fine-tune) + クイックテスト

```
┌─ Reference Audio ───────────────────────────────────────────────────┐
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                                                          │       │
│  │    Drop .wav / .flac / .mp3 files here                   │       │
│  │                                                          │       │
│  │    (FileDropArea — dashed #666 border)                   │       │
│  │    Hover: dashed #4fc3f7 border                          │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌──────────────────────────────────────┐  [Browse...]               │
│  │ ref1.wav                             │  [Remove Selected]         │
│  │ ref2.wav                             │                            │
│  │ ref3.wav                             │                            │
│  └──────────────────────────────────────┘  (QListWidget, multi-sel)  │
│                                                                      │
│  Speaker name: [Speaker_A          ]  [▶ Generate .tmrvc_speaker]   │
│  Status: Speaker file saved: Speaker_A.tmrvc_speaker                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Fine-tune (Optional) ──────────────────────────────────────────────┐
│                                                                      │
│  Student Checkpoint: [distill.pt                    ] [Browse...]    │
│                                                                      │
│  Steps: [200  ▲▼]    LR: [0.00100  ▲▼]    [✓] Use GTM              │
│                                                                      │
│  [▶ Fine-tune & Save .tmrvc_speaker]  [Cancel]                      │
│                                                                      │
│  [████████████████░░░░░░░░░░░░░░░░]                                 │
│  Step 120/200  loss=0.0234                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Quick Conversion Test ─────────────────────────────────────────────┐
│                                                                      │
│  [Record]  [Browse Test File...]  [▶ Play Result]                   │
│                                                                      │
│  ┌─ Waveform ──────────────────────────────────────────────────┐    │
│  │  (WaveformView — pyqtgraph, pen #4fc3f7, rolling 5s)       │    │
│  │  Y: amplitude ±1.0, X: time (seconds)                       │    │
│  │  min-height: 80px                                            │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**`.tmrvc_speaker` v2:** メタデータ (display_name, created_at, training_mode 等) が自動付与される。
`_SpeakerGenerateWorker` / `FinetuneWorker` それぞれが metadata dict を構築して `write_speaker_file()` に渡す。

---

### 3.6 Page 6: Realtime Demo

**目的:** リアルタイム Voice Conversion の統合デモ

```
┌─ Audio Configuration ───────────────────────────────────────────────┐
│                                                                      │
│  Input Device:  [Microphone (Realtek Audio)                    ▼]   │
│  Output Device: [Speakers (Realtek Audio)                      ▼]   │
│  Buffer Size:   [256 ▼]  (64 | 128 | 256 | 512 | 1024)             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Model Configuration ──────────────────────────────────────────────┐
│                                                                      │
│  ONNX Model Dir:   [models/fp32                          ▼] [Browse]│
│  Speaker File:     [Speaker_A.tmrvc_speaker              ▼] [Browse]│
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

[▶ Start]  [■ Stop]

┌─ Monitoring ────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌─ Levels ─────────────────────────────────────────────┐           │
│  │  Input:   [████████████░░░░░░░░] -12.3 dB            │           │
│  │  Output:  [██████████░░░░░░░░░░] -18.5 dB            │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                      │
│  Inference: 3.2 ms    Total Latency: ~23.2 ms    Buffer: OK        │
│                                                                      │
│  Dry/Wet:     [═══════════════●═══]  85%                            │
│  Output Gain: [═══════●══════════]  -3.0 dB                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Waveform Display ──────────────────────────────────────────────────┐
│                                                                      │
│  ┌─ Input (WaveformView, pen #4fc3f7, 5s rolling) ──────────────┐  │
│  │  ~~\/\/~~\/~~\/\/~~\/~~\/~~\/~~\/~~\/~~\/~~\/~~\/~~\/~~\     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌─ Output (WaveformView, pen #a6e3a1, 5s rolling) ─────────────┐  │
│  │  ~~\/~~\/\/~~\/\/~~\/~~\/~~\/~~\/\/~~\/\/~~\/~~\/~~\/~~\     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**レイテンシ計算式:**

```
Total Latency = (buffer_size / sample_rate × 1000) + inference_ms + 10ms
```

**Buffer Status:** `OK` → `#a6e3a1`、`UNDERRUN` → `#f38ba8`

**AudioEngine** は `AudioEngine(QThread)`:
- `RingBuffer` (SPSC, capacity 4096) × 2 (input/output)
- `PingPongState` per model (double-buffered state tensors)
- Causal STFT → Mel → F0 → 4 ONNX models → iSTFT OLA
- Dry/Wet mix + output gain
- Signals: `level_updated`, `timing_updated`, `buffer_status`

---

### 3.7 Page 7: ONNX Export

**目的:** ONNX エクスポート、INT8 量子化、パリティ検証、ベンチマーク

```
┌─ Export Configuration ──────────────────────────────────────────────┐
│                                                                      │
│  Checkpoint:  [distill_final.pt                            ▼]       │
│  Output Dir:  [models/                                 ] [Browse...] │
│                                                                      │
│  Models:                                                             │
│    [✓] content_encoder  [✓] converter  [✓] ir_estimator             │
│    [✓] vocoder          [✓] speaker_encoder                         │
│                                                                      │
│  [▶ Export FP32]    [▶ Export + Quantize INT8]                       │
│                                                                      │
│  [████████████████████████████████████]  5 / 5 models               │
│  Status: Export complete.                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Parity Verification ──────────────────────────────────────────────┐
│  ┌──────────────────┬────────────┬────────────┬──────────┐          │
│  │ Model            │ Max Abs Err│ Mean Abs   │ Status   │          │
│  ├──────────────────┼────────────┼────────────┼──────────┤          │
│  │ content_encoder  │  2.4e-06   │  8.1e-07   │ ● PASS   │          │
│  │ converter        │  5.1e-06   │  1.2e-06   │ ● PASS   │          │
│  │ ir_estimator     │  1.8e-06   │  5.3e-07   │ ● PASS   │          │
│  │ vocoder          │  3.3e-06   │  9.8e-07   │ ● PASS   │          │
│  │ speaker_encoder  │  --        │  --        │ ○ --     │          │
│  └──────────────────┴────────────┴────────────┴──────────┘          │
│  [▶ Run Parity Check]                                                │
└──────────────────────────────────────────────────────────────────────┘

┌─ Inference Benchmark ───────────────────────────────────────────────┐
│  ┌──────────────────┬──────────────┬──────────────┐                 │
│  │ Model            │ FP32 (ms)    │ INT8 (ms)    │                 │
│  ├──────────────────┼──────────────┼──────────────┤                 │
│  │ content_encoder  │  0.42        │  0.18        │                 │
│  │ converter        │  1.23        │  0.51        │                 │
│  │ ir_estimator     │  0.38        │  0.15        │                 │
│  │ vocoder          │  0.31        │  0.14        │                 │
│  │ Total per-frame  │  1.96        │  0.83        │                 │
│  └──────────────────┴──────────────┴──────────────┘                 │
│  [▶ Run Benchmark]                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Parity Status セル:** `● PASS` → `#a6e3a1`、`● FAIL` → `#f38ba8`、`○ --` → `#585b70`
**Worker:** `ExportWorker(BaseWorker)` — FP32 export → optional INT8 quantization

---

## 4. 画面詳細 — Realtime VC (egui/Rust)

### 4.1 パネル構成

egui は Immediate Mode GUI。上から下へパネルを順に描画する。

| # | Panel | Module | 内容 |
|---|-------|--------|------|
| 1 | Audio Device | `device_panel.rs` | Input/Output ComboBox + Buffer Size |
| 2 | Models | `model_panel.rs` | ONNX dir + Speaker file + auto-detect |
| 3 | Voice Profile | `voice_profile_panel.rs` | WAV 追加、プロファイル作成 |
| 4 | Start/Stop | `app.rs` | VC 開始/停止ボタン |
| 5 | Monitor | `monitor.rs` | レベルメーター、推論時間、フレーム統計 |
| 6 | Controls | `controls.rs` | Dry/Wet, Gain, Quality (q) |
| 7 | Status | `app.rs` | ステータステキスト |

### 4.2 Monitor パネル

```
┌─ Monitor ──────────────────────────────────────────────────┐
│                                                            │
│  Input:     [████████████░░░░░░░░░░░░] -12.3 dB           │
│  Output:    [████████░░░░░░░░░░░░░░░░] -18.5 dB           │
│                                                            │
│  Inference    Current: 3.2 ms                              │
│               P50:     2.8 ms                              │
│               P95:     4.1 ms                              │
│                                                            │
│  Estimated Latency:  ~20 ms (Live mode, q=0.00)           │
│                                                            │
│  Frames: 12345   Overruns: 0   Underruns: 0               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**レベルバー描画:** `draw_level_bar(ui, db)` — 幅比率 `(db + 60) / 60`、色は §1.4 に準拠
**レイテンシ表示:** q ≤ 0.3 → "Live mode ~20ms"、q > 0.3 → "HQ mode ~80ms"
**P50/P95:** `EvalLogger` で 512 フレームウィンドウのパーセンタイル計算

### 4.3 Controls パネル

| Slider | Range | Default | Step | Format |
|--------|-------|---------|------|--------|
| Dry/Wet | 0.0 – 1.0 | 1.0 | 0.01 | `{:.2}` |
| Output Gain | -60 – +12 dB | 0.0 | 0.5 | `{:.1} dB` |
| Quality (q) | 0.0 – 1.0 | 0.0 | 0.01 | `{:.2}` + モード表示 |

値は `Arc<AtomicF32>` で Processor Thread と共有。UI 変更は即座に反映。

### 4.4 Voice Profile パネル

```
┌─ ボイスプロファイル作成 ──────────────────────────────────────┐
│                                                              │
│  音声ファイル:                                                │
│    ref1.wav                                                  │
│    ref2.wav                                                  │
│  [Add WAV]  [Clear]                                          │
│                                                              │
│  保存先: [SpeakerB.tmrvc_speaker        ] [Save As...]       │
│                                                              │
│  Mode: (●) Embedding only                                    │
│        ( ) Fine-tune (LoRA)  (Python 利用可能時のみ表示)     │
│                                                              │
│  Checkpoint: [distill.pt              ] [Browse...]           │
│  (Fine-tune 選択時のみ表示)                                   │
│                                                              │
│  [Create Profile]                                            │
│                                                              │
│  ◌ エンコーダ推論中...                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**状態遷移:**

```
Idle ──[Create Profile]──▶ Creating (spinner + progress_message)
Creating ──[Done]──▶ Idle + "完了: path" + auto-load speaker
Creating ──[Error]──▶ Idle + "エラー: msg"
```

Embedding mode: Rust ネイティブ (`SpeakerEncoderSession` + `create_voice_profile()`)
Fine-tune mode: Python CLI (`tmrvc-finetune`) を subprocess で呼び出し

---

## 5. データフロー

### 5.1 学習パイプライン (Research Studio)

```
               ┌─────────┐
               │ Raw Audio│
               └────┬─────┘
                    │
         ┌──────────▼──────────┐
  Page 1 │    Data Prep        │  前処理 → Feature Cache
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
  Page 2 │  Teacher Training   │  Teacher U-Net 学習
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
  Page 3 │   Distillation      │  Teacher → Student 蒸留
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
  Page 7 │    ONNX Export      │  .pt → .onnx (FP32/INT8)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
  Page 5 │  Speaker Enrollment │  .tmrvc_speaker v2 生成
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
  Page 6 │   Realtime Demo     │  リアルタイム VC 実行
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
  Page 4 │    Evaluation       │  品質評価 (SECS, UTMOS, A/B)
         └────────────────────┘
```

### 5.2 Worker ライフサイクル (全 Worker 共通パターン)

```
┌─ UI Thread ──────────────┐     ┌─ Worker Thread ────────────────┐
│                          │     │                                 │
│  [Run] clicked           │     │                                 │
│    ├─ Disable [Run]      │     │                                 │
│    ├─ Enable [Cancel]    │     │                                 │
│    ├─ Show ProgressBar   │     │                                 │
│    └─ worker.start() ────┼────▶│  run()                         │
│                          │     │    ├─ Emit progress(cur, tot)   │
│  ◄── progress(c, t) ────┼─────┼────┤                            │
│    └─ ProgressBar update │     │    ├─ Check is_cancelled        │
│                          │     │    ├─ Emit log_message(text)    │
│  ◄── log_message(txt) ──┼─────┼────┤                            │
│    └─ LogViewer.append() │     │    ├─ Emit metric(name,val,step)│
│                          │     │    │                            │
│  ◄── metric(...) ────────┼─────┼────┤                            │
│    └─ MetricPlot.add()   │     │    └─ Done                      │
│                          │     │                                 │
│  ◄── finished(ok, msg) ──┼─────┼─── Emit finished               │
│    ├─ Enable [Run]       │     │                                 │
│    ├─ Disable [Cancel]   │     │                                 │
│    └─ Hide ProgressBar   │     │                                 │
└──────────────────────────┘     └─────────────────────────────────┘
```

**BaseWorker signals:**
- `progress(int, int)` — current, total
- `log_message(str)` — timestamped log
- `metric(str, float, int)` — name, value, step
- `finished(bool, str)` — success, message
- `error(str)` — error message

**Cancellation:** `threading.Event` flag, Worker loop で `is_cancelled` をチェック

### 5.3 Realtime VC データフロー (Rust)

```
┌───────────────────────────────────────────────────────────────────┐
│                       egui UI Thread (~30fps)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Device   │  │ Model    │  │ Controls │  │ Monitor  │         │
│  │ Panel    │  │ Panel    │  │ Sliders  │  │ Display  │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────▲─────┘         │
└───────┼──────────────┼──────────────┼──────────────┼─────────────┘
        │         Command TX     AtomicF32     SharedStatus
        │         (channel)      (lock-free)   (atomic read)
        │              │              │              │
┌───────┼──────────────▼──────────────▼──────────────┼─────────────┐
│       │         Processor Thread                   │             │
│       │     ┌────────────────────────┐             │             │
│       │     │   StreamingEngine      │─ write ─────┘             │
│       │     │   ├─ Content Encoder   │  SharedStatus             │
│       │     │   ├─ IR Estimator      │  (inference_ms, levels,   │
│       │     │   ├─ Converter         │   frame_count, overruns)  │
│       │     │   ├─ Converter HQ      │                           │
│       │     │   └─ Vocoder           │                           │
│       │     └──────┬────────┬────────┘                           │
│       │      Input Ring    Output Ring                           │
│       │      (SPSC)        (SPSC)                               │
└───────┼────────┬───────────────┬─────────────────────────────────┘
        │        │               │
        ▼        ▼               ▼
   cpal Config   Microphone      Speakers
                (Input Stream)   (Output Stream)
```

**EvalLogger:** フレームごとに CSV 記録 (inference_ms, q, p50, p95, overruns)

---

## 6. 共有ウィジェット一覧

| Widget | File | 基底 | 主要 API |
|--------|------|------|----------|
| `FileDropArea` | `widgets/file_drop.py` | `QFrame` | `files_dropped` signal, D&D `.wav/.flac/.mp3` |
| `LogViewer` | `widgets/log_viewer.py` | `QPlainTextEdit` | `append_log(text)`, `clear_log()`, max 5000行 |
| `ProgressPanel` | `widgets/progress_panel.py` | `QWidget` | `set_progress(cur, tot)`, `set_status(text)`, `cancel_requested` signal |
| `AudioMeter` | `widgets/audio_meter.py` | `QWidget` | `set_level(db)`, -60~0 dB, 色自動切替 |
| `MetricPlot` | `widgets/metric_plot.py` | `QWidget` (pyqtgraph) | `add_point(x, y, series)`, `clear_series()`, auto-downsample |
| `WaveformView` | `widgets/waveform_view.py` | `QWidget` (pyqtgraph) | `set_data(samples)`, `append_data(samples)`, rolling window |

---

## 7. 設定永続化

### ConfigManager (`models/config_manager.py`)

ファイル: `tmrvc_gui_config.json` (プロジェクトルート)

```json
{
  "project_root": ".",
  "audio": {
    "input_device": null,
    "output_device": null,
    "buffer_size": 512
  },
  "realtime": {
    "onnx_dir": "",
    "speaker_file": "",
    "dry_wet": 1.0,
    "output_gain_db": 0.0
  },
  "training": {
    "phase": "phase0",
    "datasets": ["VCTK", "JVS"],
    "batch_size": 16,
    "lr": 1e-4,
    "total_steps": 50000,
    "mode": "local",
    "ssh_host": "", "ssh_user": "", "ssh_key": "", "ssh_remote_dir": ""
  },
  "data_prep": {
    "n_workers": 4,
    "steps": ["resample", "normalize", "vad_trim", "segment", "features"]
  }
}
```

API: `load()`, `save()`, `get(section, key, default)`, `set(section, key, value)`

---

## 8. スレッドモデル

### Research Studio (PySide6)

```
Thread 1: GUI Event Loop (Qt main thread)
  ├─ ウィジェット描画・ユーザー操作
  ├─ Signal/Slot でワーカーと通信
  └─ pyqtgraph プロット更新

Thread 2: sounddevice Callback (PortAudio, C level)
  ├─ Input: マイク → Input Ring Buffer
  └─ Output: Output Ring Buffer → スピーカー

Thread 3: Audio Processing (QThread = AudioEngine)
  ├─ Input Ring → ONNX 推論 → Output Ring
  └─ Signal で GUI にレベル・タイミング通知

Thread 4+: Background Jobs (QThread)
  ├─ DataWorker / TrainWorker / ExportWorker / EvalWorker / FinetuneWorker
  └─ 同時に 1 ジョブ実行 (リソース競合回避)
```

### Realtime VC (egui/Rust)

```
Thread 1: egui UI (main thread)
  └─ ~30fps repaint, read SharedStatus atomics

Thread 2: tmrvc-processor
  └─ Command channel → StreamingEngine.process_one_frame()

Thread 3: cpal input callback
  └─ マイク → Input SpscRingBuffer

Thread 4: cpal output callback
  └─ Output SpscRingBuffer → スピーカー

Thread 5 (optional): voice-profile-worker
  └─ SpeakerEncoderSession → create_voice_profile()
```

---

## 9. 整合性チェックリスト

- [x] 全 7 ページが `main_window.py` の `_TABS` リストと一致
- [x] Worker signals が各ページの接続コードと整合
- [x] カラーパレットが `style.qss` の実値と一致
- [x] Rust UI パネルが `app.rs` の `update()` draw 順序と一致
- [x] メトリクス名が TeacherTrainer / EvalWorker の emit と一致
- [x] `.tmrvc_speaker` v2 が enrollment / finetune フローに反映
- [x] Quality (q) スライダーが HQ/Live モード切替に連動
- [x] AudioEngine Ring Buffer 容量 (4096) が Python 版と一致
- [x] Rust SpscRingBuffer 容量が `RING_BUFFER_CAPACITY` 定数と一致
- [x] EvalLogger CSV フォーマットがテストと一致
