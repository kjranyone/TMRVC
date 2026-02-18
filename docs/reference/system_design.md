# TMRVC System Design — IR-aware Few-shot Voice Conversion

Kojiro Tanaka — system design
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-16 — v2: 最高品質蒸留モデル路線へ全面改訂

> **Goal:** SOTA 級の高品質 Teacher モデルを構築し、
> **蒸留後も Teacher に迫る（あるいは超える）品質** を持つ Student を生成する。
> Student は CPU-only 環境で **48 kHz / 50 ms 未満** のリアルタイム推論が可能であること。

> **設計原則:** 品質に妥協しない。レイテンシ制約は Student のみに適用し、
> Teacher は品質天井を最大化することだけに集中する。蒸留プロセスで品質を保つ
> ために、最新の手法（DMDSpeech, ADCD, Shortcut FM）を組み合わせる。

---

## 0. 用語定義

| 用語 | 定義 |
|---|---|
| **Teacher** | 品質最優先の大規模モデル（GPU 学習・GPU 推論、ステップ数制限なし） |
| **Student** | Teacher から蒸留された軽量モデル（CPU リアルタイム推論） |
| **IR** | Impulse Response — 部屋の残響・マイク特性を含む音響伝達関数 |
| **RTF** | Real-Time Factor — 処理時間 / 音声時間。RTF < 1 でリアルタイム |
| **OT-CFM** | Optimal Transport Conditional Flow Matching — 最適輸送ベースのフローマッチング |
| **DiT** | Diffusion Transformer — Transformer ベースの拡散/フローモデル backbone |
| **ADCD** | Adversarial Diffusion Conversion Distillation — 変換パイプラインごと蒸留する手法 |
| **DMD** | Distribution Matching Distillation — 分布レベルで蒸留し直接指標最適化を行う手法 |
| **SECS** | Speaker Embedding Cosine Similarity — 話者類似度の客観指標 |
| **LoRA** | Low-Rank Adaptation — 少数パラメータのみ更新する効率的な適応手法 |

---

## 1. レイテンシバジェット

50 ms の内訳を厳密に設計する。**すべての処理段の合計が 50 ms を超えてはならない。**

```
                        50 ms total
├──────────┬───────────┬───────────┬──────────┤
│ Input    │ Feature   │ Inference │ Waveform │
│ Buffer   │ Extract   │ (Student) │ Synth    │
│  5 ms    │  5 ms     │  30 ms    │ 10 ms    │
└──────────┴───────────┴───────────┴──────────┘
```

| 処理段 | 上限 | 内容 |
|---|---|---|
| **Input Buffer** | 5 ms | OS オーディオコールバック + ring buffer 読み出し |
| **Feature Extraction** | 5 ms | F0 推定 + Content embedding lookup（キャッシュ活用） |
| **Student Inference** | 30 ms | 本体の NN 推論。2,400 サンプル (48 kHz × 50 ms) を生成 |
| **Waveform Synthesis** | 10 ms | Vocoder（Student に統合する場合は Inference に含む） |

### チャンク設計

| パラメータ | 値 |
|---|---|
| サンプリングレート | 48,000 Hz |
| チャンクサイズ | 2,400 samples (50 ms) |
| ホップサイズ | 2,400 samples (50 ms) |
| Look-ahead | 0 samples（因果制約: 未来情報を使わない） |
| Overlap (crossfade) | 240 samples (5 ms)、チャンク境界のクリック防止 |

---

## 2. 全体アーキテクチャ

```
┌───────────────────────────────────────────────────────────────────┐
│                      Training Phase (GPU)                         │
│                                                                   │
│  ┌─────────┐    ┌──────────────────┐    ┌───────────────────┐    │
│  │ Dataset  │───▶│ Teacher          │───▶│ Teacher Ckpt      │    │
│  │ (§6)     │    │ OT-CFM DiT (§3)  │    │ SECS ≥ 0.92       │    │
│  └─────────┘    └──────────────────┘    └────────┬──────────┘    │
│                                                   │               │
│                              ┌─────────────────────┤               │
│                              │                     │               │
│                    ┌─────────▼────────┐  ┌────────▼──────────┐   │
│                    │ Stage 1          │  │ Stage 2            │   │
│                    │ Shortcut FM      │  │ DMD + Direct       │   │
│                    │ 10→2 step (§4.2) │  │ Metric Opt (§4.3)  │   │
│                    └─────────┬────────┘  └────────┬──────────┘   │
│                              │                     │               │
│                              └──────────┬──────────┘               │
│                                         │                         │
│                              ┌──────────▼──────────┐              │
│                              │ Stage 3: ADCD       │              │
│                              │ Joint Distill (§4.4) │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                              ┌──────────▼──────────┐              │
│                              │ Student Ckpt        │              │
│                              │ ~15M, 1-step        │              │
│                              │ 品質: ≥ Teacher      │              │
│                              └──────────┬──────────┘              │
│                                         │                         │
│                              Few-shot Adaptation (§5)             │
│                                         │                         │
│                              ┌──────────▼──────────┐              │
│                              │ Adapted Student     │              │
│                              │ (target speaker)    │              │
│                              └─────────────────────┘              │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                     Inference Phase (CPU)                          │
│                                                                   │
│  Mic ─▶ [Chunk Buffer] ─▶ [Feature Ext] ─▶ [Student 1-step] ─▶ Out │
│            5 ms              5 ms              30 ms       10 ms  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Teacher モデル設計 — 品質天井の最大化

### 3.1 アーキテクチャ選定

品質最優先のため、**OT-CFM + DiT (Diffusion Transformer)** を採用する。
2024-2026 年の VC 論文で最高の話者類似度・音質を達成しているアーキテクチャ系統。

| 候補 | SECS | UTMOS | 蒸留適性 | 選定 |
|---|---|---|---|---|
| **OT-CFM + DiT** (PFlow-VC / R-VC 系) | **0.920-0.930** | **4.05-4.10** | Shortcut FM で 2-step 化可能 | **◎** |
| Flow Matching + DiT (Seed-VC 系) | 0.868 | — | OSS、実装容易 | ○ (ベース実装参考) |
| Score-based Diffusion U-Net | 0.84-0.85 | 3.9-4.0 | ADCD で 1-step 化実績あり | △ |
| Neural Codec LM (VALL-E 2 系) | human parity | — | CPU 蒸留困難 | × |

**選定理由:**
- PFlow-VC (arXiv:2502.05471) は SECS 0.920 で VC 特化の SOTA
- R-VC (arXiv:2506.01014) は Shortcut FM で 2-step でも SECS 0.930・UTMOS 4.10
- DiT backbone は F5-TTS / Seed-VC で実証済み、OSS 実装が豊富
- OT-CFM は直線軌道に近く、Shortcut FM / DMD による蒸留と相性が良い

### 3.2 Teacher の内部構成

**設計思想:** PFlow-VC の時変 timbre tokens + discrete pitch VQVAE を取り入れつつ、
Seed-VC の DiT backbone をベースに構築する。

```
Input:
  x_content  : Content embedding (WavLM-large layer 7, 1024-dim)
  x_pitch    : Discrete pitch tokens (Pitch VQVAE, 64 codebook, 128-dim)
  x_timbre   : Time-varying timbre tokens (64 tokens via cross-attention)
  x_ir       : IR conditioning (64-dim, §3.5)

       ┌────────────────────────────────────────────────┐
       │             Conditioning                        │
       │                                                 │
       │  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
       │  │ Content   │  │ Pitch   │  │ Timbre       │  │
       │  │ Proj+Conv │  │ VQVAE   │  │ Cross-Attn   │  │
       │  │ 1024→512  │  │ 128→512 │  │ 64 tokens    │  │
       │  └─────┬─────┘  └────┬────┘  └──────┬───────┘  │
       │        └──────┬──────┘               │          │
       │               ▼                      │          │
       │     Content + Pitch concat           │          │
       │     (frame-level, 512-dim)           │          │
       │               │                      │          │
       │               ▼                      ▼          │
       │     ┌─── DiT Cross-Attention ─── Timbre KV ──┐ │
       │     │    + IR FiLM conditioning               │ │
       │     └─────────────────────────────────────────┘ │
       └────────────────────┬───────────────────────────┘
                            │
       ┌────────────────────▼───────────────────────────┐
       │   OT-CFM DiT Backbone                          │
       │                                                 │
       │   Architecture: DiT with U-Net skip connections │
       │   Layers: 16 (8 down + 8 up with skip)         │
       │   Heads: 12                                     │
       │   Hidden dim: 768                               │
       │   FFN dim: 3072                                 │
       │   Position encoding: RoPE                       │
       │   Parameters: ~180M                             │
       │                                                 │
       │   Domain: mel spectrogram (128-bin, 48 kHz)     │
       └────────────────────┬───────────────────────────┘
                            │
                            ▼
                Predicted velocity v_θ (OT-CFM)
```

**パラメータ合計 (Teacher):**

| モジュール | パラメータ数 |
|---|---|
| DiT Backbone | ~180M |
| Pitch VQVAE | ~5M |
| Timbre Cross-Attention Encoder | ~15M |
| **Total (trainable)** | **~200M** |
| WavLM-large (frozen) | 315M |
| ECAPA-TDNN (frozen) | 6M |

### 3.3 音声表現

| 項目 | 選定 | 理由 |
|---|---|---|
| **中間表現** | Mel spectrogram (128-bin, 48 kHz) | 位相復元を vocoder に委ねる。Latent 圧縮は Student 側で適用。 |
| **フレームシフト** | 10 ms (480 samples) | 標準的。チャンク 50 ms = 5 フレーム |
| **FFT サイズ** | 2048 | 48 kHz で十分な周波数分解能 |
| **窓長** | 2048 samples (~42.7 ms) | FFT と一致 |
| **Vocoder (Teacher)** | BigVGAN-v2 | 48 kHz 対応、現時点で最高品質 |

### 3.4 Content Encoder / F0 / Timbre 設計

品質天井を最大化するため、Teacher では計算コストを度外視して最良のモジュールを使う。

| モジュール | 選定 | パラメータ数 | 理由 |
|---|---|---|---|
| **Content Encoder** | WavLM-large (layer 7) | 315M (frozen) | HuBERT-base (768-dim) より高品質な content 表現。PFlow-VC / R-VC で採用実績。layer 7 は content 情報が最も豊富。 |
| **Content 量子化** | K-means 1024 clusters → 512-dim proj | — | Timbre leakage 防止。StableVC / EZ-VC で有効性実証。 |
| **Pitch** | Self-supervised Pitch VQVAE | ~5M | PFlow-VC (arXiv:2502.05471) 方式。64 codebook entries, 128-dim。F0 の離散化で安定性向上、continuous F0 より robust。 |
| **Timbre Encoder** | ECAPA-TDNN → 64 time-varying tokens | ~6M (frozen) + ~15M (cross-attn) | PFlow-VC 方式。static speaker embedding ではなく time-varying cross-attention tokens で timbre を表現。品質に大きく寄与。 |

### 3.5 IR Conditioning 設計

concept.md の Design 1 (parametric) を基盤としつつ、Teacher では learned embedding も併用。

```
IR Conditioning (128-dim total):

  Parametric path (64-dim):
  ├── RT60 subband estimates (8 bands)     : 8-dim
  ├── DRR (Direct-to-Reverb Ratio)         : 1-dim
  ├── Early reflection energy              : 1-dim
  ├── Spectral tilt (mic coloration proxy) : 4-dim
  └── Learned parametric residual          : 50-dim

  Learned embedding path (64-dim):
  ├── Lightweight IR encoder (3-layer CNN) : reverberant mel → 64-dim
  └── Trained end-to-end with Teacher

  Total: concat(parametric, learned) → 128-dim → FiLM conditioning
```

**推論時の IR 取得方法:**

| モード | 方法 | レイテンシ影響 |
|---|---|---|
| **キャリブレーション（推奨）** | 起動時にテスト信号 or 数秒の音声から IR パラメータ + embedding を推定。以降固定。 | なし（事前計算） |
| **オンライン適応** | N チャンクごとに IR embedding を更新 | 微小（バックグラウンド推定） |

### 3.6 Teacher の学習手順

```
Phase 1: Base OT-CFM Training
  - データ: 多話者クリーン音声 ≥ 1000 時間 (§6)
  - 入力: (content, pitch, timbre) → mel
  - 損失: OT-CFM velocity matching loss
  - Pitch VQVAE: 事前学習 (別途 50K steps)
  - OT solver: mini-batch OT for coupling
  - ステップ: ~800K steps, batch 64, lr 1e-4, cosine schedule
  - GPU: 4-8x A100/H100

Phase 2: IR-robust 化
  - データ: Phase 1 と同じ + RIR augmentation (§6.2)
  - 追加: IR conditioning path (128-dim)
  - 損失: OT-CFM loss + IR parameter prediction auxiliary loss (λ=0.1)
  - ステップ: ~300K steps (fine-tune from Phase 1)

Phase 3: 品質最大化 (perceptual + adversarial)
  - 追加損失:
    - Multi-resolution STFT loss (λ=0.5)
    - ECAPA speaker consistency loss (λ=0.3)
    - WavLM perceptual feature loss (λ=0.2)
  - Optional: R1 regularization + discriminator (品質が頭打ちなら)
  - ステップ: ~200K steps

Phase 4: Shortcut Flow Matching 化 (蒸留準備)
  - R-VC (arXiv:2506.01014) の手法を適用
  - Teacher を step-size conditioned に再学習
  - DiT に step-size token を追加（既存の timestep 条件付けを拡張）
  - これにより 10-step → 2-step の推論が Teacher 自身で可能に
  - ステップ: ~100K steps
```

**Teacher の推論品質目標 (Phase 3 完了時):**

| 指標 | 目標 | 参考 (SOTA) |
|---|---|---|
| SECS | ≥ 0.920 | PFlow-VC: 0.920, R-VC: 0.930 |
| UTMOS | ≥ 4.05 | R-VC: 4.10, FasterVoiceGrad: 4.03 |
| CER | ≤ 2.0% | R-VC: 1.40%, PFlow-VC: -- |
| WER | ≤ 5.0% | PFlow-VC: 2.57%, Seed-VC: 11.99% |

---

## 4. 蒸留戦略 — 品質を落とさず高速化

### 4.1 蒸留の全体設計思想

**核心:** 単純な MSE 蒸留では品質が必ず劣化する。最新研究では
**蒸留後の Student が Teacher を超える** ことが実証されている (DMDSpeech, arXiv:2410.11097)。
この知見を活用し、3段階の蒸留パイプラインを構築する。

```
Teacher (200M, 10-step OT-CFM)
    │
    │ Stage 1: Shortcut Flow Matching (§4.2)
    │ アーキテクチャは Teacher と同一、step 数のみ削減
    ▼
Mid-Teacher (200M, 2-step)
    │
    │ Stage 2: DMD + Direct Metric Optimization (§4.3)
    │ Student アーキテクチャへ蒸留 + 微分可能指標で品質向上
    ▼
Student-Draft (15M, 1-step)
    │
    │ Stage 3: ADCD Joint Distillation (§4.4)
    │ Content Encoder も同時蒸留、End-to-End 最適化
    ▼
Student-Final (15M, 1-step, content encoder 込み)
```

### 4.2 Stage 1: Shortcut Flow Matching (Teacher 内で step 削減)

**参考論文:** R-VC (arXiv:2506.01014) — SECS 0.930 を 2-step で達成

Teacher Phase 4 で導入した Shortcut FM により、Teacher 自身が 2-step で高品質推論可能になる。
これが Stage 2 以降の「教師信号」となる。

```
Training:
  - Teacher の DiT に step-size token d を追加条件付け
  - d ∈ {1/T, 2/T, ..., 1} をランダムサンプル
  - 損失: E[‖v_θ(x_t, t, d) - (x_1 - x_0)/d‖²]
  - d=1 のとき 1-step、d=1/2 のとき 2-step に相当

Result:
  - 10-step Teacher quality → 2-step で維持 (UTMOS 劣化 < 0.05)
  - 2-step Teacher が以降の蒸留ターゲットとして機能
```

### 4.3 Stage 2: Distribution Matching Distillation + Direct Metric Optimization

**参考論文:** DMDSpeech (arXiv:2410.11097) — **Student が Teacher を超えた唯一の手法**

ここで **アーキテクチャ変換** を行う。200M DiT → 15M Causal ConvNet。

```
┌──────────────────────────────────────────────────────────────┐
│  2-step Teacher (200M, frozen)                               │
│       │                                                      │
│       │ generate paired data                                 │
│       ▼                                                      │
│  (content, pitch, timbre, ir) → mel_teacher                  │
│                                                              │
│  Student (15M, 1-step, trainable)                            │
│       │                                                      │
│       │ predict mel                                          │
│       ▼                                                      │
│  mel_student                                                 │
│                                                              │
│  Losses:                                                     │
│    L_dist  : Distribution matching (adversarial alignment)   │
│    L_ctc   : Differentiable CTC (intelligibility, ∇ flows)  │
│    L_sv    : Differentiable Speaker Verification (ECAPA cos) │
│    L_stft  : Multi-resolution STFT loss                      │
│    L_wlm   : WavLM perceptual feature matching               │
│                                                              │
│  Total = L_dist + λ₁·L_ctc + λ₂·L_sv + λ₃·L_stft + λ₄·L_wlm │
│  λ₁=1.0, λ₂=2.0, λ₃=0.5, λ₄=0.3                           │
│                                                              │
│  Key: L_ctc と L_sv は微分可能 → Student に直接勾配が流れる    │
│       → Teacher では不可能だった end-to-end 指標最適化          │
│       → Student が Teacher を超えるメカニズム                   │
└──────────────────────────────────────────────────────────────┘

学習: ~400K steps, batch 64, lr 5e-5
```

### 4.4 Stage 3: ADCD Joint Distillation (Content Encoder 同時蒸留)

**参考論文:** FasterVoiceGrad (arXiv:2508.17868) — content encoder も同時蒸留で 6.6x 高速化

Stage 2 では WavLM-large (315M) を content encoder として使っているが、
CPU 推論では使えない。ここで content encoder も同時に蒸留する。

```
┌────────────────────────────────────────────────────────────────┐
│  Full pipeline distillation:                                   │
│                                                                │
│  Teacher path (frozen):                                        │
│    audio → WavLM-large → content(1024d) → Teacher-2step → mel │
│                                                                │
│  Student path (trainable):                                     │
│    audio → TinyContent(256d) → Student-1step → mel             │
│                                                                │
│  Adversarial loss:                                             │
│    D(mel_teacher, mel_student) — conversion-space discriminator │
│                                                                │
│  + Stage 2 の全損失を維持                                       │
│                                                                │
│  Key insight (FasterVoiceGrad):                                │
│    蒸留を「再構成空間」ではなく「変換空間」で行う                    │
│    → content encoder と decoder の整合性が自動的に最適化される     │
└────────────────────────────────────────────────────────────────┘

学習: ~200K steps, batch 64, lr 2e-5
```

### 4.5 Student のアーキテクチャ

品質最優先のため、パラメータバジェットを **~15M** に設定。
これは MeanVC (14M) と同等であり、CPU リアルタイム実績がある。

```
Student: Quality-First Causal ConvNet (~15M)
─────────────────────────────────────────────

  ┌─────────────────────────────────────────────┐
  │  Tiny Content Encoder (§4.7)                │
  │  Causal Conv1d × 7, ch: 64→128→256→512     │
  │  + 2-layer Causal Conformer (512-dim)       │
  │  Parameters: ~4M                            │
  │  Input: mel spectrogram (128-bin)           │
  │  Output: content features (512-dim)         │
  └──────────────────┬──────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │  Conditioning Fusion                        │
  │  content(512) + pitch(128) + timbre FiLM    │
  │  + IR FiLM                                  │
  └──────────────────┬──────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │  Causal ConvNeXt-v2 Blocks × 10            │
  │  (depthwise separable, causal padding)      │
  │  ch: 512, kernel: 7, dilation: [1,2,4,8,1] │
  │  FiLM: timbre + IR → scale, shift          │
  │  Residual + GroupNorm + GELU               │
  │  Parameters: ~8M                            │
  └──────────────────┬──────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │  Mel Decoder (3 layers)                     │
  │  Conv1d: 512 → 384 → 256 → 128            │
  │  Parameters: ~1M                            │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
             Mel spectrogram (128-bin)
                     │
  ┌──────────────────▼──────────────────────────┐
  │  Vocos-based iSTFT Vocoder (§4.8)          │
  │  Mel → (magnitude, phase)                   │
  │  → iSTFT → waveform                        │
  │  Parameters: ~2M                            │
  └─────────────────────────────────────────────┘
                     │
                     ▼
             Waveform (48 kHz)
```

### 4.6 Student のパラメータバジェット

| モジュール | パラメータ数 | FLOPS/チャンク | 備考 |
|---|---|---|---|
| Tiny Content Encoder | ~4M | ~20M | Causal Conv + Conformer |
| ConvNeXt Mel Predictor | ~9M | ~40M | 10 blocks, 512ch |
| Vocos iSTFT Vocoder | ~2M | ~10M | 48kHz 対応 |
| **合計** | **~15M** | **~70M** | |

> **参考値:** MeanVC (14M) は single-core CPU で RTF 0.136 を達成。
> 本設計は同等規模のため、マルチコア + ONNX 最適化で RTF < 0.5 は十分達成可能。
> ただし 50ms チャンクでの P99 latency はベンチマーク必須。

### 4.7 Content Encoder の蒸留

WavLM-large (315M) → Tiny Content Encoder (4M) の蒸留。

FasterVoiceGrad の知見: **変換パイプライン全体で蒸留する方が、
encoder 単体で蒸留するより品質が高い。** Stage 3 (ADCD) で実施。

ただし事前準備として、単体蒸留で warm-start する:

```
WavLM-large layer 7 (315M, frozen)      Tiny Content Encoder (4M)
         │                                        │
    audio input ────────────────────▶        mel input
         │                                        │
         ▼                                        ▼
  WavLM features (1024-dim, K-means)    tiny features (512-dim)
         │                                        │
         └──── CosineEmb + MSE + CTC align ───────┘
```

- CTC alignment loss: content 表現が音素境界を保持することを促進
- 学習: 200K steps, batch 128

### 4.8 Vocoder の選定と蒸留

品質最優先のため、**Vocos** (arXiv:2306.00814) ベースの iSTFT vocoder を採用。

| 候補 | パラメータ | RTF (CPU) | 品質 | 選定 |
|---|---|---|---|---|
| BigVGAN-v2 | 112M | RTF ~5 | 最高 | Teacher 用 |
| HiFi-GAN (v1) | 14M | RTF ~0.8 | 良 | × (RTF 不足) |
| **Vocos** | **13M** | **RTF ~0.3** | **良好** | ○ (参考設計) |
| **蒸留 Vocos (2M)** | **~2M** | **RTF ~0.1** | **Teacher の 95%** | **◎ Student 用** |

蒸留手順:
```
BigVGAN-v2 (112M, frozen) → mel → wav_teacher
蒸留 Vocos (2M, trainable) → mel → wav_student

損失:
  L_wave: L1(wav_student, wav_teacher)
  L_mstft: Multi-resolution STFT loss
  L_feat: BigVGAN discriminator feature matching
  L_phase: Phase spectrum loss (iSTFT の位相精度向上)

学習: 300K steps, batch 32
```

---

## 5. Few-shot Adaptation（少数話者適応）

### 5.1 適応対象

| パラメータ | 更新 | 理由 |
|---|---|---|
| Timbre tokens (LoRA, rank=8) | **更新する** | 新話者の声質を表現。Time-varying tokens は static embedding より表現力が高い。 |
| ConvNeXt FiLM 層 (LoRA, rank=4) | **更新する** | 話者固有の変換特性を捕捉 |
| Tiny Content Encoder | **凍結** | 話者非依存であるべき |
| Pitch VQVAE | **凍結** | 話者非依存 (F0 は入力側で制御) |
| IR pathway | **凍結** | 環境条件であり話者ではない |
| Vocoder | **凍結** | 汎用波形生成であるべき |

### 5.2 Few-shot の入力要件

| 項目 | 最小 | 推奨 |
|---|---|---|
| 発話数 | 3 発話 | 10-20 発話 |
| 合計音声長 | 15 秒 | 1-3 分 |
| 音声品質 | 16 kHz 以上、SNR > 20 dB | 48 kHz、クリーン環境 |

### 5.3 適応手順

```
1. ターゲット話者の音声を収集（3-20 発話）
2. ECAPA-TDNN で global speaker embedding を抽出
3. 64 time-varying timbre tokens を初期化 (global embedding を基に)
4. LoRA アダプタを初期化
   - Timbre cross-attention: rank=8, alpha=16
   - ConvNeXt FiLM: rank=4, alpha=8
5. 適応学習:
   - 入力: ターゲット話者の (content, pitch) + 初期 timbre tokens
   - 教師: ターゲット話者の実音声 mel
   - 損失: L_mel + L_stft + L_sv + L_wlm
   - ステップ: 500-2000 steps (数分で完了)
   - 更新対象: timbre tokens + LoRA weights のみ
6. 適応済み weights を保存 (~200KB)
```

---

## 6. データセット計画

### 6.1 音声データセット

品質天井を高めるため、大規模データセットを主軸に据える。

| データセット | 言語 | 話者数 | SR | 時間 | ライセンス | 用途 |
|---|---|---|---|---|---|---|
| **Emilia** | 多言語 | 100K+ | 24 kHz | ~101Kh | Apache 2.0 | Teacher 学習 (メイン)。Seed-VC で使用実績。 |
| **VCTK** | 英語 | 109 | 48 kHz | ~44h | CC BY 4.0 | 48 kHz fine-tuning + 評価 |
| **LibriTTS-R** | 英語 | 2,456 | 24 kHz | ~585h | CC BY 4.0 | Teacher 学習 (補助) |
| **JVS** | 日本語 | 100 | 24 kHz | ~30h | CC BY-SA 4.0 | 日本語対応 |
| **JSUT** | 日本語 | 1 | 48 kHz | ~10h | CC BY-SA 4.0 | 48 kHz 日本語評価用 |

> **学習戦略:** Phase 1-3 は Emilia (24kHz) + LibriTTS-R (24kHz) で大規模学習。
> Phase 3 の後半で VCTK (48kHz) を用いた 48 kHz fine-tuning を実施。
> 24→48kHz のアップサンプルではなく、48kHz ネイティブデータでの微調整で高域品質を確保。

### 6.2 RIR データセット（IR-aware 学習用）

| データセット | RIR 数 | 条件 | ライセンス | 用途 |
|---|---|---|---|---|
| **AIR Database** | ~170 | 多様な部屋 | 学術利用可 | 学習時 augmentation |
| **BUT ReverbDB** | ~1,500 | 実測 RIR + ノイズ | CC BY 4.0 | 学習時 augmentation |
| **ACE Challenge** | ~200 | パラメータラベル付き | 学術利用可 | IR パラメータ推定器の学習 |

### 6.3 データ前処理パイプライン

```
Raw Audio
    │
    ├─▶ Resample to target SR (24 kHz for Phase 1-3, 48 kHz for fine-tune)
    ├─▶ Loudness normalization (-23 LUFS)
    ├─▶ Silence trimming (VAD-based, Silero VAD)
    ├─▶ Segment into 5-15 sec chunks
    │
    ├─▶ Feature extraction (offline, cached):
    │     ├── WavLM-large layer 7 features (1024-dim)
    │     ├── WavLM features → K-means 1024 clusters (content tokens)
    │     ├── F0 contour (RMVPE) → Pitch VQVAE tokens
    │     ├── Speaker embedding (ECAPA-TDNN, 192-dim)
    │     └── Mel spectrogram (128-bin)
    │
    └─▶ IR Augmentation (online, during training):
          ├── Random RIR convolution (AIR / BUT ReverbDB)
          ├── Random EQ (low/high shelf, ±6 dB, parametric peak ±4 dB)
          ├── Random noise addition (SNR 15-40 dB)
          ├── Random mic simulation (proximity effect, roll-off)
          └── Compute IR conditioning (parametric + learned embedding)
```

---

## 7. ストリーミング推論パイプライン

### 7.1 処理フロー

```
Audio Input (OS callback, 48 kHz)
    │
    ▼
Ring Buffer (accumulate until chunk_size = 2400 samples)
    │
    ▼
Feature Extraction:
    ├── Tiny Content Encoder: mel_chunk → content_feat (512-dim × 5 frames)
    ├── Pitch: chunk → discrete pitch tokens (lightweight tracker)
    ├── Timbre: pre-computed (固定値、few-shot で適応済み)
    └── IR: pre-computed or background update (固定値)
    │
    ▼
Student 1-step Inference:
    Input:  concat(content_feat, pitch_tokens) + timbre FiLM + ir FiLM
    Output: mel_chunk (128-bin × 5 frames)
    ※ 内部状態を保持（causal conv の履歴バッファ）
    │
    ▼
Vocos iSTFT Vocoder:
    Input:  mel_chunk → (magnitude, phase)
    Output: waveform_chunk (2400 samples)
    │
    ▼
Crossfade Buffer:
    前チャンクの末尾 240 samples と現チャンクの先頭 240 samples を
    線形クロスフェードで接合
    │
    ▼
Audio Output (OS callback)
```

### 7.2 因果性の確保

| モジュール | 因果化の方法 |
|---|---|
| Tiny Content Encoder (Conv + Conformer) | Causal Conv1d + Causal self-attention (左マスク) + 履歴バッファ |
| Pitch 推定 | チャンク内のみで推定 or 1フレーム遅延許容 |
| ConvNeXt Mel Predictor | Causal depthwise conv + 履歴バッファ |
| Vocoder (Vocos iSTFT) | フレーム単位 iSTFT、overlap-add |

### 7.3 フェイルセーフ

| 状況 | 対処 |
|---|---|
| 推論が 50 ms を超過 | 前チャンクの出力を repeat（ドロップアウト防止） |
| 連続超過 (3 チャンク以上) | 品質を下げる (ConvNeXt blocks を skip: 10→6) |
| 入力レベルが極端に低い | ゲート閉鎖（無音出力、ノイズ増幅防止） |
| NaN / Inf 検出 | 前チャンク出力にフォールバック + ログ出力 |

---

## 8. デプロイ形式

### 8.1 モデルエクスポート

```
Student (PyTorch)
    │
    ├─▶ ONNX export (opset 17)
    │     ├── student_content_encoder.onnx  (~4M params)
    │     ├── student_mel_predictor.onnx    (~9M params)
    │     └── student_vocoder.onnx          (~2M params)
    │
    └─▶ Runtime options:
          ├── ONNX Runtime (CPU) ← 推奨、クロスプラットフォーム
          ├── OpenVINO (Intel CPU 最適化)
          └── XNNPACK (ARM CPU、モバイル向け)
```

### 8.2 量子化

| 手法 | サイズ削減 | 速度向上 | 品質劣化 |
|---|---|---|---|
| FP32 (baseline) | 1× (~60MB) | 1× | なし |
| FP16 | 0.5× (~30MB) | ~1.3× | 無視できる |
| INT8 dynamic | 0.25× (~15MB) | ~1.5-2× | 要検証 |
| INT8 static (calibrated) | 0.25× (~15MB) | ~2× | 小 (calibration 次第) |

**推奨:** FP32 でまず品質確認 → FP16 → 速度不足なら INT8 static 量子化
品質最優先のため、INT8 は品質劣化が許容範囲内の場合のみ適用。

### 8.3 想定スペックと推論時間の目安

| CPU | コア | クロック | 推定推論時間 (FP32, 15M) | 50ms 達成 |
|---|---|---|---|---|
| Intel i7-12700H | 14C/20T | 4.7 GHz | ~20-28 ms | 可能 |
| Intel i5-1235U | 10C/12T | 4.4 GHz | ~28-38 ms | 可能 |
| Apple M2 | 8C | 3.5 GHz | ~18-25 ms | 可能 |
| AMD Ryzen 5 5600U | 6C/12T | 4.2 GHz | ~25-35 ms | 可能 |
| Intel i3-1115G4 | 2C/4T | 4.1 GHz | ~40-55 ms | ギリギリ |

> MeanVC (14M) の実測 RTF 0.136 (single-core) を参考とした見積もり。
> **実機ベンチマークが必須。** i3 クラスで不足する場合は FP16/INT8 量子化で対応。

---

## 9. 評価計画

### 9.1 品質指標 — SOTA 水準を目標

| 指標 | 測定内容 | ツール | **Teacher 目標** | **Student 目標** |
|---|---|---|---|---|
| **SECS** | ECAPA cosine similarity | Resemblyzer | ≥ 0.920 | ≥ 0.910 |
| **UTMOS** | MOS 推定 (NN-based) | SpeechMOS | ≥ 4.05 | ≥ 4.00 |
| **CER** | 文字誤り率 | Whisper-large | ≤ 2.0% | ≤ 2.5% |
| **WER** | 単語誤り率 | Whisper-large | ≤ 5.0% | ≤ 6.0% |
| **PESQ** | 知覚音声品質 | pypesq | ≥ 3.5 | ≥ 3.3 |
| **F0 RMSE** | ピッチ追従精度 | RMVPE | < 12 Hz | < 15 Hz |
| **High-band Energy** | 8-24 kHz 帯域保存率 | scipy | > -2 dB | > -3 dB |

### 9.2 蒸留品質の評価 — Student ≥ Teacher を目指す

DMDSpeech の先例に基づき、**Student が Teacher を超える** ことを目標とする。

| 指標 | 目標 | DMDSpeech の実績 |
|---|---|---|
| SECS | Student ≥ Teacher - 0.01 | Student > Teacher (+0.02) |
| UTMOS | Student ≥ Teacher - 0.05 | Student > Teacher |
| WER | Student ≤ Teacher + 0.5% | Student < Teacher (1.94 vs 2.19) |

**直接指標最適化が機能すれば、Student が Teacher を超える可能性がある。**
これは DMDSpeech (arXiv:2410.11097) で実証済み。

### 9.3 リアルタイム性能指標

| 指標 | 測定内容 | 目標値 |
|---|---|---|
| **RTF** | 処理時間 / 音声時間 | < 0.5 |
| **E2E Latency** | マイク入力 → スピーカー出力の遅延 | < 50 ms |
| **P99 Latency** | 99パーセンタイルのチャンク処理時間 | < 45 ms |
| **Glitch Rate** | 50ms を超過したチャンクの割合 | < 0.1% |
| **Memory Usage** | 推論時のピークメモリ使用量 | < 500 MB |

### 9.4 IR-aware 評価 (concept.md E4 拡張)

| 条件軸 | 水準 |
|---|---|
| **残響** | Dry / Short RT60 (0.3s) / Long RT60 (1.0s) |
| **マイク距離** | Near (0.5m) / Far (3m) |
| **EQ coloration** | Flat / Bright (+6dB@8kHz) / Dark (-6dB@8kHz) |
| **ノイズ** | Clean / SNR 20dB / SNR 10dB |

→ **全条件で SECS > 0.88 を維持**（Dry 時 > 0.91）

### 9.5 比較対象 (ベースライン)

| システム | 種別 | 比較理由 |
|---|---|---|
| **Seed-VC** | OSS VC SOTA | 最も直接的な比較対象 |
| **RVC v2** | OSS VC (widely used) | 実用的なベースライン |
| **FasterVoiceGrad** | 1-step distilled VC | 蒸留品質の比較 |
| **MeanVC** | Lightweight streaming VC | CPU 推論性能の比較 |

---

## 10. 実装ロードマップ

### Phase 1: 基盤構築 (2 週間)

- [ ] プロジェクト構成・依存関係セットアップ (PyTorch 2.2+)
- [ ] Emilia データセットの取得・前処理パイプライン (§6.3)
- [ ] WavLM-large / RMVPE / ECAPA-TDNN の推論ラッパー
- [ ] Mel spectrogram 抽出 + BigVGAN-v2 推論の動作確認
- [ ] K-means content tokenizer の学習 (1024 clusters)
- [ ] Pitch VQVAE の学習 (64 codebook, 50K steps)

### Phase 2: Teacher 学習 (4 週間)

- [ ] OT-CFM DiT backbone の実装 (16-layer, 768-dim)
- [ ] Time-varying timbre cross-attention の実装
- [ ] Phase 1 学習: Base OT-CFM (800K steps)
- [ ] Phase 2 学習: IR-robust 化 (300K steps)
- [ ] Phase 3 学習: Perceptual + adversarial (200K steps)
- [ ] Phase 4 学習: Shortcut FM 化 (100K steps)
- [ ] Teacher 品質検証: SECS ≥ 0.920 達成確認

### Phase 3: 蒸留 (3 週間)

- [ ] Tiny Content Encoder 事前蒸留 (warm-start, 200K steps)
- [ ] Student ConvNeXt アーキテクチャ実装
- [ ] Stage 2: DMD + Direct Metric Optimization (400K steps)
- [ ] Stage 3: ADCD Joint Distillation (200K steps)
- [ ] Vocos vocoder 蒸留 (300K steps)
- [ ] End-to-End fine-tuning (100K steps)
- [ ] 蒸留品質検証: Student ≥ Teacher - 0.01 (SECS)

### Phase 4: リアルタイム化 (2 週間)

- [ ] Causal 化 + ストリーミング推論パイプライン
- [ ] ONNX エクスポート + 量子化実験
- [ ] CPU ベンチマーク (i7, i5, M2, Ryzen で実測)
- [ ] フェイルセーフ実装
- [ ] P99 latency < 45ms 達成確認

### Phase 5: Few-shot + 評価 (2 週間)

- [ ] Few-shot adaptation パイプライン (LoRA, timbre tokens)
- [ ] 評価スイート実装 (§9 の全指標)
- [ ] IR 条件スイープ評価
- [ ] ベースライン比較 (Seed-VC, RVC v2, FasterVoiceGrad, MeanVC)
- [ ] Teacher vs Student 品質比較レポート
- [ ] 最終品質レポート + デモ音声生成

---

## 11. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| Teacher が SECS 0.92 に到達しない | Student の品質天井が下がる | Seed-VC の pretrained weights を初期値として利用 / データ増量 |
| DMD 蒸留で Student が発散 | 蒸留失敗 | Discriminator の学習率を下げる / L_dist の重みを段階的に増加 |
| ADCD で content encoder が崩壊 | 明瞭度低下 | warm-start weight を強めに維持 (lr を encoder だけ 1/10) |
| 15M Student が CPU 50ms に収まらない | レイテンシ未達 | ConvNeXt blocks を 10→8 に削減 (品質影響を検証) / INT8 量子化 |
| 48 kHz vocoder 蒸留で高域劣化 | キラキラした音質が消える | Phase spectrum loss を強化 / 高域に重みをかけた STFT loss |
| Few-shot で IR 特性が timbre に混入 | 環境変化に弱い | IR pathway の凍結を厳守 + IR augmentation テストで検証 |
| Emilia データセットが大きすぎて I/O がボトルネック | 学習速度低下 | WebDataset 形式で sharding / NVMe SSD 必須 |

---

## Appendix A: 参考論文一覧 (本設計で参照)

### Teacher アーキテクチャ
| 論文 | ArXiv | 核心技術 |
|---|---|---|
| PFlow-VC | 2502.05471 | OT-CFM + 時変 timbre tokens + discrete pitch VQVAE, SECS 0.920 |
| R-VC | 2506.01014 | Shortcut Flow Matching, 2-step で SECS 0.930 |
| Seed-VC | 2411.09943 | DiT + Flow Matching, OSS, 101K 時間学習 |
| F5-TTS | 2410.06885 | DiT + OT-CFM backbone 設計の参考 |
| StableVC | 2412.04724 | DualAGC (dual attention + adaptive gate), style control |

### 蒸留手法
| 論文 | ArXiv | 核心技術 |
|---|---|---|
| DMDSpeech | 2410.11097 | DMD2 + 直接指標最適化, **Student が Teacher を超えた** |
| FasterVoiceGrad | 2508.17868 | ADCD, content encoder 同時蒸留, GPU RTF 0.0009 |
| DSFlow | 2602.09041 | Dual Supervision + step-aware tokens, 1-step |
| SlimSpeech | 2504.07776 | Annealing reflow + パラメータ 5x 圧縮 |
| ECTSpeech | 2510.05984 | Easy Consistency Tuning, Teacher 不要の蒸留 |

### リアルタイム VC
| 論文 | ArXiv | 核心技術 |
|---|---|---|
| MeanVC | 2510.08392 | 14M, 1-step mean flows, streaming |
| RT-VC | 2506.10289 | SPARC + DDSP, CPU 61ms |
| SynthVC | 2510.09245 | Neural codec, 8.24M, CPU 72ms |
| LatentVoiceGrad | 2509.08379 | 32-dim latent FM, CPU RTF 0.075 |

### IR-aware (concept.md 由来)
| 論文 | ArXiv | 核心技術 |
|---|---|---|
| BUDDy | 2405.04272 | Parametric IR operator, blind dereverb |
| Gencho | 2602.09233 | IR encoder → latent, complex spectrogram RIR |

## Appendix B: concept.md との対応表

| concept.md セクション | 本設計書の対応 |
|---|---|
| A. Diffusion as prior (SGMSE) | §3.1 で OT-CFM DiT に変更（品質 SOTA 追求） |
| B. BUDDy (blind dereverb + IR) | §3.5 IR conditioning (parametric + learned dual path) |
| C. RIR estimation / generation | §6.2 RIR datasets, §3.5 |
| D. Datasets | §6.1 (Emilia 追加で大規模化), §6.2 |
| E1. IR-robust teacher | §3.6 Phase 2 |
| E2. Explicit IR path | §3.5 (Design 1 + 2 のハイブリッド) |
| E3. Distillation with IR fixed | §4 + §5.1 (IR 凍結明記) |
| E4. Evaluation protocol | §9 (SOTA ベースラインとの比較追加) |

## Appendix C: 主要ライブラリ (想定)

| 用途 | ライブラリ | バージョン目安 |
|---|---|---|
| 学習フレームワーク | PyTorch | >= 2.2 |
| 分散学習 | DeepSpeed / FSDP | latest |
| 音声処理 | torchaudio, librosa | latest |
| WavLM | transformers (HuggingFace) | latest |
| ECAPA-TDNN | speechbrain | >= 1.0 |
| BigVGAN-v2 | bigvgan (公式) | latest |
| Vocos | vocos (公式) | latest |
| ONNX エクスポート | torch.onnx + onnxruntime | >= 1.17 |
| F0 推定 | RMVPE (公式実装) | — |
| OT solver | POT (Python Optimal Transport) | latest |
| 評価 | pypesq, resemblyzer, speechmos | latest |
| ASR 評価 | whisper (large-v3) | latest |
| オーディオ I/O (リアルタイム) | sounddevice (PortAudio) | latest |
| データローダ | webdataset | latest |
