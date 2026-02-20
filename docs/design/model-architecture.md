# TMRVC Model Architecture Details

Kojiro Tanaka — model architecture
Created: 2026-02-16 (Asia/Tokyo)

> **Core insight:** Teacher は multi-step diffusion (non-causal U-Net) だが、
> Student は 1-step に蒸留されるため、推論時は iterative diffusion ではなく
> 単純な feedforward NN の連鎖。Frame-by-frame causal streaming で 50ms 以下を実現。

---

## 1. Content Encoder (~1.5-3M params)

### 1.1 アーキテクチャ

ContentVec / WavLM teacher から知識蒸留された軽量 causal CNN。
音声の言語的内容 (音素、韻律) を話者非依存で抽出する。

```
Input: mel_frame[1, 80, 1] + f0[1, 1, 1]
       ↓
┌──────────────────────────────────────────┐
│  Input Projection                         │
│  Conv1d(81 → 256, kernel=1)             │
│  + LayerNorm + SiLU                      │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Causal ConvNeXt Block ×4-6              │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Causal Depthwise Conv1d          │  │
│  │    channels=256, kernel=7          │  │
│  │    padding=(kernel-1, 0)  ← causal│  │
│  │  LayerNorm                        │  │
│  │  Pointwise Conv1d(256 → 1024)     │  │
│  │  SiLU                             │  │
│  │  Pointwise Conv1d(1024 → 256)     │  │
│  │  + Residual connection            │  │
│  └────────────────────────────────────┘  │
│                                          │
│  Dilation pattern: [1, 1, 2, 2, 4, 4]   │
│  (6 blocks の場合)                       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Output Projection                        │
│  Conv1d(256 → 256, kernel=1)             │
│  + LayerNorm                              │
└──────────────┬───────────────────────────┘
               ↓
Output: content[1, 256, 1]
```

### 1.2 F0 の統合方法

```
mel_frame[1, 80, 1]  +  f0[1, 1, 1]  →  concat  →  [1, 81, 1]  →  input projection
```

F0 は log-scale に変換 (log(f0+1))、unvoiced frames は 0。

### 1.3 Knowledge Distillation (from WavLM-large)

**Teacher 選択の根拠 (2026-02 更新):**

| Teacher | Params | 出力次元 | 品質 | 推奨 |
|---|---|---|---|---|
| ContentVec | 95M | 768d | 良好 | Phase 0 (検証用) |
| **WavLM-large layer 7** | **317M** | **1024d** | **最高** | **Phase 1+ (本番)** |

WavLM-large は ContentVec よりも content 表現が豊かで、speaker leakage が少ない。
layer 7 (中間層) を使用することで、content と prosody のバランスを最適化。

```
WavLM-large layer 7 (Teacher, ~317M, non-causal, offline)
     │
     │  audio → teacher_content[1, 1024, T]
     │
     ├── per-frame MSE loss: ||student_content - proj(teacher_content)||²
     ├── cosine embedding loss: 1 - cos(student, proj(teacher))
     └── CTC auxiliary loss (optional): phoneme label prediction
```

**Projection**: teacher 1024-dim → 256-dim linear projection (学習時のみ使用)。

### 1.3.1 VQ Bottleneck (Speaker Leakage 対策)

TVTSyn (arXiv:2602.09389) に倣い、Content Encoder 出力に **Factorized VQ Bottleneck** を追加。
content 特徴量を離散化することで、残留 speaker 情報を除去する。

```
Content Encoder 出力: content[1, 256, 1]
       │
       ▼
┌─────────────────────────────────────────────┐
│  Factorized VQ Bottleneck                    │
│                                              │
│  1. Split: content → [c1, c2] (128d each)   │
│  2. Quantize each:                           │
│     c1 → lookup(codebook_1) → q1            │
│     c2 → lookup(codebook_2) → q2            │
│  3. Concat: [q1, q2] → quantized[1, 256, 1] │
│                                              │
│  Codebook config:                            │
│    - n_codebooks: 2                          │
│    - codebook_size: 8192                     │
│    - codebook_dim: 128                       │
│    - commitment_loss: λ_commit = 0.25       │
└─────────────────────────────────────────────┘
       │
       ▼
Output: quantized_content[1, 256, 1]
```

**学習時:**
- Straight-through estimator で勾配伝播
- Commitment loss: ||content - detach(quantized)||²
- Codebook 更新: EMA (exponential moving average)

**推論時:**
- VQ は ONNX export 時に焼き込み可能 (lookup table)
- または runtime で VQ 実行 (追加 ~0.1ms/frame)

**効果:**
- Speaker similarity loss が低下 (leakage 除去)
- Content 一致性が向上 (離散化による正則化)

### 1.4 パラメータ・計算量見積もり

| 構成 | Params | FLOPs/frame | CPU time/frame (est.) |
|---|---|---|---|
| 4 blocks, d=256, k=7 | ~1.5M | ~3M | ~0.5ms |
| 6 blocks, d=256, k=7 | ~2.2M | ~5M | ~0.8ms |
| 6 blocks, d=256, k=7, dilated | ~2.2M | ~5M | ~0.8ms |

> **推奨:** 6 blocks (dilated) で開始。受容野: 6×(7-1)×max_dilation = ~70 frames (700ms)。

### 1.5 Streaming State

Per-frame 推論時、各 causal conv layer は `kernel_size - 1` フレーム分の過去コンテキストを保持。

```
Layer 1 (k=7, d=1): context = 6 frames
Layer 2 (k=7, d=1): context = 6 frames
Layer 3 (k=7, d=2): context = 12 frames
Layer 4 (k=7, d=2): context = 12 frames (※ dilation による場合は要調整)
...
→ Total state: [1, 256, 28] (6+6+12+2+2 の合計は設計次第)
```

実際の state shape は onnx-contract.md §3 で定義。

---

## 2. Converter / 1-Step Denoiser (~3-5M params)

### 2.1 アーキテクチャ

Content features を target speaker の音響特徴に変換する。
Teacher (multi-step diffusion U-Net) から 1-step に蒸留された causal CNN。

```
Inputs:
  content[1, 256, 1]     ← Content Encoder 出力
  spk_embed[1, 192]      ← Speaker embedding (cached)
  ir_params[1, 24]       ← IR parameters (cached)
  state_in[1, 384, 52]   ← Hidden state

       ┌────────────────────────────────────────────┐
       │  Input Projection                           │
       │  Conv1d(256 → 384, kernel=1)               │
       │  + LayerNorm + SiLU                         │
       └──────────────┬─────────────────────────────┘
                      ↓
       ┌────────────────────────────────────────────┐
       │  Causal ConvNeXt Block ×6-8                │
       │  with FiLM Conditioning                     │
       │                                             │
       │  ┌───────────────────────────────────────┐ │
       │  │ Causal Depthwise Conv1d               │ │
       │  │   channels=384, kernel=7              │ │
       │  │ LayerNorm                             │ │
       │  │                                       │ │
       │  │ FiLM (speaker + IR conditioning):     │ │
       │  │   γ, β = Linear(concat(spk, ir))     │ │
       │  │   x = γ * x + β                      │ │
       │  │                                       │ │
       │  │ Pointwise Conv1d(384 → 1536)         │ │
       │  │ SiLU                                  │ │
       │  │ Pointwise Conv1d(1536 → 384)         │ │
       │  │ + Residual                            │ │
       │  └───────────────────────────────────────┘ │
       │                                             │
       │  Dilation pattern: [1, 1, 2, 2, 4, 4, 8, 8]│
       │  (8 blocks の場合)                           │
       └──────────────┬─────────────────────────────┘
                      ↓
       ┌────────────────────────────────────────────┐
       │  Output Projection                          │
       │  Conv1d(384 → 513, kernel=1)               │
       │  (513 = n_fft/2 + 1 = STFT frequency bins) │
       └──────────────┬─────────────────────────────┘
                      ↓
Output: pred_features[1, 513, 1]
```

### 2.2 FiLM Conditioning (Feature-wise Linear Modulation)

```python
class FiLMConditioner(nn.Module):
    def __init__(self, d_cond, d_model):
        self.proj = nn.Linear(d_cond, d_model * 2)  # γ and β

    def forward(self, x, cond):
        # cond = concat(spk_embed, ir_params) → [1, 216]
        gamma_beta = self.proj(cond)           # [1, 768]
        gamma, beta = gamma_beta.chunk(2, -1)  # each [1, 384]
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
```

- `cond = concat(spk_embed[192], acoustic_params[32])` → 224-dim
- 各 ConvNeXt block 内で適用
- Speaker と acoustic (IR + voice source) の情報が各層で注入される

### 2.3 LoRA: Cross-Attention K/V への適用

Few-shot adaptation 時、Converter の一部の層に LoRA を適用する。

```
Standard FiLM layer:
  γ, β = W_film @ cond

LoRA-adapted FiLM layer (enrollment 時):
  γ, β = (W_film + ΔW) @ cond
  where ΔW = B @ A  (A: d_cond × rank, B: rank × d_model×2)
  rank = 4

Weight merge (enrollment 後):
  W_film_merged = W_film + (α/rank) × B @ A
  → 推論時は vanilla FiLM (追加コストなし)
```

**Merge 手順:**
1. Enrollment: target speaker の音声で LoRA weights (A, B) を学習
2. Merge: `W_merged = W_original + (alpha/rank) * B @ A`
3. Merged weights を converter.onnx に焼き込み、または `lora_delta` として保存
4. 推論時: merged weights で通常の FiLM を実行 (LoRA のオーバーヘッドなし)

### 2.4 Global Timbre Memory (GTM) — 代替 Converter アーキテクチャ

GTM は静的な speaker FiLM を時変の timbre cross-attention に置換する。
`ConverterStudentGTM` として実装。ONNX I/O は `ConverterStudent` と完全互換。

```
spk_embed[1, 192]
    │
    ├── GlobalTimbreMemory
    │   Linear(192 → 8×48 = 384)
    │   Reshape → memory[1, 8, 48]
    │
    └── 各 ConverterBlockGTM 内で:
        ┌────────────────────────────────────┐
        │ CausalConvNeXtBlock(384, k=3, d=?) │
        │         ↓                           │
        │ TimbreCrossAttention:               │
        │   Q: content[384, T], K/V: memory   │
        │   4-head attention, d_head=96       │
        │         ↓                           │
        │ FiLM(IR only): ir_params[24] → γ,β │
        └────────────────────────────────────┘
```

**利点:**
- 話者特徴がフレームごとに content に適応的に適用される
- IR params は FiLM のまま (時不変で OK)
- LoRA 対象は `gtm.proj` (1 層) に集約 → delta サイズ削減

**定数 (constants.yaml):**
- `gtm_n_entries: 8`, `gtm_d_entry: 48`, `gtm_n_heads: 4`
- 8 × 48 = 384 = d_converter_hidden (意図的)

### 2.5 ConverterStudentHQ — Semi-Causal HQ Mode

`ConverterStudentHQ` は HQ mode 用の semi-causal converter。
`SemiCausalConvNeXtBlock` を使用し、入力 T=7 (1 current + 6 lookahead) から T=1 を出力する。

```
SemiCausalConvNeXtBlock:
  Training: pad(left_context, right_context), output T = input T
  Streaming: state_in provides left context, input contains right context
    Output T = input T - right_context. State advances by 1 frame.
```

**right_context 配分 (greedy min(dilation, remaining), L=6):**

| Block | dilation | right_ctx | left_ctx | T in→out |
|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 7→6 |
| 1 | 1 | 1 | 1 | 6→5 |
| 2 | 2 | 2 | 2 | 5→3 |
| 3 | 2 | 2 | 2 | 3→1 |
| 4 | 4 | 0 | 8 | 1→1 |
| 5 | 4 | 0 | 8 | 1→1 |
| 6 | 6 | 0 | 12 | 1→1 |
| 7 | 6 | 0 | 12 | 1→1 |

- HQ state = sum(left_ctx) = **46 frames** (vs causal 52)
- Weight structure is identical to ConverterStudent → `from_causal()` で初期化可能
- 計算量: causal の ~18% 増 (~2.4ms/frame vs ~2.0ms/frame)

**モデルプロファイル:**

| Mode | q range | Converter | Latency | State |
|---|---|---|---|---|
| Live | q ≤ 0.3 | ConverterStudent (T=1) | 20ms | 52 frames |
| HQ | q > 0.3 | ConverterStudentHQ (T=7→1) | 80ms | 46 frames |

### 2.6 蒸留戦略: Teacher → Student

#### Teacher Model (学習時のみ、§6 参照)

Non-causal U-Net, multi-step diffusion (v-prediction)。
高品質な変換結果を生成し、Student の教師信号とする。

#### Phase 1: ODE Trajectory Pre-training

```
Teacher (20 steps) で ODE trajectory を生成:
  x_t = α_t × x_0 + σ_t × ε    (t = 0, 0.05, 0.1, ..., 1.0)

Student は 1-step で x_0 を予測するよう学習:
  Loss = ||v_student(x_1) - v_teacher||²

  where v_teacher = trajectory direction from Teacher ODE
```

- Teacher の ODE 軌道に沿った velocity を Student が模倣
- Student は noise level t=1 (pure noise) → t=0 (clean) を 1-step で実行

#### Phase 2: Distribution Matching Distillation (DMD)

```
Phase 1 で粗く学習した Student をさらに洗練:

  Loss_DMD = E[||score_fake - score_real||²]

  score_fake: Student が生成した sample の distribution score
  score_real: Teacher training data の distribution score

+ L_stft: Multi-resolution STFT loss (perceptual quality)
+ L_spk:  Speaker similarity loss (ECAPA cosine)
```

#### Causal Teacher → Causal Student の Domain Mismatch 対処

| 問題 | Teacher は bidirectional (non-causal)、Student は causal |
|---|---|
| **対策 1** | Teacher の出力を "causal-masked" teacher target として使用。Student は causal な情報のみで近似 |
| **対策 2** | Causal Teacher variant を追加学習 (Teacher の attention を causal mask で再学習) |
| **対策 3** | Progressive training: まず large context → 徐々に context を短縮 |
| **推奨** | 対策 1 から開始。品質不足なら対策 2 を追加 |

### 2.5 パラメータ・計算量見積もり

| 構成 | Params | FLOPs/frame | CPU time/frame (est.) |
|---|---|---|---|
| 6 blocks, d=384, k=7 | ~3.2M | ~8M | ~1.5ms |
| 8 blocks, d=384, k=7 | ~4.2M | ~10M | ~2.0ms |
| 8 blocks, d=384, k=7, dilated | ~4.2M | ~10M | ~2.0ms |

> **推奨:** 8 blocks (dilated) で開始。受容野を最大化して変換品質を確保。

---

## 3. Vocoder (~0.33-5M params)

### 3.1 アーキテクチャ: iSTFT-based (Vocos 系)

Converter 出力の STFT 特徴量から magnitude と phase を予測し、iSTFT で波形復元。

```
Input: features[1, 513, 1]  ← Converter 出力

       ┌────────────────────────────────────────────┐
       │  Causal ConvNeXt Backbone                   │
       │                                             │
       │  Conv1d(513 → 256, kernel=1)               │
       │  + LayerNorm + SiLU                         │
       │                                             │
       │  Causal ConvNeXt Block ×3-4                │
       │    channels=256, kernel=7                   │
       │    dilation: [1, 2, 4] or [1, 1, 2, 4]     │
       └──────────────┬─────────────────────────────┘
                      ↓
       ┌────────────────────────────────────────────┐
       │  Magnitude Head                             │
       │  Conv1d(256 → 513, kernel=1)               │
       │  + ReLU (ensure non-negative)              │
       └──────────────┬─────────────────────────────┘
                      ↓ stft_mag[1, 513, 1]
       ┌────────────────────────────────────────────┐
       │  Phase Head (cos/sin parameterization)      │
       │  Conv1d(256 → 513×2, kernel=1)             │
       │    → cos_part[1, 513, 1]                   │
       │    → sin_part[1, 513, 1]                   │
       │  phase = atan2(sin_part, cos_part)          │
       └──────────────┬─────────────────────────────┘
                      ↓ stft_phase[1, 513, 1]

Output: stft_mag[1, 513, 1], stft_phase[1, 513, 1]
  → iSTFT → 窓付き信号 (960 samples) → Overlap-Add
```

### 3.2 Phase Prediction: cos/sin Parameterization

直接 phase を予測すると [-π, π] の wraparound で学習が不安定になる。
cos/sin を別々に予測し atan2 で phase を復元する。

```python
cos_part = self.phase_head_cos(backbone_out)  # unconstrained
sin_part = self.phase_head_sin(backbone_out)  # unconstrained
phase = torch.atan2(sin_part, cos_part)       # [-π, π]
```

### 3.3 Overlap-Add 出力

```
iSTFT output (per frame):
  complex_stft = mag * exp(j * phase)        # [513, 1]
  time_signal = IFFT(complex_stft, n=1024)   # [1024]
  windowed = time_signal[:960] * hann_window  # [960]

Overlap-Add:
  output_buffer[n:n+960] += windowed
  → 240 samples (= hop) が確定し出力
```

### 3.4 候補比較

| Vocoder | Params | RTF (CPU, est.) | 品質 | Streaming |
|---|---|---|---|---|
| Vocos (original) | ~13M | ~0.3 | 良好 | 要改造 |
| Vocos-lite (stripped) | ~3M | ~0.15 | 良 | 要改造 |
| **MS-Wavehax** | **~0.33M** | **~0.05** | **十分** | **Causal 対応** |
| Streaming HiFi-GAN | ~14M | ~0.8 | 良 | 対応 |

> **推奨:** MS-Wavehax (0.33M) から開始。品質不足なら Vocos-lite (3M) に切り替え。
> MS-Wavehax は iSTFT-based で超軽量、causal streaming に最適。

### 3.5 パラメータ・計算量見積もり

| 構成 | Params | FLOPs/frame | CPU time/frame (est.) |
|---|---|---|---|
| MS-Wavehax (0.33M) | 330K | ~0.7M | ~0.1ms |
| Vocos-lite (3M) | 3M | ~6M | ~0.5ms |
| Vocos-full (13M) | 13M | ~26M | ~2.0ms |

---

## 4. IR Estimator / Acoustic Params Estimator (~1-3M params)

### 4.1 アーキテクチャ

入力音声の音響環境 (残響特性、マイク特性) および声質パラメータを推定する。
amortized 実行 (10 フレーム = 100ms ごと)。

```
Input: mel_chunk[1, 80, 10]  ← 10 frames 分の mel (accumulated)

       ┌────────────────────────────────────────────┐
       │  Causal CNN                                 │
       │                                             │
       │  Conv1d(80 → 128, kernel=3) + SiLU         │
       │  Conv1d(128 → 128, kernel=3, d=2) + SiLU   │
       │  Conv1d(128 → 128, kernel=3, d=4) + SiLU   │
       │  AdaptiveAvgPool1d(1)  ← temporal pooling   │
       └──────────────┬─────────────────────────────┘
                      ↓
       ┌────────────────────────────────────────────┐
       │  MLP Head                                   │
       │  Linear(128 → 64) + SiLU                   │
       │  Linear(64 → 32)                            │
       │  + Sigmoid/Tanh (range clamping)            │
       └──────────────┬─────────────────────────────┘
                      ↓
Output: acoustic_params[1, 32]
        (24 IR params + 8 voice source params)
```

### 4.2 出力仕様: 24 IR params + 8 Voice Source params

#### IR Parameters (indices 0-23): 8 Subbands × 3 Parameters

| Index | Parameter | Description | Range | Activation |
|---|---|---|---|---|
| 0-7 | RT60 (sec) | 各サブバンドの残響時間 | [0.05, 3.0] | sigmoid × 2.95 + 0.05 |
| 8-15 | DRR (dB) | Direct-to-Reverb Ratio | [-10, 30] | sigmoid × 40 - 10 |
| 16-23 | Spectral tilt | マイク/部屋の周波数特性傾斜 | [-6, 6] | tanh × 6 |

#### Voice Source Parameters (indices 24-31)

| Index | Parameter | Description | Range |
|---|---|---|---|
| 24 | breathiness_low | 低域ブレス成分 | [0, 1] |
| 25 | breathiness_high | 高域ブレス成分 | [0, 1] |
| 26 | tension_low | 低域声帯張力 | [0, 1] |
| 27 | tension_high | 高域声帯張力 | [0, 1] |
| 28 | jitter | 基本周波数微細変動 | [0, 1] |
| 29 | shimmer | 振幅微細変動 | [0, 1] |
| 30 | formant_shift | フォルマントシフト | [-1, 1] |
| 31 | roughness | 声の粗さ | [0, 1] |

Voice source params は `.tmrvc_speaker` のプリセット値とランタイムブレンド可能。
詳細は `docs/design/acoustic-condition-pathway.md` §Voice Source Presets を参照。

**Subbands (24kHz, 8 bands):**

| Band | Frequency Range |
|---|---|
| 0 | 0 - 375 Hz |
| 1 | 375 - 750 Hz |
| 2 | 750 - 1500 Hz |
| 3 | 1500 - 3000 Hz |
| 4 | 3000 - 4500 Hz |
| 5 | 4500 - 6000 Hz |
| 6 | 6000 - 9000 Hz |
| 7 | 9000 - 12000 Hz |

### 4.3 Amortized Execution

```
Frame  0: accumulate mel → mel_buffer[:, :, 0]
Frame  1: accumulate mel → mel_buffer[:, :, 1]
...
Frame  9: accumulate mel → mel_buffer[:, :, 9]
           → ir_estimator.Run(mel_buffer) → ir_params
           → ir_params をキャッシュ (次の 10 frames で使用)
Frame 10: accumulate mel → mel_buffer[:, :, 0]  (reset)
...
```

- 10 frames (100ms) に 1 回だけ実行
- Per-frame amortized cost: inference_time / 10 ≈ 0.2ms/frame
- 室内音響は 100ms スケールでは変化しないため、十分な更新頻度

### 4.4 パラメータ・計算量見積もり

| 構成 | Params | FLOPs/run | Amortized FLOPs/frame |
|---|---|---|---|
| Small (128ch, 3 layers) | ~1M | ~2M | ~0.2M |
| Medium (128ch, 5 layers) | ~2M | ~4M | ~0.4M |

---

## 5. Speaker Encoder (~5-10M params, offline)

### 5.1 アーキテクチャ: ECAPA-TDNN ベース

話者の声質特徴を抽出し、LoRA delta を生成する。
Enrollment 時 (offline) に 1 回だけ実行。推論パイプラインには含まれない。

```
Input: mel_ref[1, 80, T_ref]  ← 参照音声 (3-15 sec)

       ┌────────────────────────────────────────────┐
       │  ECAPA-TDNN Backbone (~5M)                  │
       │                                             │
       │  SE-Res2Net Block ×3                       │
       │    channels: 512 → 512 → 512               │
       │    kernel: 5, scale: 8                     │
       │  Channel & Context-dependent Statistics    │
       │  Attentive Statistics Pooling              │
       │    → utterance-level embedding             │
       └──────────────┬─────────────────────────────┘
                      ↓
       ┌────────────────────────────────────────────┐
       │  Speaker Embedding Head                     │
       │  Linear(1536 → 192)                        │
       │  + L2 normalization                         │
       └──────────────┬─────────────────────────────┘
                      ↓ spk_embed[1, 192]
       ┌────────────────────────────────────────────┐
       │  LoRA Delta Head                            │
       │  Linear(1536 → 512) + SiLU                 │
       │  Linear(512 → 24576)                        │
       │  (24576 = 4 layers × 2 KV × (384×4 + 4×384))│
       └──────────────┬─────────────────────────────┘
                      ↓ lora_delta[1, 24576]

Output: spk_embed[1, 192], lora_delta[1, 24576]
```

### 5.2 LoRA Delta の構造

```
lora_delta[24576] の内訳:

Layer 0:
  K_down[384, 4] = lora_delta[0:1536]
  K_up[4, 384]   = lora_delta[1536:3072]
  V_down[384, 4] = lora_delta[3072:4608]
  V_up[4, 384]   = lora_delta[4608:6144]

Layer 1:
  K_down[384, 4] = lora_delta[6144:7680]
  ...

Layer 2: ...
Layer 3: ...
```

### 5.3 `.tmrvc_speaker` ファイル生成

```python
def create_speaker_file(audio_paths: list[str], output_path: str):
    # 1. 全参照音声の mel を結合
    mel_ref = extract_and_concat_mel(audio_paths)  # [1, 80, T_total]

    # 2. Speaker encoder 実行
    spk_embed, lora_delta = speaker_encoder(mel_ref)

    # 3. バイナリ保存
    write_tmrvc_speaker(output_path, spk_embed, lora_delta)
    # Format: Magic("TMSP") + Version(1) + sizes + data + SHA256
```

### 5.4 パラメータ

| Component | Params | Note |
|---|---|---|
| ECAPA-TDNN backbone | ~5M | Pre-trained (SpeechBrain) |
| Speaker embedding head | ~0.3M | Fine-tuned |
| LoRA delta head | ~1-5M | Trained from scratch |
| **Total** | **~6-10M** | |

> Offline 実行のため、パラメータ数は CPU real-time 制約を受けない。

---

## 6. Teacher Model (学習時のみ)

### 6.1 アーキテクチャ: U-Net with Cross-Attention

非 causal (bidirectional) の高品質モデル。学習時のみ使用。

```
Inputs:
  x_t:       noisy mel[1, 80, T]     ← diffusion noise level t の mel
  t:         timestep[1]             ← diffusion timestep
  content:   [1, 1024, T]            ← WavLM-large layer 7 features
  f0:        [1, 1, T]              ← log-F0 contour
  spk_embed: [1, 192]               ← speaker embedding
  acoustic_params: [1, 32]          ← acoustic conditioning (IR + voice source)

       ┌──────────────────────────────────────┐
       │  U-Net (v-prediction, OT-CFM)        │
       │                                       │
       │  Encoder:                             │
       │    Down Block 1: 80 → 128, /2         │
       │    Down Block 2: 128 → 256, /2        │
       │    Down Block 3: 256 → 384, /2        │
       │    Down Block 4: 384 → 512, /2        │
       │                                       │
       │  Bottleneck:                          │
       │    ResBlock + Cross-Attention          │
       │    Q = x_t, K/V = content             │
       │    + F0 FiLM + Speaker FiLM           │
       │    + Acoustic FiLM + Timestep embedding│
       │                                       │
       │  Decoder (with skip connections):     │
       │    Up Block 4: 512 → 384, ×2          │
       │    Up Block 3: 384 → 256, ×2          │
       │    Up Block 2: 256 → 128, ×2          │
       │    Up Block 1: 128 → 80, ×2           │
       │                                       │
       │  Cross-Attention at:                  │
       │    Down Block 3, 4                    │
       │    Bottleneck                         │
       │    Up Block 3, 4                      │
       └──────────────┬───────────────────────┘
                      ↓
Output: v_predicted[1, 80, T]  (velocity in mel space)
```

### 6.2 OT-CFM: Optimal Transport Conditional Flow Matching (更新)

従来の Rectified Flow から **OT-CFM (Matcha-TTS, StableVC)** に変更。
minibatch 内で noise と data の最適輸送ペアを計算することで、軌道を直線化し少ステップ生成を可能にする。

```python
# Forward process (OT-CFM)
def forward_process_otcfm(x_0, noise, t):
    """
    x_0: clean mel [B, 80, T]
    noise: matched noise via optimal transport [B, 80, T]
    t: timestep in [0, 1]
    """
    # Linear interpolation (OT path)
    x_t = (1 - t) * noise + t * x_0
    
    # Velocity target (points directly to x_0)
    v_target = x_0 - noise
    
    return x_t, v_target

# Optimal Transport pairing
def compute_ot_pairs(x_0_batch, noise_batch):
    """
    minibatch 内で noise と data の最適ペアを計算
    使用: scipy.optimize.linear_sum_assignment
    """
    B = x_0_batch.shape[0]
    
    # Cost matrix: pairwise distances
    cost = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            cost[i, j] = np.linalg.norm(x_0_batch[i] - noise_batch[j])
    
    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Reorder noise to match optimal pairs
    return noise_batch[col_ind]

# Training step
def train_step_otcfm(model, batch):
    content, f0, spk_embed, acoustic_params, mel_target = batch
    B = mel_target.shape[0]
    
    # Sample noise
    noise = torch.randn_like(mel_target)
    
    # OT pairing (within batch)
    noise_matched = compute_ot_pairs(mel_target.detach().cpu().numpy(), 
                                      noise.detach().cpu().numpy())
    noise_matched = torch.from_numpy(noise_matched).to(mel_target.device)
    
    # Sample timestep
    t = torch.rand(B, 1, 1)
    
    # Forward process
    x_t, v_target = forward_process_otcfm(mel_target, noise_matched, t)
    
    # Predict velocity
    v_pred = model(x_t, t, content, f0, spk_embed, acoustic_params)
    
    # Loss
    loss_flow = F.mse_loss(v_pred, v_target)
    return loss_flow
```

**OT-CFM の利点:**
- 軌道が直線化 → 少ない推論ステップで高品質
- 1-step 蒸留がより容易 (Teacher 軌道が既に直線的)
- StableVC (AAAI 2025) で実証済み

### 6.3 損失関数

```
L_total = L_flow + λ_stft × L_stft + λ_spk × L_spk

L_flow: Flow matching loss (v-prediction MSE)
L_stft: Multi-resolution STFT loss
  - resolutions: [512, 1024, 2048] (FFT sizes)
  - spectral convergence + log magnitude L1
L_spk: Speaker consistency loss
  - cosine similarity between ECAPA embeddings of generated vs target
```

### 6.4 パラメータ

| Component | Params |
|---|---|
| U-Net encoder/decoder | ~60M |
| Cross-attention layers | ~15M |
| Conditioning projections | ~5M |
| **Total** | **~80M** |

> Teacher は GPU 学習・GPU 推論。パラメータ数制約は緩い。

---

## 7. パラメータバジェット合計 (Streaming Models)

### 7.1 推論時に実行されるモデル

| Model | Params | Execution | Per-frame FLOPs | Per-frame CPU time |
|---|---|---|---|---|
| content_encoder | 1.5-3M | Per-frame | 3-5M | 0.5-0.8ms |
| converter | 3-5M | Per-frame | 8-10M | 1.5-2.0ms |
| vocoder | 0.33-5M | Per-frame | 0.7-6M | 0.1-0.5ms |
| ir_estimator | 1-3M | /10 frames | 0.2-0.4M (amortized) | 0.02-0.05ms (amortized) |
| **合計** | **5.8-16M** | | **12-21M** | **~2.1-3.4ms** |

### 7.2 推奨構成

| Model | 推奨 Params | 根拠 |
|---|---|---|
| content_encoder | 2.2M | 6 blocks, 受容野確保のため dilated |
| converter | 4.2M | 8 blocks, 変換品質が最重要 |
| vocoder (MS-Wavehax) | 0.33M | 超軽量で十分な品質 |
| ir_estimator | 1M | amortized 実行、軽量で十分 |
| **合計** | **~7.7M** | |

### 7.3 CPU Real-time 実現性

```
推奨構成の per-frame 合計:
  FLOPs: ~12M
  CPU time: ~3ms (Intel i7-12700H, single core, FP32 estimate)

hop_length = 240 samples = 10ms
→ utilization = 3ms / 10ms = 30%

INT8 量子化後:
  CPU time: ~1.5ms → utilization = 15%
```

> 十分なマージンがあり、CPU real-time は確実に達成可能。

---

## 8. 設計整合性チェックリスト

- [x] Content Encoder 出力 (256d) が Converter 入力と一致
- [x] Content Encoder Teacher: WavLM-large layer 7 (1024d) → projection (Phase 1+)
- [x] VQ Bottleneck: 2 codebooks × 8192 × 128d で speaker leakage 対策
- [x] Converter 出力 (513d) が Vocoder 入力 (513d) と一致
- [x] Acoustic Estimator 出力 (32d: 24 IR + 8 voice source) が Converter の FiLM 入力と一致
- [x] Speaker Encoder 出力 (192d) が Converter の FiLM 入力と一致
- [x] Voice Source Preset (8d) ブレンドが acoustic_params[24..31] に適用される
- [x] Voice Source 外部蒸留が Phase 2 で有効
- [x] LoRA delta サイズ (24576) が onnx-contract.md と一致
- [x] State tensor shapes が onnx-contract.md §3 と一致
- [x] 全 Live streaming models が causal (look-ahead = 0)
- [x] ConverterStudentHQ は semi-causal (right_ctx=[1,1,2,2,0,0,0,0], state=46)
- [x] パラメータ合計 (~7.7M) が CPU real-time に十分小さい
- [x] Per-frame inference (~3ms) < hop time (10ms)
- [x] Teacher は OT-CFM v-prediction (軌道直線化)、Student は 1-step 蒸留
- [x] 品質目標: Phase 1 で SECS ≥ 0.90, UTMOS ≥ 4.0

---

## 9. Student アーキテクチャ妥当性 (Latency-Quality 前提)

### 9.1 判定方針

Student の妥当性は「単一の最速点」ではなく、`Latency-Quality` スペクトラム全域で評価する。

評価モード:

- Live: `q=0.0`, lookahead=0
- Mix: `q=0.5`, lookahead=3
- Quality: `q=1.0`, lookahead=6

### 9.2 主要評価指標

| 指標カテゴリ | 指標 | 目的 |
|---|---|---|
| レイテンシ | reported latency / frame time / overrun rate | 実時間性能の保証 |
| 明瞭性 | ASR WER, 子音区間の誤り率 | 活舌の評価 |
| イントネーション | F0 RMSE, F0 correlation, V/UV error | ピッチ・抑揚の評価 |
| 話者性 | SECS / speaker cosine | 話者一致 |
| 音質 | UTMOS / MCD / STFT loss proxy | 知覚品質 |

### 9.3 受け入れ基準 (更新)

| Mode | レイテンシ | 品質条件 |
|---|---|---|
| Live | <= 30ms (reported) | SECS >= 0.90, UTMOS >= 4.0 |
| Mix | <= 55ms (reported) | Live 比で WER 改善 or 同等, F0 corr 改善 |
| Quality | <= 85ms (reported) | Live 比で WER/子音誤り/F0 RMSE の有意改善 |

### 9.4 アーキテクチャ妥当性の判断

以下を満たす場合、現行 Student 構成を妥当とする。

- Live で real-time 制約を安定達成
- Mix/Quality でイントネーション・活舌指標が単調改善
- モード遷移時 (クロスフェード) に破綻がない
- CPU 負荷悪化時に adaptive 降格で破綻回避できる