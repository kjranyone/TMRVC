# TMRVC Codec-Latent Design: Token-Based Streaming Voice Conversion

Kojiro Tanaka — codec-latent design
Created: 2026-02-28 (Asia/Tokyo)

> **Paradigm shift:** 入力波形を直接変換するのではなく、因果ニューラル音声コーデックで潜在トークン列に変換し、そのトークン列を条件として「次トークン」を生成するストリーミング条件付き生成モデル。
> 
> **Core insight:** チャンク境界での破綻は、連続特徴量では回避困難だが、離散トークン空間では「トークン境界 = 自然な区切り」としてモデルが学習可能。

---

## 1. 設計動機: 従来手法の問題点

### 1.1 既存リアルタイムVC (RVC, Beatrice等) の課題

```
入力チャンク → 特徴量抽出 → 変換 → デコード → 出力
                ↑
           チャンク境界で情報断絶
```

- **チャンク外コンテキスト喪失**: 各チャンクを独立処理 → 文脈不整合
- **境界破綻**: チャンク境界で急激な変化 → ノイズ・歪み
- **State管理の手設計**: 連続特徴量のstateは経験則に依存

### 1.2 Codec→Codec パラダイムの利点

```
入力 → Causal Codec → 潜在トークン列 → Token LM/Flow → 出力トークン → Decoder → 出力
                          ↓
                   トークン境界 = 自然な区切り
                   時間整合性はモデルが学習
```

**VALL-E (2023) が示したこと:**
- 3秒の参照音声のみでゼロショット話者変換が可能
- 離散トークン表現が話者性・内容を適切に分離
- in-context learning で話者適応

---

## 2. アーキテクチャ概要

### 2.1 システム構成

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TMRVC Codec-Latent Pipeline                      │
│                                                                          │
│  DAW Audio In (48kHz)                                                    │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────┐                                                    │
│  │  Polyphase       │  48kHz → 24kHz                                    │
│  │  Resampler       │                                                    │
│  └────────┬─────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐     ┌─────────────────────────────────────┐       │
│  │ Streaming Codec  │────▶│ Token Buffer (N tokens)              │       │
│  │ (Causal, ~20ms)  │     │  [t-3, t-2, t-1, t]                  │       │
│  │                  │     └───────────────┬─────────────────────┘       │
│  │ Encoder: Conv1d  │                     │                             │
│  │ RVQ: 4 codebooks │                     ▼                             │
│  │ Latent: 50 Hz    │     ┌─────────────────────────────────────┐       │
│  │ (20ms frame)     │     │ Streaming Token Transformer /       │       │
│  └──────────────────┘     │ Causal Flow Matching                │       │
│                           │                                     │       │
│                           │ Input: tokens[t-K:t], spk_embed     │       │
│                           │ Output: token[t+1]                  │       │
│                           │                                     │       │
│                           │ Causal attention mask               │       │
│                           │ State: KV cache                     │       │
│                           └───────────────┬─────────────────────┘       │
│                                           │                             │
│                                           ▼                             │
│                           ┌─────────────────────────────────────┐       │
│                           │ Streaming Codec Decoder             │       │
│                           │                                     │       │
│                           │ Tokens → Conv1d Decoder → 24kHz    │       │
│                           │ (Causal, OLA output)               │       │
│                           └───────────────┬─────────────────────┘       │
│                                           │                             │
│                                           ▼                             │
│                           ┌──────────────────┐                         │
│                           │  Polyphase       │  24kHz → 48kHz          │
│                           │  Resampler       │                          │
│                           └────────┬─────────┘                         │
│                                    │                                    │
│                                    ▼                                    │
│                           DAW Audio Out (48kHz)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 コンポーネント一覧

| コンポーネント | パラメータ | 実行頻度 | レイテンシ寄与 |
|---------------|-----------|---------|--------------|
| **Streaming Codec Encoder** | ~10M | 20ms/frame | 20ms (frame) |
| **Token Buffer** | — | — | 0 (memory) |
| **Streaming Token Model** | ~20-50M | 20ms/frame | 5-10ms |
| **Streaming Codec Decoder** | ~3M | 20ms/frame | 5ms |
| **Resampler** | — | per-block | ~0 |
| **合計** | ~33-63M | — | **~30-35ms** |

---

## 3. Streaming Neural Codec

### 3.1 Codec 選択肢比較

| Codec | Frame | Bitrate | Causal | Streaming | 品質 |
|-------|-------|---------|--------|-----------|------|
| **SoundStream** | 10-20ms | 3-18kbps | ✓ | ✓ | 良 |
| **EnCodec** | ~26ms | 1.5-24kbps | ✓ | ✓ | 高 |
| **DAC** | ~10ms | 8kbps | △ | 要改造 | 最高 |
| **Lyra v2** | 20ms | 3.2-9.2kbps | ✓ | ✓ | 良 |
| **自作Causal Codec** | 20ms | 8-16kbps | ✓ | ✓ | 設計次第 |

**推奨:** EnCodec ベースの因果バリアント、または自作軽量コーデック。

### 3.2 Causal Codec Encoder

```
Input: audio[1, 480] (20ms @ 24kHz)
       │
       ▼
┌────────────────────────────────────────────┐
│  Causal Conv1d Encoder                      │
│                                            │
│  Conv1d(1 → 32, k=7, causal) + SiLU        │
│  Conv1d(32 → 64, k=5, causal, stride=2)    │
│  Conv1d(64 → 128, k=5, causal, stride=2)   │
│  Conv1d(128 → 256, k=3, causal, stride=2)  │
│  Conv1d(256 → 512, k=3, causal)            │
└──────────────────┬─────────────────────────┘
                   │
                   ▼  [1, 512, 1] (latent frame)
┌────────────────────────────────────────────┐
│  Residual Vector Quantization (RVQ)         │
│                                            │
│  n_codebooks: 4                            │
│  codebook_size: 1024                       │
│  codebook_dim: 128                         │
│                                            │
│  Output: tokens[4] (4 discrete tokens)     │
└────────────────────────────────────────────┘
```

**フレームレート変換:**
- 24kHz / 480 = 50 Hz (20ms per frame)
- 1秒あたり 50 frames → 200 tokens (4 codebooks × 50)

### 3.3 Causal Codec Decoder

```
Input: tokens[4] (from Token Model)
       │
       ▼
┌────────────────────────────────────────────┐
│  RVQ Dequantization                         │
│  lookup(tokens) → latent[1, 512, 1]        │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Causal Conv1d Decoder (Transposed)         │
│                                            │
│  ConvTranspose1d(512 → 256, k=3, stride=2) │
│  ConvTranspose1d(256 → 128, k=3, stride=2) │
│  ConvTranspose1d(128 → 64, k=5, stride=2)  │
│  ConvTranspose1d(64 → 32, k=5, stride=2)   │
│  Conv1d(32 → 1, k=7, causal)               │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
Output: audio[1, 480] (20ms @ 24kHz)
```

### 3.4 Codec 学習戦略

**Option A: Pre-trained Codec + Fine-tune**
- EnCodec (pre-trained on music/speech) を使用
- Causal attention mask で再学習

**Option B: Scratch Training (推奨)**
- VC用に特化した軽量コーデックをスクラッチ学習
- VCTK + JVS + LibriTTS-R で学習
- 損失関数:
  - Multi-scale STFT loss
  - Adversarial loss (multi-scale discriminator)
  - Commitment loss (RVQ)

---

## 4. Streaming Token Model

### 4.1 モデル選択肢

| アーキテクチャ | 因果性 | 並列性 | 品質 | 推論速度 |
|--------------|--------|--------|------|---------|
| **Causal Transformer** | ✓ | △ | 高 | 中 |
| **Mamba / SSM** | ✓ | ✓ | 高 | 高 |
| **Causal Flow Matching** | ✓ | △ | 最高 | 中 |
| **Consistency Model** | ✓ | ✓ | 高 | 高 |

**推奨:** Mamba (State Space Model) ベースのストリーミング生成。

### 4.2 Streaming Mamba Token Model

```
Input: 
  - tokens_in[t-K:t] (K context tokens, e.g., K=10)
  - spk_embed[192]
  - state_in (Mamba hidden state)

       │
       ▼
┌────────────────────────────────────────────────────────────┐
│  Token Embedding                                            │
│  Embedding(1024 → 256) × 4 codebooks → concat → [1, K, 1024]│
│  + Positional Encoding (learned, causal)                   │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  Speaker Conditioning                                       │
│  spk_embed[192] → Linear → [1, 256]                        │
│  → cross-attention or FiLM to each layer                   │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  Stacked Mamba Blocks × 6-12                                │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Mamba Block                                        │   │
│  │                                                     │   │
│  │  x → LayerNorm → Linear(x, A, B, C, D)            │   │
│  │     → Selective Scan (causal, stateful)            │   │
│  │     → Linear → output                              │   │
│  │     + Residual                                     │   │
│  │                                                     │   │
│  │  State: h[t] = A @ h[t-1] + B @ x[t]               │   │
│  │  Output: y[t] = C @ h[t] + D @ x[t]                │   │
│  │                                                     │   │
│  │  (causal: h[t] depends only on h[0:t-1])           │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  + FiLM conditioning (spk_embed) between blocks           │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  Output Heads (one per codebook)                            │
│                                                            │
│  Linear(256 → 1024) × 4 → softmax → P(token | context)    │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
Output:
  - tokens_out[4] (next 4 tokens, sampled or argmax)
  - state_out (updated Mamba state)
```

### 4.3 Mambaの利点

1. **線形時間複雑性**: O(T) vs Transformer O(T²)
2. **Causal設計**: 状態 h[t] は過去のみに依存
3. **状態サイズが小さい**: ~数KB vs Transformer KV cache ~数MB
4. **CPU推論に最適**: 効率的なscan演算

### 4.4 代替: Causal Transformer with KV Cache

Mambaが不適切な場合のフォールバック:

```
Causal Transformer:
  - n_layers: 6-8
  - d_model: 256
  - n_heads: 4
  - d_ff: 1024
  - max_context: 100 tokens (2秒)
  
KV Cache:
  - 事前確保: [n_layers, 2, 100, n_heads, d_head]
  - 推論時: append-only, sliding window
```

---

## 5. ストリーミングパイプライン詳細

### 5.1 Frame-by-Frame処理フロー

```cpp
// 20ms frame processing
void processFrame(float* input_480, float* output_480) {
    // 1. Encode to tokens
    int tokens_in[4];
    codec_encoder_->encode(input_480, tokens_in);
    
    // 2. Update token buffer
    token_buffer_.push(tokens_in);
    
    // 3. Generate next tokens
    int tokens_out[4];
    token_model_->generate(
        token_buffer_.get_context(10),  // K=10 context tokens
        spk_embed_,                      // cached speaker embedding
        mamba_state_,                    // in/out Mamba state
        tokens_out
    );
    
    // 4. Decode to audio
    codec_decoder_->decode(tokens_out, output_480);
}
```

### 5.2 レイテンシ内訳

| 段階 | 時間 | 備考 |
|-----|------|------|
| Resample in | ~0 | polyphase, negligible |
| Codec Encoder | 5ms | Conv1d encoder + RVQ |
| Token Model | 10ms | Mamba inference |
| Codec Decoder | 5ms | RVQ lookup + Conv1d decoder |
| Resample out | ~0 | polyphase, negligible |
| **Algorithmic** | **20ms** | frame size (causal) |
| **Processing** | **20ms** | inference |
| **Total** | **~40ms** | w/ DAW buffer |

### 5.3 State管理

```
State Components:
┌────────────────────────────────────────────────────┐
│  Codec Encoder State                                │
│  - Conv1d context: [kernel_size-1] per layer       │
│  - Size: ~10KB                                      │
├────────────────────────────────────────────────────┤
│  Token Buffer                                       │
│  - Recent tokens: [K, 4] (K=10)                    │
│  - Size: ~160B                                      │
├────────────────────────────────────────────────────┤
│  Mamba State                                        │
│  - Hidden state per layer: [d_state, d_model]      │
│  - Size: ~100KB (6 layers, d_state=16, d_model=256)│
├────────────────────────────────────────────────────┤
│  Codec Decoder State                                │
│  - Conv1d context: [kernel_size-1] per layer       │
│  - Size: ~10KB                                      │
└────────────────────────────────────────────────────┘
Total State: ~120KB (fixed, pre-allocated)
```

---

## 6. 学習パイプライン

### 6.1 Stage 1: Codec Pre-training

```bash
# Codec 学習 (VCTK + JVS + LibriTTS-R)
uv run tmrvc-train-codec \
  --cache-dir data/cache \
  --frame-size 480 \        # 20ms @ 24kHz
  --n-codebooks 4 \
  --codebook-size 1024 \
  --device cuda
```

**損失関数:**
- Multi-scale STFT loss (resolution: [512, 1024, 2048])
- Adversarial loss (Multi-scale discriminator)
- Feature matching loss
- Commitment loss (RVQ)

### 6.2 Stage 2: Token Model Training (Teacher)

```bash
# Teacher: Causal Flow Matching (非リアルタイム)
uv run tmrvc-train-token-teacher \
  --codec-checkpoint checkpoints/codec.pt \
  --cache-dir data/cache \
  --context-length 100 \   # 2秒コンテキスト
  --n-layers 8 \
  --d-model 256 \
  --device cuda
```

**学習目標:** 
- Flow matching で token[t+1] を予測
- 条件: tokens[0:t], spk_embed

### 6.3 Stage 3: Token Model Distillation (Student)

```bash
# Student: Mamba (リアルタイム用)
uv run tmrvc-distill-token \
  --teacher-checkpoint checkpoints/token_teacher.pt \
  --architecture mamba \
  --n-layers 6 \
  --d-model 256 \
  --device cuda
```

**蒸留戦略:**
- Teacher の出力分布を Student が模倣
- Kullback-Leibler divergence loss

### 6.4 Stage 4: Speaker Enrollment

#### 6.4.1 階層的話者適応パイプライン

```
参照音声 (3秒〜5分)
       │
       ├─→ Speaker Encoder (ECAPA-TDNN) ─→ spk_embed [192]
       │
       ├─→ Style Encoder (optional) ─→ style_embed [128]
       │
       ├─→ Codec Encoder ─→ reference_tokens [T, 4] (in-context用)
       │
       └─→ LoRA Fine-tuning (optional) ─→ lora_delta [15872]
              │
              ▼
         .tmrvc_speaker ファイル
```

#### 6.4.2 Adaptation Levels

| Level | 必要音声 | 処理時間 | 用途 |
|-------|---------|---------|------|
| **light** | 3-10秒 | <1秒 | 高速VC、簡易クローン |
| **standard** | 10-30秒 | ~5秒 | 高品質VC/TTS |
| **full** | 1-5分 | 1-5分 | キャラクター再現 |

#### 6.4.3 CLI Usage

```bash
# Light: spk_embedのみ (高速)
uv run tmrvc-create-character \
    --audio ref.wav \
    --output models/speaker.tmrvc_speaker \
    --level light

# Standard: in-context用参照トークン付き (推奨)
uv run tmrvc-create-character \
    --audio-dir data/sample_voice/ \
    --output models/speaker.tmrvc_speaker \
    --level standard \
    --max-ref-frames 150

# Full: LoRA fine-tuning付き (最高品質)
uv run tmrvc-create-character \
    --audio-dir data/sample_voice/ \
    --token-model checkpoints/token_student.pt \
    --output models/speaker.tmrvc_speaker \
    --level full \
    --finetune-steps 200 \
    --device cuda
```

#### 6.4.4 In-Context Learning at Inference

```
1. Load .tmrvc_speaker → extract spk_embed, reference_tokens
2. Pre-fill KV-Cache with reference_tokens
3. Streaming inference:
   - codec_encoder(input_audio) → input_tokens
   - token_model(input_tokens, spk_embed, kv_cache) → output_tokens
   - codec_decoder(output_tokens) → output_audio
```

#### 6.4.5 出力 (.tmrvc_speaker v3)

```json
{
  "version": 3,
  "spk_embed": [192 floats],
  "style_embed": [128 floats, optional],
  "reference_tokens": [[4 ints], ...],
  "lora_delta": [15872 floats, optional],
  "metadata": {
    "name": "Character Name",
    "enrollment_audio": "data/sample_voice/",
    "enrollment_duration_sec": 30.5,
    "adaptation_level": "standard"
  }
}
```

---

## 7. 従来設計との比較

### 7.1 アーキテクチャ比較

| 側面 | 従来 (Feature-based) | 新設計 (Codec-Latent) |
|------|---------------------|----------------------|
| **表現** | mel → 連続content[256d] | audio → 離散tokens[4] |
| **時間整合性** | State tensor (手設計) | トークン予測 (モデル学習) |
| **境界破綻耐性** | 中 | 高 |
| **話者適応** | LoRA delta | in-context + spk_embed (+ LoRA optional) |
| **話者ファイル** | ~65KB (v1) | ~1KB (light) / ~65KB (full) |
| **モデルサイズ** | ~7.7M | ~33-63M |
| **レイテンシ** | ~25ms | ~40ms |
| **実装複雑性** | 中 | 高 |

### 7.2 トレードオフ分析

**Codec-Latent の利点:**
1. チャンク境界での破綻が本質的に回避される
2. 時間整合性がモデルに学習される
3. VALL-E系のゼロショット能力を継承
4. 離散表現で speaker leakage が自然に抑制

**Codec-Latent の欠点:**
1. モデルサイズが増大 (~4-8x)
2. レイテンシが増加 (~15ms)
3. 実装複雑性が上昇
4. Codec品質に全体が依存

### 7.3 推奨移行戦略

```
Phase 1: 既存設計を維持しつつ、Codec-Latent を並行開発
Phase 2: オフライン評価で品質比較
Phase 3: 品質優位が確認されれば移行
Phase 4: ハイブリッド (軽量時は従来、高品質時はCodec-Latent)
```

---

## 8. 実装ロードマップ

### 8.1 Phase A: Codec Development (4-6週)

- [ ] Causal Codec Encoder/Decoder 実装
- [ ] RVQ 実装
- [ ] Codec 学習パイプライン構築
- [ ] 品質評価 (ViSQOL, MUSHRA)

### 8.2 Phase B: Token Model Development (6-8週)

- [ ] Mamba block 実装 (PyTorch)
- [ ] Causal Transformer 実装 (フォールバック)
- [ ] Token Model 学習パイプライン
- [ ] 蒸留パイプライン

### 8.3 Phase C: Integration (4-6週)

- [ ] ONNX Export (Codec + Token Model)
- [ ] Rust Streaming Engine 更新
- [ ] VST3 Plugin 統合
- [ ] レイテンシ・品質評価

### 8.4 Phase D: Optimization (2-4週)

- [ ] INT8 量子化
- [ ] 推論最適化
- [ ] メモリフットプリント削減

---

## 9. 設計整合性チェックリスト

- [ ] Codec frame size (20ms) が hop time と整合
- [ ] トークンフレームレート (50 Hz) がリアルタイム処理に適合
- [ ] Mamba state が固定サイズで事前確保可能
- [ ] 全コンポーネントが causal (lookahead = 0)
- [ ] Total state < 1MB (CPU real-time 実現可能)
- [ ] Codec decoder 出力がOLAなしで連続 (causal conv)
- [ ] Speaker enrollment が offline で実行可能
- [ ] ONNX export 時に Mamba が適切に変換可能

---

## 10. 参考文献

1. **VALL-E** (2023): Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. arXiv:2301.02111
2. **SoundStream** (2021): An End-to-End Neural Audio Codec. arXiv:2107.03312
3. **EnCodec** (2022): High Fidelity Neural Audio Compression. arXiv:2210.13438
4. **DAC** (2023): High-Fidelity Audio Compression with Improved RVQGAN. arXiv:2306.06546
5. **Mamba** (2023): Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752
6. **StreamVC** (2023): Streaming Voice Conversion via Causal Convolution.
7. **LLVC** (2021): Low-Latency Voice Conversion.
