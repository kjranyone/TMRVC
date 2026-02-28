# TMRVC Model Architecture Details (Codec-Latent Pipeline)

Kojiro Tanaka — model architecture
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-28 — Codec-Latent パラダイムに一本化

> **Core insight:** 入力音声を因果ニューラルコーデックで離散トークン列に変換し、
> Mamba ベースのトークン予測モデルで次トークンを生成、デコーダで波形復元。

---

## 1. Streaming Codec Encoder (~10M params)

### 1.1 アーキテクチャ

Causal Conv1d エンコーダ + Residual Vector Quantization (RVQ)。

```
Input: audio[1, 1, 480] (20ms @ 24kHz)
       │
       ▼
┌────────────────────────────────────────────┐
│  Causal Conv1d Encoder                      │
│                                            │
│  Conv1d(1 → 32, k=7, causal) + LayerNorm + SiLU  │
│  Conv1d(32 → 64, k=7, causal, stride=2)    │
│  Conv1d(64 → 128, k=7, causal, stride=2)   │
│  Conv1d(128 → 256, k=7, causal, stride=2)  │
│  Conv1d(256 → 512, k=7, causal)            │
└──────────────┬─────────────────────────────┘
               │
               ▼  [1, 512, 1]
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

### 1.2 Frame Rate

| パラメータ | 値 | 備考 |
|-----------|---|------|
| Frame size | 480 samples | 20ms @ 24kHz |
| Frame rate | 50 Hz | 50 frames/sec |
| Tokens per frame | 4 | RVQ (4 codebooks) |
| Token rate | 200 tokens/sec | 4 × 50 |

### 1.3 Causal Conv1d

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, 
                              stride=stride, dilation=dilation)
    
    def forward(self, x, state=None):
        if state is None:
            state = torch.zeros(x.shape[0], x.shape[1], self.padding)
        x = torch.cat([state, x], dim=2)
        new_state = x[:, :, -self.padding:]
        return self.conv(x), new_state
```

### 1.4 State Size

```
Encoder State:
  Layer 1: 1 × 6 = 6
  Layer 2: 32 × 6 = 192
  Layer 3: 64 × 12 = 768
  Layer 4: 128 × 24 = 3072
  Layer 5: 256 × 48 = 12288
  ────────────────────────
  Total: ~16KB
```

---

## 2. Token Model (~20-50M params)

### 2.1 Mamba Architecture

Selective State Space Model (SSM) による causal な次トークン予測。

```
Input:
  - tokens_in[K, 4] (K context tokens)
  - spk_embed[192]
  - mamba_state (hidden state)

       │
       ▼
┌────────────────────────────────────────────────────────────┐
│  Token Embedding                                            │
│  Embedding(1024 → 256) × 4 codebooks → sum → [K, 256]     │
│  + Positional Encoding (learned, causal)                   │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  Speaker Projection                                         │
│  spk_embed[192] → Linear → [256] → FiLM to each layer     │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  Stacked Mamba Blocks × 6                                   │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Mamba Block                                        │   │
│  │                                                     │   │
│  │  x → LayerNorm → in_proj → [x, z]                 │   │
│  │  x → conv1d (causal) → silu                        │   │
│  │  x → x_proj → [B, C, delta]                        │   │
│  │  delta → dt_proj                                   │   │
│  │                                                     │   │
│  │  State update (causal):                            │   │
│  │    h[t] = A @ h[t-1] + B @ x[t]                   │   │
│  │  Output:                                           │   │
│  │    y[t] = C @ h[t] + D @ x[t]                     │   │
│  │                                                     │   │
│  │  y = y * silu(z)                                   │   │
│  │  out = out_proj(y) + residual                      │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  + FiLM(spk_embed) between blocks                         │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  Output Heads (one per codebook)                            │
│  Linear(256 → 1024) × 4 → softmax → P(token | context)    │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
Output:
  - logits[K, 4, 1024]
  - next_tokens[4] (sampled)
  - mamba_state (updated)
```

### 2.2 Mamba State Size

```
Mamba State (per layer):
  [d_inner, d_state] = [512, 16] = 8192 floats

Total (6 layers):
  6 × 8192 × 4 bytes = ~200KB
```

### 2.3 Streaming Inference

```python
@torch.no_grad()
def streaming_generate(self, tokens, spk_embed, state, temperature=1.0):
    logits, new_states = self.model(tokens, spk_embed, state['mamba_states'])
    
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    next_tokens = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
    
    return next_tokens, {'mamba_states': new_states}
```

---

## 3. Streaming Codec Decoder (~3M params)

### 3.1 アーキテクチャ

RVQ dequantization + Causal ConvTranspose1d デコーダ。

```
Input: tokens[4]
       │
       ▼
┌────────────────────────────────────────────┐
│  RVQ Dequantization                         │
│  for each codebook:                         │
│    quant_i = embedding(tokens[i])          │
│    latent += quant_i                       │
│  Output: latent[1, 128, 1]                 │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Causal Conv1d Decoder (Transposed)         │
│                                            │
│  Conv1d(128 → 256, k=7, causal)            │
│  ConvTranspose1d(256 → 128, k=7, stride=2) │
│  ConvTranspose1d(128 → 64, k=7, stride=2)  │
│  ConvTranspose1d(64 → 32, k=7, stride=2)   │
│  Conv1d(32 → 1, k=7, causal) + Tanh        │
└──────────────┬─────────────────────────────┘
               │
               ▼
Output: audio[1, 1, 480] (20ms @ 24kHz)
```

### 3.2 State Size

```
Decoder State:
  Layer 1: 256 × 6 = 1536
  Layer 5: 32 × 6 = 192
  ────────────────────────
  Total: ~8KB
```

---

## 4. Speaker Encoder (~5M params)

### 4.1 アーキテクチャ

ECAPA-TDNN ベースの話者埋め込み抽出。Enrollment 時 (offline) にのみ実行。

```
Input: mel[1, 80, T] (参照音声)

       │
       ▼
┌────────────────────────────────────────────┐
│  ECAPA-TDNN Backbone (~5M)                  │
│  SE-Res2Net Block ×3                       │
│  Attentive Statistics Pooling              │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Speaker Embedding Head                     │
│  Linear(1536 → 192) + L2 normalization     │
└──────────────┬─────────────────────────────┘
               │
               ▼
Output: spk_embed[192]
```

---

## 5. パラメータバジェット合計

| Model | Params | Execution | State Size |
|-------|--------|-----------|------------|
| Codec Encoder | ~10M | Per-frame (20ms) | ~16KB |
| Token Model (Mamba) | ~20-50M | Per-frame (20ms) | ~200KB |
| Codec Decoder | ~3M | Per-frame (20ms) | ~8KB |
| Speaker Encoder | ~5M | Offline only | — |
| **Total (streaming)** | **~33-63M** | — | **~224KB** |

---

## 6. 設計整合性チェックリスト

- [ ] Codec frame size (20ms = 480 samples) がストリーミングパイプラインと整合
- [ ] Token rate (50 Hz × 4 = 200 tokens/sec) がリアルタイム処理に適合
- [ ] Mamba state が固定サイズで事前確保可能
- [ ] 全コンポーネントが causal (lookahead = 0)
- [ ] Total state < 1MB (CPU real-time 実現可能)
- [ ] Codec encoder/decoder state が ping-pong buffering 可能
- [ ] Speaker enrollment が offline で実行可能
- [ ] ONNX export 時に Mamba scan 演算が適切に変換可能

---

## 7. 参考文献

1. **SoundStream** (2021): An End-to-End Neural Audio Codec. arXiv:2107.03312
2. **EnCodec** (2022): High Fidelity Neural Audio Compression. arXiv:2210.13438
3. **VALL-E** (2023): Neural Codec Language Models are Zero-Shot TTS. arXiv:2301.02111
4. **DAC** (2023): High-Fidelity Audio Compression with Improved RVQGAN. arXiv:2306.06546
5. **Mamba** (2023): Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752
