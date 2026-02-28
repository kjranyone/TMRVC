# TMRVC ONNX Model I/O Contract (Codec-Latent Pipeline)

Kojiro Tanaka — ONNX contract
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-28 — Codec-Latent パラダイムに一本化

> **Purpose:** Python (tmrvc-export) と Rust (tmrvc-engine-rs) の間のインターフェース仕様。
> 3つの ONNX モデルの入出力テンソル形状、state 管理、数値パリティ基準を定義する。

---

## 1. 共有定数

### 1.1 Source of Truth: `configs/constants.yaml`

```yaml
# Audio parameters
sample_rate: 24000
frame_size: 480            # 20ms frame
frame_rate: 50             # sample_rate / frame_size

# Codec parameters
n_codebooks: 4
codebook_size: 1024
codebook_dim: 128
latent_dim: 512

# Token Model parameters
d_model: 256
d_state: 16                # Mamba state dimension
d_conv: 4                  # Mamba conv kernel
expand: 2                  # Mamba expansion factor
n_layers: 6
context_length: 10         # Token context window (frames)

# Speaker parameters
d_speaker: 192
```

### 1.2 Rust Constants (`tmrvc-engine-rs/src/constants.rs`)

```rust
// Auto-generated from configs/constants.yaml — DO NOT EDIT
pub const SAMPLE_RATE: usize = 24000;
pub const FRAME_SIZE: usize = 480;
pub const FRAME_RATE: usize = 50;

pub const N_CODEBOOKS: usize = 4;
pub const CODEBOOK_SIZE: usize = 1024;
pub const CODEBOOK_DIM: usize = 128;
pub const LATENT_DIM: usize = 512;

// Token Model (Transformer)
pub const D_MODEL: usize = 256;
pub const N_HEADS: usize = 4;
pub const HEAD_DIM: usize = D_MODEL / N_HEADS;  // 64
pub const N_LAYERS: usize = 6;
pub const CONTEXT_LENGTH: usize = 10;

pub const D_SPEAKER: usize = 192;

// State sizes (in f32 elements)
pub const CODEC_ENCODER_STATE_SIZE: usize = 16384;   // 1 * 512 * 32
pub const CODEC_DECODER_STATE_SIZE: usize = 8192;    // 1 * 256 * 32
pub const KV_CACHE_SIZE: usize = N_LAYERS * 2 * N_HEADS * CONTEXT_LENGTH * HEAD_DIM;  // 30720
pub const TOTAL_STATE_SIZE: usize = 
    CODEC_ENCODER_STATE_SIZE + KV_CACHE_SIZE + CODEC_DECODER_STATE_SIZE;  // 55196 (~216KB)
```

---

## 2. 3 モデルの I/O 仕様

### 2.1 一覧表

| Model | File | Inputs | Outputs | Execution |
|---|---|---|---|---|
| **codec_encoder** | `codec_encoder.onnx` | audio_frame, state_in | tokens, state_out | Per-frame (20ms) |
| **token_model** | `token_model.onnx` | tokens_in, spk_embed, state_in | logits, state_out | Per-frame (20ms) |
| **codec_decoder** | `codec_decoder.onnx` | tokens, state_in | audio_frame, state_out | Per-frame (20ms) |
| **speaker_encoder** | `speaker_encoder.onnx` | mel_ref | spk_embed | Offline only |

---

### 2.2 codec_encoder

音声フレームを離散トークン列に変換する Causal Conv1d Encoder + RVQ。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `audio_frame` | `[1, 1, 480]` | float32 | 20ms 音声フレーム (24kHz) |
| `state_in` | `[1, STATE_DIM, STATE_FRAMES]` | float32 | Causal conv の hidden state |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `tokens` | `[1, 4]` | int64 | 4つの離散トークン (RVQ codebook indices) |
| `state_out` | `[1, STATE_DIM, STATE_FRAMES]` | float32 | 更新された hidden state |

**State Layout:**

```
state_in / state_out:  [1, 512, 32]
  - Encoder conv layers の context
  - 合計 ~16KB
```

---

### 2.3 token_model

Transformer (Causal Self-Attention) による次トークン予測。KV-cache でストリーミング対応。
F0条件付けでピッチ制御対応（歌唱VC用）。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `tokens_in` | `[1, K, L]` | int64 | Context tokens (K=4 codebooks, L=10 frames = 200ms) |
| `spk_embed` | `[1, 192]` | float32 | Speaker embedding (L2 normalized) |
| `f0_condition` | `[1, L, 2]` | float32 | F0 conditioning: [f0_normalized, pitch_shift] per frame |
| `kv_cache_in` | `[12, 1, 4, 10, 64]` | float32 | KV-cache (6 layers × 2 for K/V) |

**Layout Clarification:**

```
tokens_in: [1, 4, 10]
           │  │  └── context_length (10 frames)
           │  └───── n_codebooks (4 codebooks)
           └──────── batch_size

f0_condition: [1, 10, 2]
              │  │   └── [f0_normalized, pitch_shift]
              │  └────── context_length (10 frames)
              └───────── batch_size

Memory layout: [cb0_f0, cb0_f1, ..., cb0_f9, cb1_f0, cb1_f1, ..., cb3_f9]
```

**F0 Conditioning Details:**

```
f0_normalized: log2(f0 / f0_mean)
  - f0_mean は話者ごとの平均F0 (speaker file に保存)
  - 0 = 話者の平均ピッチ、+1 = 1オクターブ上、-1 = 1オクターブ下

pitch_shift: セミトーン単位のピフト
  - +12 = 1オクターブ上、-12 = 1オクターブ下
  - 0 = ピッチシフトなし（元のメロディを維持）

実行時処理:
  f0_target = f0_mean * 2^(f0_normalized + pitch_shift/12)
```

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `logits` | `[1, 4, 1024]` | float32 | 次トークンの確率分布 (logits) |
| `kv_cache_out` | `[12, 1, 4, 10, 64]` | float32 | 更新された KV-cache |

**KV-Cache Layout:**

```
kv_cache_in / kv_cache_out:  [12, 1, 4, 10, 64]
  - 12 = n_layers(6) × 2 (K and V)
  - 1 = batch size
  - 4 = n_heads
  - 10 = context_length
  - 64 = head_dim (d_model/n_heads = 256/4)
  - 合計 ~120KB
```

**Sampling:**

```rust
// Rust側で softmax + sampling を実行
fn sample_tokens(logits: &[f32], temperature: f32, top_k: usize) -> [i64; 4] {
    let mut tokens = [0i64; 4];
    for cb in 0..4 {
        let cb_logits = &logits[cb * 1024..(cb + 1) * 1024];
        let probs = softmax(&cb_logits.map(|x| x / temperature));
        tokens[cb] = sample_top_k(&probs, top_k);
    }
    tokens
}
```

**F0 Extraction (Rust側):**

```rust
// Streaming F0 extraction using CREPE-lite or PYIN
struct F0Tracker {
    f0_buffer: Vec<f32>,
    f0_mean: f32,  // from .tmrvc_speaker
}

impl F0Tracker {
    fn process_frame(&mut self, frame: &[f32], pitch_shift: f32) -> [f32; 2] {
        let f0 = self.detect_f0(frame);  // CREPE-lite
        let f0_norm = (f0 / self.f0_mean).log2();
        [f0_norm, pitch_shift]
    }
}
```
tokens_in: [1, 4, 10]
           │  │  └── context_length (10 frames)
           │  └───── n_codebooks (4 codebooks)
           └──────── batch_size

Memory layout: [cb0_f0, cb0_f1, ..., cb0_f9, cb1_f0, cb1_f1, ..., cb3_f9]
```

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `logits` | `[1, 4, 1024]` | float32 | 次トークンの確率分布 (logits) |
| `kv_cache_out` | `[12, 1, 4, 10, 64]` | float32 | 更新された KV-cache |

**KV-Cache Layout:**

```
kv_cache_in / kv_cache_out:  [12, 1, 4, 10, 64]
  - 12 = n_layers(6) × 2 (K and V)
  - 1 = batch size
  - 4 = n_heads
  - 10 = context_length
  - 64 = head_dim (d_model/n_heads = 256/4)
  - 合計 ~120KB
```

**Sampling:**

```rust
// Rust側で softmax + sampling を実行
fn sample_tokens(logits: &[f32], temperature: f32, top_k: usize) -> [i64; 4] {
    let mut tokens = [0i64; 4];
    for cb in 0..4 {
        let cb_logits = &logits[cb * 1024..(cb + 1) * 1024];
        let probs = softmax(&cb_logits.map(|x| x / temperature));
        tokens[cb] = sample_top_k(&probs, top_k);
    }
    tokens
}
```

---

### 2.4 codec_decoder

離散トークンから音声フレームを復元する RVQ Dequantization + Causal Conv1d Decoder。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `tokens` | `[1, 4]` | int64 | 4つの離散トークン |
| `state_in` | `[1, STATE_DIM, STATE_FRAMES]` | float32 | Causal conv の hidden state |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `audio_frame` | `[1, 1, 480]` | float32 | 20ms 音声フレーム (24kHz) |
| `state_out` | `[1, STATE_DIM, STATE_FRAMES]` | float32 | 更新された hidden state |

**State Layout:**

```
state_in / state_out:  [1, 256, 32]
  - Decoder conv layers の context
  - 合計 ~8KB
```

---

### 2.5 speaker_encoder (Offline)

参照音声から話者埋め込みを抽出。リアルタイム推論では使用しない。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `mel_ref` | `[1, 80, T]` | float32 | 参照音声の log-mel (T frames, 任意長) |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `spk_embed` | `[1, 192]` | float32 | Speaker embedding (L2 normalized) |

---

## 3. Speaker File Format (`.tmrvc_speaker`)

### 3.1 v3 Format (Current)

```json
{
  "version": 3,
  "spk_embed": [192 floats],
  "f0_mean": 220.0,
  "style_embed": [128 floats, optional],
  "reference_tokens": [[4 ints, ...], optional],
  "lora_delta": [15872 floats, optional],
  "voice_source_preset": [8 floats, optional],
  "metadata": {
    "name": "Speaker Name",
    "enrollment_audio": "path/to/ref.wav",
    "enrollment_duration_sec": 30.5,
    "adaptation_level": "standard",
    "created_at": "2026-02-28T12:00:00Z"
  }
}
```

### 3.2 Adaptation Levels

| Level | Fields | Use Case | Min Audio |
|-------|--------|----------|-----------|
| **light** | `spk_embed`, `f0_mean` only | High-speed VC | 3-10 sec |
| **standard** | + `style_embed`, `reference_tokens` | High-quality VC/TTS | 10-30 sec |
| **full** | + `lora_delta` | Character reproduction | 1-5 min + fine-tune |

### 3.3 Field Descriptions

| Field | Shape | Description |
|-------|-------|-------------|
| `spk_embed` | [192] | Speaker timbre embedding (ECAPA-TDNN, L2 normalized) |
| `f0_mean` | scalar | Speaker's average F0 in Hz (for pitch normalization) |
| `style_embed` | [128] | Prosody/style embedding (optional) |
| `reference_tokens` | [T, 4] | Codec tokens from reference audio for in-context (optional) |
| `lora_delta` | [15872] | LoRA adapter weights for Token Model (optional) |
| `voice_source_preset` | [8] | Voice source parameters (breathiness, tension, etc.) |

### 3.4 Binary Format

```
Offset   Size (bytes)    Field
──────   ────────────    ──────────────────────────
0x0000   4               Magic: "TMSP" (0x544D5350)
0x0004   4               Version: uint32_le = 3
0x0008   4               flags: uint32_le (bit 0: has_style, bit 1: has_ref_tokens, bit 2: has_lora)
0x000C   4               spk_embed_size: uint32_le = 192
0x0010   4               f0_mean: float32_le
0x0014   4               style_embed_size: uint32_le = 128 (0 if not present)
0x0018   4               ref_tokens_frames: uint32_le (0 if not present)
0x001C   4               lora_size: uint32_le = 15872 (0 if not present)
0x0020   4               metadata_size: uint32_le
0x0024   768             spk_embed: float32_le[192]
0x0324   4               f0_mean: float32_le (redundant, for alignment)
0x0328   512             style_embed: float32_le[128] (if present)
         N*16            reference_tokens: int32_le[N, 4] (if present)
         63488           lora_delta: float32_le[15872] (if present)
         metadata_size   metadata_json: UTF-8 JSON
         32              checksum: SHA-256
```

### 3.5 In-Context Usage

When `reference_tokens` is present, the Token Model can use it for in-context learning:

```
1. Load reference_tokens [T_ref, 4] from .tmrvc_speaker
2. Pre-fill KV-Cache with reference_tokens
3. Start streaming inference with spk_embed conditioning
```

This enables zero-shot speaker adaptation without fine-tuning.

---

## 4. Streaming State Management

### 4.1 Ping-Pong Double Buffering

```
┌─────────────────────────────────────────────────────────┐
│  Frame N                                                │
│                                                         │
│  Read:  encoder_state_A, mamba_state_A, decoder_state_A│
│  Write: encoder_state_B, mamba_state_B, decoder_state_B│
│                                                         │
│  After frame: swap A ↔ B                               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Total State Size

| Component | Elements | Bytes |
|-----------|----------|-------|
| Codec Encoder State | 16,384 | 64 KB |
| Token Context Buffer | 40 | 160 B |
| KV-Cache (6 layers × 2 × 4 × 10 × 64) | 30,720 | 120 KB |
| Codec Decoder State | 8,192 | 32 KB |
| **Total** | **55,336** | **~216 KB** |

> 1MB 以下を維持。事前確保で RT-safe。

---

## 5. 数値パリティ基準

### 5.1 許容誤差

| Tensor Type | Metric | Threshold |
|-------------|--------|-----------|
| Float output | L∞ norm | < 1e-4 |
| Float output | Cosine similarity | > 0.999 |
| Int64 output | Exact match | 100% |

### 5.2 検証スクリプト

```bash
uv run python -m tmrvc_export.verify_parity \
  --codec-encoder models/fp32/codec_encoder.onnx \
  --token-model models/fp32/token_model.onnx \
  --codec-decoder models/fp32/codec_decoder.onnx
```

---

## 6. 設計整合性チェックリスト

- [ ] codec_encoder: audio_frame[480] → tokens[4]
- [ ] token_model: tokens[1,4,L] + spk_embed[192] + f0_condition[1,L,2] + kv_cache → logits[4,1024]
- [ ] codec_decoder: tokens[4] → audio_frame[480]
- [ ] KV-cache が固定サイズ [12, 1, 4, 10, 64]
- [ ] Total state < 300KB (現在 ~216KB)
- [ ] Frame size (480 samples = 20ms) が streaming pipeline と整合
- [ ] Token sampling (softmax + top-k) が Rust 側で実装済み
- [ ] .tmrvc_speaker v3 format: spk_embed + f0_mean + optional (style_embed, reference_tokens, lora_delta)
- [ ] In-context learning: reference_tokens で KV-Cache pre-fill 可能
- [ ] Adaptation levels: light / standard / full が実装済み
- [ ] F0 extraction: Rust側で CREPE-lite または PYIN 実装
- [ ] F0 normalization: f0_mean from .tmrvc_speaker 使用
