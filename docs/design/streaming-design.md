# TMRVC Streaming Design & Latency Budget (Codec-Latent Pipeline)

Kojiro Tanaka — streaming design
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-28 — Codec-Latent パラダイムに一本化

> **Target:** End-to-end latency ≤ 50ms (DAW buffer 込み)、CPU-only。
> Token-based streaming で実現。

---

## 1. 基本パラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| **内部サンプルレート** | 24,000 Hz | モデル処理はすべてこのレートで実行 |
| **Frame size** | 480 samples (20ms) | 1 frame = 1 処理単位 |
| **Frame rate** | 50 Hz | 1秒あたり50フレーム |
| **Tokens per frame** | 4 | RVQ (4 codebooks) |
| **Token rate** | 200 tokens/sec | 4 × 50 |

---

## 2. Token-Based Streaming Pipeline

### 2.1 Frame Processing Flow

```cpp
// 20ms frame processing
void processFrame(float* input_480, float* output_480) {
    // 1. Codec Encoder: audio → tokens
    int tokens_in[4];
    codec_encoder_->Run(input_480, tokens_in);
    // Latency: ~5ms
    
    // 2. Update token context buffer
    token_buffer_.push(tokens_in);
    // Now: [t-K, ..., t-1, t] where K=10
    
    // 3. Token Model: generate next tokens
    int tokens_out[4];
    token_model_->Run(
        token_buffer_.context(),    // [K, 4] tokens
        spk_embed_,                 // [192]
        mamba_state_in_,            // in
        tokens_out,                 // out
        mamba_state_out_            // out
    );
    // Latency: ~10ms (Mamba inference)
    
    // 4. Codec Decoder: tokens → audio
    codec_decoder_->Run(tokens_out, output_480);
    // Latency: ~5ms
}
```

### 2.2 System Pipeline

```
processBlock(buffer, numSamples)
│
├─ 1. Downsample: 48kHz → 24kHz (polyphase)
│
├─ 2. Input Ring Buffer に書き込み
│
├─ 3. while (inputRing.available() >= 480)  ← Frame trigger
│     │
│     ├─ 3a. inputRing から 480 samples を読み出し
│     │
│     ├─ 3b. Codec Encoder → tokens[4]
│     │
│     ├─ 3c. Token Buffer 更新
│     │
│     ├─ 3d. Token Model (Mamba) → next_tokens[4]
│     │      + Mamba state update
│     │
│     ├─ 3e. Codec Decoder → audio[480]
│     │
│     └─ 3f. Output Ring Buffer に書き込み
│
├─ 4. Output Ring Buffer から numOutputSamples 読み出し
│
├─ 5. Upsample: 24kHz → 48kHz (polyphase)
│
└─ 6. buffer に書き戻し
```

---

## 3. レイテンシバジェット

### 3.1 Latency Breakdown

| 構成要素 | 時間 | 備考 |
|---------|------|------|
| DAW Input Buffer | 2.7-5.8ms | 128-256 samples @ 48kHz |
| Resample in | ~0 | polyphase, negligible |
| **Codec Encoder** | **5ms** | Conv1d encoder + RVQ |
| **Token Model** | **10ms** | Mamba inference |
| **Codec Decoder** | **5ms** | RVQ lookup + Conv1d decoder |
| **Algorithmic** | **20ms** | frame size (inherent) |
| Resample out | ~0 | polyphase, negligible |
| DAW Output Buffer | 2.7-5.8ms | 128-256 samples @ 48kHz |
| **Total (nominal)** | **~45-50ms** | |

### 3.2 Latency Configuration Table

| 構成 | DAW SR | Buffer Size | Input | Algo | Process | Output | **Total** |
|---|---|---|---|---|---|---|---|
| **Best case** | 48kHz | 128 | 2.67ms | 20ms | 20ms | 2.67ms | **~45ms** |
| **Nominal** | 48kHz | 256 | 5.33ms | 20ms | 20ms | 5.33ms | **~51ms** |
| **Worst case** | 44.1kHz | 256 | 5.80ms | 20ms | 20ms | 5.80ms | **~52ms** |

> Worst case でも 55ms 以下を維持。

---

## 4. State Management

### 4.1 State Components

```
Token-Based State Layout:
┌────────────────────────────────────────────────────┐
│  Codec Encoder State                                │
│  - Conv1d context: ~16KB                           │
├────────────────────────────────────────────────────┤
│  Token Buffer (Context Window)                      │
│  - [K, 4] tokens: K=10 → 40 ints → 160B           │
├────────────────────────────────────────────────────┤
│  Mamba State                                        │
│  - Per-layer: [d_inner, d_state]                   │
│  - 6 layers, d_inner=512, d_state=16 → ~200KB     │
├────────────────────────────────────────────────────┤
│  Codec Decoder State                                │
│  - Conv1d context: ~8KB                            │
└────────────────────────────────────────────────────┘
Total: ~224KB (fixed, pre-allocated)
```

### 4.2 Ring Buffer Sizes

| Ring Buffer | Capacity | 根拠 |
|---|---|---|
| **Input Ring** | 2048 samples | frame (480) + 余裕。最大 ~85ms 分 |
| **Output Ring** | 2048 samples | frame (480) + 数フレーム分のバッファ |

### 4.3 Memory Allocation

すべての State と Ring Buffer は構築時に固定サイズで確保。
Audio thread 内で `malloc` / `free` は **一切呼ばない**。

---

## 5. Sample Rate Conversion

### 5.1 48kHz ↔ 24kHz (整数比 2:1)

```
Downsample (48→24): 2:1 polyphase decimation
  - FIR lowpass filter (cutoff 12kHz)
  - Filter order: 48 taps (24 per phase)
  - 1 output sample ごとに 24 multiply-add

Upsample (24→48): 1:2 polyphase interpolation
  - 同じ FIR フィルタの polyphase 分解
  - Zero-stuffing + filtering
```

### 5.2 44.1kHz ↔ 24kHz (有理数比 147:80)

```
Rational polyphase resampler:
  44100 / 24000 = 441/240 = 147/80
```

---

## 6. Graceful Degradation

### 6.1 Inference Time 監視

```cpp
struct FrameTimingStats {
    float lastFrameMs;       // 直近 frame の処理時間
    float avgFrameMs;        // 移動平均 (過去 100 frames)
    float maxFrameMs;        // 過去 100 frames の最大値
    int overrunCount;        // frame_time (20ms) 超過回数
};
```

### 6.2 過負荷時の対処

| 条件 | アクション |
|---|---|
| `lastFrameMs > frameTimeMs` (20ms 超過) | overrunCount++、ログ記録 |
| `overrunCount > 3` (連続超過) | Token model のコンテキスト長を短縮 |
| `overrunCount > 10` (深刻な過負荷) | Dry bypass モードに切り替え |
| `avgFrameMs < frameTimeMs × 0.8` (回復) | 通常モードに復帰 |

---

## 7. スレッドモデル

### 7.1 概要

```
┌─────────────────────────────────────────────────┐
│  Audio Thread (RT priority)                      │
│                                                  │
│  ✓ Ring Buffer read/write                       │
│  ✓ Codec Encoder/Decoder inference              │
│  ✓ Token Model inference                        │
│  ✓ Resampling                                   │
│                                                  │
│  ✗ NO malloc / free                             │
│  ✗ NO mutex lock                                │
│  ✗ NO file I/O                                  │
└─────────────────────────────────────────────────┘
        │                           ▲
        │  SPSC Queue (commands)    │  SPSC Queue (responses)
        ▼                           │
┌─────────────────────────────────────────────────┐
│  Worker Thread                                   │
│                                                  │
│  • Speaker file loading (.tmrvc_speaker)        │
│  • Model loading (ONNX session creation)        │
│  • Few-shot enrollment computation              │
└─────────────────────────────────────────────────┘
```

---

## 8. DAW Latency Reporting

### 8.1 setLatencySamples 算出式

```cpp
int TMRVCProcessor::getLatencyInSamples() const {
    // Algorithmic latency: 1 frame (20ms)
    const double algoLatencySec = kFrameSize / kInternalSampleRate;
    // 20ms in samples at DAW sample rate
    return static_cast<int>(std::round(algoLatencySec * currentSampleRate_));
}
```

| DAW SR | Latency (samples) | Latency (ms) |
|---|---|---|
| 48,000 Hz | 960 | 20.0ms |
| 44,100 Hz | 882 | 20.0ms |
| 96,000 Hz | 1920 | 20.0ms |

---

## 9. 設計整合性チェックリスト

- [x] 全構成でレイテンシ 55ms 以下
- [x] Inference time (~20ms) ≈ frame time (20ms) — マージンあり
- [x] Audio thread は RT-safe (no malloc, no lock, no I/O)
- [x] Ring Buffer は pre-allocated、固定サイズ
- [x] Worker thread との通信は lock-free SPSC queue
- [x] DAW latency reporting は algorithmic latency のみ (20ms)
- [x] 全コンポーネントが causal (look-ahead = 0)
- [x] Mamba state が固定サイズ (~200KB) で事前確保可能
- [x] Total state < 1MB (CPU real-time 実現可能)
