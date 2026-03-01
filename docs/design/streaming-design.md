# TMRVC Streaming Design & Latency Budget (UCLM v2)

Kojiro Tanaka — streaming design
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-03-01 — UCLM Token Spec v2 を反映

> **Target:** End-to-end latency ≤ 50ms (DAW buffer 込み)、CPU-only。
> Dual-stream token-based streaming (`A_t` + `B_t`) で実現。
> **Sync:** canonical token contract は `emotion-aware-codec.md` / `onnx-contract.md` を参照。

---

## 1. 基本パラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| **内部サンプルレート** | 24,000 Hz | モデル処理はすべてこのレートで実行 |
| **Frame size** | 240 samples (10ms) | 1 frame = 1 処理単位 |
| **Frame rate** | 100 Hz | 1秒あたり100フレーム |
| **Acoustic tokens per frame** | 8 | RVQ (8 codebooks) |
| **Control tokens per frame** | 4 | `[op, type, dur, int]` |
| **Acoustic token rate** | 800 tokens/sec | 8 × 100 |
| **Control token rate** | 400 tokens/sec | 4 × 100 |

---

## 2. Token-Based Streaming Pipeline

### 2.1 Frame Processing Flow

```cpp
// 10ms frame processing (UCLM v2)
void processFrame(float* input_240, float* output_240) {
    // 1. Codec Encoder: audio -> acoustic tokens A_t
    int a_src[8];
    codec_encoder_->Run(input_240, a_src);
    // Latency: ~5ms
    
    // 2. Update rolling context buffers (A/B)
    ctxA_.push(a_src);
    
    // 3. UCLM core: generate target A_t and B_t
    int a_tgt[8];
    int b_tgt[4];
    uclm_core_->Run(
        ctxA_.context(),            // [K, 8]
        ctxB_.context(),            // [K, 4]
        spk_embed_,                 // [192]
        voice_state_,               // [8]
        delta_voice_state_,         // [8]
        kv_cache_in_,               // in
        a_tgt, b_tgt,               // out
        kv_cache_out_               // out
    );
    // Latency: ~8-10ms (core inference)
    
    // 4. Codec Decoder: A_t/B_t -> audio
    codec_decoder_->Run(a_tgt, b_tgt, voice_state_, event_trace_in_, output_240);
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
├─ 3. while (inputRing.available() >= 240)  ← Frame trigger
│     │
│     ├─ 3a. inputRing から 240 samples を読み出し
│     │
│     ├─ 3b. Codec Encoder → A_t[8]
│     │
│     ├─ 3c. Context Buffer (A/B) 更新
│     │
│     ├─ 3d. UCLM Core → next_A_t[8], next_B_t[4]
│     │      + KV cache update
│     │
│     ├─ 3e. Codec Decoder → audio[240]
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
| **UCLM Core** | **10ms** | Transformer inference (KV cache) |
| **Codec Decoder** | **5ms** | RVQ lookup + Conv1d decoder |
| **Algorithmic** | **10ms** | frame size (inherent) |
| Resample out | ~0 | polyphase, negligible |
| DAW Output Buffer | 2.7-5.8ms | 128-256 samples @ 48kHz |
| **Total (nominal)** | **~45-50ms** | |

### 3.2 Latency Configuration Table

| 構成 | DAW SR | Buffer Size | Input | Algo | Process | Output | **Total** |
|---|---|---|---|---|---|---|---|
| **Best case** | 48kHz | 128 | 2.67ms | 10ms | 18ms | 2.67ms | **~33ms** |
| **Nominal** | 48kHz | 256 | 5.33ms | 10ms | 18ms | 5.33ms | **~39ms** |
| **Worst case** | 44.1kHz | 256 | 5.80ms | 10ms | 20ms | 5.80ms | **~42ms** |

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
    int overrunCount;        // frame_time (10ms) 超過回数
};
```

### 6.2 過負荷時の対処

| 条件 | アクション |
|---|---|
| `lastFrameMs > frameTimeMs` (10ms 超過) | overrunCount++、ログ記録 |
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
    // Algorithmic latency: 1 frame (10ms)
    const double algoLatencySec = kFrameSize / kInternalSampleRate;
    // 10ms in samples at DAW sample rate
    return static_cast<int>(std::round(algoLatencySec * currentSampleRate_));
}
```

| DAW SR | Latency (samples) | Latency (ms) |
|---|---|---|
| 48,000 Hz | 480 | 10.0ms |
| 44,100 Hz | 441 | 10.0ms |
| 96,000 Hz | 960 | 10.0ms |

---

## 9. 設計整合性チェックリスト

- [x] 全構成でレイテンシ 55ms 以下
- [x] Inference time (~14-20ms) と 10ms frame の並列処理戦略を定義
- [x] Audio thread は RT-safe (no malloc, no lock, no I/O)
- [x] Ring Buffer は pre-allocated、固定サイズ
- [x] Worker thread との通信は lock-free SPSC queue
- [x] DAW latency reporting は algorithmic latency のみ (10ms)
- [x] 全コンポーネントが causal (look-ahead = 0)
- [x] Mamba state が固定サイズ (~200KB) で事前確保可能
- [x] Total state < 1MB (CPU real-time 実現可能)
