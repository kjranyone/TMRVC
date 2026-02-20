# TMRVC Streaming Design & Latency Budget

Kojiro Tanaka — streaming design
Created: 2026-02-16 (Asia/Tokyo)

> **Target:** End-to-end latency ≤ 50ms (DAW buffer 込み)、CPU-only。
> Frame-by-frame causal streaming で実現。

---

## 1. 基本パラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| **内部サンプルレート** | 24,000 Hz | モデル処理はすべてこのレートで実行 |
| **hop_length** | 240 samples (10ms) | 1 frame = 1 hop。最小処理単位 |
| **n_fft** | 1024 | 周波数ビン数: 513 (n_fft/2 + 1) |
| **window_length** | 960 samples (40ms) | Hann 窓。n_fft より短く zero-padding |
| **n_mels** | 80 | Mel フィルタバンク数 |
| **mel_fmin** | 0 Hz | |
| **mel_fmax** | 12,000 Hz | Nyquist (24kHz / 2) |
| **Causal windowing** | 右端 = 現在フレーム | 過去コンテキストのみ使用 (look-ahead = 0) |

### Causal Window の配置

```
Time ────────────────────────────────▶
                                      │ current sample
         ┌───── window (960 samples) ─────┐
         │  past context   │  current hop  │
         │  720 samples    │  240 samples  │
         └─────────────────┴───────────────┘
                                      │
                        ┌── zero-pad to n_fft ──┐
                        │ 960 samples + 64 zeros │
                        └────────────────────────┘
```

- 窓の右端が現在の hop 位置に一致
- 過去 720 samples (30ms) + 現在 hop 240 samples (10ms) = 窓長 960 samples (40ms)
- n_fft = 1024 へは右側 zero-padding (64 samples)

---

## 2. レイテンシバジェット

### 2.1 レイテンシ構成要素

```
Total Latency = DAW Input Buffer + Resample + Algorithmic + Inference + Resample + DAW Output Buffer

  DAW Input Buffer:  DAW buffer size / DAW sample rate
  Resample (in):     ~0 (polyphase, negligible)
  Algorithmic:       hop_length / internal_sr = 240/24000 = 10ms
  Inference:         content_enc + converter + vocoder ≈ 3-5ms
  Resample (out):    ~0 (polyphase, negligible)
  DAW Output Buffer: DAW buffer size / DAW sample rate
```

### 2.2 レイテンシバジェット表

| 構成 | DAW SR | Buffer Size | Input Buffer | Algo | Inference | Output Buffer | **Total** |
|---|---|---|---|---|---|---|---|
| **Best case** | 48,000 Hz | 128 samples | 2.67ms | 10ms | 4ms | 2.67ms | **~19ms** |
| **Nominal** | 48,000 Hz | 256 samples | 5.33ms | 10ms | 4ms | 5.33ms | **~25ms** |
| **Worst case** | 44,100 Hz | 256 samples | 5.80ms | 10ms | 4ms | 5.80ms | **~26ms** |
| **Stress case** | 44,100 Hz | 512 samples | 11.61ms | 10ms | 4ms | 11.61ms | **~37ms** |

> 全構成で **50ms 以下**を達成。Stress case でも十分なマージンがある。

### 2.3 Inference 内訳 (per-frame target)

| Model | Target | Note |
|---|---|---|
| Causal STFT + Mel | 0.2ms | FFT + mel filterbank (固定演算) |
| content_encoder | 1.0ms | ~1.5-3M params, causal ConvNeXt |
| converter (1-step) | 2.0ms | ~3-5M params, causal CNN + FiLM |
| vocoder | 0.5ms | ~0.3-3M params, iSTFT-based |
| iSTFT + OLA | 0.1ms | 固定演算 |
| ir_estimator (amortized) | 0.2ms | ~1-3M params, 10 frame ごと → 0.2ms/frame |
| **合計** | **~4.0ms** | 10ms hop 内で完了 (utilization ~40%) |

---

## 3. Audio Thread パイプライン

### 3.1 processBlock 内の信号フロー

```
processBlock(buffer, numSamples)
│
├─ 1. Downsample: 48kHz → 24kHz (polyphase)
│     numSamples=256 → ~128 output samples
│
├─ 2. Input Ring Buffer に書き込み
│
├─ 3. while (inputRing.available() >= hopSize)  ← STFT トリガー
│     │
│     ├─ 3a. inputRing から hopSize (240) samples を読み出し
│     │      (窓関数適用時は past context 含め 960 samples 参照)
│     │
│     ├─ 3b. Causal STFT → mel_frame[1, 80, 1]
│     │
│     ├─ 3c. F0 estimation (YIN-based, causal)
│     │      → f0_frame[1, 1, 1]
│     │
│     ├─ 3d. content_encoder.Run(mel_frame, f0, state_in)
│     │      → content[1, 256, 1], state_out
│     │
│     ├─ 3e. (every 10 frames) ir_estimator.Run(mel_accumulated)
│     │      → acoustic_params[1, 32]  (cached between runs)
│     │      (24 IR params + 8 voice source params)
│     │
│     ├─ 3e'. Voice Source Blend (if preset loaded):
│     │      acoustic_params[24..31] = lerp(estimated, preset, α)
│     │      (RT-safe: stack copy of [f32; 32] + 8 mul-add)
│     │
│     ├─ 3f. converter.Run(content, spk_embed, acoustic_params, state_in)
│     │      → pred_features[1, F, 1], state_out
│     │
│     ├─ 3g. vocoder.Run(pred_features, state_in)
│     │      → stft_mag[1, 513, 1], stft_phase[1, 513, 1], state_out
│     │
│     ├─ 3h. iSTFT (mag, phase) → hop samples (240)
│     │
│     ├─ 3i. Overlap-Add → Output Ring Buffer
│     │
│     └─ 3j. state ping-pong swap (A ↔ B)
│
├─ 4. Output Ring Buffer から numOutputSamples 読み出し
│
├─ 5. Upsample: 24kHz → 48kHz (polyphase)
│
├─ 6. Dry/Wet Mix + Output Gain
│
└─ 7. buffer に書き戻し
```

### 3.2 STFT トリガーロジック

```cpp
// processBlock 内
while (inputRing_.available() >= kHopSize) {
    // 1 frame 分の処理を実行
    processOneFrame();
}
```

DAW buffer size が hop_length の整数倍でない場合にも対応:
- 48kHz / 256 samples → 24kHz で ~128 samples → hop 240 の 0.53 倍
- 2 回の processBlock で ~256 samples → 1 frame 処理 + 余り 16 samples
- 余りは Ring Buffer に蓄積され、次回以降の processBlock で消費

### 3.3 Output Pre-fill

初期化時に Output Ring Buffer に 1 hop 分 (240 samples) の silence を pre-fill する。

```cpp
void StreamingEngine::initialize() {
    // Pre-fill output ring with silence (1 hop = 10ms)
    float silence[kHopSize] = {0};
    outputRing_.write(silence, kHopSize);
}
```

**目的:**
- 初回 inference spike の吸収 (初回実行は JIT warmup で遅い場合がある)
- Output Ring Buffer underrun 防止
- 追加レイテンシ: +10ms (レイテンシバジェットの algorithmic latency に含まれる)

---

## 4. Ring Buffer 設計

### 4.1 FixedRingBuffer

```cpp
template<typename T, size_t Capacity>
class FixedRingBuffer {
    alignas(64) T buffer_[Capacity];  // Cache-line aligned
    size_t readPos_ = 0;
    size_t writePos_ = 0;

public:
    size_t available() const;         // 読み取り可能サンプル数
    size_t freeSpace() const;         // 書き込み可能サンプル数
    void write(const T* data, size_t n);
    void read(T* data, size_t n);
    void peek(T* data, size_t n, size_t offset = 0) const;  // 読み取り位置を進めずに参照
};
```

### 4.2 バッファサイズ

| Ring Buffer | Capacity | 根拠 |
|---|---|---|
| **Input Ring** | 2048 samples | 窓長 (960) + 余裕。最大 ~85ms 分 |
| **Output Ring** | 2048 samples | pre-fill (240) + 数フレーム分のバッファ |
| **STFT Context Buffer** | 960 samples | 窓長分。causal windowing 用の過去コンテキスト保持 |

### 4.3 メモリ確保

すべての Ring Buffer は構築時に固定サイズで確保 (pre-allocated)。
Audio thread 内で `malloc` / `free` は **一切呼ばない**。

```
Total Ring Buffer Memory:
  Input:   2048 × 4 bytes =  8 KB
  Output:  2048 × 4 bytes =  8 KB
  Context:  960 × 4 bytes ≈  4 KB
  ────────────────────────────────
  Total:                   ≈ 20 KB
```

---

## 5. Sample Rate Conversion

### 5.1 48kHz ↔ 24kHz (整数比 2:1)

```
Downsample (48→24): 2:1 polyphase decimation
  - FIR lowpass filter (cutoff 12kHz, transition band 12-16kHz)
  - Filter order: 48 taps (24 per phase)
  - Polyphase decomposition: 2 phases
  - 1 output sample ごとに 24 multiply-add

Upsample (24→48): 1:2 polyphase interpolation
  - 同じ FIR フィルタの polyphase 分解
  - Zero-stuffing + filtering
  - 1 output sample ごとに 24 multiply-add
```

### 5.2 44.1kHz ↔ 24kHz (有理数比 147:80)

```
Rational polyphase resampler:
  44100 / 24000 = 441/240 = 147/80

Downsample (44.1→24):
  - L=80, M=147 rational resampler
  - Upsample by 80, filter, downsample by 147
  - Polyphase implementation: 80 phases × ~32 taps each
  - 実効フィルタ長: ~2560 taps (polyphase で分散)

Upsample (24→44.1):
  - L=147, M=80 rational resampler
  - 147 phases × ~32 taps each
```

### 5.3 PolyphaseResampler API

```cpp
class PolyphaseResampler {
public:
    PolyphaseResampler(int srcRate, int dstRate, int filterOrder = 48);

    // pre-allocated output buffer に書き込み
    // 戻り値: 実際に書き込まれた output samples 数
    int process(const float* input, int numInputSamples,
                float* output, int maxOutputSamples);

    int getMaxOutputSamples(int numInputSamples) const;
    void reset();
};
```

---

## 6. Graceful Degradation

### 6.1 Inference Time 監視

```cpp
struct FrameTimingStats {
    float lastFrameMs;       // 直近 frame の処理時間
    float avgFrameMs;        // 移動平均 (過去 100 frames)
    float maxFrameMs;        // 過去 100 frames の最大値
    int overrunCount;        // hop_time (10ms) 超過回数
};
```

毎フレーム `std::chrono::steady_clock` で計測。

### 6.2 過負荷時の対処

| 条件 | アクション |
|---|---|
| `lastFrameMs > hopTimeMs` (10ms 超過) | overrunCount++ 、ログ記録 |
| `overrunCount > 3` (連続超過) | IR estimator を一時停止 (キャッシュ値を継続使用) |
| `overrunCount > 10` (深刻な過負荷) | Dry bypass モードに切り替え (入力をそのまま出力) |
| `avgFrameMs < hopTimeMs × 0.8` (回復) | 通常モードに復帰 |

### 6.3 NaN/Inf ガード

```cpp
// 各モデル出力後にチェック
if (containsNanOrInf(outputTensor)) {
    // 前フレームの出力を再利用
    std::copy(prevOutput_, prevOutput_ + kHopSize, currentOutput_);
    overrunCount_++;
    LOG_WARN("NaN/Inf detected in model output, using previous frame");
}
```

---

## 7. スレッドモデル

### 7.1 概要

```
┌─────────────────────────────────────────────────┐
│  Audio Thread (RT priority)                      │
│                                                  │
│  ✓ Ring Buffer read/write                       │
│  ✓ STFT / mel computation                       │
│  ✓ ONNX Runtime inference (per-frame models)    │
│  ✓ iSTFT + overlap-add                          │
│  ✓ Resampling                                   │
│                                                  │
│  ✗ NO malloc / free                             │
│  ✗ NO mutex lock                                │
│  ✗ NO file I/O                                  │
│  ✗ NO system calls (beyond clock)               │
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
│  • LoRA weight merging                          │
│  • IR calibration (startup)                     │
│  • Timing stats reporting                       │
└─────────────────────────────────────────────────┘
```

### 7.2 RT-safe 制約

Audio thread で禁止される操作:

| 禁止事項 | 理由 | 代替手段 |
|---|---|---|
| `malloc` / `free` / `new` / `delete` | Priority inversion (OS allocator lock) | Pre-allocated buffers (TensorPool) |
| `std::mutex::lock` | Priority inversion | Lock-free SPSC queue |
| File I/O (`fopen`, `read`, etc.) | Unbounded blocking | Worker thread に委譲 |
| `std::cout` / logging | I/O blocking | Lock-free ring buffer ログ (best-effort) |
| ONNX Runtime session creation | Memory allocation | Worker thread で事前作成 |
| Exception throwing | Stack unwinding cost | Error codes / flags |

### 7.3 Lock-free SPSC Queue

Audio thread ↔ Worker thread 間の通信:

```cpp
template<typename T, size_t Capacity>
class SPSCQueue {
    // Single-Producer Single-Consumer lock-free queue
    // Based on Lamport's algorithm with cache-line padding

    alignas(64) std::atomic<size_t> readPos_{0};
    alignas(64) std::atomic<size_t> writePos_{0};
    alignas(64) T buffer_[Capacity];

public:
    bool tryPush(const T& item);  // Producer (non-blocking)
    bool tryPop(T& item);         // Consumer (non-blocking)
};
```

### 7.4 Command / Response Protocol

```cpp
// Audio thread → Worker thread
enum class Command : uint8_t {
    LoadSpeaker,       // path to .tmrvc_speaker
    LoadModel,         // path to .onnx directory
    CalibrateIR,       // run IR calibration
    Shutdown,          // graceful shutdown
};

struct CommandMessage {
    Command type;
    char payload[256]; // file path, etc.
};

// Worker thread → Audio thread
enum class Response : uint8_t {
    SpeakerReady,      // speaker data loaded into staging slot
    ModelReady,        // model loaded into staging slot
    IRCalibrated,      // IR params ready
    Error,             // error description
};

struct ResponseMessage {
    Response type;
    char payload[256]; // error message, etc.
};
```

Audio thread は毎 processBlock 冒頭で Response queue を non-blocking poll:

```cpp
ResponseMessage resp;
while (responseQueue_.tryPop(resp)) {
    switch (resp.type) {
        case Response::SpeakerReady:
            speakerManager_.swapToStaging();  // atomic pointer swap
            break;
        case Response::ModelReady:
            modelSlot_.swap();                // atomic pointer swap
            break;
        // ...
    }
}
```

---

## 8. DAW Latency Reporting

### 8.1 setLatencySamples 算出式

```cpp
int TMRVCProcessor::getLatencyInSamples() const {
    // Algorithmic latency: 1 hop (accumulation) + 1 hop (pre-fill)
    const double algoLatencySec = 2.0 * kHopSize / kInternalSampleRate;
    // 20ms in samples at DAW sample rate
    return static_cast<int>(std::round(algoLatencySec * currentSampleRate_));
}
```

| DAW SR | Latency (samples) | Latency (ms) |
|---|---|---|
| 48,000 Hz | 960 | 20.0ms |
| 44,100 Hz | 882 | 20.0ms |
| 96,000 Hz | 1920 | 20.0ms |

> DAW は `setLatencySamples()` を参照して他トラックとの位相補正 (PDC) を行う。
> 報告するのは algorithmic latency のみ (DAW buffer latency は DAW 側で把握済み)。

### 8.2 内訳

| 要素 | 値 | 説明 |
|---|---|---|
| Input accumulation | 10ms (1 hop) | hop 分のサンプルが溜まるまでの待ち時間 |
| Output pre-fill | 10ms (1 hop) | 初期化時の silence pre-fill |
| Inference time | ~4ms | DAW latency reporting には含めない (processing time) |
| **Reported latency** | **20ms** | |

---

## 9. Overlap-Add (OLA) 出力合成

### 9.1 iSTFT + OLA

```
Frame N-1:    ├──── window (960 samples) ────┤
                                    ├─── hop (240) ───┤
Frame N:           ├──── window (960 samples) ────┤
                                         ├─ hop ─┤
                   ◄──── overlap (720) ────►
```

- vocoder が出力する STFT (mag, phase) → iSTFT で 960 samples の窓付き信号を得る
- 前フレームの出力と 720 samples 重複 → 加算で合成
- hop_length (240 samples) ごとに新しい出力サンプルが確定

### 9.2 OLA Buffer

```cpp
class OverlapAddBuffer {
    alignas(64) float buffer_[kWindowSize];  // 960 samples
    int writePos_ = 0;

public:
    // 新しい窓付き信号を加算し、確定した hop samples を返す
    void addFrame(const float* windowedSignal, float* hopOutput);
    void reset();
};
```

```cpp
void OverlapAddBuffer::addFrame(const float* windowedSignal, float* hopOutput) {
    // 1. 確定した hop 部分を出力にコピー
    std::copy(buffer_, buffer_ + kHopSize, hopOutput);

    // 2. overlap 部分をシフト
    std::memmove(buffer_, buffer_ + kHopSize, (kWindowSize - kHopSize) * sizeof(float));

    // 3. 末尾をゼロクリア
    std::fill(buffer_ + kWindowSize - kHopSize, buffer_ + kWindowSize, 0.0f);

    // 4. 新しいフレームを加算
    for (int i = 0; i < kWindowSize; ++i) {
        buffer_[i] += windowedSignal[i];
    }
}
```

---

## 10. Causal STFT 実装

### 10.1 per-frame Causal STFT

```cpp
void computeCausalSTFT(
    const float* contextBuffer,   // 過去 windowSize samples
    float* stftReal,              // [n_fft/2+1]
    float* stftImag               // [n_fft/2+1]
) {
    alignas(64) float windowed[kNFFT] = {0};

    // 1. 窓関数適用 (Hann window on windowSize samples)
    for (int i = 0; i < kWindowSize; ++i) {
        windowed[i] = contextBuffer[i] * hannWindow_[i];
    }
    // windowed[kWindowSize..kNFFT-1] は zero-padded

    // 2. Real FFT (size = kNFFT = 1024)
    fft_.forward(windowed, stftReal, stftImag);
}
```

### 10.2 per-frame Mel 変換

```cpp
void computeMel(
    const float* stftReal,    // [n_fft/2+1]
    const float* stftImag,    // [n_fft/2+1]
    float* melFrame           // [n_mels]
) {
    // 1. Power spectrum
    alignas(64) float power[kNFFT / 2 + 1];
    for (int i = 0; i <= kNFFT / 2; ++i) {
        power[i] = stftReal[i] * stftReal[i] + stftImag[i] * stftImag[i];
    }

    // 2. Mel filterbank (pre-computed, sparse matrix)
    // melFilterbank_ は [n_mels, n_fft/2+1] sparse matrix
    for (int m = 0; m < kNMels; ++m) {
        float sum = 0.0f;
        for (auto& [bin, weight] : melFilterbank_[m]) {
            sum += weight * power[bin];
        }
        melFrame[m] = std::log(std::max(sum, 1e-10f));  // log-mel
    }
}
```

---

## 11. 設計整合性チェックリスト

- [x] 全 4 構成でレイテンシ 50ms 以下 (Best: ~19ms, Nominal: ~25ms, Worst: ~26ms, Stress: ~37ms)
- [x] Inference time (~4ms) < hop time (10ms) — 十分なマージン
- [x] Audio thread は RT-safe (no malloc, no lock, no I/O)
- [x] Ring Buffer は pre-allocated、固定サイズ
- [x] Worker thread との通信は lock-free SPSC queue
- [x] DAW latency reporting は algorithmic latency のみ (20ms)
- [x] Causal windowing — look-ahead = 0
- [x] 48kHz↔24kHz は整数比 polyphase (効率的)
- [x] 44.1kHz↔24kHz は有理数比 polyphase (正確)
- [x] Graceful degradation: overrun 検知 → IR停止 → bypass
- [x] Voice Source Blend は RT-safe (stack copy + 8 mul-add, zero allocation)

---

## 12. Latency-Quality Spectrum Control

### 12.1 目的

リアルタイム用途では低レイテンシを優先し、品質重視用途ではイントネーション・活舌の再現性を優先できるように、
`Latency-Quality` を連続可変にする。

### 12.2 制御パラメータ

単一ノブ `q` を導入する。`q` は `[0.0, 1.0]` で、`0.0 = Live`、`1.0 = Quality`。

| Parameter | q=0.0 (Live) | q=1.0 (Quality) | 備考 |
|---|---|---|---|
| lookahead_hops | 0 | 6 | 0-60ms (10ms/hop) を連続可変 |
| f0_window_ms | 20 | 80 | 高品質側で F0 安定化 |
| ir_update_interval | 10 frames | 5 frames | 高品質側で IR 追従を強化 |
| converter profile | causal_1step_live | lookahead_1step_hq | モデルを段階的に切替 |
| vocoder profile | vocoder_live | vocoder_hq | 品質側で表現力優先 |

補間例:

```text
lookahead_hops = round(lerp(0, 6, q))
f0_window_ms   = round(lerp(20, 80, q))
```

### 12.3 レイテンシ算出

既存の報告レイテンシ (20ms = input accumulation 10ms + output pre-fill 10ms) に
`lookahead_hops * hop_ms` を加算する。

```text
reported_latency_ms = 20 + lookahead_hops * 10
```

例:

- `q=0.0` (`lookahead=0`) -> `20ms`
- `q=0.5` (`lookahead=3`) -> `50ms`
- `q=1.0` (`lookahead=6`) -> `80ms`

### 12.4 推論パイプライン拡張

`processOneFrame()` で現在フレームのみ使うのではなく、`lookahead_hops > 0` の場合は
リングバッファ先読みで未来コンテキストを取り込み、`content/converter/vocoder` に渡す。

実装要件:

- Audio thread 内で追加 `malloc` は禁止 (先読みバッファは固定長で事前確保)
- `lookahead_hops` 変更は atomic パラメータとして反映
- モデルプロファイル切替は Worker thread でロードし、Audio thread では atomic swap

### 12.5 モード遷移と破綻防止

`q` の変更時に音色・位相破綻を避けるため、`crossfade_ms = 100` の二重実行クロスフェードを行う。

```text
old_profile_out * (1 - a) + new_profile_out * a
```

`a` は 0 -> 1 を 100ms で線形遷移。

### 12.6 過負荷時の自動降格

高品質側で `lastFrameMs > hopTimeMs` が連続した場合は `q` を自動で下げる。

ルール:

- 連続 overrun 3 回: `q = max(0, q - 0.2)`
- 連続 overrun 10 回: `q = 0` + dry bypass 優先

これにより「品質優先で始めて、CPU が足りなければ自動で低遅延側に寄せる」運用を可能にする。

### 12.7 実装ステータス

**定数 (constants.yaml):**

```yaml
max_lookahead_hops: 6          # HQ mode lookahead (60ms)
converter_hq_state_frames: 46  # sum of left_ctx for semi-causal
hq_threshold_q: 0.3            # q > 0.3 で HQ mode 有効
crossfade_frames: 10            # 100ms crossfade on mode switch
```

**実装済みコンポーネント:**

| Component | Status | Details |
|---|---|---|
| `SemiCausalConvNeXtBlock` | Done | `modules.py` — asymmetric padding, state advance by 1 |
| `ConverterStudentHQ` | Done | `converter.py` — T_in=7, T_out=1, state=46, `from_causal()` |
| ONNX export (`converter_hq.onnx`) | Done | `export_onnx.py` — fixed T=7 input |
| Rust `ContentBuffer` | Done | `processor.rs` — circular buffer for 7 content frames |
| Rust HQ pipeline | Done | `processor.rs` — mode selection, crossfade, adaptive degradation |
| Monitor HQ display | Done | `monitor.rs` — Live/HQ mode label in latency display |

**モデルプロファイル切替:**

- `q <= 0.3` → Live mode: causal converter (T=1), 20ms latency
- `q > 0.3` → HQ mode: semi-causal converter (T=7), 80ms latency
- 閾値は `HQ_THRESHOLD_Q = 0.3` (2段階。将来的に連続補間を検討)