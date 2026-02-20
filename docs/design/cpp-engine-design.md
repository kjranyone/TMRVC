# TMRVC C++ Engine Design

Kojiro Tanaka — C++ engine design
Created: 2026-02-16 (Asia/Tokyo)

> **Scope:** tmrvc-engine (JUCE 非依存の C++ ストリーミング推論ライブラリ) と
> tmrvc-plugin (JUCE VST3 ラッパー) の設計。

---

## 1. クラス図

```
┌───────────────────────────────────────────────────────────────────┐
│  tmrvc-plugin (JUCE VST3)                                         │
│                                                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐              │
│  │  TMRVCProcessor     │    │  TMRVCEditor        │              │
│  │  (AudioProcessor)   │◄──▶│  (AudioProcessorEditor)            │
│  │                     │    │                     │              │
│  │  - prepareToPlay()  │    │  - Speaker selector │              │
│  │  - processBlock()   │    │  - Dry/Wet slider   │              │
│  │  - releaseResources │    │  - Gain slider      │              │
│  │  - getState/setState│    │  - IR mode toggle   │              │
│  │                     │    │  - Latency display  │              │
│  └────────┬────────────┘    └─────────────────────┘              │
│           │ owns                                                  │
│           ▼                                                       │
└───────────┼───────────────────────────────────────────────────────┘
            │
            │ uses (header-only include)
            ▼
┌───────────────────────────────────────────────────────────────────┐
│  tmrvc-engine (JUCE-free C++ library)                             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  StreamingEngine                                         │     │
│  │  (central orchestrator)                                  │     │
│  │                                                          │     │
│  │  - initialize(sampleRate, bufferSize, modelDir)          │     │
│  │  - process(inputBuffer, outputBuffer, numSamples)        │     │
│  │  - reset()                                               │     │
│  │  - getLatencyInSamples()                                 │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  OrtSessionBundle   │  │  TensorPool           │     │     │
│  │  │                     │  │                       │     │     │
│  │  │  - content_encoder  │  │  - contiguous alloc   │     │     │
│  │  │  - ir_estimator     │  │  - getMelFrame()      │     │     │
│  │  │  - converter        │  │  - getContent()       │     │     │
│  │  │  - vocoder          │  │  - getIRParams()      │     │     │
│  │  │  - runWithBinding() │  │  - getSTFTMag()       │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  FixedRingBuffer    │  │  PolyphaseResampler   │     │     │
│  │  │  <float, 2048>      │  │                       │     │     │
│  │  │                     │  │  - process()          │     │     │
│  │  │  - inputRing_       │  │  - 48↔24, 44.1↔24    │     │     │
│  │  │  - outputRing_      │  │  - reset()            │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  SpeakerManager     │  │  CrossFader           │     │     │
│  │  │                     │  │                       │     │     │
│  │  │  - double-buffered  │  │  - speaker switch     │     │     │
│  │  │  - loadSpeaker()    │  │  - model hot-reload   │     │     │
│  │  │  - swapToStaging()  │  │  - fade(src, dst, n)  │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐                                │     │
│  │  │  SPSCQueue          │                                │     │
│  │  │  <Command, 32>      │  Audio thread → Worker thread  │     │
│  │  │  <Response, 32>     │  Worker thread → Audio thread  │     │
│  │  └─────────────────────┘                                │     │
│  └─────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. メモリレイアウト: TensorPool

### 2.1 設計方針

すべてのテンソルバッファを **単一の contiguous allocation** で確保。
Audio thread 内での動的メモリ確保を完全に排除する。

### 2.2 Contiguous Allocation 図

```
TensorPool: single malloc (total ~300 KB)
┌────────────────────────────────────────────────────────────┐
│ Offset   Size     Name                    Shape            │
│ ──────   ─────    ──────────────────────  ──────────────── │
│ 0x0000   320 B    mel_frame               [1, 80, 1]       │
│ 0x0140   4 B      f0_frame                [1, 1, 1]        │
│ 0x0144   1024 B   content                 [1, 256, 1]      │
│ 0x0544   128 B    acoustic_params         [1, 32]          │
│ 0x05C4   768 B    spk_embed               [1, 192]         │
│ 0x0874   2052 B   pred_features           [1, 513, 1]      │
│ 0x1078   2052 B   stft_mag                [1, 513, 1]      │
│ 0x187C   2052 B   stft_phase              [1, 513, 1]      │
│ 0x2080   3200 B   mel_chunk (IR est.)     [1, 80, 10]      │
│                                                             │
│ === Ping-Pong State Buffers ===                             │
│                                                             │
│ 0x2C80   28672 B  content_enc_state_A     [1, 256, 28]     │
│ 0x9C80   28672 B  content_enc_state_B     [1, 256, 28]     │
│ 0x10C80  3072 B   ir_est_state_A          [1, 128, 6]      │
│ 0x11880  3072 B   ir_est_state_B          [1, 128, 6]      │
│ 0x12480  79872 B  converter_state_A       [1, 384, 52]     │
│ 0x25E80  79872 B  converter_state_B       [1, 384, 52]     │
│ 0x39880  14336 B  vocoder_state_A         [1, 256, 14]     │
│ 0x3D080  14336 B  vocoder_state_B         [1, 256, 14]     │
│                                                             │
│ === Work Buffers ===                                        │
│                                                             │
│ 0x40880  4096 B   fft_buffer              [1024] complex    │
│ 0x41880  3840 B   ola_buffer              [960]             │
│ 0x42780  3840 B   hann_window             [960] (pre-comp.) │
│ 0x43680  8192 B   resample_buffer         [2048]            │
│ ──────                                                      │
│ Total:   ~281 KB                                            │
└────────────────────────────────────────────────────────────┘
```

> **Note (Rust Implementation):** The Rust implementation in `tmrvc-engine-rs` uses separate `Vec<f32>` 
> allocations for each buffer rather than a single contiguous allocation. State buffers are managed 
> via `PingPongState` structs with independent `Vec` backing. Both approaches achieve the same goal: 
> **no dynamic allocation during inference**. All buffers are pre-allocated at initialization time.

### 2.3 TensorPool API

```cpp
class TensorPool {
public:
    TensorPool();   // Single allocation in constructor
    ~TensorPool();  // Single free in destructor

    // Typed accessors (return pre-offset pointers)
    float* getMelFrame()       { return base_ + kOffsetMelFrame; }
    float* getF0Frame()        { return base_ + kOffsetF0Frame; }
    float* getContent()        { return base_ + kOffsetContent; }
    float* getAcousticParams() { return base_ + kOffsetAcousticParams; }
    float* getSpkEmbed()       { return base_ + kOffsetSpkEmbed; }
    float* getPredFeatures()   { return base_ + kOffsetPredFeatures; }
    float* getSTFTMag()        { return base_ + kOffsetSTFTMag; }
    float* getSTFTPhase()      { return base_ + kOffsetSTFTPhase; }
    float* getMelChunk()       { return base_ + kOffsetMelChunk; }

    PingPongState& contentEncState() { return contentEncState_; }
    PingPongState& irEstState()      { return irEstState_; }
    PingPongState& converterState()  { return converterState_; }
    PingPongState& vocoderState()    { return vocoderState_; }

    void resetAllStates();  // Zero all state buffers

private:
    float* base_;            // Single contiguous allocation
    PingPongState contentEncState_;
    PingPongState irEstState_;
    PingPongState converterState_;
    PingPongState vocoderState_;
};
```

---

## 3. OrtSessionBundle: ONNX Runtime セッション管理

### 3.1 API

```cpp
class OrtSessionBundle {
public:
    // Worker thread で呼び出し (メモリ確保あり)
    bool loadModels(const char* modelDir, const OrtSessionOptions* options);
    void unloadModels();

    // Audio thread で呼び出し (RT-safe)
    bool runContentEncoder(
        const float* melFrame, const float* f0,
        const float* stateIn, float* content, float* stateOut);

    bool runIREstimator(
        const float* melChunk, const float* stateIn,
        float* acousticParams, float* stateOut);

    bool runConverter(
        const float* content, const float* spkEmbed,
        const float* acousticParams, const float* stateIn,
        float* predFeatures, float* stateOut);

    bool runVocoder(
        const float* features, const float* stateIn,
        float* stftMag, float* stftPhase, float* stateOut);

    bool isLoaded() const;

private:
    OrtEnv* env_ = nullptr;
    OrtSession* contentEncSession_ = nullptr;
    OrtSession* irEstSession_ = nullptr;
    OrtSession* converterSession_ = nullptr;
    OrtSession* vocoderSession_ = nullptr;
    OrtIoBinding* bindings_[4] = {};  // Pre-created bindings
};
```

### 3.2 IO Binding による Zero-Copy 推論

```cpp
bool OrtSessionBundle::runContentEncoder(
    const float* melFrame, const float* f0,
    const float* stateIn, float* content, float* stateOut)
{
    // 1. Wrap pre-allocated buffers as OrtValue (no allocation)
    OrtValue* inputs[3];
    CreateTensorWithData(melFrame, {1,80,1},   &inputs[0]);
    CreateTensorWithData(f0,       {1,1,1},    &inputs[1]);
    CreateTensorWithData(stateIn,  {1,256,28}, &inputs[2]);

    OrtValue* outputs[2];
    CreateTensorWithData(content,  {1,256,1},  &outputs[0]);
    CreateTensorWithData(stateOut, {1,256,28}, &outputs[1]);

    // 2. Bind to pre-created binding
    OrtBindInput(binding_, "mel_frame", inputs[0]);
    OrtBindInput(binding_, "f0",        inputs[1]);
    OrtBindInput(binding_, "state_in",  inputs[2]);
    OrtBindOutput(binding_, "content",   outputs[0]);
    OrtBindOutput(binding_, "state_out", outputs[1]);

    // 3. Run (reads from melFrame buffer, writes to content buffer)
    OrtStatus* status = OrtRunWithBinding(contentEncSession_, binding_);

    // 4. Cleanup OrtValues (no data freed, just metadata)
    for (auto v : inputs) OrtReleaseValue(v);
    for (auto v : outputs) OrtReleaseValue(v);

    return status == nullptr;
}
```

### 3.3 Session Options

```cpp
OrtSessionOptions* createSessionOptions() {
    OrtSessionOptions* opts;
    OrtCreateSessionOptions(&opts);

    // Single-threaded execution (audio thread で使用)
    OrtSetIntraOpNumThreads(opts, 1);
    OrtSetInterOpNumThreads(opts, 1);

    // Graph optimizations
    OrtSetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    // Memory pattern optimization
    OrtEnableMemPattern(opts);

    return opts;
}
```

---

## 4. Double-Buffered Model Hot-Reload

### 4.1 ModelSlot

```cpp
struct ModelSlot {
    OrtSessionBundle bundle;
    std::atomic<bool> ready{false};
};

class ModelManager {
    ModelSlot slots_[2];
    std::atomic<int> activeSlot_{0};

public:
    // Audio thread: 現在のアクティブ slot を取得
    OrtSessionBundle& getActive() {
        return slots_[activeSlot_.load(std::memory_order_acquire)].bundle;
    }

    // Worker thread: staging slot にロード → swap
    bool loadToStaging(const char* modelDir) {
        int staging = 1 - activeSlot_.load(std::memory_order_relaxed);
        ModelSlot& slot = slots_[staging];

        // 1. 新モデルをロード (blocking, Worker thread)
        if (!slot.bundle.loadModels(modelDir, createSessionOptions())) {
            return false;
        }
        slot.ready.store(true, std::memory_order_release);
        return true;
    }

    // Audio thread: staging が ready なら swap
    void trySwap() {
        int staging = 1 - activeSlot_.load(std::memory_order_relaxed);
        if (slots_[staging].ready.load(std::memory_order_acquire)) {
            // Atomic swap
            activeSlot_.store(staging, std::memory_order_release);
            slots_[1 - staging].ready.store(false, std::memory_order_relaxed);
        }
    }
};
```

### 4.2 Hot-reload フロー

```
1. User selects new model directory (GUI → Worker thread)
2. Worker thread: loadToStaging(newModelDir)
   - Creates new ORT sessions
   - Loads ONNX files
   - Sets staging.ready = true
3. Audio thread (next processBlock): trySwap()
   - Detects staging.ready == true
   - Atomic swap of active slot index
   - Old sessions remain valid until next reload
4. CrossFader: smooth transition between old/new output
```

---

## 5. Speaker 管理

### 5.1 SpeakerManager

```cpp
class SpeakerManager {
public:
    // Worker thread: .tmrvc_speaker ファイルをロード
    bool loadSpeakerFile(const char* path);

    // Audio thread: 現在のアクティブ speaker データを取得
    const float* getSpkEmbed() const;     // [192]
    const float* getLoRADelta() const;    // [24576]

    // Audio thread: staging → active swap
    void swapToStaging();

    bool hasSpeaker() const;

private:
    struct SpeakerData {
        float spkEmbed[kDSpeaker];             // 192
        float loraDelta[kLoRATotalSize];       // 24576
        std::atomic<bool> valid{false};
    };

    SpeakerData slots_[2];
    std::atomic<int> activeSlot_{0};
};
```

### 5.2 LoRA Weight Merge

Speaker ロード時に LoRA delta を Converter の weights に merge する。

```cpp
bool SpeakerManager::loadSpeakerFile(const char* path) {
    // 1. ファイル読み込み・検証 (Worker thread)
    SpeakerFileHeader header;
    if (!readAndValidate(path, &header)) return false;

    int staging = 1 - activeSlot_.load();
    SpeakerData& slot = slots_[staging];

    // 2. spk_embed コピー
    readFloats(file, slot.spkEmbed, kDSpeaker);

    // 3. lora_delta コピー
    readFloats(file, slot.loraDelta, kLoRATotalSize);

    // 4. LoRA merge into converter weights
    //    W_merged = W_original + (alpha/rank) * B @ A
    //    → 新しい converter session を staging ModelSlot に作成
    mergeLoRAIntoConverter(slot.loraDelta);

    slot.valid.store(true, std::memory_order_release);
    return true;
}
```

### 5.3 Double-Buffered Speaker Slots

```
Slot 0 (active): Speaker A の spk_embed + LoRA-merged converter
Slot 1 (staging): Speaker B をバックグラウンドでロード中...

ロード完了 → Audio thread で swapToStaging() → Slot 1 が active に
→ CrossFader で Speaker A → B への滑らかな遷移
```

### 5.4 .tmrvc_speaker Format

onnx-contract.md §6 で定義。ここでは C++ 側の読み込みを記載。

```cpp
struct SpeakerFileHeader {
    char magic[4];        // "TMSP"
    uint32_t version;     // 1
    uint32_t embedSize;   // 192
    uint32_t loraSize;    // 24576
};

bool SpeakerManager::readAndValidate(const char* path, SpeakerFileHeader* header) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    fread(header, sizeof(SpeakerFileHeader), 1, f);

    // Magic check
    if (memcmp(header->magic, "TMSP", 4) != 0) { fclose(f); return false; }

    // Version check
    if (header->version != 1) { fclose(f); return false; }

    // Size check
    if (header->embedSize != kDSpeaker) { fclose(f); return false; }
    if (header->loraSize != kLoRATotalSize) { fclose(f); return false; }

    // SHA-256 verification
    // ... (read all data, compute SHA-256, compare with stored checksum)

    fclose(f);
    return true;
}
```

---

## 6. CrossFader: Glitch-free Transition

### 6.1 用途

- Speaker 切り替え時の滑らかな遷移
- Model hot-reload 時のクリック防止

### 6.2 実装

```cpp
class CrossFader {
public:
    CrossFader(int fadeLengthSamples = 240);  // default: 1 hop = 10ms

    // Trigger a crossfade from current output to new source
    void startFade();

    // Apply crossfade (called per sample or per block)
    void process(const float* newOutput, float* buffer, int numSamples);

    bool isFading() const { return fadePos_ < fadeLength_; }

private:
    int fadeLength_;
    int fadePos_ = 0;
    bool fading_ = false;
    float prevOutput_[960];  // Store previous output for crossfade
};
```

```cpp
void CrossFader::process(const float* newOutput, float* buffer, int numSamples) {
    if (!fading_) {
        // No fade: just copy new output
        std::copy(newOutput, newOutput + numSamples, buffer);
        return;
    }

    for (int i = 0; i < numSamples && fadePos_ < fadeLength_; ++i, ++fadePos_) {
        float t = static_cast<float>(fadePos_) / fadeLength_;
        // Equal-power crossfade
        float gainOld = std::cos(t * M_PI * 0.5f);
        float gainNew = std::sin(t * M_PI * 0.5f);
        buffer[i] = gainOld * prevOutput_[fadePos_] + gainNew * newOutput[i];
    }

    // Remaining samples after fade completes
    if (fadePos_ >= fadeLength_) {
        int remaining = numSamples - fadePos_;
        if (remaining > 0) {
            std::copy(newOutput + fadePos_, newOutput + numSamples, buffer + fadePos_);
        }
        fading_ = false;
    }
}
```

---

## 7. StreamingEngine: Central Orchestrator

### 7.1 API

```cpp
class StreamingEngine {
public:
    struct Config {
        double hostSampleRate;     // DAW sample rate (44100 or 48000)
        int maxBufferSize;         // DAW max buffer size
        const char* modelDir;      // Path to ONNX models directory
        bool useQuantized;         // Use INT8 quantized models
    };

    // Lifecycle
    bool initialize(const Config& config);
    void reset();
    void shutdown();

    // Audio thread (RT-safe)
    void process(const float* input, float* output, int numSamples);
    int getLatencyInSamples() const;

    // Parameter control (thread-safe via atomic)
    void setDryWetMix(float mix);      // 0.0 = dry, 1.0 = wet
    void setOutputGain(float gainDb);  // -inf to +12 dB
    void setIRMode(IRMode mode);       // Auto / Manual
    void setIRParams(const float* params, int n);  // Manual IR override

    // Async operations (delegated to Worker thread)
    void loadSpeakerAsync(const char* path);
    void loadModelAsync(const char* modelDir);

    // Monitoring
    FrameTimingStats getTimingStats() const;

private:
    TensorPool tensorPool_;
    ModelManager modelManager_;
    SpeakerManager speakerManager_;
    FixedRingBuffer<float, 2048> inputRing_;
    FixedRingBuffer<float, 2048> outputRing_;
    PolyphaseResampler downsampler_;
    PolyphaseResampler upsampler_;
    OverlapAddBuffer olaBuffer_;
    CrossFader crossFader_;
    SPSCQueue<CommandMessage, 32> commandQueue_;
    SPSCQueue<ResponseMessage, 32> responseQueue_;

    // Frame processing state
    int irFrameCounter_ = 0;
    float cachedAcousticParams_[kNAcousticParams] = {};
    float voiceSourcePreset_[kNVoiceSourceParams] = {};  // from .tmrvc_speaker
    bool hasVoiceSourcePreset_ = false;

    // Atomic parameters
    std::atomic<float> dryWetMix_{1.0f};
    std::atomic<float> outputGain_{1.0f};

    void processOneFrame();
    void pollResponses();
};
```

### 7.2 processBlock 実装概要

```cpp
void StreamingEngine::process(
    const float* input, float* output, int numSamples)
{
    // 0. Poll worker thread responses (non-blocking)
    pollResponses();

    // 1. Downsample input (host rate → 24kHz)
    float downsampled[1024];  // TensorPool work buffer
    int numDown = downsampler_.process(input, numSamples,
                                        downsampled, 1024);

    // 2. Write to input ring buffer
    inputRing_.write(downsampled, numDown);

    // 3. Process frames while enough data available
    while (inputRing_.available() >= kHopSize) {
        processOneFrame();
    }

    // 4. Read from output ring buffer
    float upsampled_in[1024];
    int numToRead = upsampler_.getInputSamplesNeeded(numSamples);
    int numRead = std::min(numToRead,
                           static_cast<int>(outputRing_.available()));

    if (numRead > 0) {
        outputRing_.read(upsampled_in, numRead);
    }
    // Zero-fill if underrun
    if (numRead < numToRead) {
        std::fill(upsampled_in + numRead,
                  upsampled_in + numToRead, 0.0f);
    }

    // 5. Upsample (24kHz → host rate)
    float wet[2048];
    upsampler_.process(upsampled_in, numToRead, wet, numSamples);

    // 6. Dry/wet mix
    float mix = dryWetMix_.load(std::memory_order_relaxed);
    float gain = outputGain_.load(std::memory_order_relaxed);
    for (int i = 0; i < numSamples; ++i) {
        output[i] = gain * ((1.0f - mix) * input[i] + mix * wet[i]);
    }
}
```

---

## 8. VST3 パラメータ

### 8.1 パラメータ一覧

| Parameter | ID | Range | Default | Type |
|---|---|---|---|---|
| **Dry/Wet** | 0 | 0.0 - 1.0 | 1.0 (100% wet) | Float |
| **Output Gain** | 1 | -60.0 - +12.0 dB | 0.0 dB | Float |
| **IR Mode** | 2 | 0=Auto, 1=Manual | 0 (Auto) | Choice |
| **IR RT60** | 3 | 0.05 - 3.0 sec | 0.5 sec | Float |
| **IR DRR** | 4 | -10.0 - +30.0 dB | 10.0 dB | Float |
| **IR Tilt** | 5 | -6.0 - +6.0 dB/oct | 0.0 | Float |
| **Voice Preset** | 6 | 0.0 - 1.0 | 0.0 (off) | Float |

### 8.2 パラメータの Audio Thread への伝達

```cpp
// TMRVCProcessor.cpp
void TMRVCProcessor::parameterChanged(int paramID, float newValue) {
    switch (paramID) {
        case kParamDryWet:
            engine_.setDryWetMix(newValue);
            break;
        case kParamGain:
            engine_.setOutputGain(dbToLinear(newValue));
            break;
        case kParamIRMode:
            engine_.setIRMode(newValue < 0.5f ? IRMode::Auto : IRMode::Manual);
            break;
        case kParamIRRT60:
        case kParamIRDRR:
        case kParamIRTilt:
            if (irMode_ == IRMode::Manual) {
                // Construct manual IR params from GUI sliders
                float params[kNIRParams];
                buildManualIRParams(params);
                engine_.setIRParams(params, kNIRParams);
            }
            break;
    }
}
```

全パラメータは `std::atomic` 経由で Audio thread に伝達。Lock-free。

---

## 9. DAW 統合

### 9.1 prepareToPlay

```cpp
void TMRVCProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    StreamingEngine::Config config;
    config.hostSampleRate = sampleRate;
    config.maxBufferSize = samplesPerBlock;
    config.modelDir = currentModelDir_.c_str();
    config.useQuantized = useQuantizedModels_;

    engine_.initialize(config);
    setLatencySamples(engine_.getLatencyInSamples());
}
```

### 9.2 processBlock

```cpp
void TMRVCProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& midi) {
    auto* channelData = buffer.getWritePointer(0);  // Mono processing
    int numSamples = buffer.getNumSamples();

    engine_.process(channelData, channelData, numSamples);

    // If stereo, copy mono to second channel
    if (buffer.getNumChannels() > 1) {
        buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
    }
}
```

### 9.3 setLatencySamples

```cpp
int TMRVCProcessor::getLatencyInSamples() const {
    return engine_.getLatencyInSamples();
}

// StreamingEngine 内部:
int StreamingEngine::getLatencyInSamples() const {
    // Algorithmic latency: 2 hops (accumulation + pre-fill)
    double algoLatencySec = 2.0 * kHopSize / static_cast<double>(kSampleRate);
    return static_cast<int>(std::round(algoLatencySec * hostSampleRate_));
}
```

### 9.4 State Persistence (getStateInformation / setStateInformation)

```cpp
void TMRVCProcessor::getStateInformation(juce::MemoryBlock& destData) {
    juce::ValueTree state("TMRVCState");
    state.setProperty("version", 1, nullptr);
    state.setProperty("dryWet", getParameter(kParamDryWet)->getValue(), nullptr);
    state.setProperty("gain", getParameter(kParamGain)->getValue(), nullptr);
    state.setProperty("irMode", static_cast<int>(irMode_), nullptr);
    state.setProperty("speakerPath", currentSpeakerPath_, nullptr);
    state.setProperty("modelDir", currentModelDir_, nullptr);

    juce::MemoryOutputStream stream(destData, false);
    state.writeToStream(stream);
}

void TMRVCProcessor::setStateInformation(const void* data, int sizeInBytes) {
    auto state = juce::ValueTree::readFromData(data, sizeInBytes);
    if (!state.isValid()) return;

    // Restore parameters
    if (state.hasProperty("dryWet"))
        getParameter(kParamDryWet)->setValueNotifyingHost(state["dryWet"]);
    if (state.hasProperty("gain"))
        getParameter(kParamGain)->setValueNotifyingHost(state["gain"]);

    // Reload speaker/model asynchronously
    if (state.hasProperty("speakerPath"))
        engine_.loadSpeakerAsync(state["speakerPath"].toString().toRawUTF8());
    if (state.hasProperty("modelDir"))
        engine_.loadModelAsync(state["modelDir"].toString().toRawUTF8());
}
```

---

## 10. Edge Cases

### 10.1 Sample Rate 変更

```
DAW が sample rate を変更した場合:
  1. releaseResources() が呼ばれる
  2. prepareToPlay(newSampleRate, newBufferSize) が呼ばれる
  3. StreamingEngine を re-initialize
     - PolyphaseResampler を新しいレート比で再構築
     - Ring Buffer を reset
     - State tensors を zero-clear
     - setLatencySamples() を更新
```

### 10.2 Buffer Size 変更

```
DAW が buffer size を変更した場合:
  1. prepareToPlay(sameRate, newBufferSize) が呼ばれる
  2. Ring Buffer の capacity は固定 (2048) のため再確保不要
  3. PolyphaseResampler の内部バッファは maxBufferSize に基づくため再構築
  4. Latency samples は変わらない (algorithmic latency のみ報告)
```

### 10.3 Inference Overrun

```
processOneFrame() が 10ms を超過した場合:
  - Output Ring Buffer への書き込みが遅延
  - 次の processBlock で underrun が発生する可能性
  - 対処:
    1. Output Ring Buffer が空なら前フレーム出力を再利用
    2. FrameTimingStats.overrunCount を increment
    3. 連続 3 回超過で IR estimator を停止
    4. 連続 10 回超過で dry bypass
  - 復帰:
    1. avgFrameMs < hopTimeMs × 0.8 で通常モードに復帰
    2. IR estimator を再開
```

### 10.4 Speaker ファイルが見つからない場合

```
- engine_.loadSpeakerAsync() が失敗
- Worker thread が Error response を送信
- Audio thread は「speaker なし」状態を検知
- 対処: 入力をそのまま出力 (bypass mode)
- GUI に "No speaker loaded" を表示
```

---

## 11. ビルドシステム

### 11.1 tmrvc-engine CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(tmrvc-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime (static link via ort-builder)
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "")
find_library(ORT_LIB onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH)

add_library(tmrvc-engine STATIC
    src/streaming_engine.cpp
    src/ort_session_bundle.cpp
    src/tensor_pool.cpp
    src/fixed_ring_buffer.cpp
    src/polyphase_resampler.cpp
    src/spsc_queue.cpp
    src/speaker_manager.cpp
    src/cross_fader.cpp
)

target_include_directories(tmrvc-engine
    PUBLIC include
    PRIVATE "${ORT_ROOT}/include"
)

target_link_libraries(tmrvc-engine
    PRIVATE ${ORT_LIB}
)
```

### 11.2 tmrvc-plugin CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(tmrvc-plugin LANGUAGES CXX)

# JUCE
find_package(JUCE CONFIG REQUIRED)

juce_add_plugin(TMRVCPlugin
    PLUGIN_MANUFACTURER_CODE Tmrv
    PLUGIN_CODE Tmvc
    FORMATS VST3 Standalone
    PRODUCT_NAME "TMRVC"
)

target_sources(TMRVCPlugin PRIVATE
    src/TMRVCProcessor.cpp
    src/TMRVCEditor.cpp
)

target_link_libraries(TMRVCPlugin
    PRIVATE
        tmrvc-engine
        juce::juce_audio_utils
        juce::juce_dsp
)
```

---

## 12. 整合性チェックリスト

- [x] StreamingEngine は JUCE に非依存 (architecture.md §5.2)
- [x] ONNX Runtime は C API only / 静的リンク (architecture.md §5.3)
- [x] TensorPool の shapes が onnx-contract.md §2-3 と一致 (ir_params → acoustic_params[32])
- [x] Ring Buffer サイズが streaming-design.md §4 と一致
- [x] Audio thread は RT-safe (streaming-design.md §7.2)
- [x] SPSC Queue protocol が streaming-design.md §7.3-7.4 と一致
- [x] Speaker format が onnx-contract.md §6 と一致
- [x] Latency reporting が streaming-design.md §8 と一致
- [x] Graceful degradation が streaming-design.md §6 と一致
- [x] Model dimensions が model-architecture.md と一致
- [x] Voice Source Preset: speaker ファイルから読み込み、converter 呼び出し前にブレンド (RT-safe)
- [x] Voice Preset パラメータ (ID=6) が VST3 パラメータ一覧に追加済み
