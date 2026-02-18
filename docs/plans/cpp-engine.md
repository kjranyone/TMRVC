# Stream 4: C++ Engine (tmrvc-engine)

## Goal

`docs/design/cpp-engine-design.md` に基づき、JUCE 非依存の C++ ストリーミング推論ライブラリを実装する。

## Important Context: tmrvc-rt (Rust) が参考実装として存在

tmrvc-rt は **完全に動作する Rust 実装** で、C++ エンジンと同一アーキテクチャ:

| Component | tmrvc-rt (Rust) | tmrvc-engine (C++) |
|---|---|---|
| TensorPool | `engine/tensor_pool.rs` (single contiguous alloc) | Planned |
| ORT Bundle | `engine/ort_bundle.rs` (ort crate) | ORT C API |
| DSP | `engine/dsp.rs` (rustfft) | PFFFT or KissFFT |
| Ring Buffer | `engine/ring_buffer.rs` (lock-free SPSC) | Planned |
| Ping-Pong | `engine/ping_pong.rs` | Planned |
| Speaker | `engine/speaker.rs` (.tmrvc_speaker) | Planned |
| Processor | `engine/processor.rs` (359 lines) | StreamingEngine |
| Resampler | Not yet | PolyphaseResampler |
| CrossFader | Not yet | Planned |

**Rust 実装をテスト oracle として使える** — 同一入力に対する C++ と Rust の出力を比較。

## Decision Point: C++ Engine は本当に必要か？

| Option | Pros | Cons |
|---|---|---|
| **A: C++ engine 実装** | VST3 plugin に必要、JUCE 統合、プロダクション品質 | 大きな工数、Rust と重複 |
| **B: Rust → C FFI で VST3 接続** | Rust コード再利用、安全性 | JUCE との接続が複雑、cbindgen が必要 |
| **C: VST3 不要、tmrvc-rt で完結** | 工数ゼロ | DAW 統合なし |

**推奨:** 当面は Option C (tmrvc-rt で十分)。VST3 が必要になった時点で Option A or B を決定。

## Implementation Plan (Option A を選択した場合)

### Phase 4.1: Scaffolding

```
tmrvc-engine/
├── CMakeLists.txt
├── include/tmrvc/
│   ├── constants.h          ← scripts/generate_constants.py で生成
│   ├── tensor_pool.h
│   ├── fixed_ring_buffer.h
│   ├── ping_pong_state.h
│   ├── ort_session_bundle.h
│   ├── dsp.h
│   ├── speaker_manager.h
│   ├── cross_fader.h
│   ├── polyphase_resampler.h
│   └── streaming_engine.h
└── src/
    ├── tensor_pool.cpp
    ├── fixed_ring_buffer.cpp
    ├── ort_session_bundle.cpp
    ├── dsp.cpp
    ├── speaker_manager.cpp
    ├── cross_fader.cpp
    ├── polyphase_resampler.cpp
    └── streaming_engine.cpp
```

### Phase 4.2: Core Components (RT-safe, no allocation)

1. **TensorPool** — Single malloc, offset-based accessors
   - Reference: `tmrvc-rt/src/engine/tensor_pool.rs`
   - Reference: `docs/design/cpp-engine-design.md` §2

2. **FixedRingBuffer** — Lock-free SPSC, power-of-2 capacity
   - Reference: `tmrvc-rt/src/engine/ring_buffer.rs`

3. **PingPongState** — Double-buffered state management
   - Reference: `tmrvc-rt/src/engine/ping_pong.rs`

4. **CrossFader** — Equal-power crossfade
   - Reference: `docs/design/cpp-engine-design.md` §6

### Phase 4.3: DSP

1. **Causal STFT** — Hann window, left-padded FFT
   - Reference: `tmrvc-rt/src/engine/dsp.rs`
   - FFT library: PFFFT (single-header, no allocation) or KissFFT

2. **Mel filterbank** — HTK-style triangular filters
   - Pre-computed at init time, stored in TensorPool

3. **iSTFT + Overlap-Add** — Phase reconstruction, OLA buffer

4. **PolyphaseResampler** — 48↔24kHz, 44.1↔24kHz

### Phase 4.4: ONNX Runtime Integration

1. **OrtSessionBundle** — C API wrapper
   - `OrtCreateTensorWithDataAsOrtValue` for zero-copy
   - Pre-created `OrtIoBinding` per model
   - Single-threaded execution (intra=1, inter=1)

2. **ModelManager** — Double-buffered hot-reload
   - Reference: `docs/design/cpp-engine-design.md` §4

### Phase 4.5: StreamingEngine

Central orchestrator combining all components:
- `initialize()` / `process()` / `reset()` / `shutdown()`
- Reference: `docs/design/cpp-engine-design.md` §7

### Phase 4.6: Tests

```
tests/cpp/
├── test_tensor_pool.cpp
├── test_ring_buffer.cpp
├── test_ping_pong.cpp
├── test_dsp.cpp           # Causal STFT vs Python reference
├── test_ort_bundle.cpp    # Load + run dummy ONNX
└── test_streaming.cpp     # End-to-end parity with tmrvc-rt
```

Testing framework: Catch2 (single-header)

## Dependencies

- CMake >= 3.20
- C++17 compiler
- ONNX Runtime (ort-builder for static link, CPU EP only)
- FFT library (PFFFT or KissFFT)
- Catch2 (testing)

## Acceptance Criteria

- [ ] `cmake --build build` が成功
- [ ] TensorPool の total allocation が ~281KB
- [ ] ONNX Runtime で 4 モデルの推論が成功
- [ ] Causal STFT が Python 実装と bit-exact (float32 precision)
- [ ] End-to-end: 同一入力に対して tmrvc-rt と数値的に一致 (atol < 1e-5)
- [ ] Audio thread で malloc/free が一切発生しない (valgrind or ASan)
- [ ] 全テスト pass
