# TMRVC Rust Engine Design (UCLM Edition)

Kojiro Tanaka — Rust engine design
Created: 2026-03-01 (Asia/Tokyo)

> **Scope:** tmrvc-engine-rs (nih-plug 非依存の Rust ストリーミング推論ライブラリ) と
> tmrvc-vst (nih-plug VST3 ラッパー) の最新設計。
> **Current Paradigm:** Disentangled UCLM (`codec_encoder -> uclm_core -> codec_decoder`, `A_t/B_t` dual-stream, 10ms frames)

---

## 1. クラス図 (UCLM)

```
┌───────────────────────────────────────────────────────────────────┐
│  tmrvc-vst (nih-plug VST3)                                        │
│  - TMRVCPlugin / TMRVCParams                                      │
└───────────┬───────────────────────────────────────────────────────┘
            │ owns/uses
            ▼
┌───────────────────────────────────────────────────────────────────┐
│  tmrvc-engine-rs (nih-plug-free Rust crate)                       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  StreamingEngine                                         │     │
│  │  (central orchestrator)                                  │     │
│  │                                                          │     │
│  │  - process_one_frame(input, output, params)              │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  OrtBundle          │  │  TokenBuffers       │     │     │
│  │  │                     │  │                      │     │     │
│  │  │  - codec_encoder    │  │  - past_A_t          │     │     │
│  │  │  - vc_encoder (VQ)  │  │  - past_B_t          │     │     │
│  │  │  - voice_state_enc  │  │  - voice_state_hist  │     │     │
│  │  │  - uclm_core        │  │  - kv_cache          │     │     │
│  │  │  - codec_decoder    │  └──────────────────────┘     │     │
│  │  └─────────────────────┘                                │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  SpeakerFile        │  │  SharedStatus        │     │     │
│  │  │                     │  │  (Atomic stats GUI)  │     │     │
│  │  │  - spk_embed[192]   │  │                      │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  └─────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

## 2. 推論フロー (10ms Frame)

1. **Audio In**: 48kHz -> Resample to 24kHz (240 samples = 10ms).
2. **Codec Encode**: `codec_encoder.onnx`
   - Input: Audio frame `[1, 1, 240]`
   - Output: `source_A_t` `[1, 8, 1]`
3. **VC Encode (Disentanglement)**: `vc_encoder.onnx`
   - Input: `source_A_t`
   - Output: `vq_content_features` `[1, 1, d_model]`
4. **Voice State / CFG**:
   - `voice_state_enc.onnx` にて明示的パラメータ(UI)とWavLM(SSL)から `state_cond` を生成。
5. **UCLM Core (AR Token Generation)**: `uclm_core.onnx`
   - Input: `vq_content_features` + `state_cond` + `spk_embed` + `past_A_t/B_t` + `kv_cache`
   - Output: `target_A_t[1, 8, 1]`, `target_B_t[1, 4, 1]`
   - *CFG Scale適用可能*
6. **Codec Decode**: `codec_decoder.onnx`
   - Input: `target_A_t`, `target_B_t`, `voice_state`
   - Output: Generated audio frame `[1, 1, 240]`
7. **Audio Out**: Resample to 48kHz -> DAW Out.

## 3. リアルタイム制約 (RT-Safe)
- 全ての ONNX セッションの入力/出力テンソルバッファは初期化時に事前確保 (Pre-allocation)。
- オーディオスレッド中での `Vec::new()` や Mutex のロックは厳禁。
