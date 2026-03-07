# Worker 04: Serving and Runtime Pointer Execution

## Scope

Replace duration expansion in runtime TTS with causal pointer execution and external pacing controls across Python serve, Rust runtime, ONNX export, and VST-facing control surfaces.


## Primary Files

- `tmrvc-serve/src/tmrvc_serve/uclm_engine.py`
- `tmrvc-serve/src/tmrvc_serve/routes/tts.py`
- `tmrvc-serve/src/tmrvc_serve/routes/admin.py` (New)
- `tmrvc-serve/src/tmrvc_serve/app.py`
- `tmrvc-serve/src/tmrvc_serve/routes/vc_streaming.py`
- `tmrvc-engine-rs/src/processor.rs`
- `tmrvc-engine-rs/src/ort_bundle.rs`
- `tmrvc-export/src/tmrvc_export/export_uclm.py`
- `tmrvc-vst/src/plugin.rs`
- tests under `tests/serve/`
- tests under `tmrvc-engine-rs/tests/`


## Required Outcomes

- TTS path no longer depends on predicted durations as the primary runtime mechanism
- pointer state persists across frame steps
- runtime control surface exposes timing and physical-state controls
- text-side cache behavior is explicit and bounded
- health and smoke tests still pass
- runtime exposes the controls needed for drama-like delivery, not only neutral reading
- management APIs exist for the Gradio control plane
- Python serve, Rust runtime, ONNX export, and VST share the same pointer / `voice_state` semantics
- runtime supports short-reference few-shot speaker adaptation in a reproducible way
- CFG-style control scaling is exposed through a documented inference contract
- multilingual / code-switch requests are supported through explicit language conditioning


## Concrete Tasks

1. Add pointer state object in `uclm_engine.py`.
2. Replace duration-prediction-based target-length bootstrap with pointer-driven progression.
3. Add request parameters to TTS route:
   - `pace`, `hold_bias`, `boundary_bias`
   - `cfg_scale`
   - `explicit_voice_state` (8-D preset or curve)
   - optional `delta_voice_state`
   - `reference_audio` or precomputed `speaker_embed`
   - `speaker_profile_id` (loading pre-exported speaker prompts)
   - optional `reference_text`
   - `language_id`
4. Implement Low-Latency Streaming Protocol:
   - support **Server-Sent Events (SSE)** or **WebSockets** for chunked audio delivery.
   - ensure pointer-state telemetry is interleaved with audio chunks for UI/VST feedback.
5. Port the same canonical state contract to:
   - `tmrvc-engine-rs`
   - ONNX export wrappers
   - **VST parameter/control bridge (expose all 8-D controls and pacing to DAW automation).**
6. Ensure streaming-safe cache update:
   - no hidden re-expansion of full text sequence per step
   - no full-sequence recomputation when not needed
7. Add Zero-Shot / Few-Shot Voice Cloning API support:
   - accept `reference_audio` (base64 or file upload) and optional `reference_text` in the TTS request.
   - run the `Speaker Prompt Encoder` on the fly to extract `speaker_embed` and `prompt_kv_cache`.
   - cache these prompt features across conversational turns for the same speaker to avoid redundant extraction.
8. Define text-side cache contract:
   - text encoder outputs are computed once per request or once per turn boundary
   - cross-attention keys/values for active text/context window are cached and reused across 10 ms steps
   - pointer advancement updates window indices or masks, not full encoded text recomputation
   - dialogue-context refresh happens only when the external turn/context input changes
9. Freeze the runtime finite-state behavior:
   - max one text-unit advance per frame in the initial mainline
   - explicit EOS / stop condition
   - explicit behavior on terminal hold
   - explicit serialization of `stall_frames` or equivalent deadlock-detection state
   - explicit `max_frames_per_unit` guardrail and force-advance behavior
   - explicit skip-protection when advance confidence is weak or inconsistent
10. Review VC path for semantic-context hooks so the new runtime state shape does not block future VC improvements.
11. Add dialogue-performance controls to runtime contract where supported:
   - scene or context embedding input
   - phrase pressure / interruption pressure
   - optional breath tendency / release bias
12. Implement `routes/admin.py` for Worker 12:
   - `GET /admin/health` (detailed system health)
   - `GET /admin/telemetry` (VRAM, latency, model state)
   - `POST /admin/load_model` (switch checkpoints)
   - `GET /admin/models` (list available checkpoints)
   - `GET /admin/runtime_contract` (pointer / voice-state contract introspection)
13. Define few-shot runtime behavior:
   - when `reference_audio` is encoded
   - how `speaker_embed` or prompt cache is reused across turns
   - how prompt encoding latency is measured separately from steady-state generation latency
   - how runtime logs enough metadata to reproduce external-baseline comparisons
   - how conflicts between `speaker_embed` and `prompt_codec_tokens` are resolved at runtime
   - how much authority prompt texture has relative to speaker-identity anchoring
14. Define CFG runtime behavior:
   - whether guidance blends logits, hidden states, or selected heads
   - how unconditional passes are computed or cached
   - safe `cfg_scale` bounds to avoid breaking pointer stability
15. Define cache synchronization behavior:
   - sliding-window text attention policy or deterministic cache re-indexing
   - explicit invalidation rules when the pointer crosses a window boundary
   - interaction between prompt cache, text cache, and autoregressive acoustic cache
16. Define waveform-decoder runtime policy:
   - whether an external vocoder / enhancer stage exists
   - its streaming latency budget
   - fallback path if the high-quality decoder is unavailable
17. Define multilingual / code-switch runtime policy:
   - utterance-level vs token-level language conditioning inputs
   - cross-lingual few-shot prompting when prompt language differs from target language
18. Define pointer fallback runtime policy:
   - configurable `max_frames_per_unit`
   - forced-advance after excessive stall
   - low-`boundary_confidence` fallback behavior
   - observability hooks so stalls/skips are visible in telemetry rather than silent


## Runtime Contract

The engine must be able to do:

- initialize pointer from text tokens
- advance or hold per 10 ms step
- emit codec tokens and control tokens causally
- expose current text position for debugging
- expose current explicit / delta voice-state controls for debugging
- expose enough runtime telemetry to inspect why a line sounded flat
- keep text-side encoded context and attention cache stable across frame steps without O(N) recomputation
- preserve the same state transition semantics in Python serve, Rust runtime, ONNX export, and VST parameter mapping
- expose the active few-shot speaker-conditioning source (`reference_audio` derived or precomputed embedding) for debugging
- expose active guidance scale and cache policy for debugging
- expose stall / skip fallback counters and the active pointer-fallback reason


## Guardrails

- do not regress route compatibility without explicit fallback
- do not add large look-ahead
- do not make server depend on MFA-generated files
- do not hide expressive controls behind undocumented presets only
- do not demote 8-D physical control to undocumented labels or GUI-only presets
- do not recompute full text-side K/V every 10 ms step unless a measured fallback path explicitly says so
- do not let runtime invent pointer-state semantics that differ from Worker 01 state-transition definitions
- do not let Python, Rust, ONNX, and VST drift into separate runtime contracts
- do not expose `cfg_scale` without defining unconditional-pass semantics and safety limits
- do not leave waveform quality to an unspecified decoder fallback
- do not let silent stall-recovery or skip-recovery heuristics diverge across runtimes


## Handoff Contract

- serving works with new v3 checkpoints
- API parameters are documented and stable
- Python serve, Rust runtime, ONNX export, and VST are all wired to the same control/state contract
- worker 05 can wire `dev.py` serve defaults to the new mode


## Required Tests

- TTS endpoint smoke with pointer mode
- engine pointer-state step test
- cache-behavior test for text-side K/V reuse across frame steps
- backward compatibility test for legacy checkpoint loading
- route schema test for new pacing controls
- route/engine test for dialogue-performance control propagation
- route/engine test for short-reference speaker adaptation path
- route/engine test for CFG parameter propagation and safety clamping
- cache-behavior test for sliding-window or cache re-indexing under pointer movement
- route/engine test for multilingual/code-switch propagation
- route/engine test for `max_frames_per_unit` forced-advance behavior
- route/engine test for low-`boundary_confidence` fallback behavior
- Python vs Rust pointer-state parity smoke test
- PyTorch vs ONNX runtime-contract parity test for pointer / `voice_state` fields
- streaming numerical parity test for batch vs frame-by-frame CausalConv1d path
