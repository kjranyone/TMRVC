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
- `tmrvc-core/src/tmrvc_core/types.py`
- `tmrvc-core/src/tmrvc_core/dialogue_types.py`
- tests under `tests/serve/`
- tests under `tmrvc-engine-rs/tests/`


## Required Outcomes

- TTS path no longer depends on predicted durations as the primary runtime mechanism
- pointer state persists across frame steps
- runtime control surface exposes timing and physical-state controls
- text-side cache behavior is explicit and bounded
- health and smoke tests still pass
- runtime exposes the controls needed for drama-like delivery, not only neutral reading
- management APIs exist for the Gradio/WebUI control plane
- Python serve, Rust runtime, ONNX export, and VST share the same pointer / `voice_state` semantics
- all runtimes consume the shared serializable schema owned by `tmrvc-core`
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
   - explicit force-advance side-effect policy:
     - how `progress_value` is reset or clipped
     - how `acoustic_history` remains continuity-safe to avoid glitches (must include **Acoustic Hidden State Smoothing** or cross-fade logic to ensure glitch-free/click-free audio during forced skips)
     - how forced-advance is surfaced in telemetry and parity tests
   - explicit numeric advance-decision rule:
     - thresholding method for `advance_logit` / `advance_prob`
     - tie-break and near-threshold behavior
     - parity-safe comparison semantics shared by Python and Rust
10. Review VC path for semantic-context hooks so the new runtime state shape does not block future VC improvements.
   - define the causal fusion path for VC semantic features with optional dialogue/text context
   - ensure this path is explicit even when VC bypasses pointer progression
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
13. Implement WebUI-facing orchestration and evaluation routes:
   - `POST /ui/datasets/upload`
   - `POST /ui/datasets/register`
   - `GET /ui/jobs/{job_id}`
   - `GET /ui/jobs/{job_id}/events`
   - `POST /ui/curation/runs`
   - `POST /ui/curation/runs/{run_id}/resume`
   - `POST /ui/curation/runs/{run_id}/stop`
   - `GET /ui/curation/records`
   - `POST /ui/curation/records/{record_id}/action`
   - `POST /ui/workshop/generate`
   - `POST /ui/workshop/takes/{take_id}/pin`
   - `POST /ui/workshop/takes/{take_id}/export`
   - `POST /ui/workshop/sessions`
   - `POST /ui/eval/sessions`
   - `GET /ui/eval/assignments/{assignment_id}`
   - `POST /ui/eval/assignments/{assignment_id}/submit`
14. Define few-shot runtime behavior:
   - when `reference_audio` is encoded
   - how `speaker_embed` or prompt cache is reused across turns
   - how prompt encoding latency is measured separately from steady-state generation latency
   - how runtime logs enough metadata to reproduce external-baseline comparisons
   - how conflicts between `speaker_embed` and `prompt_codec_tokens` are resolved at runtime
   - how much authority prompt texture has relative to speaker-identity anchoring
15. Define CFG runtime behavior:
   - guidance blends logits: `guided_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)`
   - unconditional pass is produced by zeroing out: `explicit_voice_state`, `ssl_voice_state`, `speaker_embed`, and `dialogue_context` (the same conditioning inputs dropped during CFG training dropout)
   - unconditional pass may reuse the same KV cache structure but with zeroed conditioning; it must not require a separate model instance
   - safe `cfg_scale` bounds: clamp to `[1.0, 3.0]` by default to avoid pointer instability; values above 3.0 require explicit opt-in
   - low-latency runtime modes:
     - `off`
     - `full` two-pass CFG
     - `lazy` CFG refresh every N frames with bounded drift policy
     - optional `distilled` one-pass CFG-compatible path
   - define when VST / real-time engine must clamp or disable expensive CFG modes
   - real-time priority policy:
     - Rust engine / VST default to `off`, `lazy`, or `distilled`
     - `full` two-pass CFG is non-default in hard real-time paths and must justify itself against the 10 ms budget
16. Define cache synchronization behavior:
   - sliding-window text attention policy or deterministic cache re-indexing
   - explicit invalidation rules when the pointer crosses a window boundary
   - interaction between prompt cache, text cache, and autoregressive acoustic cache
17. Define waveform-decoder runtime policy:
   - initial v3 policy: the codec-native decoder (inverse RVQ + neural vocoder from the codec model) is the primary waveform path
   - candidates for future enhancement: `Vocos`, `HiFi-GAN` variants, or `DAC` decoder
   - streaming latency budget: waveform decoding must complete within the 10 ms frame budget; if an enhanced decoder exceeds this, fall back to the codec-native decoder
   - fallback path: always maintain the codec-native decoder as a guaranteed-available fast path
   - quality measurement: Worker 06 must include waveform artifact rate (clicks, buzzing, metallic artifacts) in the TTS quality gate
18. Define multilingual / code-switch runtime policy:
   - utterance-level vs token-level language conditioning inputs
   - cross-lingual few-shot prompting when prompt language differs from target language
19. Define pointer fallback runtime policy:
   - configurable `max_frames_per_unit`
   - forced-advance after excessive stall
   - low-`boundary_confidence` fallback behavior
   - observability hooks so stalls/skips are visible in telemetry rather than silent
20. Define UI event-stream contract:
   - SSE event types at minimum:
     - `job_progress`
     - `job_blocked_human`
     - `job_failed`
     - `job_completed`
     - `take_ready`
     - `telemetry_update`
   - every event includes:
     - `event_type`
     - `job_id`
     - `object_type`
     - `object_id`
     - `timestamp`
     - `payload_version`
   - event stream must be resumable using `Last-Event-ID` or equivalent cursor
21. Define API idempotency and conflict behavior for UI-originated writes:
   - upload/register/run/export/create-session actions accept `idempotency_key`
   - review/edit/policy actions accept `object_version`
   - conflict responses must be typed, not plain text:
     - `stale_version`
     - `locked_by_other`
     - `already_submitted`
     - `policy_forbidden`
22. Define artifact download contract for WebUI:
   - artifact responses return:
     - `artifact_id`
     - `artifact_type`
     - `download_url`
     - `expires_at`
     - `provenance_summary`
   - artifacts must be retrievable without shell access


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
- expose active CFG runtime mode (`off | full | lazy | distilled`) and refresh interval for debugging
- expose stall / skip fallback counters and the active pointer-fallback reason
- expose active job state and resumable event cursor for WebUI-driven long-running tasks
- expose conflict / lock metadata needed by optimistic-lock forms


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
- do not expose lazy or distilled CFG modes without numerical and perceptual validation against the full mode
- do not make `full` two-pass CFG the default in Rust/VST unless latency proof exists on the target budget
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
- route schema test for WebUI dataset / curation / evaluation endpoints
- SSE or WebSocket event-schema test for `job_progress` / `job_failed` / `take_ready`
- route/engine test for dialogue-performance control propagation
- route/engine test for short-reference speaker adaptation path
- route/engine test for CFG parameter propagation and safety clamping
- route/engine test for lazy-CFG refresh policy and mode clamping
- cache-behavior test for sliding-window or cache re-indexing under pointer movement
- route/engine test for multilingual/code-switch propagation
- route/engine test for `max_frames_per_unit` forced-advance behavior
- route/engine test that force-advance does not violate serialized state invariants
- Python vs Rust parity test for advance-threshold and near-threshold decisions
- route/engine test for low-`boundary_confidence` fallback behavior
- route conflict test for stale object-version submissions
- idempotency-key test for retried UI write requests
- Python vs Rust pointer-state parity smoke test
- PyTorch vs ONNX runtime-contract parity test for pointer / `voice_state` fields
- streaming numerical parity test for batch vs frame-by-frame CausalConv1d path
