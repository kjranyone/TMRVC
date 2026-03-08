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
- `tmrvc-serve` is the single authoritative backend API surface for multi-user WebUI workflows
- v3.0 claim scope is stratified by runtime class:
  - Python serve may claim CFG-enhanced drama-grade acting after validation
  - Rust / VST / strict real-time paths must claim pointer/control parity and causal latency first; CFG-enhanced drama claims are deferred until `lazy` or `distilled` CFG pass validation


## Concrete Tasks

1. Add pointer state object in `uclm_engine.py`.
2. Replace duration-prediction-based target-length bootstrap with pointer-driven progression.
3. Add request parameters to TTS route:
   - `pace`, `hold_bias`, `boundary_bias`
   - `cfg_scale`
   - `explicit_voice_state` (8-D preset or curve)
   - optional `delta_voice_state`
   - `reference_audio` (for on-the-fly extraction)
   - **`speaker_profile_id` (loads the canonical `SpeakerProfile` from the Casting Gallery, see `docs/design/speaker-profile-spec.md`)**
   - optional `reference_text`
   - `language_id`
4. Implement Low-Latency Streaming Protocol:
   - support **Server-Sent Events (SSE)** or **WebSockets** for chunked audio delivery.
   - ensure pointer-state telemetry is interleaved with audio chunks for UI/VST feedback.
5. Port the same canonical state contract to:
   - `tmrvc-engine-rs`
   - ONNX export wrappers
   - **VST parameter/control bridge (expose all 8-D controls and pacing to DAW automation).**
   - **VST Deployment Constraint:** VST plugin distribution strictly requires an ONNX/Rust backend. Deploying full PyTorch inside a VST is forbidden due to binary size, dependency conflicts, and unpredictable latency. The VST runtime must execute the ONNX-exported graph exclusively.
   - **Standalone Host Implementation:** Develop a lightweight, Python-free executable ("Standalone Host") using the `tmrvc-engine-rs` and a Rust-based GUI (e.g., `egui`). This serves as the primary real-time VC tool for general users/streamers, sharing the same ONNX backend and `SpeakerProfile` logic as the VST.
   - **Rust-Native G2P Requirement:** To support TTS in VST/Standalone without Python, the Rust engine MUST integrate a native G2P library (e.g., `lindera` + `openjtalk-sys` or `rust-jieba`). Relying on a Python server for text analysis is forbidden due to network/IPC jitter breaking the 10ms budget.
   - **Frozen frontend parity obligation:** A Rust-native G2P is acceptable only if it is proven to emit the same canonical `phoneme_ids` and `text_suprasegmentals` as the frozen Python training frontend on the same normalized text. "Close enough" segmentation or accent extraction is not acceptable for sign-off.
   - Worker 04 must therefore expose or consume a shared golden-fixture suite for text frontend parity, including tokenization, accent/tone features, language routing, and fallback markers.
6. **v3.0 `SpeakerProfile` boundary:**
   - v3.0 runtime voice switching is prompt-based only: `speaker_embed`, `prompt_codec_tokens`, and optional `prompt_kv_cache`.
   - `SpeakerProfile` must not carry runtime weight deltas or require model hot-swaps in the v3.0 mainline.
   - post-v3.0 per-actor adaptor/model variants may be explored as a separate artifact-management track, but they must not change the canonical v3.0 `SpeakerProfile` serialization contract.
7. Ensure streaming-safe cache update:
   - no hidden re-expansion of full text sequence per step
   - no full-sequence recomputation when not needed
8. Add Zero-Shot / Few-Shot Voice Cloning API support:
   - accept `reference_audio` (base64 or file upload) and optional `reference_text` in the TTS request.
   - run the `Speaker Prompt Encoder` on the fly to extract `speaker_embed` and `prompt_kv_cache`.
   - **Persist or retrieve these prompt features via the canonical `SpeakerProfile` contract managed in `models/characters/` (Casting Gallery).**
   - cache these prompt features across conversational turns for the same speaker to avoid redundant extraction.
9. Define text-side cache contract:
   - text frontend outputs (`phoneme_ids` and `text_suprasegmentals` when supported) are computed once per request or once per turn boundary
   - text encoder outputs are computed once per request or once per turn boundary
   - cross-attention keys/values for active text/context window are cached and reused across 10 ms steps
   - pointer advancement updates window indices or masks, not full encoded text recomputation
   - dialogue-context refresh happens only when the external turn/context input changes
10. Freeze the runtime finite-state behavior:
   - max one text-unit advance per frame in the initial mainline
   - explicit EOS / stop condition
   - explicit behavior on terminal hold
   - explicit serialization of `stall_frames` or equivalent deadlock-detection state
   - explicit `max_frames_per_unit` guardrail and force-advance behavior
   - explicit skip-protection when advance confidence is weak or inconsistent
   - explicit force-advance side-effect policy:
     - how `progress_value` is reset or clipped
     - how `acoustic_history` remains continuity-safe without hidden-state smoothing, cache rewriting, or decoder cross-fade heuristics outside the shared serializable contract
     - how forced-advance is surfaced in telemetry and parity tests
   - explicit numeric advance-decision rule:
     - thresholding method for `advance_logit` / `advance_prob`
     - tie-break and near-threshold behavior
     - parity-safe comparison semantics shared by Python and Rust
11. Review VC path for semantic-context hooks so the new runtime state shape does not block future VC improvements.
   - define the causal fusion path for VC semantic features with optional dialogue/text context
   - ensure this path is explicit even when VC bypasses pointer progression
12. Add dialogue-performance controls to runtime contract where supported:
   - scene or context embedding input
   - phrase pressure / interruption pressure
   - optional breath tendency / release bias
13. Implement `routes/admin.py` for Worker 12:
   - `GET /admin/health` (detailed system health)
   - `GET /admin/telemetry` (VRAM, latency, model state)
   - `POST /admin/load_model` (switch checkpoints)
   - `GET /admin/models` (list available checkpoints)
   - `GET /admin/runtime_contract` (pointer / voice-state contract introspection)
14. Implement WebUI-facing orchestration and evaluation routes:
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
   - these routes are the authoritative multi-user contract; direct manifest file access is not a mainline API
   - Worker 04 owns HTTP transport, auth/middleware integration, and typed response semantics
   - Worker 07 owns curation record storage semantics and mutation correctness behind this API boundary
15. Define few-shot runtime behavior:
   - when `reference_audio` is encoded
   - **Few-Shot Latency Management:** Acknowledge that encoding a 3-10s prompt takes >10ms.
     - **Preferred Workflow:** Require humans to pre-encode and save profiles via the **Casting Gallery** (Worker 12) for production-grade low-latency starts.
     - **Real-time extraction:** If `reference_audio` is provided on-the-fly, implement a `wait_for_prompt` API flag in the TTS request. When `true`, the server waits for encoding before starting the 10ms-step generation; when `false`, the server returns an error if encoding exceeds the budget.
   - how **`speaker_profile_id`** loads the precomputed embedding and prompt tokens from the Casting Gallery
   - how `speaker_embed` or prompt cache is reused across turns
   - freeze `max_prompt_seconds_active`, `max_prompt_frames`, `max_prompt_kv_tokens`, and `max_prompt_cache_bytes` as runtime-enforced limits owned by `configs/constants.yaml`
   - if enrollment evidence exceeds runtime limits, apply a deterministic prompt-selection/compression rule and record it in `SpeakerProfile`
   - how prompt encoding latency is measured separately from steady-state generation latency
   - how runtime logs enough metadata to reproduce external-baseline comparisons
   - how conflicts between `speaker_embed` and `prompt_codec_tokens` are resolved at runtime
   - how much authority prompt texture has relative to speaker-identity anchoring
   - how Rust/VST degrade deterministically to reduced prompt summary or `speaker_embed`-only mode when the prompt budget would break real-time guarantees
16. Freeze runtime-budget constants in `configs/constants.yaml`:
   - `max_text_units_active`
   - `max_dialogue_context_units`
   - `max_prompt_seconds_active`
   - `max_prompt_frames`
   - `n_prompt_summary_tokens` (frozen initial value: 32)
   - `max_prompt_kv_tokens`
   - `max_prompt_cache_bytes`
   - `max_acoustic_history_frames`
   - `max_cross_attn_kv_bytes`
   - `streaming_latency_budget_ms`
   - `streaming_hardware_class_primary`
   - these constants are part of the public runtime contract and must not be re-declared independently in Python, Rust, ONNX, or VST
   - these entries must be numeric and versioned before Worker 04 implementation is considered stable; placeholder names without values are not acceptable
   - the measured values and the engineering margin used to choose them must be recorded in docs so Worker 06 can validate against the same budget
17. Define CFG runtime behavior:
   - guidance must be defined for all conditioned outputs used by the runtime:
     - `guided_logits_a`
     - `guided_logits_b`
     - `guided_advance_logit`
     - `guided_progress_delta`
     - `guided_boundary_confidence` when enabled
   - the canonical formula is `guided_x = uncond_x + cfg_scale * (cond_x - uncond_x)` followed by documented post-guidance clamps where required for numeric stability
   - unconditional pass is produced by zeroing out or dropping exactly the Worker 01 mask set:
     - `explicit_voice_state`
     - `delta_voice_state`
     - `ssl_voice_state`
     - `speaker_embed`
     - `prompt_codec_tokens` or `prompt_kv_cache`
     - `dialogue_context`
     - `acting_intent`
     - `local_prosody_latent`
   - unconditional pass may reuse the same KV cache structure but with zeroed conditioning; it must not require a separate model instance
   - safe `cfg_scale` bounds: clamp to `[1.0, 3.0]` by default to avoid pointer instability; values above 3.0 require explicit opt-in
   - v3.0 required modes:
     - `off`
     - `full` two-pass CFG
   - post-v3.0 optimization modes:
     - `lazy` CFG refresh every N frames with bounded drift policy
     - `distilled` one-pass CFG-compatible path
   - define when VST / real-time engine must clamp or disable expensive CFG modes
   - real-time priority policy:
     - Rust engine (ONNX) / VST default to `off` in v3.0 mainline; this means **drama-grade CFG-enhanced acting is available only via Python serve (GPU) in v3.0**. DAW / VST users receive pointer-driven pacing and 8-D physical control but not CFG-boosted expressiveness until `lazy` or `distilled` modes are validated and promoted, as running full 2-pass CFG on CPU/ONNX would severely violate the 10ms real-time budget.
     - **CFG Circuit Breaker:** In all runtimes, implement a monitor for the 10ms budget. If frame-step latency exceeds 8ms, the runtime must automatically downgrade from `full` to `lazy` or `off` for the remainder of the utterance.
     - **Benchmark / claim mode exception:** Any benchmark, parity, external-baseline, or release-signoff run must use a frozen CFG mode with the circuit breaker disabled. If the requested mode exceeds budget during a claim-valid run, the run fails or is marked invalid; it must not silently downgrade and remain eligible for comparison.
     - `lazy` or `distilled` may become the default only after they ship and pass the full-mode validation gates; promotion is a priority for post-v3.0 because it directly gates the drama-grade claim for Rust/VST paths
     - `full` two-pass CFG is non-default in hard real-time paths and must justify itself against the 10 ms budget
     - **Python serve GPU latency verification:** `full` two-pass CFG doubles the per-step compute (two forward passes per 10 ms frame). Worker 04 must measure whether GPU inference on the frozen primary hardware class can complete both passes within the 10 ms frame budget. If it cannot, the Python serve drama-grade CFG claim must either (a) accept increased latency with documented trade-off, or (b) adopt batch-wise guidance (compute unconditional pass every N frames) as the default `full` mode. Worker 06 must include this measurement in the latency benchmark suite before Stage B.
   - the pointer update rule must consume `guided_advance_logit` / `guided_progress_delta` in `full` mode; using guided acoustic logits with unguided pointer outputs is forbidden unless explicitly labeled an ablation
18. Define cache synchronization behavior:
   - sliding-window text attention policy or deterministic cache re-indexing
   - explicit invalidation rules when the pointer crosses a window boundary
   - interaction between prompt cache, text cache, and autoregressive acoustic cache
19. Define waveform-decoder runtime policy:
   - initial v3 policy: the codec-native decoder (inverse RVQ + neural vocoder from the codec model) is the primary waveform path
   - candidates for future enhancement: `Vocos`, `HiFi-GAN` variants, or `DAC` decoder
   - streaming latency budget: waveform decoding must complete within the 10 ms frame budget; if an enhanced decoder exceeds this, fall back to the codec-native decoder
   - fallback path: always maintain the codec-native decoder as a guaranteed-available fast path
   - quality measurement: Worker 06 must include waveform artifact rate (clicks, buzzing, metallic artifacts) in the TTS quality gate
   - **v3.1 acoustic refinement runtime preparation:**
     - define the runtime integration point for the v3.1 acoustic refinement module that takes coarse AR codec tokens and produces refined full-RVQ tokens
     - the codec-native decoder remains as fallback when refinement is disabled or latency-constrained
20. Define multilingual / code-switch runtime policy:
   - utterance-level vs token-level language conditioning inputs
   - cross-lingual few-shot prompting when prompt language differs from target language
21. Add text frontend parity contract for Python vs Rust:
   - freeze the exact normalization -> G2P -> `phoneme_ids` + `text_suprasegmentals` mapping used by training
   - export a golden test corpus covering Japanese accent phrases, tonal-language examples, code-switch cases, punctuation/number normalization, and fallback cases
   - Rust runtime must either:
     - reproduce the frozen Python outputs exactly on that corpus, or
     - reject the request / declare a downgraded fallback mode that is excluded from parity/claim-valid runs
   - shipping a Rust text frontend without this proof is forbidden because it would create a train/serve contract split
22. Define pointer fallback runtime policy:
   - configurable `max_frames_per_unit`
   - forced-advance after excessive stall
   - low-`boundary_confidence` fallback behavior
   - observability hooks so stalls/skips are visible in telemetry rather than silent
   - force-advance may update only serialized pointer fields and documented bias terms; runtime-only hidden-state smoothing, transformer-cache rewriting, or decoder cross-fade heuristics are forbidden in the mainline contract
23. Define UI event-stream contract:
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
24. Define API idempotency and conflict behavior for UI-originated writes:
   - upload/register/run/export/create-session actions accept `idempotency_key`
   - review/edit/policy actions accept canonical `metadata_version`
   - conflict responses must be typed, not plain text:
     - `stale_version`
     - `locked_by_other`
     - `already_submitted`
     - `policy_forbidden`
   - Worker 04 owns HTTP status codes, middleware, and response payload shapes; Worker 07 owns underlying record-version checks for curation data
25. Define artifact download contract for WebUI:
   - artifact responses return:
     - `artifact_id`
     - `artifact_type`
     - `download_url`
     - `expires_at`
     - `provenance_summary`
   - artifacts must be retrievable without shell access
26. Define `SpeakerProfile` runtime behavior:
   - canonical persistence keys and ownership metadata
   - exact prompt-cache invalidation rule when the prompt encoder or tokenizer changes
   - compatibility behavior for missing or stale `speaker_profile_id`
   - explicit rule that v3.0 profile loading must not mutate model weights at request time
27. Define frame-index contract for any runtime-visible alignment telemetry:
   - `sample_rate = 24000`
   - `hop_length = 240`
   - inclusive `start_frame`, exclusive `end_frame`
   - parity with `tmrvc-core` frame alignment tests is mandatory


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
- expose the active few-shot speaker-conditioning source (**`speaker_profile_id`** or on-the-fly reference) for debugging
- expose the active prompt-budget mode and any prompt downselection/compression policy for debugging
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
- do not leave the 10 ms causal claim as a prose promise without frozen runtime-budget constants
- do not claim real-time few-shot support if prompt conditioning requires automatic degradation on the primary hardware class
- do not claim real-time CFG support if the required CFG mode misses the frozen streaming budget
- do not let unconditional CFG retain prompt or prosody conditioning through an undocumented side path
- do not expose lazy or distilled CFG modes without numerical and perceptual validation against the full mode
- do not make `full` two-pass CFG the default in Rust/VST unless latency proof exists on the target budget
- do not leave waveform quality to an unspecified decoder fallback
- do not let silent stall-recovery or skip-recovery heuristics diverge across runtimes
- do not preserve real-time latency by performing undocumented hidden-state surgery during force-advance


## Handoff Contract

- serving works with new v3 checkpoints
- API parameters are documented and stable
- Python serve, Rust runtime, ONNX export, and VST are all wired to the same control/state contract
- worker 05 can wire `dev.py` serve defaults to the new mode
- SSE event-stream implementation (task 22) is owned by worker 04 as part of `tmrvc-serve` routes; worker 12 is the consumer, not the implementer
- idempotency and conflict handling (task 23) are enforced in `tmrvc-serve` middleware; worker 12 sends `idempotency_key` and canonical `metadata_version` but does not implement the check


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
- route/engine test for full unconditional-conditioning mask parity
- route/engine test for lazy-CFG refresh policy and mode clamping
- cache-behavior test for sliding-window or cache re-indexing under pointer movement
- route/engine test for multilingual/code-switch propagation
- route/engine test for `max_frames_per_unit` forced-advance behavior
- route/engine test that force-advance does not violate serialized state invariants
- Python vs Rust parity test for advance-threshold and near-threshold decisions
- route/engine test for low-`boundary_confidence` fallback behavior
- route conflict test for stale `metadata_version` submissions
- idempotency-key test for retried UI write requests
- Python vs Rust pointer-state parity smoke test
- PyTorch vs ONNX runtime-contract parity test for pointer / `voice_state` fields
- streaming numerical parity test for batch vs frame-by-frame CausalConv1d path
- `SpeakerProfile` stale-version / invalidation test
