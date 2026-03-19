# Track: v4 Serving And Runtime Cutover

## Scope

This track owns the serve/export/runtime cutover for `v4`.
This is not a request to keep extending the current `v3` request surface.

The critical-path `v4` slice is:

- canonical `v4` request/response schemas
- physical-plus-latent conditioning in serve and runtime
- trajectory import/export/replay
- real causal streaming on the serve path
- canonical backend speaker enrollment and profile handling

## Primary Files

- `tmrvc-serve/src/tmrvc_serve/schemas.py`
- `tmrvc-serve/src/tmrvc_serve/routes/tts.py`
- `tmrvc-serve/src/tmrvc_serve/routes/ui.py`
- `tmrvc-serve/src/tmrvc_serve/uclm_engine.py`
- `tmrvc-export/src/tmrvc_export/`
- `tmrvc-engine-rs/src/processor.rs`
- `tmrvc-engine-rs/src/ort_bundle.rs`
- `tests/serve/`
- `tmrvc-engine-rs/tests/`

## Open Tasks

### 1. Replace the serve schema with the `v4` contract

Required request surfaces:

- simple mode
- advanced physical-control mode
- prompt / acting mode
- trajectory replay mode

Required control fields:

- physical controls
- acting latent or acting macro controls
- pacing controls
- speaker profile or speaker embedding anchor
- optional reference-audio inputs

Rules:

- no `v3`-only `explicit_voice_state [8]` assumption in the new mainline route
- prompt-driven inference and deterministic replay must be distinct route semantics
- schema-version mismatch must fail explicitly

### 2. Add the `v4` compile / replay / patch / transfer API surface

Required routes:

- `POST /ui/workshop/compile`
- `POST /ui/workshop/generate`
- `POST /ui/workshop/trajectories/{trajectory_id}/patch`
- `POST /ui/workshop/trajectories/{trajectory_id}/replay`
- `POST /ui/workshop/trajectories/{trajectory_id}/transfer`

Rules:

- creator-facing compile may accept prompts, tags, and context
- replay and transfer must accept canonical artifacts directly
- deterministic replay must not silently reinterpret prompts
- transfer is part of the `v4` target scope, not an optional appendix

### 3. Persist `v4` trajectory artifacts with optimistic versioning

This track owns the authoritative persistence layer for trajectory artifacts.

Minimum requirements:

- stable `trajectory_id`
- schema-versioned artifact persistence
- optimistic patch versioning
- explicit provenance:
  - compile
  - replay
  - transfer
  - patched replay

UI state is not an authoritative store.

### 4. Update ONNX/export boundaries for the `v4` conditioning split

Export must preserve separate inputs for:

- explicit physical controls
- acting latent path
- speaker conditioning
- pacing / pointer-side control signals, where applicable

Rules:

- export wrappers must not collapse physical and latent controls into one unnamed blob
- Python, ONNX, and Rust must consume the same ordering and shape conventions

### 5. Deliver real causal serve-path streaming

`v4` does not permit batch fallback to count as claim-valid streaming.

Required behavior:

- actual incremental generation on the serve path
- pointer/control telemetry on that path
- parity with the declared runtime contract
- measurable latency and RTF on the frozen hardware class

With Mimi at 12.5 Hz, each audio frame spans 80 ms. First-token latency target: < 160 ms (2 codec frames).
Streaming granularity is 80 ms per chunk, not 10 ms. Control telemetry can still be emitted at higher rate.

### 6. Keep canonical backend speaker enrollment

Serving must provide the authoritative encode/persist behavior used by the UI.
Few-shot enrollment must not depend on frontend-local dummy embeddings.

### 7. Support the raw-audio bootstrap output contract

Serve/runtime must be able to consume speaker profiles and reference artifacts produced by the new bootstrap/curation pipeline.

At minimum, the runtime surface must tolerate:

- pseudo speaker identities
- confidence-bearing compiled controls
- reference-derived acting initialization

## Required Tests

- route schema tests for all `v4` request modes
- compile determinism test, if the compiler is deterministic for a given frozen mode
- trajectory persistence roundtrip
- optimistic-version conflict test
- deterministic replay test from frozen `TrajectoryRecord`
- transfer route test using the same acting artifact on another speaker anchor
- backend enrollment path test
- real causal streaming test
- Python vs ONNX vs Rust input-contract parity test

### 8. Migrate `ContextStylePredictor` from Claude API to open-weight LLM

> See `track_architecture.md` §6 for model selection rationale. This section covers implementation migration only.

The current `context_predictor.py` uses `claude-haiku-4-5-20251001` via Anthropic API.
This must be replaced with the open-weight Intent Compiler model selected in `track_architecture.md`:

- Primary: `Qwen/Qwen3.5-35B-A3B`
- Fallback: `Qwen/Qwen3.5-4B`

Required changes:

- replace `anthropic` client with `transformers` or `vllm` backend
- keep the existing rule-based fallback as a no-GPU safety net
- ensure compile output is deterministic for a given model + prompt
- add model version to `IntentCompilerOutput.provenance`

## Out Of Scope

Do not reopen:

- preserving the `v3` `8-D` request ABI
- batch-chunked pseudo-streaming as a release-valid substitute
- frontend-local speaker embedding hacks

## Exit Criteria

- the `v4` serve schema is canonical and versioned (schema version field present and validated)
- trajectory compile / replay / patch / transfer routes exist and return correct HTTP status codes
- deterministic replay produces bit-identical output for the same `TrajectoryRecordV4` input (see `docs/design/acceptance-thresholds.md` §V4-7)
- real causal streaming exists on the serve path and meets thresholds defined in `docs/design/acceptance-thresholds.md` §V4-6
- export and Rust runtime consume the same conditioning tensor shapes (verified by cross-runtime parity test)
- canonical backend enrollment is the only claim-valid path (no dummy embedding code paths remain)
- `ContextStylePredictor` uses open-weight LLM (no Anthropic API dependency in mainline)
