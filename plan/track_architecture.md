# Track: v4 Architecture Cutover

## Scope

This track owns the core schema and contract cutover for `v4`.
This is not a request to extend the current `v3` `8-D voice_state` contract.

This track must define the canonical `v4` acting architecture boundary used by:

- `tmrvc-core`
- `tmrvc-train`
- `tmrvc-export`
- `tmrvc-serve`
- `tmrvc-engine-rs`
- `tmrvc-gui`

## Primary Files

- `tmrvc-core/src/tmrvc_core/types.py`
- `tmrvc-core/src/tmrvc_core/constants.py`
- `configs/constants.yaml`
- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/onnx-contract.md`
- `docs/design/v4-master-plan.md`

## Open Tasks

### 1. Freeze the `v4` conditioning decomposition

This track must define the canonical conditioning split for `v4`:

- `speaker identity`
- `explicit physical controls`
- `acting texture latent`
- `dialogue / semantic intent`
- `pacing controls`

Rules:

- physical controls and latent controls must be different tensors
- latent controls must not be aliased as "more physical dimensions"
- no hidden second timing path outside pointer semantics
- causal operation remains non-negotiable; audio frames at 80 ms (12.5 Hz), control at 10 ms (100 Hz)

### 2. Replace the public `8-D` acting contract with the `v4` core schema

This track must define the new public schema for:

- explicit physical control vector
- delta physical control vector, if retained
- acting texture latent vector
- intent compiler output
- trajectory record

Minimum requirements:

- schema versioning
- serialization roundtrip
- provenance
- confidence and mask support where pseudo-labels or compiler uncertainty exist
- deterministic replay compatibility

### 3. Freeze the physical control registry

This track must freeze the first `v4` explicit physical control set.
`v4.0` freezes at `12-D`. Expansion to `16-D` is a post-`v4.0` option, not a `v4.0` deliverable.

The chosen set must:

- stay interpretable
- stay editable
- avoid near-duplicate semantics
- map onto measurable audio-derived observables

The registry must explicitly state:

- canonical order
- display labels
- physical interpretation
- default values
- frame-local vs slower-varying semantics
- extraction provenance

### 4. Freeze the acting texture latent contract

This track must define the `v4` latent path as a first-class contract.
`v4.0` freezes at `24-D` (`D_ACTING_LATENT`). The master plan's initial range was `16-D` to `32-D`; `24-D` is the selected point within that range.

Minimum fields:

- latent dimension (`24-D`)
- inference-time source:
  - prompt-derived prior
  - reference-derived estimate
  - direct replay artifact
- admissible macro controls
- serialization rule in trajectory artifacts

Rules:

- latent axes are not a public beginner-facing ABI
- replay must be possible without recompiling natural language
- latent must support same-speaker replay and cross-speaker reuse

### 5. Freeze `IntentCompilerOutput`

`IntentCompilerOutput` is still required in `v4`, but it must target the new contract.

Minimum fields:

- source prompt or tags
- compiled physical controls
- compiled acting latent prior
- pacing controls
- optional dialogue state
- warnings
- provenance
- schema version

Rules:

- raw prompt text is not the runtime contract
- compile output must be serializable and inspectable
- uncertain prompt interpretations must surface explicit warnings

### 6. Intent Compiler model selection

- Primary: `Qwen/Qwen3.5-35B-A3B` (MoE, 3B active)
- Fallback: `Qwen/Qwen3.5-4B` (dense, MoE infra がない場合)
- 選定理由:
  - 35B-A3B は active 3B で 4B dense より軽いが品質は 35B 相当
  - inference 時に毎回走るため latency が critical — MoE の低 active パラメータが有利
  - Qwen ファミリー統一により bootstrap annotation (Qwen3.5-9B) との言語能力が揃う
  - 201 言語/方言対応
- 移行対象: 現在の `context_predictor.py` は Claude Haiku API 依存 → open-weight に置換必須
  - API 依存はオフライン不可、再現性なし、コスト発生の三重苦
  - v4 では Intent Compiler の出力は deterministic かつ serializable でなければならない

### 7. Freeze `TrajectoryRecord`

This track must define the `v4` replay artifact.

Minimum fields:

- trajectory id
- source compile artifact
- realized pointer trace
- realized physical trajectory
- realized acting latent trajectory or resolved latent state
- realized acoustic/control tokens
- provenance and schema version

Rules:

- replay/edit/transfer must consume this artifact directly
- wall-clock-only addressing is forbidden
- opaque unversioned latent blobs are forbidden
- patching a local region must be a first-class use case

### 8. Extend text encoder contract for inline acting tags

The `v4` text encoder must accept enriched transcripts containing inline acting tags
alongside phoneme sequences. This follows the Fish Audio S2 rich-transcription approach
adapted for the `v4` hybrid architecture.

Required changes:

- extend the text encoder vocabulary to include acting tag tokens
- define a frozen acting tag vocabulary with categories:
  - vocal events: `[inhale]`, `[exhale]`, `[laugh]`, `[sigh]`, etc.
  - prosodic markers: `[emphasis]`, `[prolonged]`, `[pause]`
  - acting directives: `[angry]`, `[whisper]`, `[calm]`, `[excited]`, etc.
  - free-form acting brackets: `[...]` with learned embedding
- the pointer must treat acting tags as consumed text units (not skipped)
- acting tags must co-exist with physical controls and acting latent — they are a third conditioning path, not a replacement

Design constraints:

- tag embedding dimension must match the existing text encoder hidden dimension
- tag tokens must participate in the pointer attention mechanism
- the model must learn that `[angry]` predicts certain physical-control tendencies without overriding explicit physical slider values
- at inference time, if both an inline tag and an explicit physical override exist, the physical override takes precedence (editable-first principle)

### 9. Add schema roundtrip tests

Required tests:

- physical control registry serialization roundtrip
- `IntentCompilerOutput` roundtrip
- `TrajectoryRecord` roundtrip
- pointer-synchronous edit target roundtrip
- compatibility checks between core schema and export schema

### 10. Adopt Mimi codec and dual-rate time axis

v4.0 replaces the internal `EmotionAwareCodec` with **Mimi** (Kyutai, 2024).

Selected codec:

- **Mimi** (`kyutai/mimi` on HuggingFace)
- 24kHz, 79.3M params, pre-trained (CC-BY-4.0)
- 8 RVQ quantizers × 2048 bins
- 12.5 Hz frame rate (80ms per frame)
- Fully streaming / causal

Selection rationale:

- pre-trained weights public — no codec training required
- 12.5 Hz reduces UCLM sequence length by 8× vs 100 Hz → faster training and inference
- semantic codebook at position 0 aligns with v4's intent-first design
- fully streaming and causal — compatible with v4 runtime contract
- 2024 vintage, adopted by Moshi — proven in production

Dual-rate time axis:

- **audio stream** (codec tokens): 12.5 Hz — predicted by UCLM autoregressively
- **control stream** (physical controls, acting latent): 50–100 Hz — kept high for fine-grained editability
- **pointer**: operates at audio frame rate (12.5 Hz) for text-to-frame alignment
- control values are interpolated or held between audio frames
- TrajectoryRecord stores both rates: `physical_trajectory` at control rate, `acoustic_trace` at codec rate

Constant changes:

- `rvq_vocab_size`: 1024 → 2048
- `hop_length`: 240 → 1920
- `codec_frame_rate`: 12.5 (new)
- `control_frame_rate`: 100 (new, explicit)
- `n_codebooks`: 8 (unchanged)
- `control_slots`: 4 (unchanged)

What does NOT change:

- 12-D physical control registry
- 24-D acting texture latent
- IntentCompilerOutput / TrajectoryRecord schema (fields are rate-agnostic)
- Pointer semantics (still advance/hold per frame — just fewer frames)

## Out Of Scope

Do not reopen:

- preserving `v3` `8-D` compatibility
- generic pointer-head invention from zero
- generic `SpeakerProfile` history
- old `v3` prompt-only serving behavior

## Exit Criteria

- `tmrvc-core` exposes the canonical `v4` acting schema (merged to main, import test passes)
- the 12-D physical registry is frozen: canonical order, display labels, defaults documented in `voice_state.py` and `constants.yaml`
- `IntentCompilerOutputV4` and `TrajectoryRecordV4` pass JSON/msgpack serialization roundtrip tests
- docs point to the `v4` schema as the only active mainline boundary
- roundtrip tests exist and pass for: physical control registry, `IntentCompilerOutputV4`, `TrajectoryRecordV4`, pointer-synchronous edit target
- "frozen" means: merged to main, reviewed, roundtrip test suite green, no open blocking issues
