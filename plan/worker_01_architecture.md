# Worker 01: v4 Architecture Cutover

## Scope

Worker 01 owns the core schema and contract cutover for `v4`.
This is not a request to extend the current `v3` `8-D voice_state` contract.

Worker 01 must define the canonical `v4` acting architecture boundary used by:

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

Worker 01 must define the canonical conditioning split for `v4`:

- `speaker identity`
- `explicit physical controls`
- `acting texture latent`
- `dialogue / semantic intent`
- `pacing controls`

Rules:

- physical controls and latent controls must be different tensors
- latent controls must not be aliased as "more physical dimensions"
- no hidden second timing path outside pointer semantics
- `10 ms` causal operation remains non-negotiable

### 2. Replace the public `8-D` acting contract with the `v4` core schema

Worker 01 must define the new public schema for:

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

Worker 01 must freeze the first `v4` explicit physical control set.
Target range is `12-D` to `16-D`.

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

Worker 01 must define the `v4` latent path as a first-class contract.

Minimum fields:

- latent dimension
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

### 6. Freeze `TrajectoryRecord`

Worker 01 must define the `v4` replay artifact.

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

### 7. Add schema roundtrip tests

Required tests:

- physical control registry serialization roundtrip
- `IntentCompilerOutput` roundtrip
- `TrajectoryRecord` roundtrip
- pointer-synchronous edit target roundtrip
- compatibility checks between core schema and export schema

## Out Of Scope

Do not reopen these as Worker 01 tasks:

- preserving `v3` `8-D` compatibility
- generic pointer-head invention from zero
- generic `SpeakerProfile` history
- old `v3` prompt-only serving behavior

## Exit Criteria

- `tmrvc-core` exposes the canonical `v4` acting schema
- the physical registry is frozen and documented
- `IntentCompilerOutput` and `TrajectoryRecord` are versioned
- docs point to the `v4` schema as the only active mainline boundary
- roundtrip tests exist for the new artifacts
