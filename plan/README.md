# UCLM v3 Implementation Plan

## Goal

This plan translates the MFA-free `UCLM v3` direction into executable engineering work. The target state is:

- no MFA dependency in the main training or inference path
- no `durations.npy` requirement for the main TTS path
- causal pointer-based text consumption at 10 ms steps
- **High-fidelity Zero-Shot / Few-Shot speaker adaptation is a mainline requirement (not optional)**
- preserved low-latency VC path with better semantic-context handling
- drama-grade conversational acting is supported by architecture, data, and validation
- **Measured parity or superiority against a fixed, version-pinned external baseline suite**

This directory is intended for parallel execution by multiple workers. Each worker file is standalone and names the files it is allowed to touch first.


## Non-Negotiable Constraints

- MFA is forbidden in the v3 mainline architecture.
- `DurationPredictor` must not remain the main TTS control path.
- Inference must stay causal and 10 ms step based.
- VC must not regress into large look-ahead or offline-only behavior.
- v2 legacy behavior may remain, but only as compatibility or ablation mode.
- online MAS/CTC or bootstrap labels may assist training only as auxiliary or transitional supervision; they must not become a permanent release dependency for pointer progression
- Python / Rust / ONNX / VST must consume one shared serializable contract owned in `tmrvc-core`, not re-declare it independently
- all human-operated workflows from dataset ingest/upload through curation, export, audition, and evaluation must be available from the WebUI without requiring CLI usage


## Execution Reality

- plans must target the existing monorepo structure first
- shared runtime and tensor contracts live in `tmrvc-core`
- dataset and cache contracts live in `tmrvc-data`
- the repository currently contains `tmrvc-gui` (PySide6 / Qt), but the v3 HITL control plane is a separate Gradio/WebUI mainline because multi-user audit, blind evaluation, and role separation are first-class requirements


## Target End State

### Training

- dataset can train TTS without `durations.npy`
- trainer supports pointer loss and optional auxiliary alignment regularization
- model can learn online text progression
- classifier-free guidance compatible training is built into the mainline conditioning path
- codec/token hierarchy is explicit rather than treated as one undifferentiated token stream
- v2 duration branch is optional and explicitly marked legacy
- high-quality local-model pseudo-annotation can bootstrap supervision from raw short-utterance corpora
- **few-shot speaker adaptation is supported as a first-class mainline path, not an afterthought**

### Inference

- server TTS route uses pointer state rather than duration expansion
- engine exposes pace / hold / boundary-bias controls
- engine, ONNX export, and VST preserve the same pointer / voice-state contract
- pointer state is incremental and stream-safe
- pointer and attention caches are updated with an explicit sliding-window / re-indexing policy
- runtime can express turn-taking tempo, hesitation, overlap pressure, and phrase-final release
- **runtime can consume short reference-speaker evidence for few-shot speaker adaptation via prompt-cache**
- waveform decoding quality is treated as a first-class bottleneck, with an explicit vocoder / codec-decoder quality plan
- multilingual and code-switching inference is supported under a documented language-conditioning contract

### Tooling

- `dev.py` separates `v2 legacy` and `v3 pointer`
- docs and test suites reflect v3 as the default forward path
- human operators do not need `dev.py` or any CLI for normal ingest / review / evaluation workflows


## Worker Split

- `worker_01_architecture.md`
  - model graph changes
  - pointer state definition
  - compatibility boundaries
  - `voice_state` / `delta_state` contract
- `worker_02_training.md`
  - trainer, losses, sampling, config migration
- `worker_03_dataset_alignment.md`
  - dataset contracts, removal of duration hard dependency, phone coverage metrics
- `worker_04_serving.md`
  - server, Python runtime, Rust runtime, ONNX export, VST control path
- `worker_05_devops_docs.md`
  - `dev.py`, config flow, docs, legacy labeling
- `worker_06_validation.md`
  - tests, parity, latency, quality gates, experiment protocol
- `dramatic_acting_requirements.md`
  - cross-cutting requirements for drama-grade TTS
  - proof obligations for expressive quality claims
- `ai_curation_system.md`
  - full-system plan for multi-stage, iterative, selective AI curation
- `worker_07_curation_orchestration.md`
  - manifest, scheduling, retries, resumability
- `worker_08_curation_providers.md`
  - ASR, diarization, separation, refinement provider integration
- `worker_09_curation_selection.md`
  - scoring, rejection, subset promotion, anti-noise policy
- `worker_10_curation_export.md`
  - export to cache, metadata contracts, dev.py integration
- `worker_11_curation_validation.md`
  - curation system evaluation and acceptance
- `worker_12_gradio_control_plane.md`
  - Gradio/WebUI control plane for interactive drama workshop, HITL evaluation, and auditable human workflows

## Recommended Execution Order

### Stage A: Freeze architecture contracts

- worker 01 defines new interfaces
- worker 03 defines dataset/input contract
- worker 04 aligns Python/Rust/export runtime state expectations
- worker 07 defines curation manifest and stage contracts
- worker 08 defines provider contracts
- worker 12 drafts only the minimum admin/eval UI contract; it must not freeze unsupported inputs

### Stage B: Build training path

- worker 02 implements pointer training path
- worker 06 adds failing tests for new contracts
- worker 09 defines promotion/rejection policy for curated data
- worker 10 defines how curated outputs become trainable assets
- worker 12 prototypes the interactive evaluation arena

### Stage C: Migrate runtime and tools

- worker 04 switches inference path
- worker 05 rewrites menus/docs/config expectations
- worker 07 and worker 10 wire curation entrypoints and resumable operations
- worker 12 connects the reduced drama workshop to the stabilized runtime APIs

### Stage D: Validate and deprecate legacy path

- worker 06 runs integration matrix
- worker 05 marks v2/MFA flow as legacy-only
- worker 11 validates the curation system and signs off promotion policy
- worker 12 hosts the final blind A/B evaluation session

## Dependency Graph

- worker 01 blocks worker 02 and worker 04 on interface names
- worker 03 blocks worker 02 on dataset batch contract
- worker 02 blocks worker 04 on checkpoint schema and blocks worker 12 on training artifacts and evaluation data
- worker 04 blocks worker 05 and worker 12 on final CLI/API/Admin flags and blocks worker 06 on runtime/export parity targets
- worker 06 depends on all workers for final integration
- worker 07 blocks worker 08, worker 09, and worker 10 on manifest contract
- worker 08 blocks worker 09 on provider outputs and confidence semantics
- worker 09 blocks worker 10 on promotion/rejection outcomes
- worker 10 blocks worker 11 and worker 12 on exported artifact schema and manifest browsing

### Core v3 Path

```text
worker_01_architecture   worker_03_dataset_alignment
        |   \                      |
        |    \                     |
        v     +-------+    +------+
worker_04_serving     |    |
        |             v    v
        |       worker_02_training ---> worker_12_gradio
        |             |                 (WebUI Control Plane)
        v             v
     worker_06_validation
```

- `worker_01` and `worker_03` may start in parallel: `worker_01` fixes the model/runtime contract; `worker_03` fixes the dataset/batch contract.
- `worker_02` depends on both `worker_01` (pointer/state interfaces) and `worker_03` (batch/cache schema).
- `worker_04` consumes the final model/checkpoint/runtime contract across Python serve, Rust engine, ONNX export, and VST. It also depends on `worker_02` for checkpoint schema.
- `worker_12` depends on `worker_02` (training artifacts) and `worker_04` (runtime/admin APIs).
- `worker_06` is the final proof layer for the integrated v3 path.

### AI Curation Path

```text
worker_07_curation_orchestration
        |
        v
worker_08_curation_providers
        |
        v
worker_09_curation_selection
        |
        v
worker_10_curation_export
        |
        v
worker_11_curation_validation
```

- `worker_07` owns the manifest, stage boundaries, resumability, and legality state.
- `worker_08` produces raw annotations and refined pseudo-labels.
- `worker_09` decides what is promotable and to which bucket.
- `worker_10` converts promoted records into trainable/exportable assets.
- `worker_11` proves the curation system is trustworthy enough to feed training.

### Cross-Cutting Tooling and Policy

```text
worker_05_devops_docs
   ^            ^
   |            |
worker_04    worker_10
```

- `worker_05` should not freeze menus, docs, or operator flows until runtime and export contracts are stable.
- `dramatic_acting_requirements.md` and `ai_curation_system.md` constrain all workers and are not optional reference material.


## Worker Start Conditions

Use this section as the handoff gate. A worker should not start coding until its prerequisites below are satisfied.

### worker_01_architecture

- may start immediately
- must treat existing v2 runtime as compatibility-only
- must publish canonical names and tensor/state contracts before downstream implementation begins

### worker_02_training

- starts only after `worker_01` defines pointer/state interfaces
- starts only after `worker_03` defines batch/cache contracts
- must not invent new runtime-only fields without feeding them back to `worker_01`

### worker_03_dataset_alignment

- may start immediately
- must consume `dramatic_acting_requirements.md` and `ai_curation_system.md` as hard requirements
- must publish cache/dataset schema before `worker_02` hardens loaders

### worker_04_serving

- can prototype early against `worker_01`
- must not freeze public API or runtime state layout until `worker_02` checkpoint schema is stable
- must preserve causal 10 ms semantics and pointer-driven termination
- must carry the same state semantics into Python serve, Rust runtime, ONNX export, and VST-facing control surfaces

### worker_05_devops_docs

- starts in draft mode early, but final operator-facing flow waits for `worker_04` and `worker_10`
- must present `v3 pointer` as the default forward path and `v2 legacy` as compatibility
- must not document workflows that depend on MFA in the mainline

### worker_06_validation

- may write failing tests early
- final sign-off waits for `worker_01` through `worker_05`
- must validate both correctness and policy invariants, not just tensor shapes

### worker_07_curation_orchestration

- may start immediately
- must publish manifest, stage-state, split-state, and legality-state contracts before `worker_08`, `worker_09`, and `worker_10` freeze their outputs
- must keep reruns/resume semantics deterministic

### worker_08_curation_providers

- starts only after `worker_07` defines manifest and stage contracts
- must publish provider output schema, disagreement semantics, and transcript-refinement engine contract before `worker_09` scoring is finalized
- must not treat separated audio as automatically trusted waveform ground truth

### worker_09_curation_selection

- starts only after `worker_08` output/confidence semantics are defined
- must encode numeric thresholds and legality gates in config, not in prose only
- must define bucket promotion policy before `worker_10` export rules are finalized

### worker_10_curation_export

- starts only after `worker_09` promotion states and bucket semantics are stable
- must preserve conversation graph, provenance, split, and legality fields into exported assets
- must not silently drop fields required for drama-conditioned training

### worker_11_curation_validation

- may scaffold evaluation harnesses early
- final acceptance waits for `worker_07` through `worker_10`
- must prove annotation benefit, promotion correctness, split integrity, and legality enforcement

### dramatic_acting_requirements.md

- always in force
- any worker that drops dialogue context, local prosody capacity, or controllable pacing is non-compliant
- any worker that hides or demotes the 8-D physical `voice_state` behind vague style labels is non-compliant

### ai_curation_system.md

- always in force for workers 03 and 07 through 11
- any implementation that reduces curation to one-shot ASR without refinement, rejection, and promotion is non-compliant


## Global Deliverables

- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/external-baseline-registry.md`
- shared pointer / `voice_state` / cache-state schema in `tmrvc-core`
- model and training code for pointer-based TTS
- serving/runtime/export code for pointer-based TTS across Python, Rust, ONNX, and VST
- updated `dev.py` plan and menus
- integration tests for v2 legacy and v3 pointer modes
- numerical parity tests for batch vs streaming and PyTorch vs ONNX runtime paths
- measurable protocol for proving dramatic conversational acting
- measurable protocol for a fixed external-baseline comparison, including few-shot speaker adaptation
- pinned external-baseline registry and evaluation protocol with exact artifact/version/settings
- a production-grade AI curation plan for unlabeled wav-only corpora


## Definition of Done

- a v3 experiment can preprocess, train, finalize, and serve without MFA
- main TTS path does not require `durations.npy`
- if auxiliary alignment supervision is used during training, release sign-off still includes at least one alignment-free pointer configuration
- canonical v3 control path keeps 8-D `voice_state` and `delta_state` first-class from train through runtime
- code clearly separates `v2 legacy` from `v3 pointer`
- **Zero-shot speaker cloning works from a 3-10s reference clip with high similarity and disentangled prosody**
- latency and correctness tests pass
- Rust/ONNX/VST paths match the canonical runtime contract closely enough to pass parity gates
- docs state the new default path unambiguously
- expressive-quality claims are backed by reproducible evaluation
- **Blind A/B evaluation against the fixed external baseline artifact is completed and documented**
- the curation path can transform raw wav-only corpora into quality-gated trainable assets
