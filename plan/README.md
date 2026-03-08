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
- all authoritative multi-user human workflows must go through one typed backend API boundary; filesystem-direct UI access is dev-only and is not the mainline contract
- frame-indexed artifacts must use the same audio stepping contract as `tmrvc-core`: `sample_rate=24000`, `hop_length=240`, and `T = ceil(num_samples / hop_length)` with exact parity to validated frame-alignment tests

## Architectural Reset Policy

UCLM v3 is not an incremental refinement of v2.
It is a clean architectural reset motivated by the conclusion that v2 is architecturally incomplete for the project's target claims.

Implications:

- v2 is retained only for compatibility, ablation, and historical comparison.
- v2 design constraints must not limit v3 interface, runtime, or training decisions.
- any reuse of v2 code is an implementation convenience only and does not imply contract inheritance.
- sign-off for v3 depends on v3 meeting its own proof obligations, not on relative improvement over v2.


## Release Scope

The scope is split into three layers:

- `v3.0 core proof obligations`
- `v3.0 quality amplifiers`
- `post-v3.0 optimization tracks`

This prevents future quality/performance upgrades from being confused with the proof obligations of the initial MFA-free pointer release.

## SOTA Claim Policy

TMRVC may claim SOTA or state-of-the-art competitiveness only under the following conditions:

- the primary external baseline is frozen before large-scale Stage B training begins
- the baseline artifact, tokenizer, prompt rule, reference lengths, inference settings, language set, and hardware class are fixed and versioned
- evaluation is run on the same frozen protocol for TMRVC and the baseline
- no release report may replace a failed direct comparison with internal-only comparisons against v2

If TMRVC fails to match or exceed the pinned primary baseline on any metric declared as a primary claim axis, the corresponding claim is blocked.
Acceptable fallback is to narrow the claim explicitly.
Unacceptable fallback is to keep the broad claim and explain the deficit away as a trade-off after the fact.

### Codec Model Ownership

The v3 plan assumes a stable RVQ codec model (`n_codebooks=8`, 24 kHz, `hop_length=240`). Codec model selection, version pinning, and any retraining are owned by Worker 01 (artifact identity) and Worker 06 (quality validation). The codec artifact and its decoder must be frozen before Stage B large-scale training begins. Codec-level factorization or replacement is a v3.1/v4 investigation path (see Worker 01 § Acoustic Refinement Roadmap), not an initial v3 requirement.

### v3.0 Core Proof Obligations

- causal pointer-driven TTS without MFA
- differentiable teacher-forced pointer-training surrogate explicitly mapped to the hard runtime pointer state-transition contract
- shared pointer / cache / `voice_state` / `speaker_profile` schema in `tmrvc-core`
- first-class 8-D physical control with trainable targets, masks, provenance, and validation
- bounded dialogue-context path with runtime budgets frozen in `configs/constants.yaml`
- canonical suprasegmental text-feature contract for accent / tone / phrase-boundary cues in languages that require it
- few-shot speaker adaptation from a short reference clip under a reproducible and budget-bounded contract
- v3.0 `SpeakerProfile` is prompt-evidence based (`speaker_embed`, prompt tokens/cache) and must not require runtime weight mutation
- deterministic bootstrap-alignment artifact contract for transitional supervision
- one authoritative WebUI/backend API surface for ingest, curation, generation, export, and evaluation
- parity and latency validation across Python / Rust / ONNX / VST for the shared pointer / `voice_state` / cache-state contract
- direct evaluation against at least one fixed, version-pinned public external baseline
- no SOTA-quality claim unless the pinned baseline comparison passes on the declared claim axes

### v3.0 Quality Amplifiers

- modern transformer backbone (`RoPE`, `GQA`, `SwiGLU`, `RMSNorm`, `FlashAttention2`) as a candidate default; rollback is allowed if the frozen core-proof gates pass without it and the fallback wins the quality/latency tradeoff
- flow-matching prosody predictor as the preferred expressive path, with a simpler fallback path retained until the pointer-core proof is independently closed
- CFG mainline modes: `off` and `full`
- richer WebUI ergonomics

### v3.0 Proof Layering Policy

To keep the initial MFA-free pointer release scientifically interpretable, the plan must distinguish:

- `v3.0 core proof obligations`
  - pointer-driven causal TTS
  - shared serializable contracts in `tmrvc-core`
  - frozen 8-D `voice_state` semantics
  - frozen runtime budgets and parity gates
  - deterministic bootstrap and few-shot evaluation contracts
- `v3.0 quality amplifiers`
  - backbone modernization
  - flow-matching prosody
  - `full` CFG
  - richer WebUI ergonomics

Rule:

- a failed quality amplifier must not be allowed to obscure whether the pointer architecture itself passed or failed
- every amplifier promoted into the shipping `v3.0` release must retain an ablation or rollback path so Worker 06 can isolate regressions cleanly
- worker reports must separate `core-proof pass/fail` from `quality-amplifier pass/fail`

### Post-v3.0 Optimization Tracks

- CFG acceleration modes `lazy` and `distilled`
- alternative vocoders / acoustic refinement stages (`v3.1` upgrade path)
- advanced quantization strategies


## SOTA Landscape Awareness (2025-2026)

This plan is informed by the dominant trends in SOTA TTS systems as of early 2026. Exact paper references and dates are tracked in `plan/arxiv_survey_2026_03.md`. The high-level conclusions used by this plan are:

### Dominant pattern: 2-stage AR + Non-AR

Many recent strong systems use a two-stage pipeline:
1. AR language model over discrete semantic/codec tokens
2. Non-AR refinement (flow matching, diffusion, or DiT) for high-fidelity continuous acoustic generation

TMRVC v3 initial mainline uses flattened single-stage codec prediction for simplicity and streaming compatibility. A 2-stage flow-matching acoustic refinement module is planned as the v3.1 quality upgrade path (see worker_01 § Acoustic Refinement Roadmap).

### Additional implications

- progress-aware alignment structure is a promising direction, but it does not remove the need for explicit parity gates in a 10 ms causal runtime
- disentanglement is now a first-class design problem, but TMRVC still needs repository-local proof that prompt conditioning does not leak prosody through hidden paths

### Scale reality

SOTA systems train on 100K-5M hours of data with 0.5B-1.5B+ parameter LMs. TMRVC must define its target scale assumptions explicitly (see § Scale Assumptions below).


## Scale Assumptions

The plan must operate under explicit scale assumptions so architecture, data, and compute decisions are grounded.

- **Target model scale (initial v3):** 100M-300M parameter UCLM core + auxiliary modules. Sub-1B total.
- **Target data scale (initial v3):** 10K-100K hours curated data. The curation system is designed to maximize quality at this scale rather than competing on raw volume.
- **Scaling strategy:** Quality-first at modest scale, with architecture designed to scale up without redesign. The 2-stage refinement upgrade path (v3.1) is the primary quality lever before scaling data.
- **Non-goal for initial v3:** Matching 1M+ hour training runs or 1B+ parameter models. Competitive quality is pursued through architectural efficiency (pointer alignment, disentanglement, CFG, flow-matching prosody) rather than scale alone.
- **Scale gap risk:** Comparing a 300M single-stage AR model against a 1.7B 2-stage model (Qwen3-TTS) on zero-shot similarity or waveform quality is structurally disadvantaged. The frozen baseline assignment is: **CosyVoice 3 (0.5B) is the `primary` baseline** for fair scale-aligned evaluation; **Qwen3-TTS (1.7B) is the `secondary` ceiling baseline** for aspirational comparison. If TMRVC cannot match a baseline on a declared primary claim axis, the honest response is to narrow the claim rather than blame scale. Worker 06 must pre-register which axes are "scale-sensitive" (e.g., raw MOS on unseen languages, pure audio fidelity) versus "architecture-sensitive" (e.g., controllability, disentanglement, streaming latency) so that sign-off language is grounded in the actual gap, not post-hoc rationalization.
- **Data augmentation policy:** Data augmentation (speed perturbation, noise injection, pitch shifting, SpecAugment, etc.) is not adopted as a default mainline training technique in initial v3. The curation system is the primary data-quality lever. If Worker 06 quality gates show that data diversity is a binding constraint at the target scale, augmentation strategies may be evaluated as a Stage C or post-v3.0 addition. Any adopted augmentation must be documented, switchable, and its effect on pointer-alignment stability must be validated before mainline adoption.


## Execution Reality

- plans must target the existing monorepo structure first
- shared runtime and tensor contracts live in `tmrvc-core`
- dataset and cache contracts live in `tmrvc-data`
- `tmrvc-gui` is a Gradio-only WebUI; PySide6/Qt has been fully deprecated and removed. The Gradio control plane is the sole HITL surface for multi-user audit, blind evaluation, role separation, and all operational workflows
- `plan/arxiv_survey_2026_03.md` records the dated research survey used to inform architecture decisions; workers should update it when a new paper materially changes a design choice


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
- runtime enforces explicit prompt-conditioning budgets (`max_prompt_seconds_active`, `max_prompt_frames`, `max_prompt_cache_bytes`) so few-shot conditioning cannot silently break the 10 ms causal budget
- waveform decoding quality is treated as a first-class bottleneck, with an explicit vocoder / codec-decoder quality plan
- **v3.1 upgrade path: 2-stage acoustic refinement (flow matching / DiT) is the planned quality ceiling lift**
- multilingual and code-switching inference is supported under a documented language-conditioning contract
- v3.0 drama-grade claim is scoped per runtime class:
  - Python serve may use `full` CFG and is the only path eligible for CFG-enhanced drama-grade acting claims in v3.0
  - Rust / VST / strict real-time paths must preserve pointer/control parity and causal latency, but are not eligible for CFG-enhanced drama claims until `lazy` or `distilled` CFG are validated and promoted

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
- worker 01 freezes the canonical 8-D `voice_state` dimension registry before any downstream worker binds UI, losses, or export fields to dimension order
- worker 04 freezes numeric runtime budgets in `configs/constants.yaml` before latency-sensitive implementation is considered stable
- worker 06 freezes the primary external baseline registry entry, evaluation protocol version, and hardware class before large-scale Stage B training
- worker 06 freezes the metric extractor/version choices used by acting and disentanglement metrics before large-scale Stage B training
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
        |    \        v    v
        |     +-> worker_02_training
        |             |         |
        |             |         | (training artifacts,
        |             |         |  evaluation data)
        |             v         v
        +-------> worker_12_gradio
        |         (WebUI Control Plane)
        |             ^
        |             | (artifact schema)
        |         worker_10_curation_export
        v
     worker_06_validation
```

- `worker_01` and `worker_03` may start in parallel: `worker_01` fixes the model/runtime contract; `worker_03` fixes the dataset/batch contract.
- `worker_02` depends on both `worker_01` (pointer/state interfaces) and `worker_03` (batch/cache schema).
- `worker_04` consumes the final model/checkpoint/runtime contract across Python serve, Rust engine, ONNX export, and VST. It also depends on `worker_02` for checkpoint schema.
- `worker_12` depends on `worker_02` (training artifacts), `worker_04` (runtime/admin APIs, SSE, idempotency), and `worker_10` (exported artifact schema).
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
      ^       ^        ^
      |       |        |
worker_04  worker_10  worker_12
```

- `worker_05` should not freeze menus, docs, or operator flows until runtime, export, and WebUI contracts are stable.
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

- starts in draft mode early, but final operator-facing flow waits for `worker_04`, `worker_10`, and `worker_12`
- WebUI operator guides (tasks 7-10) must not freeze until `worker_12` UI contract is stable
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
- owns `docs/design/auth-spec.md` for optimistic locking, audit-trail, and concurrency-control contracts; Worker 04 implements the network-level auth/RBAC middleware in `tmrvc-serve`, and Worker 12 consumes both

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

### worker_12_gradio_control_plane

- may draft minimum admin/eval UI contracts early (Stage A)
- must not freeze unsupported control inputs before `worker_01` and `worker_04` stabilize runtime contracts
- depends on `worker_02` for training artifacts and evaluation data
- depends on `worker_04` for runtime/admin API contracts, SSE event schema, and idempotency middleware
- depends on `worker_10` for exported artifact schema and manifest browsing contract
- must consume backend-owned schemas (`SpeakerProfile`, `voice_state`, pointer state) rather than inventing frontend-only shapes

### ai_curation_system.md

- always in force for workers 03 and 07 through 11
- any implementation that reduces curation to one-shot ASR without refinement, rejection, and promotion is non-compliant


## Global Deliverables

- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/external-baseline-registry.md`
- `docs/design/auth-spec.md`
- `docs/design/speaker-profile-spec.md`
- `docs/design/curation-contract.md`
- `docs/design/onnx-contract.md`
- `docs/design/rust-engine-design.md`
- `docs/design/gui-design.md`
- `plan/arxiv_survey_2026_03.md`
- shared pointer / `voice_state` / cache-state schema in `tmrvc-core`
- shared `speaker_profile` and prompt-cache metadata schema in `tmrvc-core`
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
- canonical v3 control path keeps 8-D `voice_state` and `delta_state` first-class from data curation through train and runtime
- 8-D `voice_state` targets, masks, confidences, and provenance survive export and are validated as usable supervision rather than UI-only controls
- code clearly separates `v2 legacy` from `v3 pointer`
- **Zero-shot speaker cloning works from a 3-10s reference clip with high similarity and disentangled prosody**
- latency and correctness tests pass
- Rust/ONNX/VST paths match the canonical runtime contract closely enough to pass parity gates
- all human-facing workflows use one authoritative typed backend API surface; no mainline workflow depends on direct manifest file reads from the UI
- all frame-indexed alignment artifacts use the same `24kHz`, `hop_length=240`, `T = ceil(N / 240)` convention as `tmrvc-core`
- docs state the new default path unambiguously
- expressive-quality claims are backed by reproducible evaluation
- **Blind A/B evaluation against the fixed external baseline artifact is completed and documented**
- the curation path can transform raw wav-only corpora into quality-gated trainable assets
