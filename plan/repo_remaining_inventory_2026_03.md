# Repository Remaining Inventory (2026-03)

## Purpose

This note is the repository-wide pending-only inventory under the `v4` single-cutover program.

It exists because much of the old backlog language assumed:

- `v3` as the active mainline
- speaker-separated curated datasets as the default starting point
- extension of the current `8-D voice_state` contract

Those assumptions are now stale.

## Method

Status is assigned from the current repository state, not from old plan text.

Three buckets are used:

- `implemented enough`
- `partial`
- `missing and required`

The meaning of "implemented enough" is narrow:
the code may still be superseded by `v4`, but it should not remain on the active backlog as if it were missing from scratch.

## Corrections To Stale Assumptions

### 1. The current `8-D voice_state` stack is no longer the target mainline

The repository already contains:

- canonical `8-D` registry
- pointer-oriented train/serve/runtime plumbing
- GUI support for the current `8-D` controls

This is not missing work.
It is an implemented `v3` baseline that will be superseded by `v4`.

### 2. The next blocker is not "more `v3` features"

The real repository-wide blocker is:

- no end-to-end `v4` path from raw unlabeled audio corpus to validated acting-centric runtime

### 3. Existing replay-oriented plan text is too narrow

Compile / replay / patch remain relevant, but the new critical delta is broader:

- raw-audio bootstrap
- `v4` contract cutover
- physical-plus-latent control split
- cross-stack parity

## 1. Implemented Enough

These should not remain as immediate backlog items "from zero".

### A. `v3` pointer and serving baseline

- pointer-state contracts exist
- pointer-mode training exists
- baseline `/tts` and workshop generation routes exist
- parity and runtime tests exist

### B. `v3` dataset and curation baseline

- dataset loading exists
- G2P and bootstrap-alignment support exist
- curation orchestration/provider/export stack exists

### C. baseline WebUI surface

- Gradio control plane exists
- Workshop, Casting Gallery, and Evaluation Arena exist

### D. baseline external comparison infrastructure

- baseline registry exists
- evaluation protocol exists
- Fish Audio S2 baseline entry is already frozen

## 2. Partial

These are real capabilities, but the remaining delta matters and must stay active.

### A. Architecture is implemented for `v3`, not frozen for `v4`

Implemented:

- pointer-oriented `v3` contracts
- current `8-D` registry
- existing prompt/context conditioning paths

Missing delta:

- `v4` physical registry
- `v4` acting texture latent contract
- `v4` `IntentCompilerOutput`
- `v4` `TrajectoryRecord`

### B. Data stack is functional, but not `v4`-ready

Implemented:

- dataset preparation and curation baseline
- text supervision and feature caching

Missing delta:

- raw-audio bootstrap pipeline from mixed-speaker corpora
- pseudo speakerization quality gates
- `v4` train-ready cache contract with physical-plus-semantic supervision tiers

### C. Serve/runtime is functional, but not `v4`-complete

Implemented:

- current `/tts` path
- workshop generation path
- backend-owned speaker profile loading

Missing delta:

- `v4` request surface
- `v4` trajectory artifact persistence
- transfer path as a first-class capability
- real causal serve-path streaming under the `v4` contract

### D. Validation is strong, but not aligned to the new thesis

Implemented:

- parity and many regression suites
- baseline freeze and acceptance docs

Missing delta:

- bootstrap QC gates
- physical calibration metrics
- replay fidelity and edit locality as `v4` sign-off artifacts
- transfer quality metrics
- Fish S2 claim matrix tied to the `v4` protocol

### E. GUI is usable, but still presents the old model

Implemented:

- current expressive TTS workshop
- speaker-profile-facing UI

Missing delta:

- `v4` basic and advanced physical panels
- acting prompt/macro panels
- trajectory-first editing and transfer flow
- removal of all claim-invalid dummy enrollment shortcuts

## 3. Missing And Required

These are the repository-wide items that still justify staying in the immediate backlog.

### 1. Canonical `v4` raw-audio bootstrap contract

Required location:

- `docs/design/`
- `tmrvc-data`

Why it remains:

- without it, `v4` cannot start from unlabeled raw corpora

### 2. Canonical `v4` acting contract

Required location:

- `tmrvc-core`

Why it remains:

- without it, the repository has no frozen mainline schema for physical-plus-latent acting control

### 3. Canonical `v4` serve/export/runtime contract

Required location:

- `tmrvc-serve`
- `tmrvc-export`
- `tmrvc-engine-rs`

Why it remains:

- without it, the stack cannot consume the same conditioning surface end-to-end

### 4. `v4` trajectory-first Workshop UI

Required location:

- `tmrvc-gui`

Why it remains:

- without it, the main UI still exposes the old control abstraction

### 5. `v4` sign-off harnesses

Required location:

- `tests/`
- `docs/design/`

Why it remains:

- without measured evidence, the new acting-centric thesis is unproven
