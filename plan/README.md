# Active Plan

`plan/` is the active backlog for the `v4` single-cutover program.

The design source of truth is
[`docs/design/v4-master-plan.md`](/home/kojirotanaka/kjranyone/TMRVC/docs/design/v4-master-plan.md).
This directory keeps only open execution work.

Rules:

- if a file remains in `plan/`, it is still open
- implemented contracts belong in `docs/design/`, code, or tests
- `v3` compatibility is not a planning requirement for `v4`
- no public "beats Fish Audio S2" language is allowed until Worker 06 sign-off

## Program Frame

`v4` is not an incremental migration.
It is a coordinated replacement of the current data, model, runtime, and UI assumptions.

The program is organized around these execution tracks:

1. raw-audio bootstrap and curation
2. v4 acting architecture and contracts
3. v4 serve/runtime/export cutover
4. v4 GUI/control-plane cutover
5. validation and competitive sign-off

## Active Files

- `worker_01_architecture.md`
  - freeze the `v4` conditioning contract and core schema cutover
- `worker_04_serving.md`
  - replace `v3` request/runtime boundaries with the `v4` serve/export/runtime contract
- `worker_06_validation.md`
  - add raw-audio bootstrap QC, controllability, parity, and Fish S2 claim gates
- `worker_12_gradio_control_plane.md`
  - replace the current `8-D` workshop with the `v4` physical-plus-latent control plane
- `dramatic_acting_requirements.md`
  - remaining drama-grade claim blockers only
- `fish_audio_s2_competitive_strategy.md`
  - strategic rationale for the differentiated `v4` path
- `repo_remaining_inventory_2026_03.md`
  - repository-wide pending-only inventory under the `v4` program
- `arxiv_survey_2026_03.md`
  - research context, not a backlog file

## Immediate Backlog

1. freeze the `v4` dataset/bootstrap contract for raw unlabeled audio corpora
2. freeze the `v4` acting contract:
   - explicit physical controls
   - acting texture latent
   - intent compiler output
   - trajectory record
3. replace the serve/export/runtime contract with the `v4` conditioning surface
4. replace the GUI control model with:
   - basic physical controls
   - advanced physical controls
   - acting prompt/macro controls
   - trajectory editing
5. add `v4` sign-off gates for:
   - bootstrap quality
   - physical controllability
   - Python/ONNX/Rust parity
   - Fish S2 competitor-facing claim discipline

## Critical Path

The critical path is:

1. raw-audio bootstrap
2. core contract freeze
3. train/runtime/UI cutover
4. end-to-end validation
5. `v4` checkpoint training

The following are not allowed to jump the queue:

- polishing the old `v3` 8-D UX
- extending `v3`-only routes
- preserving old checkpoint compatibility

## Competitive Claim Rule

No unqualified "SOTA" language is allowed from this plan alone.

Allowed only after Worker 06 sign-off:

- "beats Fish Audio S2 on the declared programmable expressive-speech axes"
- "beats Fish Audio S2 on the declared acting-editability axes"

Blocked unless separately proven:

- broad overall TTS SOTA
- broad first-take naturalness SOTA
- broad streaming SOTA

## Execution Order

1. freeze survey and competitor protocol
2. freeze `v4` raw-audio bootstrap contract
3. freeze `v4` core conditioning contract
4. implement train/export/serve/runtime/gui cutover
5. run `v4` validation and competitor reports
6. train and sign off the first `v4` checkpoint
