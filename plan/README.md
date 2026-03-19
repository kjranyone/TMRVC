# Active Plan

`plan/` is the active backlog for the `v4` single-cutover program.

The design source of truth is
[`docs/design/v4-master-plan.md`](/home/kojirotanaka/kjranyone/TMRVC/docs/design/v4-master-plan.md).
This directory keeps only open execution work.

Rules:

- if a file remains in `plan/`, it is still open
- implemented contracts belong in `docs/design/`, code, or tests
- `v3` compatibility is not a planning requirement for `v4`
- no public "beats Fish Audio S2" language is allowed until track_validation sign-off

## Program Frame

`v4` is not an incremental migration.
It is a coordinated replacement of the current data, model, runtime, and UI assumptions.

## File Organization

Files are organized by prefix:

- `track_*` — execution tracks (what to build)
- `strategy_*` — strategic rationale (why)
- `reference_*` — background context (surveys, inventories)

## Execution Tracks

- `track_survey.md`
  - freeze competitor analysis, control taxonomy, and evaluation protocol for v4
- `track_data_bootstrap.md`
  - raw-audio bootstrap pipeline from unlabeled corpora to train-ready cache
- `track_architecture.md`
  - freeze the `v4` conditioning contract and core schema cutover
- `track_training.md`
  - v4 training pipeline: 12-D physical, acting latent, biological constraints, supervision tiers
- `track_serving.md`
  - replace `v3` request/runtime boundaries with the `v4` serve/export/runtime contract
- `track_validation.md`
  - add raw-audio bootstrap QC, controllability, parity, and Fish S2 claim gates
- `track_gui.md`
  - replace the current `8-D` workshop with the `v4` physical-plus-latent control plane
- `track_codec_strategy.md`
  - codec selection and generation design: EnCodec baseline vs Mimi AR/NAR vs single-codebook
- `track_workflow.md`
  - idempotent bootstrap/train workflow with phase-based state management

## Strategy

- `strategy_competitive.md`
  - strategic rationale for the differentiated `v4` path vs Fish Audio S2
- `strategy_acting_claims.md`
  - remaining drama-grade claim blockers only

## Reference (archived to docs/)

- `docs/design/arxiv-survey.md` — research survey (frozen, moved from plan/)
- `docs/design/decision-model-scale.md` — 248M model scale decision record (moved from plan/)

## Critical Path

1. survey freeze (`track_survey`)
2. v4 dataset contract freeze (`track_data_bootstrap`)
3. raw-audio bootstrap pipeline implementation (`track_data_bootstrap`)
4. v4 model contract freeze (`track_architecture`)
5. train / export / serve / runtime / gui implementation (`track_training`, `track_serving`, `track_gui`)
6. end-to-end validation (`track_validation`)
7. v4 checkpoint training (`track_training`)
8. release sign-off (`track_validation`)

The following are not allowed to jump the queue:

- polishing the old `v3` 8-D UX
- extending `v3`-only routes
- preserving old checkpoint compatibility

## Track Dependencies

```
track_survey
  └─► track_data_bootstrap (dataset contract freeze)
        └─► track_architecture (model contract freeze)
              ├─► track_training (model + loss implementation)
              ├─► track_serving (serve/export/runtime cutover)
              │     └─ depends on track_architecture §6 for Intent Compiler model
              └─► track_gui (control plane cutover)
                    └─ depends on track_serving for backend routes
        └─► track_validation (gates depend on all above)
```

No track may declare exit without its upstream dependencies also meeting exit criteria.

## Competitive Claim Rule

No unqualified "SOTA" language is allowed from this plan alone.

Allowed only after track_validation sign-off:

- "beats Fish Audio S2 on the declared programmable expressive-speech axes"
- "beats Fish Audio S2 on the declared acting-editability axes"

Blocked unless separately proven:

- broad overall TTS SOTA
- broad first-take naturalness SOTA
- broad streaming SOTA
