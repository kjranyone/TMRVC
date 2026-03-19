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

## Remaining Open Tracks

- `track_survey.md`
  - freeze competitor analysis, control taxonomy, and evaluation protocol for v4
- `track_codec_strategy.md`
  - codec selection: EnCodec baseline vs Mimi AR/NAR vs single-codebook (experiments pending)
- `track_validation.md`
  - bootstrap QC, controllability, parity, and Fish S2 claim gates (sign-off pending)

## Remaining Strategy

- `strategy_acting_claims.md`
  - remaining drama-grade claim blockers (validation gates pending)

## Completed Tracks (removed from plan/)

The following tracks have been fully implemented and removed:

- `track_architecture.md` — v4 schema frozen in code
- `track_data_bootstrap.md` — 13-stage bootstrap pipeline implemented
- `track_training.md` — 9-loss trainer, RL phase, enriched transcripts implemented
- `track_serving.md` — compile/replay/patch/transfer routes, Qwen LLM backend
- `track_gui.md` — 6-panel workshop with trajectory UI
- `track_workflow.md` — idempotent phase-based workflow
- `strategy_competitive.md` — competitive strategy realized in architecture
- `IMPLEMENTATION_INSTRUCTIONS.md` — all phases consumed

## Competitive Claim Rule

No unqualified "SOTA" language is allowed from this plan alone.

Allowed only after track_validation sign-off:

- "beats Fish Audio S2 on the declared programmable expressive-speech axes"
- "beats Fish Audio S2 on the declared acting-editability axes"

Blocked unless separately proven:

- broad overall TTS SOTA
- broad first-take naturalness SOTA
- broad streaming SOTA
