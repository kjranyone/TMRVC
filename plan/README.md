# Active Plan

`plan/` is the active backlog for the `v4` single-cutover program.

This directory keeps only open execution work.

Rules:

- if a file remains in `plan/`, it is still open
- implemented contracts belong in `docs/design/`, code, or tests

## Remaining Open Tracks

- `track_codec_strategy.md`
  - codec selection: EnCodec baseline vs Mimi AR/NAR vs single-codebook (code ready, experiments pending)
- `track_validation.md`
  - bootstrap QC, controllability, parity gates (sign-off pending; Fish S2 comparison descoped)

## Completed (removed from plan/)

- `track_architecture.md` — v4 schema frozen in code
- `track_data_bootstrap.md` — 13-stage bootstrap pipeline
- `track_training.md` — 9-loss trainer, RL, enriched transcripts
- `track_serving.md` — compile/replay/patch/transfer, Qwen LLM backend
- `track_gui.md` — 6-panel workshop with trajectory UI
- `track_workflow.md` — idempotent phase-based workflow
- `track_survey.md` — competitive analysis in docs/design/ and eval scripts
- `strategy_competitive.md` — strategy realized in architecture
- `strategy_acting_claims.md` — all 6 blockers implemented, sign-off via track_validation
- `track_character_intent.md` — conditioning split realized end-to-end (model, trainer, export, Rust, UI)
- `IMPLEMENTATION_INSTRUCTIONS.md` — all phases consumed
