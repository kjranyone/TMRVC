# Track: v4 Survey And Competitive Freeze

## Scope

This track owns the `v4` survey phase: competitor analysis, control taxonomy, and evaluation protocol alignment.
This must complete BEFORE dataset contract freeze.
This is not ongoing research — it has concrete deliverables and a hard freeze point.

## Primary Files

- `docs/design/external-baseline-registry.md`
- `docs/design/evaluation-protocol.md`
- `plan/reference_arxiv_survey.md`
- `plan/strategy_competitive.md`

## Open Tasks

### 1. Freeze competitor summary for `v4`

Required entries:

- Fish Audio S2: input formats, control surfaces, few-shot conditions, streaming
  - rich-transcription ASR: inline acting tag injection in training data
  - inline instruction following via text-conditioned generation (not external LLM)
  - RL fine-tuning with rich-transcription ASR as instruction-following reward
- CosyVoice 2/3: streaming architecture, multilingual coverage
- Qwen3-TTS: 2-stage pipeline, scale comparison
- other relevant systems from `reference_arxiv_survey.md`

Each entry must have:

- frozen version/date
- public artifact reference
- capability summary

### 2. Freeze prompt/control taxonomy

Required analysis:

- classify control methods across competitors: prompt-only, physical-only, hybrid, reference-driven
- map TMRVC `v4`'s hybrid (physical + latent) position relative to competitors
- define which control axes are unique to `v4`

### 3. Freeze evaluation subset mapping

Required behavior:

- map each `v4` claim to a specific evaluation subset and protocol
- ensure every public claim has a testable evaluation path
- cross-reference with `track_validation` metrics

### 4. Define "beats Fish Audio S2" conditions

Mandatory win axes:

- acting editability
- trajectory replay fidelity
- edit locality

Mandatory guardrail axes:

- first-take naturalness
- few-shot speaker similarity
- latency

Rules:

- narrow claim if win is only on editability
- explicit axis definitions must be frozen

### 5. Update `reference_arxiv_survey.md` for `v4`

Required additions:

- `v4`-relevant literature: physical+latent hybrid, biological constraints, acting texture residual
- update "Decisions Taken" section from `v3` to `v4`
- flag papers relevant to the new `12-D` to `16-D` physical registry

## Required Deliverables

- frozen competitor summary document
- prompt/control taxonomy matrix
- evaluation subset mapping table
- Fish S2 claim condition document
- updated arxiv survey with `v4` sections

## Freeze Management

- Freeze date: TBD
- Sign-off owner: TBD
- After freeze, no changes to competitor summaries, taxonomy, or evaluation mappings without re-review

## Out Of Scope

Do not reopen:

- ongoing literature monitoring (this is a freeze, not a continuous process)
- implementation of evaluation harnesses (that is `track_validation`)
- broad SOTA claims (blocked per competitive claim rule)

## Exit Criteria

- competitor summary covers at minimum Fish S2, CosyVoice 3, Qwen3-TTS with frozen version pins
- every `v4` public claim maps to exactly one evaluation protocol entry
- Fish S2 win/loss conditions are frozen with explicit axis definitions
- survey freeze date is recorded and no further changes accepted without re-review
