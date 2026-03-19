# Track: v4 Validation And Claim Gating

## Scope

This track owns the validation gates for the `v4` single-cutover program.
This includes both repository-internal release gates and competitor-facing claim gates.

This track is the only owner allowed to unblock "beats Fish Audio S2" language.

## Primary Files

- `tests/data/`
- `tests/train/`
- `tests/serve/`
- `tmrvc-engine-rs/tests/`
- `docs/design/evaluation-protocol.md`
- `docs/design/acceptance-thresholds.md`
- `docs/design/external-baseline-registry.md`
- `docs/design/curation-contract.md`
- `docs/design/v4-master-plan.md`

## Open Tasks

### 1. Add raw-audio bootstrap quality gates

Required `v4` bootstrap metrics:

- diarization purity
- speaker-cluster consistency
- overlap rejection quality
- transcript quality proxy
- physical-label coverage
- physical-label confidence calibration
- language coverage and code-switch reporting

Rules:

- low-confidence pseudo-labels must be measured, not silently treated as dense truth
- bootstrap quality is a release gate, not an offline curiosity

Ownership: track_validation defines the acceptance thresholds and produces the sign-off report. track_data_bootstrap implements the measurement code within the bootstrap pipeline. If thresholds are not met, track_data_bootstrap must fix the pipeline; track_validation must not lower thresholds without re-review.

### 2. Add `v4` controllability and editability metrics

Required metrics:

- physical control response monotonicity
- physical calibration error
- trajectory replay fidelity
- edit locality
- cross-speaker acting transfer quality
- semantic prompt-following quality
- inline acting tag instruction-following rate
- RL reward compliance (instruction-following + physical compliance + intelligibility)

Each metric needs:

- frozen definition
- reproducible harness
- threshold report for sign-off

### 3. Separate creator variance from deterministic replay variance

Required policy:

- prompt compile variance is measured separately
- deterministic replay runs only from frozen compile artifacts or trajectory artifacts
- transfer variance is measured separately from same-speaker replay variance

No report may mix these buckets under a single uncontrolled "quality" number.

### 4. Add runtime parity gates

Required `v4` parity suites:

- Python vs ONNX parity
- Python vs Rust parity
- batch vs streaming numerical parity
- physical-control ordering and scaling parity
- acting-latent ordering and scaling parity
- codec encode/decode parity: Python Mimi vs ONNX export (if applicable)

If parity fails, the release is blocked.

### 5. Freeze claim taxonomy in actual reports

Every evaluation report must declare which claim is being tested:

- raw-audio bootstrap readiness
- broad external-baseline TTS competitiveness
- acting controllability
- programmable expressive speech
- cross-speaker acting transfer
- real-time causal runtime
- inline instruction following

If a claim lacks its required evidence, it is blocked rather than softened in prose.

### 6. Freeze the Fish S2 head-to-head protocol for `v4`

This track must maintain the frozen Fish S2 competitor protocol.

Mandatory axes for any "beats Fish Audio S2" language:

- acting editability
- trajectory replay fidelity
- edit locality
- transfer quality, if transfer is part of the claim

Mandatory guardrail axes:

- first-take naturalness or preference
- few-shot speaker similarity
- latency class disclosure

Claim rule:

- TMRVC may claim it beats Fish S2 only on the frozen declared axes
- if TMRVC wins on editability but loses clearly on first-take quality, the claim must narrow to editability / programmability only
- broad SOTA language remains blocked unless separately proven

## Required Tests And Reports

- bootstrap-quality report
- physical-calibration report
- replay-fidelity report
- edit-locality report
- acting-transfer report
- runtime-parity report
- claim matrix mapping each public claim to evidence
- Fish S2 head-to-head report with frozen artifact/version, prompt rule, and subset definition
- instruction-following report: tag compliance rate before and after RL, broken down by tag category
- physical compliance under RL: monotonicity and calibration preserved after RL phase

## Out Of Scope

Do not reopen:

- preserving old `v3` claim wording
- generic regression checks that are already covered elsewhere
- using plan text itself as substitute for measured evidence

## Exit Criteria

- bootstrap QC gates exist and produce reports with pass/fail against defined thresholds
- controllability metrics are implemented and meet thresholds defined in `docs/design/acceptance-thresholds.md` §V4-1
- parity reports cover Python, ONNX, Rust, and streaming paths and meet thresholds defined in `docs/design/acceptance-thresholds.md` §V4-5
- reports explicitly separate compile variance, replay variance, and transfer variance (no mixed buckets)
- Fish S2 claim rules are frozen per `docs/design/acceptance-thresholds.md` §V4-9 with explicit axis definitions, subset sizes, and statistical significance requirements
- each public claim maps to a specific report with measured evidence
