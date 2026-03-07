# Worker 11: Curation Validation and Acceptance

## Purpose

Prove that the AI curation system improves training data quality rather than only producing impressive metadata.

## Intent

This worker is the brake pedal. It prevents the project from claiming success because the pipeline runs. It must prove that curation improves usable data quality and downstream model behavior.

## Primary Files

- evaluation scripts
- validation reports
- curation acceptance checklist

## Required Outcomes

- provider comparison report
- pseudo-label audit report
- promoted subset quality report
- downstream uplift report
- human-audit process report

## Validation Layers

### Layer 1: Stage Quality

Measure:

- ASR spot-check accuracy (via Gradio Auditor)
- diarization purity (via Gradio Auditor)
- separation artifact rate
- transcript refinement uplift

### Layer 2: Promotion Policy Quality

Measure:

- promote/review/reject distribution
- false promote rate on audited samples (manual audit via UI)
- false reject rate where possible
- reviewer agreement / disagreement patterns
- approval-latency and queue health

### Layer 3: Downstream Utility

Measure:

- TTS text coverage uplift
- pointer-training stability uplift
- VC prior stability uplift
- expressive metrics uplift where available

### Layer 4: Human Process Integrity

Measure:

- audit-trail completeness
- role separation compliance
- override frequency and justification quality
- rater quality-control pass rate

## Concrete Tasks

1. Define stage-by-stage benchmark protocol.
2. Define sample audit protocol:
   - promoted sample audit
   - review sample audit
   - rejected sample audit
   - dual-review or override audit where policy requires it
3. Define downstream comparison:
   - naive raw ingestion baseline
   - curated subset baseline
4. Define acceptance thresholds for adopting each provider stack.
5. Define failure conditions that block rollout.
6. Validate legality gating and split integrity:
   - no unknown-rights source in mainline export
   - no holdout leakage into train buckets
7. Validate the human workflow itself:
   - every critical promotion/export action has actor, timestamp, and rationale
   - required role separation is enforced
   - blinded rating protocol is actually followed in stored records

## Acceptance Criteria

The curation system is acceptable only if:

1. promoted subsets show better annotation quality than naive one-pass ASR
2. provider fusion or refinement yields measurable uplift
3. downstream training benefits measurably from curated subsets
4. rejected samples are mostly truly bad or unsuitable
5. provenance is sufficient to explain failures
6. legality and split gates are enforced with zero critical violations
7. human audit and rating records are sufficient to reconstruct who approved what and why

## Guardrails

- do not sign off based only on pipeline completion
- do not sign off based only on one cherry-picked demo
- do not ignore false promotions
- do not accept opaque score formulas
- do not accept a HITL claim if approvals or ratings are not reconstructible from stored audit data

## Deliverables

- benchmark commands
- audit checklist
- provider comparison matrix
- downstream uplift report template
- human workflow integrity checklist
