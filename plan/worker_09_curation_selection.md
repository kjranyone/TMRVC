# Worker 09: Quality Scoring, Rejection, and Promotion

## Purpose

Turn raw provider outputs into actionable curation decisions.

## Intent

The system is only useful if it can say "use this", "re-run this", or "throw this away" without handwaving. This worker defines that policy.

## Primary Files

- curation scoring module
- promotion policy config
- quality summary reports

## Required Outcomes

- stable quality score formula
- explicit hard-reject rules
- explicit review rules
- explicit promotion rules
- subset promotion lists
- bucket-specific numeric thresholds encoded in config
- explicit human-override and approval policy
- explicit policy for partial or missing `voice_state` supervision

## Scoring Dimensions

Each sample should be scored on:

- transcript confidence
- transcript agreement across providers
- diarization trust
- overlap severity
- language consistency
- duration sanity
- separation damage estimate
- separation confidence
- style / event extraction completeness
- `voice_state` supervision density
- `voice_state` confidence quality
- audio technical quality

## Decision Policy

### Reject

Examples:

- transcript empty
- severe overlap
- separation damage above threshold
- wrong language
- clipping or corruption

### Review

Examples:

- provider disagreement
- marginal transcript confidence
- partial speaker uncertainty
- partial event extraction failure

### Promote

Requires:

- acceptable hard constraints
- quality score above threshold
- provenance complete

## Subset Types

The system must promote into different buckets:

1. `tts_mainline`
2. `vc_prior`
3. `expressive_prior`
4. `holdout_eval`

Not all promoted data belong in the same training path.

## Concrete Tasks

1. Define score weights and threshold config.
2. Define bucket-specific promotion policy.
   - initial mainline rule:
     - separation-derived waveforms are annotation aids, not default waveform teachers
   - `vc_prior` may consume labels or metadata recovered from separation, but not separated waveform teachers in the initial mainline
   - any future use of separated waveforms as teachers must live in an explicit research / ablation bucket gated by:
     - high `separation_confidence`
     - low `waveform_artifact_score`
     - speaker/timbre preservation pass
     - explicit human approval
   - `tts_mainline` does not auto-adopt separation-derived waveform teachers in the initial mainline
3. Define anti-contamination rules:
   - holdout must not leak into train
4. Encode legality gates into promotion policy.
5. Define same-text clustering policy:
   - preserve same-text multi-context examples for expressive evaluation
6. Define reporting:
   - score histogram
   - top rejection reasons
   - provider disagreement summary
7. Define human approval policy:
   - which buckets allow direct auto-promotion
   - which buckets require auditor approval
   - which buckets require double approval or admin override
   - how overrides are justified and logged
8. Define `voice_state` promotion policy:
   - which buckets require minimum `voice_state` coverage
   - when missing `voice_state` is acceptable with explicit absence metadata
   - when low-confidence `voice_state` forces `review` rather than `promote`

## Guardrails

- do not use one scalar score without rejection reasons
- do not promote mixed-quality data into one main bucket
- do not treat VC-usable data as automatically TTS-usable
- do not let high separation confidence bypass waveform-artifact or timbre-damage checks
- do not let separated waveforms silently enter initial mainline teacher sets
- do not let workers invent bucket thresholds ad hoc
- do not leave human override rules implicit if the UI exposes promotion controls
- do not promote samples as physical-control-ready when `voice_state` coverage/confidence is unknown

## Handoff Contract

- worker 10 gets stable promoted subset lists
- worker 11 gets measurable policy outputs for evaluation
