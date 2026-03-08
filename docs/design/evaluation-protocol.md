# UCLM v3 Evaluation Protocol

This document defines the reproducible evaluation procedure for UCLM v3.
It must remain consistent with:

- `plan/worker_06_validation.md`
- `docs/design/acceptance-thresholds.md`
- `docs/design/external-baseline-registry.md`

---

## 1. Evaluation Principles

- use frozen held-out prompt sets and report their version IDs
- pin the external baseline artifact before any comparison run starts
- separate exploratory runs from release-sign-off runs
- report both automatic metrics and human evaluation
- report failure counters such as `forced_advance_count` and `skip_protection_count`

---

## 2. Automated Metrics

### 2.1 Training Quality Gates

| Metric | Target | Source |
|--------|--------|--------|
| Text supervision coverage | > 80% per dataset | dataset supervision report |
| Pointer-state sanity | no monotonicity violation | pointer contract tests |
| Checkpoint schema validity | all required pointer-head keys present | checkpoint load test |
| Bootstrap projection validity | 100% monotonic in canonical phoneme space | bootstrap alignment projection tests |

### 2.2 Latency and Runtime

| Metric | Target | Tool |
|--------|--------|------|
| Steady-state frame compute p95 | <= 10 ms on the release real-time path | `scripts/benchmark_latency.py` or equivalent |
| End-to-end chunk latency p95 | <= 50 ms on the release streaming server path | streaming endpoint latency harness |
| Time-to-first-audio | < 200 ms | streaming endpoint measurement |
| Steady-state VRAM | within pinned deployment budget | runtime telemetry |
| RTF | < 1.0 on streaming path | engine metrics |

Measure prompt extraction / `SpeakerProfile` re-encode latency separately from steady-state generation.

### 2.3 Pointer Diagnostics

| Metric | Description |
|--------|-------------|
| Completion ratio | fraction of canonical text units consumed before stop |
| Average frames per text unit | should correlate with reference speech rate |
| Advance probability histogram | should separate hold vs advance modes |
| `forced_advance_count` | fallback trigger count during evaluation |
| `skip_protection_count` | skip-protection trigger count during evaluation |

---

## 3. Integration Matrix

| Config | Train | Serve | Status |
|--------|-------|-------|--------|
| v3 pointer, alignment-free release recipe | `--tts-mode pointer` | `tts_mode=\"pointer\"` | Primary |
| v3 pointer, transitional bootstrap supervision | `--tts-mode pointer` with supervised bootstrap source | `tts_mode=\"pointer\"` | Transitional / ablation |
| v2 legacy | `--tts-mode legacy_duration` | `tts_mode=\"legacy_duration\"` | Legacy |

Release sign-off must include the alignment-free release recipe, not only transitional recipes.

---

## 4. TTS Quality Evaluation

### 4.1 Naturalness

- evaluate on a frozen prompt set spanning neutral, dialogue, and expressive cases
- report MOS with 95% confidence intervals
- release-signoff subjective runs require at least 30 unique raters

### 4.2 Intelligibility

- compute CER / WER on ASR re-transcription of synthesized audio
- stratify by language and by code-switch subset

### 4.3 Control Responsiveness

For each exposed control (`pace`, `hold_bias`, `boundary_bias`, selected `voice_state` dimensions):

1. generate a frozen control sweep
2. measure duration, pause behavior, and selected acoustic metrics
3. compute monotonic correlation and significance
4. report both aggregate and failure cases

### 4.4 Context Sensitivity

- same text, different dialogue context
- measure:
  - `context_separation_score`
  - `prosody_collapse_score`
  - pause / boundary placement differences
- include human appropriateness rating on the same frozen subset

---

## 5. Few-Shot Speaker Adaptation

### 5.1 Reference Lengths

- fixed reference durations: 3 s, 5 s, 10 s
- trimmed from the same source session where possible
- reuse the same frozen prompt set across all systems

### 5.2 Metrics

- `few_shot_speaker_score`
- CER degradation vs text-prompted synthesis
- `timbre_prosody_disentanglement_score`
- `prosody_transfer_leakage_score`

### 5.3 Leakage Protocol

- use the same speaker prompt while varying target text and dialogue context
- compare generated pitch / duration contours against the reference prompt contours
- low leakage is required when the target context differs materially from the prompt context

---

## 6. External Baseline Comparison

### 6.1 Frozen Baseline Requirement

Every release-signoff run must cite an entry in
`docs/design/external-baseline-registry.md`.

The following must be frozen:

- baseline model name
- exact artifact / checkpoint identifier
- tokenizer / text normalization settings
- prompt trimming rules
- inference parameters
- evaluation prompt set version

### 6.2 Dimensions

| Dimension | Description |
|-----------|-------------|
| Naturalness | overall speech quality and human-likeness |
| Controllability / Acting | ability to respond to dialogue context and explicit controls |
| Few-shot similarity | speaker identity retention under short references |
| Leakage resistance | ability to avoid copying prompt prosody when context differs |

### 6.3 Blind Preference Protocol

- minimum 30 unique raters for release sign-off
- randomize order and hide system identities
- allow `no preference`, but report it separately
- separate `director` notes from blinded rater judgments

---

## 7. CFG Evaluation

### 7.1 Modes

Evaluate all enabled runtime modes explicitly:

- `off`
- `full`
- `lazy`
- `distilled`

### 7.2 Checks

- pointer monotonicity under CFG sweeps
- EOS reachability under CFG sweeps
- cache consistency under CFG sweeps
- perceptual delta between `full` and any accelerated mode

`full` two-pass CFG is not the default for hard real-time Rust / VST paths unless latency proof is recorded.

---

## 8. VC Evaluation

### 8.1 Core Metrics

- speaker similarity
- intelligibility
- streaming stability
- semantic-context contract coverage

### 8.2 Context Protocol

- if VC semantic-context fusion is enabled, evaluate with and without surrounding turn context
- confirm the path remains causal and does not rely on offline look-ahead

---

## 9. Multilingual and Code-Switch Regression

- after adding or reweighting a language, rerun the frozen multilingual held-out suite
- report regression deltas for previously supported languages
- block sign-off if existing-language metrics fall beyond the documented regression budget

---

## 10. Curation and Separation Evaluation

### 10.1 Pseudo-Annotation Audit

- ASR spot-check
- text normalization audit
- speaker clustering audit
- quality-score calibration

### 10.2 Separation Policy

- separation is evaluated first as an annotation aid
- initial mainline release does not use separated waveforms as default waveform teachers
- any research-bucket use of separated waveforms must report:
  - `separation_confidence`
  - `waveform_artifact_score`
  - speaker / timbre preservation result
  - explicit human approval record

---

## 11. Statistical Procedure

- pre-register sample count, rater count, duplicate ratio, and hypothesis test
- report confidence intervals alongside `p` values
- store the protocol version with every release-signoff result bundle

---

## 12. Required Artifacts

Every sign-off bundle must include:

- metric report
- latency report
- pointer fallback counters
- external-baseline registry reference
- human evaluation export
- rater QC report
- failure-case appendix
