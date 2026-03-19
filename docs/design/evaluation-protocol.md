# UCLM v3 Evaluation Protocol

This document defines the reproducible evaluation procedure for UCLM v3.
It must remain consistent with:

- `plan/worker_06_validation.md`
- `docs/design/acceptance-thresholds.md`
- `docs/design/external-baseline-registry.md`
- `docs/design/evaluation-set-spec.md`

---

## 1. Evaluation Principles

- use frozen held-out prompt sets and report their version IDs
- pin the external baseline artifact before any comparison run starts
- separate exploratory runs from release-sign-off runs
- report both automatic metrics and human evaluation
- report failure counters such as `forced_advance_count` and `skip_protection_count`

### 1.1 Active Frozen Evaluation Set

The active release-signoff set is:

- `evaluation_set_version = tmrvc_eval_public_v1_2026_03_08`

Its exact contents, subset counts, language scope, few-shot prompt construction,
and human-rating assignment rules are frozen in
`docs/design/evaluation-set-spec.md`.
No release-signoff run may cite the set version without using that frozen spec.

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
- hardware class

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

---

## 13. v4 Control Taxonomy Matrix

> **Status: FROZEN** (2026-03-17)

### 13.1 Control Mode Classification

| Control Mode | Description | Example Systems |
|-------------|-------------|-----------------|
| **Prompt-only** | Natural language prompt describes the desired output; no explicit physical parameters | ChatTTS, Bark |
| **Physical-only** | Explicit numeric parameters for acoustic properties (F0, energy, etc.); no semantic prompt | Traditional parametric TTS |
| **Hybrid** | Combines prompt-based control with explicit physical overrides | CosyVoice 3, Qwen3-TTS |
| **Reference-driven** | Clones style from a reference audio; no explicit control axes | YourTTS, VALL-E |

### 13.2 Competitor Mapping

| System | Primary Mode | Physical Dims | Inline Tags | Latent Control | Trajectory Replay |
|--------|-------------|---------------|-------------|----------------|-------------------|
| Fish Audio S2 | Hybrid (rich-transcript + RL) | None explicit | Yes (via ASR) | No | No |
| CosyVoice 3 | Hybrid (prompt + reference) | Partial | Limited | No | No |
| Qwen3-TTS | Prompt-only | None | Limited | No | No |
| ChatTTS | Prompt-only | None | Limited | No | No |
| **TMRVC v4** | **3-layer** | **12-D explicit** | **35 acting tags** | **24-D acting latent** | **Yes (deterministic)** |

### 13.3 TMRVC v4 Unique 3-Layer Control Architecture

TMRVC v4 implements a unique 3-layer control architecture:

**Layer 1: Inline Acting Tags**
- 35 frozen acting tags embedded directly in the enriched transcript
- Tags include vocal events (`[laugh]`, `[inhale]`), prosodic markers (`[emphasis]`, `[pause]`), and acting directives (`[angry]`, `[whisper]`)
- Consumed by the text encoder alongside phoneme tokens
- Enables frame-precise control over local speech events

**Layer 2: Explicit Physical Controls (12-D)**
- 12 named physical dimensions: pitch_level, pitch_range, energy_level, pressedness, spectral_tilt, breathiness, voice_irregularity, openness, aperiodicity, formant_shift, vocal_effort, creak
- Each dimension is [0, 1] normalised with per-dimension confidence from bootstrap
- Supports per-frame or per-utterance specification
- Regularised by biological plausibility constraints (covariance prior + transition penalty)

**Layer 3: Acting Texture Latent (24-D)**
- Learned latent space capturing residual acting qualities not covered by physical controls
- User-facing via 6 macro controls: intensity, instability, tenderness, tension, spontaneity, reference_mix
- Can also be derived from reference audio (reference-driven mode)
- Disentangled from physical controls via explicit training objective

**Unique Properties:**
- All three layers compose: a single generation can use inline tags + physical targets + latent modulation simultaneously
- TrajectoryRecord captures all three layers for deterministic replay
- Edit locality: patching any layer in a local frame range does not affect other frames
- Cross-speaker transfer: acting trajectory (physical + latent) can be transferred to a different speaker identity

---

## 14. Fish S2 Victory Conditions

> **Status: FROZEN** (2026-03-17)

### 14.1 Victory Axes (TMRVC v4 Must Win)

| # | Axis | Metric | Target |
|---|------|--------|--------|
| V1 | Acting editability | Trajectory distance between 3+ acting configurations | TMRVC > Fish S2 |
| V2 | Trajectory replay fidelity | Bit-exact token match rate on deterministic replay | TMRVC = 1.0 (Fish S2 has no replay) |
| V3 | Edit locality | Max change outside patched region | TMRVC < 1e-5 (Fish S2 has no patch API) |

### 14.2 Guardrail Axes (TMRVC v4 Must Not Clearly Lose)

| # | Axis | Metric | Threshold |
|---|------|--------|-----------|
| G1 | First-take naturalness | Blind A/B preference (30+ raters) | No clear deficit vs Fish S2 |
| G2 | Few-shot speaker similarity | ECAPA-TDNN cosine sim at 3s/5s/10s reference | No clear deficit vs Fish S2 |
| G3 | Latency class disclosure | RTF + time-to-first-audio + hardware class | Must be disclosed; no evasion |

### 14.3 Claim Narrowing Rule

If TMRVC v4 wins on a victory axis but loses on a guardrail axis, the public claim **must** be narrowed:

- Win V1+V2+V3, no guardrail deficit: **Broad claim** ("TMRVC v4 beats Fish S2 on acting programmability")
- Win V1+V2 only, no deficit: **Narrow claim** ("TMRVC v4 beats Fish S2 on editability and replay fidelity")
- Win V1 but G1 deficit: **Caveat claim** ("TMRVC v4 beats Fish S2 on editability; naturalness gap noted")
- No victory wins: **No claim permitted**

### 14.4 Protocol

1. Freeze Fish S2 artifact version before any comparison run
2. Use the frozen evaluation set (`tmrvc_eval_public_v1_2026_03_08`)
3. Run `scripts/eval/fish_s2_comparison.py` to generate blind bundle
4. Collect human ratings (30+ unique raters for release sign-off)
5. Apply claim narrowing rule to determine valid claim scope
6. Store the full evaluation bundle as a release artifact
