# Acceptance Thresholds

This document defines the explicit release gates for UCLM.
If a threshold changes here, the validation plan and benchmark scripts must be
updated in the same change.

---

# v4 Acceptance Thresholds

These gates must stay consistent with `plan/track_validation.md` and `plan/track_training.md`.
All `v4` track files reference this section as the single source of truth for numeric thresholds.

Threshold ownership follows a tiered freeze policy:

- `Tier 0`: must freeze before v4 dataset contract freeze
- `Tier 1`: may be estimated from pilot runs, must freeze before v4 checkpoint training
- `Tier 2`: protocol freezes early, final numeric cutoffs freeze before release sign-off

---

## V4-1. Physical Control Responsiveness

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Physical control response monotonicity | > 0.8 correlation | 1 | Sweep each of the 12-D controls independently, measure output feature correlation |
| Physical calibration error | < 0.15 RMSE | 1 | Compare realized physical features against explicit physical targets on held-out set |

**Pass condition:** All 12-D physical controls show monotonic response and calibration within budget.

---

## V4-2. Acting Latent Health

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Acting latent utilization | > 0.1 | 1 | Measure residual latent variance usage vs total acting variance |
| Disentanglement: latent does not duplicate physical | Qualitative + ablation | 1 | Remove latent path, measure physical-only reconstruction gap |

**Pass condition:** Latent path is measurably used and captures non-physical acting residue.

---

## V4-3. Supervision Tier Weighting

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Tier D loss contribution | < 10% of total loss | 0 | Log per-tier loss fraction during training |
| Low-confidence pseudo-label masking | Active | 0 | Verify mask/downweight code path in trainer |

**Pass condition:** Low-quality supervision does not dominate training signal.

---

## V4-4. Biological Constraint Regularization

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Implausible combination reduction | > 50% vs unconstrained baseline | 1 | Compare violation count with and without biological regularization on held-out set |
| Non-zero gradients | All constraint terms produce non-zero gradients | 0 | Smoke test gradient check |

**Pass condition:** Biological priors produce measurable reduction in implausible outputs.

---

## V4-5. Cross-Runtime Parity

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Python vs ONNX parity | < 1e-4 max abs diff | 0 | Paired output comparison on frozen inputs |
| Python vs Rust parity | < 1e-4 max abs diff | 0 | Paired output comparison on frozen inputs |
| Batch vs streaming numerical parity | < 1e-4 max abs diff | 0 | Paired output comparison on frozen inputs |
| Physical-control ordering parity | Identical | 0 | Cross-runtime shape and order assertion |
| Acting-latent ordering parity | Identical | 0 | Cross-runtime shape and order assertion |

**Pass condition:** All runtime paths produce numerically identical results within tolerance.

---

## V4-6. Serve-Path Streaming

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| First-token latency | < 500 ms | 1 | Streaming endpoint measurement on target GPU class |
| Real-time factor (RTF) | < 1.0 | 1 | End-to-end RTF measurement on target GPU class |

**Pass condition:** Real causal streaming meets latency budget on the frozen hardware class.

---

## V4-7. Replay And Transfer Fidelity

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Deterministic replay fidelity | Bit-identical output for same TrajectoryRecordV4 | 0 | Automated replay comparison |
| Edit locality | TBD (freeze before release sign-off) | 2 | Measure change outside patched region |
| Cross-speaker transfer quality | TBD (freeze before release sign-off) | 2 | A/B evaluation on transferred vs fresh compile |

**Pass condition:** Replay is deterministic; edit and transfer quality meet frozen thresholds.

---

## V4-8. RL Fine-Tuning Safety

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Instruction-following improvement | > 20% relative vs supervised-only baseline | 1 | Tag compliance rate before and after RL |
| Physical control monotonicity after RL | > 0.8 | 1 | Same protocol as V4-1 after RL phase |
| Plain-text naturalness degradation | < 5% | 1 | Held-out no-tag TTS quality comparison |

**Pass condition:** RL improves instruction following without degrading physical editability or plain-text quality.

---

## V4-9. Fish S2 Head-To-Head

| Gate | Threshold | Tier | Method |
|------|-----------|------|--------|
| Acting editability win | Required for any "beats Fish S2" claim | 2 | Frozen evaluation bundle |
| Trajectory replay fidelity win | Required for any "beats Fish S2" claim | 2 | Frozen evaluation bundle |
| Edit locality win | Required for any "beats Fish S2" claim | 2 | Frozen evaluation bundle |
| First-take naturalness guardrail | No clear loss | 2 | Blind A/B preference test |
| Few-shot speaker similarity guardrail | No clear loss | 2 | Frozen speaker similarity protocol |
| Latency class disclosure | Required | 0 | Report hardware class and RTF |

**Pass condition:** Claim is scoped to axes where TMRVC wins; guardrail axes show no clear deficit.

---

## V4 Summary Checklist

| # | Criterion | Status |
|---|-----------|--------|
| V4-1 | Physical control responsiveness | [ ] |
| V4-2 | Acting latent health | [ ] |
| V4-3 | Supervision tier weighting | [ ] |
| V4-4 | Biological constraint regularization | [ ] |
| V4-5 | Cross-runtime parity | [ ] |
| V4-6 | Serve-path streaming | [ ] |
| V4-7 | Replay and transfer fidelity | [ ] |
| V4-8 | RL fine-tuning safety | [ ] |
| V4-9 | Fish S2 head-to-head | [ ] |

All v4 criteria must pass before the v4 release candidate is promoted.

---
---

# v3 Acceptance Thresholds (Historical)

> **Superseded by v4.** Retained for reference only.

The gates below were defined for the UCLM v3 release.
These gates were consistent with the former `plan/worker_06_validation.md`.

Threshold ownership follows a tiered freeze policy:

- `Tier 0`: must freeze before Stage B large-scale training
- `Tier 1`: may be estimated from pilot runs, but must freeze before large-scale claim-making or release-candidate training
- `Tier 2`: protocol freezes early, final numeric cutoffs freeze before release sign-off

At minimum:

- `Tier 0`
  - runtime budgets
  - parity tolerances
  - frame/alignment conventions
  - prompt-target evaluation pairings
  - language set and code-switch pairs
  - provider registry entries and hardware classes
- `Tier 1`
  - few-shot score bundle thresholds
  - context-separation thresholds
  - control-response thresholds
  - leakage/disentanglement thresholds
- `Tier 2`
  - MOS/preference acceptance cutoffs
  - rater QC cutoffs
  - duplicate-consistency cutoffs

---

## 1. Pointer TTS Mainline Integrity

| Gate | Threshold | Method |
|------|-----------|--------|
| Hidden MFA dependency | None | Confirm `v3 pointer` train / serve path runs without MFA artifacts |
| Hidden duration dependency | None | Confirm mainline pointer train / serve path runs without `durations.npy` |
| Alignment-free release recipe | Required | Validate at least one convergent recipe where pointer progression no longer depends on frame-level aligner labels at release time |
| Pointer-state monotonicity | 100% valid | Automated pointer trace checks on held-out synthesis set |

**Pass condition:** All four gates pass on the release-candidate configuration.

---

## 2. Streaming Latency Budget

| Gate | Threshold | Method |
|------|-----------|--------|
| Steady-state frame compute latency | <= 10 ms p95 on the release target real-time path | Dedicated latency harness on the pinned deployment hardware |
| End-to-end chunk latency | <= 50 ms p95 on the release streaming server path | Streaming endpoint measurement on the pinned deployment hardware |
| Time-to-first-audio | < 200 ms | Measure via streaming TTS endpoint on held-out prompts |
| Load degradation | < 15% p95 increase at 32 concurrent streams vs 1 stream | Load test on the release configuration |
| Real-time CFG default | `off`, `lazy`, or `distilled` | Verify Rust / VST default mode and runtime logs |

`Steady-state frame compute latency` excludes one-time prompt extraction and model load, but includes the
pointer update, `uclm_core`, and the active waveform decoder used in the real-time path.

**Pass condition:** The release target stays within budget without enabling a non-validated `full` two-pass CFG default.

---

## 3. Pointer Runtime Robustness

| Gate | Threshold | Method |
|------|-----------|--------|
| Forced-advance rate | Warning if > 5% of advances; fail if sustained and unexplained | Pointer fallback counters on held-out synthesis set |
| Skip-protection rate | Warning if > 10% of holds; fail if sustained and unexplained | Pointer fallback counters on held-out synthesis set |
| Python vs Rust threshold parity | No contract divergence | Near-threshold pointer-decision parity tests |
| Cache-sync error rate | < 0.1% | Cached vs recomputed outputs across pointer-boundary transitions |

**Pass condition:** No contract divergence, and fallback counters remain within the documented release envelope.

---

## 4. TTS Quality and Control Responsiveness

| Gate | Threshold | Method |
|------|-----------|--------|
| Pace responsiveness | Duration ratio between pace sweep endpoints differs by >= 30% | Control responsiveness benchmark |
| Hold-bias responsiveness | Mean hold duration changes monotonically across 3+ levels | Control responsiveness benchmark |
| Boundary-bias responsiveness | Boundary behavior changes directionally across sweep | Control responsiveness benchmark |
| Explicit 8-D control responsiveness | Each control dimension shows directional movement in at least one registered metric | Voice-state responsiveness benchmark |

**Pass condition:** All exposed runtime controls produce measurable, directional, reproducible changes.

---

## 5. Context Sensitivity and Drama Acting

| Gate | Threshold | Method |
|------|-----------|--------|
| Context separation | `context_separation_score` above the frozen threshold | Same-text different-context evaluation |
| Prosody collapse | `prosody_collapse_score` below the frozen threshold | Same-text grouped evaluation |
| Acting alignment | `acting_alignment_score` above the frozen threshold | Context-to-acoustics correlation benchmark |
| Human dramatic appropriateness | Above chance and above internal neutral baseline | Blind rating protocol |

**Pass condition:** Objective metrics and human evaluation both show non-trivial context-driven delivery changes.

**Scope note:** In v3.0 this section applies to the Python serve path. Rust / VST / strict real-time ONNX paths are validated under shared pointer/control parity and latency gates unless a validated fast-CFG path is shipped.

---

## 6. Few-Shot Speaker Adaptation

| Gate | Threshold | Method |
|------|-----------|--------|
| Speaker similarity | >= 0.80 at 3 s reference, >= 0.85 at 10 s | Frozen speaker-embedding protocol |
| Intelligibility degradation | < 3% vs text-prompted baseline | ASR re-transcription |
| Timbre / prosody disentanglement | `timbre_prosody_disentanglement_score` above threshold | Same prompt, different context protocol |
| Prompt prosody leakage | `prosody_transfer_leakage_score` below threshold | Cross-context prompt leakage benchmark |

**Pass condition:** The model preserves speaker identity while allowing context and controls to override reference prosody.

---

## 7. Voice Conversion Stability

| Gate | Threshold | Method |
|------|-----------|--------|
| Speaker similarity | >= v2 baseline | Frozen VC eval set |
| Intelligibility | <= v2 CER on converted speech | ASR re-transcription |
| Streaming behavior | No regression in causal runtime path | Streaming VC smoke and latency tests |
| Semantic-context path | Contract present and test-covered | Architecture and runtime contract tests |

**Pass condition:** VC does not regress while semantic-context handling remains causal and explicit.

---

## 8. Multilingual and Code-Switch Regression

| Gate | Threshold | Method |
|------|-----------|--------|
| Existing-language regression | Must remain within frozen regression budget | Re-run held-out tests after language inventory changes |
| Code-switch intelligibility | Within 2% CER of monolingual baseline per segment | Mixed-language evaluation set |
| Speaker identity retention | No material regression on mixed-language prompts | Speaker-similarity benchmark |

**Pass condition:** Adding or reweighting languages must not silently degrade previously supported languages.

---

## 9. Pseudo-Annotation and Curation Quality

| Gate | Threshold | Method |
|------|-----------|--------|
| ASR transcript accuracy | WER < 10% on audited sample | Manual spot audit |
| Text normalization correctness | >= 95% pass rate | Checklist audit |
| Speaker clustering purity | NMI >= 0.80 | Labeled clustering subset |
| Quality-score calibration | False reject < 5%, false accept < 10% | Labeled quality subset |

**Pass condition:** Pseudo-labeled data is accepted only after audit metrics meet the curation gate.

---

## 10. Separation Policy

| Gate | Threshold | Method |
|------|-----------|--------|
| Annotation uplift | Must exceed measured artifact cost | Raw vs separated comparison protocol |
| Initial mainline waveform-teacher use | Forbidden | Export-policy validation |
| Research-bucket teacher use | Only with artifact + timbre + human approval gates | Research / ablation bucket audit |

**Pass condition:** Separation improves annotation quality without silently contaminating mainline waveform teachers.

---

## 11. External Baseline Parity

| Gate | Threshold | Method |
|------|-----------|--------|
| Baseline artifact pinning | Required | Must reference the active `primary` and `secondary` entries in `docs/design/external-baseline-registry.md` |
| Primary-axis deficit vs primary baseline | None allowed on declared primary claim axes | Frozen evaluation bundle against the pinned `primary` baseline |
| Human preference vs primary baseline | >= 50% win rate on declared preference axes | Blind A/B test with the pinned `primary` baseline artifact |
| Naturalness MOS gap vs primary baseline | >= 0.0 delta on declared primary naturalness axis | 5-point MOS protocol |
| Few-shot score bundle vs primary baseline | >= 0.0 normalized delta on the frozen few-shot bundle | Speaker similarity + intelligibility + leakage bundle |
| Streaming comparison vs secondary baseline | No unexplained deficit on declared streaming claim axes | Streaming latency and streaming quality protocol against the pinned `secondary` baseline |

**Pass condition:** External-baseline comparison is reproducible, pinned, and shows no deficit on any declared primary claim axis. If a deficit exists, the corresponding public claim is blocked until the claim scope is narrowed and re-approved.

---

## 12. Human Evaluation Quality Control

| Gate | Threshold | Method |
|------|-----------|--------|
| Rater count | Minimum 30 unique raters for release sign-off | Stored evaluation records |
| Duplicate-sample ratio | 10-15% | Evaluation-set construction |
| Rater self-consistency | >= 0.7 | Cohen's kappa or agreement-rate QC |
| Statistics procedure | Pre-registered | Stored protocol + final report |

**Pass condition:** Subjective evaluation is auditable, statistically named, and QC-validated.

---

## Summary Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pointer TTS mainline integrity | [ ] |
| 2 | Streaming latency budget | [ ] |
| 3 | Pointer runtime robustness | [ ] |
| 4 | TTS control responsiveness | [ ] |
| 5 | Context sensitivity and drama acting | [ ] |
| 6 | Few-shot speaker adaptation | [ ] |
| 7 | Voice conversion stability | [ ] |
| 8 | Multilingual and code-switch regression | [ ] |
| 9 | Pseudo-annotation and curation quality | [ ] |
| 10 | Separation policy | [ ] |
| 11 | External baseline parity | [ ] |
| 12 | Human evaluation QC | [ ] |

All criteria must pass before the v3 release candidate is promoted.

---

## v4 Bootstrap Quality Thresholds

> **Status: FROZEN** (2026-03-17)
>
> These thresholds gate the acceptance of v4 bootstrap-generated training data.
> They correspond to the seven quality metrics computed by
> `tmrvc_data.bootstrap.quality_gates` (Phase 1-9) and are enforced by
> `tests/test_bootstrap_quality.py`.
>
> Changes to any threshold below require simultaneous updates to
> `tmrvc_data/bootstrap/quality_gates.py`, `tests/test_bootstrap_quality.py`,
> and this document.

| # | Metric | Threshold | Direction | Method |
|---|--------|-----------|-----------|--------|
| B1 | `diarization_purity` | >= 0.85 | higher is better | NMI / purity score on labeled diarization subset |
| B2 | `speaker_cluster_consistency` | >= 0.80 | higher is better | Cross-file cluster agreement on known-speaker subset |
| B3 | `overlap_rejection_precision` | >= 0.90 | higher is better | Precision on annotated overlap segments |
| B4 | `transcript_wer` | <= 0.15 | lower is better | WER on audited transcript subset (Whisper re-transcription) |
| B5 | `physical_label_coverage` | >= 0.70 | higher is better | Fraction of (utterance, dimension) pairs with `observed_mask=True` |
| B6 | `physical_confidence_calibration_error` | <= 0.20 | lower is better | ECE between reported confidence and actual error quantile |
| B7 | `language_coverage` | all target languages >= 100 utterances | per-language floor | Bootstrap corpus inventory count per language |

### Pass Condition

All seven thresholds must be met on the bootstrap corpus before the
resulting `v4_cache/` is eligible for supervised training.  A corpus
that fails any single gate is quarantined until the offending stage is
re-run or the corpus is excluded.

### Tier Interaction

Bootstrap quality gates are orthogonal to per-utterance supervision
tiers (A/B/C/D).  A corpus can pass all seven gates while still
containing individual Tier-D utterances; those utterances receive
reduced loss weight during training (see Phase 3-1).

---

## Summary Checklist (v4 addendum)

| # | Criterion | Status |
|---|-----------|--------|
| B1 | Diarization purity | [ ] |
| B2 | Speaker cluster consistency | [ ] |
| B3 | Overlap rejection precision | [ ] |
| B4 | Transcript WER | [ ] |
| B5 | Physical label coverage | [ ] |
| B6 | Physical confidence calibration error | [ ] |
| B7 | Language coverage | [ ] |

All seven bootstrap gates must pass before v4 training data is accepted.
