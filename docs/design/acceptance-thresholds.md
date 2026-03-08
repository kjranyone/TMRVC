# v3 Acceptance Thresholds

This document defines the explicit release gates for UCLM v3.
These gates must stay consistent with `plan/worker_06_validation.md`.
If a threshold changes here, the validation plan and benchmark scripts must be
updated in the same change.

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
