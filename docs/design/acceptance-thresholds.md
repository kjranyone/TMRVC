# v3 Acceptance Thresholds

This document defines the explicit acceptance criteria that must be satisfied
before the v3 release is considered validated. Each criterion has a measurable
gate and a designated verification method.

---

## 1. TTS Training Without MFA Artifacts

| Gate | Threshold | Method |
|------|-----------|--------|
| Forced-alignment error rate | < 2% of utterances flagged as misaligned | Run `scripts/annotate/run_forced_alignment.py` on validation split; count utterances where boundary drift exceeds 40 ms |
| Silence insertion artifacts | 0 inserted silences > 300 ms that do not appear in the reference | Compare synthesised audio pause locations against reference transcript pause markers |
| Spectrogram discontinuity | No visible glitch frames at alignment boundaries | Spot-check 50 random utterances visually; automated delta-energy check at boundary frames (threshold: < 3 dB spike) |

**Pass condition:** All three sub-gates met on the held-out validation set.

---

## 2. Streaming TTS Latency Budget

| Gate | Threshold | Method |
|------|-----------|--------|
| Per-frame-step latency | < 50 ms (p95) | Benchmark with `scripts/evaluate_drama_acting.py` or dedicated latency harness on target GPU (A100 / RTX 4090) |
| Time-to-first-audio | < 200 ms from request receipt | Measure via the `/tts/stream` endpoint under `tmrvc-serve` with 10 concurrent requests |
| Throughput degradation under load | < 15% increase in p95 latency at 32 concurrent streams vs 1 stream | Load test with `locust` or equivalent |

**Pass condition:** p95 per-frame-step latency stays under 50 ms across all test
utterances (minimum 200 utterances, diverse lengths).

---

## 3. No Voice Conversion (VC) Regression

| Gate | Threshold | Method |
|------|-----------|--------|
| Speaker similarity (cosine) | >= v2 baseline (target >= 0.85) | Compute speaker-embedding cosine similarity between converted and target speaker on the standard VC eval set |
| MOS naturalness | No statistically significant drop vs v2 (paired t-test, p < 0.05) | Crowdsourced MOS evaluation, minimum 30 listeners, 20 utterances |
| Character error rate (ASR) | <= v2 CER on converted speech | Transcribe converted audio with Whisper-large-v3; compare CER |

**Pass condition:** All three metrics equal or exceed v2 baselines.

---

## 4. Pacing Controls Produce Measurable Directional Changes

| Gate | Threshold | Method |
|------|-----------|--------|
| Pace parameter effect | Duration ratio between pace=0.7 and pace=1.3 differs by >= 30% | `evaluate_control_responsiveness()` in `scripts/evaluate_drama_acting.py` |
| Hold-bias parameter effect | Mean hold duration increases monotonically across 3+ hold_bias levels | Same evaluation script |
| Boundary-bias parameter effect | Number of detected phrase boundaries changes directionally with boundary_bias | Same evaluation script |

**Pass condition:** Each control parameter produces a statistically significant
directional change (paired t-test, p < 0.01) on at least 20 test utterances.

---

## 5. Context Sensitivity on Held-Out Dialogue

| Gate | Threshold | Method |
|------|-----------|--------|
| Pacing variance across contexts | Coefficient of variation in utterance duration >= 0.10 when the same text is spoken in different dialogue contexts | `evaluate_context_sensitivity()` in `scripts/evaluate_drama_acting.py` |
| F0 contour variance | Mean F0 standard deviation across contexts >= 5 Hz | Extract F0 with CREPE or WORLD; compare across context conditions |
| Perceptual differentiation | Listeners correctly identify intended emotion/context above chance (> 60% accuracy, 4-way forced choice) | ABX listening test, minimum 20 listeners |

**Pass condition:** Both objective metrics met, and perceptual test exceeds
chance level.

---

## 6. Human Evaluation Preference Tests

### 6a. A/B Preference: v3 vs v2

- **Sample:** 30 utterance pairs, balanced across speakers and content types.
- **Listeners:** Minimum 30 unique raters (crowdsourced or internal).
- **Protocol:** Randomised presentation order; forced-choice preference with
  optional "no preference" allowed.
- **Pass condition:** v3 preferred >= 50% of the time (non-inferiority) with
  95% confidence interval lower bound >= 45%.

### 6b. MOS Naturalness

- **Scale:** 1-5 MOS.
- **Pass condition:** v3 MOS >= 3.8 and no statistically significant drop from
  v2.

### 6c. Drama-Acting Expressiveness

- **Protocol:** Raters score expressiveness on a 1-5 Likert scale for 20
  context-varied utterances.
- **Pass condition:** Mean expressiveness score >= 3.5.

---

## 7. Pseudo-Annotation Audit Thresholds

| Gate | Threshold | Method |
|------|-----------|--------|
| ASR transcript accuracy | WER < 10% on spot-check sample (N >= 100) | Manual review against ground-truth transcripts; see `docs/design/pseudo-annotation-validation.md` |
| Text normalization correctness | >= 95% of sampled utterances pass manual audit | Checklist-based review |
| Speaker clustering purity | NMI >= 0.80 | Compare predicted clusters against ground-truth speaker labels |
| Quality-score filtering | False-reject rate < 5%, false-accept rate < 10% | Calibration on labeled quality subset |

**Pass condition:** All four sub-gates met before pseudo-annotated data is used
in any training run that feeds the release candidate.

---

## 8. Few-Shot Speaker Adaptation

| Gate | Threshold | Method |
|------|-----------|--------|
| Speaker similarity (cosine) | >= 0.80 at 3s reference, >= 0.85 at 10s | Cosine similarity of speaker embeddings |
| Intelligibility preservation | CER degradation < 3% vs text-prompted | ASR re-transcription |
| Timbre-prosody disentanglement | F0 CV >= 0.08 across contexts with same prompt | Extract F0 with CREPE/WORLD |
| External baseline parity | Within 0.05 cosine similarity of Qwen3-TTS class | Same protocol, fixed reference lengths |

**Pass condition:** Speaker similarity meets the per-reference-length thresholds,
intelligibility degradation stays below 3%, and disentanglement CV is met. The
external baseline parity gate is advisory in early iterations but mandatory for
release candidates.

---

## 9. External Baseline Parity

| Gate | Threshold | Method |
|------|-----------|--------|
| Human preference vs external SOTA | >= 40% preference (non-inferiority) | Blind A/B test, 30+ raters |
| Naturalness MOS gap | <= 0.3 MOS below external SOTA | 5-point MOS, 30+ raters |
| Controllability advantage | Statistically significant improvement in at least 2 of {pace, hold_bias, voice_state} | Control sweep evaluation |

**Pass condition:** Non-inferiority on human preference (>= 40%), naturalness
MOS within 0.3 of the external SOTA system, and a statistically significant
controllability advantage on at least two control dimensions.

---

## 10. Guidance and Cache Stability

| Gate | Threshold | Method |
|------|-----------|--------|
| Pointer monotonicity under CFG | 100% maintained for cfg_scale in [1.0, 3.0] | Automated pointer trace check |
| Cache sync error rate | < 0.1% output mismatch | Compare cached vs recomputed on 100 utterances |

**Pass condition:** Pointer monotonicity is never violated within the safe
cfg_scale range, and cached outputs match recomputed outputs on all test
utterances.

---

## Summary Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | TTS training without MFA artifacts | [ ] |
| 2 | Streaming TTS < 50 ms per frame step | [ ] |
| 3 | No VC regression | [ ] |
| 4 | Pacing controls directional | [ ] |
| 5 | Context sensitivity measurable | [ ] |
| 6 | Human evaluation preference | [ ] |
| 7 | Pseudo-annotation audit | [ ] |
| 8 | Few-shot speaker adaptation | [ ] |
| 9 | External baseline parity | [ ] |
| 10 | Guidance and cache stability | [ ] |

All ten criteria must pass before the v3 release candidate is promoted to
production.
