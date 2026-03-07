# Pseudo-Annotation Validation Protocol

This document describes the validation steps required before any
pseudo-annotated corpus is used in v3 training. Every dataset that passes
through the automatic annotation pipeline must clear all gates below.

---

## 1. ASR Spot-Check Accuracy

**Goal:** Verify that the ASR transcripts are accurate enough to serve as
training labels.

### Procedure

1. Randomly sample **N = 100** utterances (stratified by speaker and duration
   bucket: short < 3 s, medium 3-10 s, long > 10 s).
2. A human reviewer listens to each utterance and writes a reference
   transcript.
3. Compute word error rate (WER) between the ASR output and the human
   reference.
4. Record per-utterance WER and aggregate statistics.

### Acceptance criteria

| Metric | Threshold |
|--------|-----------|
| Mean WER | < 10% |
| Proportion of utterances with WER > 20% | < 5% |
| Proportion of utterances with WER = 0% | > 60% |

### Escalation

If the mean WER is between 10% and 15%, increase sample size to N = 200 and
re-evaluate. If still above 10%, the ASR model or decoding parameters must be
updated before the corpus is accepted.

---

## 2. Text Normalization Audit Checklist

For each sampled utterance, verify that the normalized transcript satisfies all
of the following:

- [ ] Numbers are spelled out correctly (e.g., "42" -> "forty-two").
- [ ] Currency symbols are expanded (e.g., "$5" -> "five dollars").
- [ ] Abbreviations are expanded where contextually appropriate.
- [ ] Punctuation is consistent with the project style guide (sentence-final
      period, no trailing whitespace).
- [ ] Unicode characters are normalized to NFC form.
- [ ] Profanity or PII tokens are handled per the data policy (redacted or
      tagged, not silently dropped).
- [ ] Hyphenated compounds are treated consistently (either always split or
      always joined, per the lexicon).
- [ ] Repeated whitespace, tabs, and BOM characters are removed.

### Acceptance criteria

- >= 95% of sampled utterances pass all checklist items.
- Any systematic failure (same rule broken > 10 times) requires a pipeline fix
  and full re-run of normalization.

---

## 3. Speaker Clustering Purity

**Goal:** Ensure that automatic speaker diarization / clustering assigns
utterances to the correct speaker identity.

### Procedure

1. Select a subset with known ground-truth speaker labels (minimum 500
   utterances across >= 10 speakers).
2. Run the speaker clustering pipeline on this subset.
3. Compute **Normalized Mutual Information (NMI)** between predicted cluster
   labels and ground-truth speaker labels.

### Acceptance criteria

| Metric | Threshold |
|--------|-----------|
| NMI | >= 0.80 |
| Cluster count accuracy | Predicted K within +/- 20% of true K |
| Over-segmentation rate | < 15% of speakers split into 2+ clusters |
| Under-segmentation rate | < 10% of clusters contain 2+ speakers |

### Diagnostic steps on failure

- Inspect the embedding space with t-SNE; look for overlapping clusters.
- Check whether failures concentrate on speakers with similar voice
  characteristics (e.g., same gender, similar age).
- If NMI is between 0.75 and 0.80, re-tune the clustering threshold and
  re-evaluate once.

---

## 4. Quality-Score Threshold Selection Methodology

The annotation pipeline assigns a scalar quality score to each utterance. The
threshold determines which utterances are kept for training.

### Procedure

1. **Label a calibration set.** Select 200 utterances spanning the full
   quality-score range. Have a human rate each as "usable" or "unusable" for
   TTS training.
2. **Plot the precision-recall curve** of the quality filter at varying
   thresholds.
3. **Select the operating point** that satisfies:
   - False-reject rate (good utterances discarded) < 5%.
   - False-accept rate (bad utterances kept) < 10%.
4. **Document the chosen threshold** and the precision/recall at that point.

### Acceptance criteria

| Metric | Threshold |
|--------|-----------|
| False-reject rate | < 5% |
| False-accept rate | < 10% |
| AUC of precision-recall curve | > 0.90 |

### Recalibration triggers

Re-run threshold selection whenever:
- The ASR model is updated.
- A new data source is added to the corpus.
- The feature extraction pipeline changes (e.g., new mel parameters).

---

## 5. Confidence Calibration Checks

ASR and quality-score models output confidence values. These must be
well-calibrated: a confidence of 0.9 should mean approximately 90% of those
predictions are correct.

### Procedure

1. Bin predictions into 10 equal-width confidence bins (0.0-0.1, ..., 0.9-1.0).
2. For each bin, compute the actual accuracy (fraction correct based on human
   labels).
3. Compute **Expected Calibration Error (ECE):**

   ```
   ECE = sum over bins of (|bin_size / N| * |accuracy_bin - confidence_bin|)
   ```

4. Plot a reliability diagram (accuracy vs. confidence).

### Acceptance criteria

| Metric | Threshold |
|--------|-----------|
| ECE | < 0.05 |
| Maximum bin deviation | < 0.15 |
| No empty high-confidence bins | Bins 0.8-1.0 must contain >= 5% of samples |

### Remediation

If ECE > 0.05, apply temperature scaling on a held-out calibration set and
re-evaluate. Document the temperature parameter used.

---

## 6. Separation-Front-End Comparison Criteria

When multiple source-separation or speech-enhancement front-ends are available,
use the following criteria to select the best one before running annotation.

### Comparison protocol

1. Take a standard noisy test set (minimum 100 utterances with known clean
   references).
2. Run each candidate front-end.
3. Measure:

| Metric | Weight | Notes |
|--------|--------|-------|
| SI-SDR improvement (dB) | 0.30 | Scale-invariant signal-to-distortion ratio |
| PESQ (wideband) | 0.25 | Perceptual quality |
| Downstream ASR WER | 0.25 | Transcription accuracy on separated output |
| Wall-clock time (relative) | 0.10 | Processing speed |
| Artifact rate | 0.10 | Human count of audible artifacts per 10 utterances |

4. Compute a weighted composite score.
5. Select the front-end with the highest composite score, provided no single
   metric is more than 20% worse than the best candidate on that metric.

### Documentation

Record the chosen front-end, its version, the composite score, and per-metric
results in the integration report (see `docs/design/integration-report-template.md`).

---

## Workflow Summary

```
Raw audio
  |
  v
[Source separation / enhancement]  -->  Gate 6: front-end comparison
  |
  v
[ASR transcription]                -->  Gate 1: spot-check WER
  |                                     Gate 5: confidence calibration
  v
[Text normalization]               -->  Gate 2: normalization audit
  |
  v
[Speaker clustering]               -->  Gate 3: NMI check
  |
  v
[Quality scoring & filtering]      -->  Gate 4: threshold calibration
  |                                     Gate 5: confidence calibration
  v
Validated pseudo-annotated corpus
```

All gates must pass before the corpus is approved for training use.
