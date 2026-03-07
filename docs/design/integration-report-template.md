# Integration Report — v3 Release Candidate

**Report date:** _YYYY-MM-DD_
**Prepared by:** _Name / Worker ID_
**Release candidate:** _RC tag or commit hash_

---

## 1. Worker Completion Status

| Worker | Scope | Status | Notes |
|--------|-------|--------|-------|
| W01 | UCLM architecture & training loop | _TODO / IN PROGRESS / DONE_ | |
| W02 | Data pipeline & pseudo-annotation | _TODO / IN PROGRESS / DONE_ | |
| W03 | Forced alignment & MFA removal | _TODO / IN PROGRESS / DONE_ | |
| W04 | Streaming TTS serving | _TODO / IN PROGRESS / DONE_ | |
| W05 | Voice conversion integration | _TODO / IN PROGRESS / DONE_ | |
| W06 | Validation & evaluation | _TODO / IN PROGRESS / DONE_ | |

For each worker that is not DONE, list the remaining items in the Notes
column or in the Remaining Blockers section below.

---

## 2. Test Results Summary

### 2a. Unit and Integration Tests

| Test suite | Passed | Failed | Skipped | Total | Notes |
|------------|--------|--------|---------|-------|-------|
| `tests/data/` | | | | | |
| `tests/train/` | | | | | |
| `tests/serve/` | | | | | |
| `tests/scripts/` | | | | | |

**Command used:**
```bash
pytest --tb=short -q 2>&1 | tail -20
```

### 2b. Evaluation Script Results

| Evaluation | Passed | Key metric | Value | Threshold |
|------------|--------|-----------|-------|-----------|
| `evaluate_drama_acting.py` — context_sensitivity | | mean CV | | >= 0.10 |
| `evaluate_drama_acting.py` — control_responsiveness | | pace ratio | | >= 1.30 |
| `evaluate_drama_acting.py` — pause_realism | | mean pause (s) | | 0.05-0.40 |

Attach or link the full JSON output:
- Path: `_results/drama_eval.json_`

---

## 3. Quality Gate Results

### 3a. Acceptance Thresholds (from `docs/design/acceptance-thresholds.md`)

| # | Criterion | Pass/Fail | Evidence |
|---|-----------|-----------|----------|
| 1 | TTS training without MFA artifacts | | _Link to alignment error report_ |
| 2 | Streaming TTS < 50 ms per frame step | | _p95 latency value_ |
| 3 | No VC regression | | _Cosine similarity, MOS, CER values_ |
| 4 | Pacing controls directional | | _evaluate_drama_acting.py output_ |
| 5 | Context sensitivity measurable | | _evaluate_drama_acting.py output_ |
| 6 | Human evaluation preference | | _MOS and preference % with CIs_ |
| 7 | Pseudo-annotation audit | | _Validation protocol results_ |

### 3b. Pseudo-Annotation Validation (from `docs/design/pseudo-annotation-validation.md`)

| Gate | Pass/Fail | Measured value | Threshold |
|------|-----------|----------------|-----------|
| ASR spot-check WER | | | < 10% |
| Text normalization audit | | | >= 95% pass |
| Speaker clustering NMI | | | >= 0.80 |
| Quality-score threshold | | | FR < 5%, FA < 10% |
| Confidence calibration ECE | | | < 0.05 |
| Separation front-end selected | | | Composite score documented |

---

## 4. Remaining Blockers

List anything that prevents promotion of this RC to production.

| ID | Blocker | Owner | Severity | ETA |
|----|---------|-------|----------|-----|
| B1 | _Description_ | _Worker / Name_ | _Critical / High / Medium_ | _Date_ |
| B2 | | | | |

If there are no blockers, write: **No remaining blockers.**

---

## 5. Acceptance Criteria Pass/Fail Checklist

This is the final sign-off checklist. Every item must be marked PASS before the
RC is promoted.

- [ ] **AC-1** TTS training completes without MFA artifacts
- [ ] **AC-2** Streaming TTS p95 frame-step latency < 50 ms
- [ ] **AC-3** VC metrics equal or exceed v2 baselines
- [ ] **AC-4** Pacing controls produce significant directional changes
- [ ] **AC-5** Context sensitivity CV >= 0.10 on held-out dialogue
- [ ] **AC-6** Human evaluation: v3 preferred >= 50% (non-inferiority)
- [ ] **AC-7** Pseudo-annotation audit: all six gates passed
- [ ] **AC-8** All unit and integration tests pass
- [ ] **AC-9** No critical or high-severity blockers remain open
- [ ] **AC-10** This report is reviewed and signed off by at least two team members

---

## 6. Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Validation lead | | | |
| Training lead | | | |
| Serving lead | | | |
| Project lead | | | |

---

_End of integration report template._
