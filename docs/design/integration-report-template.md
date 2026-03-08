# Integration Report — v3 Release Candidate

**Report date:** _YYYY-MM-DD_
**Prepared by:** _Name / Worker ID_
**Release candidate:** _RC tag or commit hash_
**Primary external baseline:** _baseline_id from external-baseline-registry_
**Secondary external baseline:** _baseline_id from external-baseline-registry_
**Frozen hardware class:** _hardware_class id_

---

## 1. Worker Completion Status

| Worker | Scope | Status | Notes |
|--------|-------|--------|-------|
| W01 | Architecture and model contract | _TODO / IN PROGRESS / DONE_ | |
| W02 | Training pipeline and losses | _TODO / IN PROGRESS / DONE_ | |
| W03 | Dataset contract, text supervision, and metrics | _TODO / IN PROGRESS / DONE_ | |
| W04 | Serving and runtime pointer execution | _TODO / IN PROGRESS / DONE_ | |
| W05 | DevOps and operator docs | _TODO / IN PROGRESS / DONE_ | |
| W06 | Validation and release gates | _TODO / IN PROGRESS / DONE_ | |
| W07 | Curation orchestration | _TODO / IN PROGRESS / DONE_ | |
| W08 | Curation providers | _TODO / IN PROGRESS / DONE_ | |
| W09 | Curation selection policy | _TODO / IN PROGRESS / DONE_ | |
| W10 | Curation export contract | _TODO / IN PROGRESS / DONE_ | |
| W11 | Curation validation | _TODO / IN PROGRESS / DONE_ | |
| W12 | Gradio / WebUI control plane | _TODO / IN PROGRESS / DONE_ | |

For each worker that is not `DONE`, list the remaining items in the Notes column or in the Remaining Blockers section below.

---

## 2. Test Results Summary

### 2.1 Unit and Integration Tests

| Test suite | Passed | Failed | Skipped | Total | Notes |
|------------|--------|--------|---------|-------|-------|
| `tests/data/` | | | | | |
| `tests/train/` | | | | | |
| `tests/serve/` | | | | | |
| `tests/runtime/` | | | | | |
| `tmrvc-engine-rs/tests/` | | | | | |

**Command used:**

```bash
pytest --tb=short -q
```

### 2.2 Evaluation / Benchmark Results

| Evaluation | Passed | Key metric | Value | Threshold |
|------------|--------|-----------|-------|-----------|
| Pointer monotonicity | | violation count | | 0 |
| Real-time frame compute | | p95 ms | | <= 10 ms |
| Streaming chunk latency | | p95 ms | | <= 50 ms |
| Time-to-first-audio | | ms | | < 200 ms |
| Voice-state responsiveness | | directional dims | | 8 / 8 required |
| Few-shot leakage | | leakage score | | below frozen threshold |
| Curation pseudo-label utility | | utility score | | non-negative vs ablation |

Attach or link the full outputs:

- Path: `_results/release_eval.json_`
- Path: `_results/latency.json_`
- Path: `_results/curation_validation.json_`

---

## 3. Acceptance Threshold Results

### 3.1 Release Gates

| # | Criterion | Pass/Fail | Evidence |
|---|-----------|-----------|----------|
| 1 | Pointer TTS mainline integrity | | |
| 2 | Streaming latency budget | | |
| 3 | Pointer runtime robustness | | |
| 4 | TTS quality and control responsiveness | | |
| 5 | Context sensitivity and drama acting | | |
| 6 | Few-shot speaker adaptation | | |
| 7 | Voice conversion stability | | |
| 8 | Multilingual and code-switch regression | | |
| 9 | Pseudo-annotation and curation quality | | |
| 10 | Separation policy | | |
| 11 | External baseline parity | | |
| 12 | Human evaluation quality control | | |

### 3.2 Latency Detail

| Path | Pass/Fail | Measured value | Threshold |
|------|-----------|----------------|-----------|
| Real-time steady-state frame compute | | | <= 10 ms p95 |
| Streaming server end-to-end chunk | | | <= 50 ms p95 |
| TTFA | | | < 200 ms |
| Rust / VST default CFG mode | | | `off`, `lazy`, or `distilled` |

### 3.3 Curation / Supervision Detail

| Gate | Pass/Fail | Measured value | Threshold |
|------|-----------|----------------|-----------|
| ASR spot-check WER | | | < 10% |
| Text normalization audit | | | >= 95% pass |
| Speaker clustering NMI | | | >= 0.80 |
| Quality-score threshold | | | FR < 5%, FA < 10% |
| `voice_state` calibration ECE | | | frozen threshold |
| `voice_state` utility uplift | | | non-negative vs no-label ablation |
| Export contract completeness | | | masks + provenance present |

---

## 4. Contract Parity Checklist

| Contract | Pass/Fail | Evidence |
|----------|-----------|----------|
| PyTorch / ONNX / Rust share canonical CFG mask set | | |
| `uclm_core` uses canonical raw-state inputs, not divergent `state_cond` public API | | |
| Pointer state update parity near threshold | | |
| `SpeakerProfile` fingerprint invalidation works | | |
| Frame convention parity (`24kHz`, `240`, inclusive/exclusive) | | |
| WebUI uses `tmrvc-serve` authoritative API, not direct manifest mutation | | |

---

## 5. Remaining Blockers

List anything that prevents promotion of this RC.

| ID | Blocker | Owner | Severity | ETA |
|----|---------|-------|----------|-----|
| B1 | _Description_ | _Worker / Name_ | _Critical / High / Medium_ | _Date_ |
| B2 | | | | |

If there are no blockers, write: **No remaining blockers.**

---

## 6. Final Sign-Off Checklist

- [ ] **AC-1** Pointer TTS train / serve path runs without MFA artifacts
- [ ] **AC-2** Real-time steady-state frame compute latency is <= 10 ms p95
- [ ] **AC-3** Streaming server chunk latency is <= 50 ms p95
- [ ] **AC-4** VC metrics match or exceed the pinned regression target
- [ ] **AC-5** All 8 explicit `voice_state` dimensions show directional responsiveness
- [ ] **AC-6** Context-sensitive acting metrics and human ratings pass the frozen threshold
- [ ] **AC-7** Few-shot prompt leakage remains below the frozen threshold
- [ ] **AC-8** Multilingual / code-switch regression gates pass
- [ ] **AC-9** Curation audit and `voice_state` supervision utility gates pass
- [ ] **AC-10** All unit, integration, parity, and runtime tests pass
- [ ] **AC-11** No critical or high-severity blockers remain open
- [ ] **AC-12** This report is reviewed and signed off by the required leads

---

## 7. Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Validation lead | | | |
| Training lead | | | |
| Serving lead | | | |
| Curation lead | | | |
| Project lead | | | |

---

_End of integration report template._
