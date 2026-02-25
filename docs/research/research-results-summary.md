# Research Results Summary

Status: **Template** (populate with actual values after evaluation)
Created: 2026-02-25
Last updated: 2026-02-25

## 1. Claims and Linked Metrics

### C1: Scene State Latent (SSL) preserves turn-to-turn acting consistency

| Metric | B0 (baseline) | B1 (+SSL) | Delta | p-value | CI (95%) |
|--------|--------------|-----------|-------|---------|----------|
| turn_coherence_score | — | — | — | — | — |
| mel_MSE | — | — | — | — | — |
| UTMOS | — | — | — | — | — |

**Gate**: B1 must statistically improve turn_coherence_score over B0 (p < 0.05).

### C2: Breath-Pause Event Head (BPEH) improves expressive realism

| Metric | B0 (baseline) | B2 (+BPEH) | Delta | p-value | CI (95%) |
|--------|--------------|------------|-------|---------|----------|
| breath_event_f1 | — | — | — | — | — |
| pause_timing_mae | — | — | — | — | — |
| UTMOS | — | — | — | — | — |

**Gate**: B2 must statistically improve breath_event_f1 over B0 (p < 0.05).

### C3: LCD enables monotonic quality-latency control

| Metric | B3 (SSL+BPEH) | B4 (+LCD) | Delta | p-value | CI (95%) |
|--------|---------------|-----------|-------|---------|----------|
| latency_p50 | — | — | — | — | — |
| latency_p95 | — | — | — | — | — |
| monotonicity_pass_rate | — | — | — | — | — |
| UTMOS | — | — | — | — | — |

**Gate**: B4 monotonicity_pass_rate >= 0.90. B4 UTMOS not significantly below B3.

### Combined: Full proposal (B4) vs Baseline (B0)

| Metric | B0 | B4 | Delta | p-value | CI (95%) |
|--------|----|----|-------|---------|----------|
| mel_MSE | — | — | — | — | — |
| SECS | — | — | — | — | — |
| F0 Corr | — | — | — | — | — |
| UTMOS | — | — | — | — | — |

**Gate**: B4 must not degrade SECS or F0 Corr below acceptable bounds.

## 2. Ablation Table (B0-B4)

> Populated by `scripts/eval_research_ablation.py` output.

| Variant | SSL | BPEH | LCD | mel_MSE [CI] | SECS [CI] | F0 Corr [CI] | UTMOS [CI] |
|---------|-----|------|-----|-------------|-----------|--------------|------------|
| B0 | — | — | — | — | — | — | — |
| B1 | + | — | — | — | — | — | — |
| B2 | — | + | — | — | — | — | — |
| B3 | + | + | — | — | — | — | — |
| B4 | + | + | + | — | — | — | — |

## 3. Statistical Analysis

> Populated by `scripts/stats_research.py` output.

### 3.1 Bootstrap 95% Confidence Intervals

See `eval/research/ci_table.md`.

### 3.2 Paired Significance Tests (Wilcoxon signed-rank)

See `eval/research/significance_table.md`.

## 4. Frozen Configuration

### Seeds

| Item | Value |
|------|-------|
| Eval seed | 42 |
| Bootstrap seed | 42 |
| Bootstrap resamples | 10,000 |

### Test Split

Defined in `configs/research/b0.yaml` (`test_split` section):

- **Datasets**: VCTK, JVS
- **Speakers**: vctk_p225, vctk_p226, vctk_p227, vctk_p228, jvs_jvs001, jvs_jvs002, jvs_jvs003, jvs_jvs004
- **Max utterances per speaker**: 10

All B0-B4 configs share the same test split.

### Research Configs

| Variant | Config | Features |
|---------|--------|----------|
| B0 | `configs/research/b0.yaml` | Baseline (no extensions) |
| B1 | `configs/research/b1.yaml` | + SSL (SceneStateUpdate, d_state=64) |
| B2 | `configs/research/b2.yaml` | + BPEH (BreathEventHead, d_hidden=128) |
| B3 | `configs/research/b3.yaml` | + SSL + BPEH |
| B4 | `configs/research/b4.yaml` | + SSL + BPEH + LCD (LatencyConditioner) |

### Model Checkpoints

| Checkpoint | Path | Step | Notes |
|-----------|------|------|-------|
| Teacher | `checkpoints/teacher_step*.pt` | — | Flow matching U-Net (17.2M params) |
| TTS | `checkpoints/tts_best.pt` | — | TextEncoder + DurationPredictor + F0Predictor + ContentSynthesizer |

## 5. Reproduction Commands

### 5.1 Prerequisites

```bash
# Install dependencies
uv sync

# Verify GPU
uv run python -c "import torch; print(torch.xpu.is_available())"

# Prepare data (if not cached)
uv run python scripts/prepare_datasets.py --config configs/datasets.yaml --device xpu
```

### 5.2 Run All Tests

```bash
uv run pytest tests/python/ -x --tb=short
```

### 5.3 Batch Ablation Evaluation

```bash
uv run python scripts/eval_research_ablation.py \
    --variants b0 b1 b2 b3 b4 \
    --checkpoint b0=checkpoints/b0_teacher.pt \
    --checkpoint b1=checkpoints/b1_teacher.pt \
    --checkpoint b2=checkpoints/b2_teacher.pt \
    --checkpoint b3=checkpoints/b3_teacher.pt \
    --checkpoint b4=checkpoints/b4_teacher.pt \
    --cache-dir data/cache \
    --seed 42 \
    --device xpu \
    --output-dir eval/research
```

Output: `eval/research/ablation_results.json`, `eval/research/ablation_table.md`

### 5.4 Statistical Analysis

```bash
uv run python scripts/stats_research.py \
    --input eval/research \
    --baseline b0 \
    --n-bootstrap 10000 \
    --seed 42
```

Output: `eval/research/stats.json`, `eval/research/ci_table.md`, `eval/research/significance_table.md`

### 5.5 Single Variant Evaluation

```bash
uv run python scripts/eval_research_baseline.py \
    --config configs/research/b0.yaml \
    --checkpoint checkpoints/teacher_step100000.pt \
    --cache-dir data/cache \
    --seed 42 \
    --device xpu \
    --output-dir eval/research/b0
```

### 5.6 Verify Reproducibility

```bash
# Run twice with same seed, compare results
uv run python scripts/eval_research_baseline.py \
    --config configs/research/b0.yaml \
    --checkpoint checkpoints/teacher_step100000.pt \
    --cache-dir data/cache --seed 42 --device xpu \
    --output-dir eval/research/b0_check

# JSON diff should be empty (within floating-point tolerance)
python -c "
import json
with open('eval/research/b0/results.json') as a, open('eval/research/b0_check/results.json') as b:
    r1, r2 = json.load(a), json.load(b)
    for u1, u2 in zip(r1['per_utterance'], r2['per_utterance']):
        assert u1['mel_mse'] == u2['mel_mse'], f'Mismatch: {u1} vs {u2}'
    print('Reproducibility verified.')
"
```

## 6. Implementation Inventory

### New Modules (Research-Specific)

| Module | Path | Purpose |
|--------|------|---------|
| SceneStateUpdate | `tmrvc-train/src/tmrvc_train/models/scene_state.py` | GRU-based SSL state update |
| BreathEventHead | `tmrvc-train/src/tmrvc_train/models/breath_event_head.py` | 4-head breath/pause prediction |
| LCD losses | `tmrvc-train/src/tmrvc_train/lcd.py` | LatencyConditioner + L_latency + L_mono |
| BPEH events | `tmrvc-data/src/tmrvc_data/bpeh_events.py` | Breath/pause event extraction |

### Modified Modules

| Module | Changes |
|--------|---------|
| `tts_trainer.py` | SSL + BPEH integration (forward, loss, checkpoint) |
| `distillation.py` | LCD phase, q conditioning, monotonicity loss |
| `app.py` | Per-session scene state cache in WebSocket |
| `tts_engine.py` | SceneStateUpdate loading + `update_scene_state()` |
| `schemas.py` | `scene_reset` field in WSConfigureRequest |
| `cli/train_tts.py` | `--enable-ssl`, `--enable-bpeh` flags |

### Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_research_baseline.py` | Deterministic single-variant eval |
| `scripts/eval_research_ablation.py` | Batch B0-B4 evaluation |
| `scripts/stats_research.py` | Bootstrap CI + Wilcoxon paired tests |

### Research Configs

| File | Variant |
|------|---------|
| `configs/research/b0.yaml` | Baseline |
| `configs/research/b1.yaml` | +SSL |
| `configs/research/b2.yaml` | +BPEH |
| `configs/research/b3.yaml` | +SSL+BPEH |
| `configs/research/b4.yaml` | +SSL+BPEH+LCD |

### Test Coverage

| Test File | Count | Scope |
|-----------|-------|-------|
| `test_scene_state.py` | 13 | SceneStateUpdate, DialogueHistoryEncoder, SSL losses |
| `test_bpeh_events.py` | 19 | Breath/pause event extraction + tensor conversion |
| `test_breath_event_head.py` | 14 | BreathEventHead model + BreathEventLoss |
| `test_tts_trainer.py` | 7 (BPEH) | BPEH integration in TTSTrainer |
| `test_research_configs.py` | 17 | B0-B3 config structure + training + CLI |
| `test_lcd.py` | 21 | LCD losses + conditioner + distillation integration + B4 config |
| `test_serve.py` | 12 (new) | Scene state engine + WSConfigureRequest scene_reset |
| `test_eval_ablation.py` | 23 | Ablation eval + bootstrap CI + Wilcoxon + stats tables |
| `test_eval_research_baseline.py` | 7 | Baseline eval seeding + reproducibility |

## 7. Publication Readiness Checklist

- [ ] All B0-B4 checkpoints trained
- [ ] `eval/research/ablation_results.json` generated
- [ ] `eval/research/stats.json` generated with CI and p-values
- [ ] C1 gate passed: B1 turn_coherence > B0 (p < 0.05)
- [ ] C2 gate passed: B2 breath_event_f1 > B0 (p < 0.05)
- [ ] C3 gate passed: B4 monotonicity_pass_rate >= 0.90
- [ ] B4 does not degrade SECS or F0 Corr
- [ ] Reproducibility verified (two seed-42 runs match)
- [ ] All tests pass: `uv run pytest tests/python/ -x`
