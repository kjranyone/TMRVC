# Research Development Plan (for Claude Code)

Created: 2026-02-25
Scope: SSL + BPEH + LCD implementation and paper-ready validation
Primary spec: `docs/research/research-novelty-plan.md`

## 0. Goal

Implement and validate the research contribution set:

- SSL: Scene State Latent
- BPEH: Breath-Pause Event Head
- LCD: Latency-conditioned distillation

Target outcome:

- Reproducible ablation results (B0/B1/B2/B3/B4)
- Statistical evidence for novelty claims
- Ready-to-write paper package

## 1. Non-Negotiable Project Rules

- Pre-release policy: no backward compatibility.
- Legacy checkpoint compatibility is not required.
- No migration guide is required.
- Any old path that blocks clarity should be deleted or replaced.

## 2. Execution Model

Use small, verifiable work packages (WP).

- One WP per Claude Code run.
- Each WP must finish with:
  - code changes
  - tests/validation command results
  - updated docs/checklists

Recommended branch naming:

- `research/wp1-ssl-scaffold`
- `research/wp2-bpeh-data`
- `research/wp3-ssl-bpeh-train`
- `research/wp4-lcd-distill`
- `research/wp5-ablation-eval`

## 3. Work Packages

## WP0 - Baseline Freeze

### Task

- Freeze baseline config/checkpoints/eval scripts.
- Add deterministic eval entrypoint for B0.

### Expected file changes

- `configs/` (baseline frozen configs)
- `scripts/` (deterministic evaluation runner)
- `docs/` (frozen baseline manifest)

### Done criteria

- Baseline manifest exists.
- Same command gives reproducible metrics within tolerance.

### Validation

```bash
uv run python scripts/eval_research_baseline.py --config configs/research/b0.yaml --seed 42
uv run python scripts/eval_research_baseline.py --config configs/research/b0.yaml --seed 42
```

---

## WP1 - SSL Model Scaffold

### Task

- Add `scene_state.py` and integrate state input/output in TTS path.
- Add state reset policy (scene boundary).

### Expected file changes

- `tmrvc-train/src/tmrvc_train/models/scene_state.py` (new)
- `tmrvc-train/src/tmrvc_train/tts_trainer.py`
- `tmrvc-core/src/tmrvc_core/types.py` (if state batch type needed)
- `configs/constants.yaml`

### Done criteria

- Training graph runs with SSL enabled.
- State tensor shape and reset behavior are unit-tested.

### Validation

```bash
uv run pytest tests/python -k "scene_state or tts_trainer"
```

---

## WP2 - BPEH Label Pipeline

### Task

- Add `events.json` schema and cache writer/reader.
- Support breath onset/duration/intensity and pause duration labels.

### Expected file changes

- `tmrvc-data/src/tmrvc_data/` (event preprocessing)
- `tmrvc-data/src/tmrvc_data/tts_dataset.py`
- `scripts/` (event extraction/validation tools)
- `docs/` (event schema doc)

### Done criteria

- Event labels are generated for target dataset split.
- Dataset loader exposes event tensors to trainer.

### Validation

```bash
uv run python scripts/build_breath_event_cache.py --cache-dir data/cache --dataset <dataset>
uv run pytest tests/python -k "event or tts_dataset"
```

---

## WP3 - BPEH Model + Losses

### Task

- Add `breath_event_head.py` and losses (`L_onset`, `L_dur`, `L_amp`).
- Plug event conditioning into TTS training path.

### Expected file changes

- `tmrvc-train/src/tmrvc_train/models/breath_event_head.py` (new)
- `tmrvc-train/src/tmrvc_train/tts_trainer.py`
- `tmrvc-train/src/tmrvc_train/losses.py`

### Done criteria

- Training step computes event losses.
- Loss values are logged and non-NaN.

### Validation

```bash
uv run pytest tests/python -k "breath_event_head or losses or tts_trainer"
```

---

## WP4 - SSL + BPEH Integration (B3)

### Task

- Joint training path for SSL and BPEH.
- Add config toggles for B0/B1/B2/B3.

### Expected file changes

- `configs/research/*.yaml`
- `tmrvc-train/src/tmrvc_train/cli/train_tts.py`
- trainer integration code

### Done criteria

- B0/B1/B2/B3 all train from CLI.
- Metrics emitted with variant tags.

### Validation

```bash
uv run tmrvc-train-tts --config configs/research/b3.yaml --cache-dir data/cache --device cuda --max-steps 1000
```

---

## WP5 - LCD in Distillation (B4)

### Task

- Inject latency condition `q` into distillation path.
- Implement `L_latency` and monotonicity loss `L_mono`.

### Expected file changes

- `tmrvc-train/src/tmrvc_train/distillation.py`
- `tmrvc-train/src/tmrvc_train/models/converter.py`
- `tmrvc-train/src/tmrvc_train/losses.py`
- research configs

### Done criteria

- Distillation supports sampled `q`.
- Quality/latency monotonicity check runs.

### Validation

```bash
uv run tmrvc-distill --config configs/research/b4.yaml --cache-dir data/cache --teacher-ckpt <ckpt> --phase B --device cuda
uv run pytest tests/python -k "distillation or converter"
```

---

## WP6 - Runtime Session State (Serve)

### Task

- Add per-session scene state cache in websocket path.
- Keep `hint` optional and soft-bias only.

### Expected file changes

- `tmrvc-serve/src/tmrvc_serve/app.py`
- `tmrvc-serve/src/tmrvc_serve/schemas.py` (if needed)
- serve tests

### Done criteria

- Multi-turn speak requests preserve scene state.
- Reset behavior works on explicit scene reset.

### Validation

```bash
uv run pytest tests/python -k "serve or ws"
```

---

## WP7 - Ablation + Statistics Pipeline

### Task

- Implement automated B0/B1/B2/B3/B4 batch eval.
- Add bootstrap CI and paired significance tests.

### Expected file changes

- `scripts/eval_research_ablation.py` (new)
- `scripts/stats_research.py` (new)
- `eval/research/` output structure definition

### Done criteria

- One command generates complete ablation table and stats.

### Validation

```bash
uv run python scripts/eval_research_ablation.py --variants b0 b1 b2 b3 b4 \
  --checkpoint b0=checkpoints/b0_teacher.pt \
  --checkpoint b1=checkpoints/b1_teacher.pt \
  --checkpoint b2=checkpoints/b2_teacher.pt \
  --checkpoint b3=checkpoints/b3_teacher.pt \
  --checkpoint b4=checkpoints/b4_teacher.pt \
  --device cuda --output-dir eval/research
uv run python scripts/stats_research.py --input eval/research
```

---

## WP8 - Paper Package Freeze

### Task

- Freeze tables/figures/configs/seeds.
- Produce final markdown summary for manuscript import.

### Expected file changes

- `docs/research/research-results-summary.md` (new)
- `eval/research/final/*`

### Done criteria

- All claim-linked metrics are present with CI and p-values.
- Repro command list is complete.

## 4. Evaluation Gates

Gate G1 (after WP4):

- B3 beats B0 on event metrics and turn coherence.

Gate G2 (after WP5):

- B4 satisfies latency control and monotonicity pass threshold.

Gate G3 (after WP7):

- Statistical significance for main claim comparisons.

Gate G4 (after WP8):

- Full reproducibility bundle assembled.

## 5. Claude Code Prompt Templates

Use these prompts per WP.

### Template A: Implementation WP

```text
You are working in C:\\lib\\github\\kjranyone\\TMRVC.
Implement WP<id> from docs/plans/research-development-plan-claude-code.md.

Hard constraints:
- No backward compatibility work.
- Remove/replace legacy paths if they conflict with clarity.
- Update docs and tests in the same run.

Output requirements:
1) list of modified files
2) exact commands run
3) test results summary
4) unresolved risks
```

### Template B: Evaluation WP

```text
You are working in C:\\lib\\github\\kjranyone\\TMRVC.
Execute WP7 (ablation + statistics) from docs/plans/research-development-plan-claude-code.md.

Produce:
- raw metric files
- aggregated table (CSV/Markdown)
- bootstrap CI
- paired significance tests for B0 vs B4
```

## 6. Immediate Next Command (Recommended)

Start with WP0:

```text
Implement WP0 (Baseline Freeze) from docs/plans/research-development-plan-claude-code.md.
Create deterministic baseline evaluation scripts and frozen config manifest.
```
