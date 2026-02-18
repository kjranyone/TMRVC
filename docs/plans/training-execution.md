# Stream 3: Training Execution

## Goal

実際にモデルを学習する。Phase 0 (検証) → Phase 1 (Base Teacher) → Phase 2 (IR-robust) → 蒸留。

詳細な学習パラメータは `docs/design/training-plan.md` を参照。
ここでは実行手順と準備チェックリストに集中する。

## Prerequisites

### Data Preparation

| Dataset | Size | Download | Status |
|---|---|---|---|
| VCTK | ~11GB | https://datashare.ed.ac.uk/handle/10283/3443 | Not downloaded |
| JVS | ~3GB | https://sites.google.com/site/shinaborumiethlab/page3/jvs_corpus | Not downloaded |
| LibriTTS-R | ~65GB | https://www.openslr.org/141/ | Not downloaded (Phase 1) |
| AIR Database | ~200MB | OpenSLR #20 | Not downloaded (Phase 2) |
| BUT ReverbDB | ~1GB | butria.fit.vutbr.cz | Not downloaded (Phase 2) |

### Data Preprocessing

```bash
# 1. Download datasets to data/ directory
# 2. Run preprocessing pipeline
uv run python -m tmrvc_data.cli.preprocess \
    --input-dir data/VCTK-Corpus-0.92 \
    --output-dir data/processed/vctk \
    --steps resample normalize vad_trim segment features \
    --workers 8

uv run python -m tmrvc_data.cli.preprocess \
    --input-dir data/jvs_corpus \
    --output-dir data/processed/jvs \
    --steps resample normalize vad_trim segment features \
    --workers 8
```

### GPU Environment

```bash
# Cloud GPU setup (Lambda / RunPod)
# 1. Clone repo
git clone <repo-url> && cd TMRVC

# 2. Install dependencies
pip install uv && uv sync

# 3. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Execution Plan

### Phase 0: Architecture Verification (~$20-50, 1-2 days)

```bash
uv run tmrvc-train-teacher \
    --config configs/train_teacher.yaml \
    --phase 0 \
    --data-dir data/processed \
    --datasets vctk jvs \
    --steps 100000 \
    --batch-size 32 \
    --lr 2e-4 \
    --output-dir checkpoints/phase0
```

**Checkpoints:**
- [ ] Loss が単調減少する
- [ ] 生成 mel が妥当な形状
- [ ] SECS > 0.7 (self-reconstruction)

### Phase 1a: Base Flow Matching (~$80-185, 2-4 days)

```bash
uv run tmrvc-train-teacher \
    --config configs/train_teacher.yaml \
    --phase 1a \
    --resume checkpoints/phase0/latest.pt \
    --data-dir data/processed \
    --datasets vctk jvs libritts_r \
    --steps 500000 \
    --batch-size 64 \
    --lr 1e-4 \
    --output-dir checkpoints/phase1a
```

### Phase 1b: + Perceptual Losses (~$55-80, 1-2 days)

```bash
uv run tmrvc-train-teacher \
    --config configs/train_teacher.yaml \
    --phase 1b \
    --resume checkpoints/phase1a/latest.pt \
    --steps 200000 \
    --lr 5e-5 \
    --lambda-stft 0.5 \
    --lambda-spk 0.3 \
    --output-dir checkpoints/phase1b
```

**Checkpoints:**
- [ ] SECS >= 0.88 (10-step sampling)
- [ ] UTMOS >= 3.8

### Phase 2: IR-robust (~$55-80, 2-3 days)

```bash
uv run tmrvc-train-teacher \
    --config configs/train_teacher.yaml \
    --phase 2 \
    --resume checkpoints/phase1b/latest.pt \
    --rir-dir data/rir \
    --steps 200000 \
    --lr 5e-5 \
    --lambda-ir 0.1 \
    --output-dir checkpoints/phase2
```

### Distillation (~$50-100, 2-3 days)

```bash
# Phase A: ODE Trajectory
uv run tmrvc-distill \
    --teacher-checkpoint checkpoints/phase2/best.pt \
    --phase A \
    --steps 200000 \
    --output-dir checkpoints/distill_a

# Phase B: DMD
uv run tmrvc-distill \
    --teacher-checkpoint checkpoints/phase2/best.pt \
    --student-checkpoint checkpoints/distill_a/latest.pt \
    --phase B \
    --steps 100000 \
    --output-dir checkpoints/distill_b
```

### ONNX Export

```bash
uv run tmrvc-export \
    --checkpoint checkpoints/distill_b/best.pt \
    --output-dir models/onnx \
    --quantize
```

## Cost Summary

| Phase | GPU | Hours | Cost (spot) |
|---|---|---|---|
| Phase 0 | 1x A100 | 12-24h | ~$15-30 |
| Phase 1 | 1x A100 | 72-168h | ~$80-185 |
| Phase 2 | 1x A100 | 48-72h | ~$55-80 |
| Distillation | 1x A100 | 48-72h | ~$50-100 |
| **Total** | | | **~$200-400** |

## Notes

- 学習設定の YAML ファイル: `configs/train_teacher.yaml`, `configs/train_student.yaml`, `configs/export.yaml` 作成済み
- CLI (`tmrvc-train-teacher`, `tmrvc-distill`) は `--config` YAML 対応済みだが、実データでの動作未検証
- Accelerate (multi-GPU) 対応は trainer.py に未実装
