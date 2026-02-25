# Baseline Manifest (B0)

Created: 2026-02-25
Status: frozen

## Overview

B0 is the unmodified baseline for the ablation study. It uses the standard
TMRVC VC pipeline with no research extensions (no SSL, no BPEH, no LCD).

## Model

| Component | Architecture | Params | Notes |
|---|---|---|---|
| TeacherUNet | 4-stage U-Net, cross-attention | 17.2M | v-prediction flow matching |
| ContentEncoder (WavLM) | Pre-trained WavLM Large | - | Frozen feature extractor, 768d |
| SpeakerEncoder (ECAPA) | Pre-trained ECAPA-TDNN | - | Frozen, 192d |

## Conditioning

| Signal | Dim | Source |
|---|---|---|
| content | 768 | WavLM Large (ContentVec) |
| f0 | 1 | CREPE/RMVPE |
| spk_embed | 192 | ECAPA-TDNN |
| acoustic_params | 32 | IR(24) + voice_source(8) |

## Sampling

| Parameter | Value |
|---|---|
| ODE solver | Euler |
| Steps | 32 |
| Sway coefficient | 1.0 |
| CFG scale | 1.0 (no guidance) |

## Test Split

8 speakers (4 VCTK, 4 JVS), 10 utterances each = 80 utterances max.

| Speaker | Dataset | Gender | Notes |
|---|---|---|---|
| vctk_p225 | VCTK | F | Southern English |
| vctk_p226 | VCTK | M | Surrey |
| vctk_p227 | VCTK | M | Cumbria |
| vctk_p228 | VCTK | F | Southern English |
| jvs_jvs001 | JVS | F | Japanese |
| jvs_jvs002 | JVS | M | Japanese |
| jvs_jvs003 | JVS | F | Japanese |
| jvs_jvs004 | JVS | M | Japanese |

## Metrics

| Metric | Description | Direction |
|---|---|---|
| mel_mse | MSE between predicted and GT mel spectrogram | lower is better |
| secs | Speaker Embedding Cosine Similarity | higher is better |
| f0_correlation | Pearson correlation on voiced F0 frames | higher is better |
| utmos_proxy | Mel-distance proxy for MOS (0-5 scale) | higher is better |

## Reproducibility

```bash
# Run 1
uv run python scripts/eval_research_baseline.py \
    --config configs/research/b0.yaml \
    --checkpoint <checkpoint_path> \
    --cache-dir data/cache \
    --seed 42 \
    --device cuda \
    --output-dir eval/research/b0_run1

# Run 2 (must match Run 1)
uv run python scripts/eval_research_baseline.py \
    --config configs/research/b0.yaml \
    --checkpoint <checkpoint_path> \
    --cache-dir data/cache \
    --seed 42 \
    --device cuda \
    --output-dir eval/research/b0_run2

# Verify
diff eval/research/b0_run1/results.json eval/research/b0_run2/results.json
```

## Variants

| Variant | SSL | BPEH | LCD | Config |
|---|---|---|---|---|
| **B0** | - | - | - | `configs/research/b0.yaml` |
| B1 | yes | - | - | `configs/research/b1.yaml` |
| B2 | - | yes | - | `configs/research/b2.yaml` |
| B3 | yes | yes | - | `configs/research/b3.yaml` |
| B4 | yes | yes | yes | `configs/research/b4.yaml` |

## Frozen Files

- `configs/research/b0.yaml` — evaluation config
- `scripts/eval_research_baseline.py` — deterministic evaluation runner
- `tmrvc_train/eval_metrics.py` — metric implementations
- This manifest (`docs/research/baseline-manifest.md`)
