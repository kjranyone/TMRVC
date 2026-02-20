# Acoustic Condition Pathway

## Overview

The original IR Pathway (24 dim) estimates room impulse-response parameters
(RT60, DRR, Spectral Tilt) per 8 subbands. This is sufficient for environment
normalization, but does not capture **voice source** characteristics —
breathiness, vocal tension, micro-perturbations, etc.

This document describes the generalization of the IR Pathway into an
**Acoustic Condition Pathway** that outputs a unified 32-dim conditioning
vector: 24 dim environmental + 8 dim voice source.

## Motivation

Expressive voice styles (e.g., soft/breathy, tense/bright) are largely
determined by voice source parameters that cannot be represented by room
acoustics alone. Adding these parameters to the conditioning vector allows
the Converter to adapt its spectral shaping based on both the acoustic
environment **and** the source voice quality.

## Parameter Layout

| Index | Name | Range | Activation | Description |
|-------|------|-------|------------|-------------|
| 0-7 | RT60 (8 subbands) | [0.05, 3.0] | sigmoid×2.95+0.05 | Reverberation time |
| 8-15 | DRR (8 subbands) | [-10, 30] | sigmoid×40-10 | Direct-to-reverberant ratio |
| 16-23 | Tilt (8 subbands) | [-6, 6] | tanh×6 | Spectral tilt |
| 24-25 | breathiness_low/high | [0, 1] | sigmoid | H1-H2 (aspiration), 2 subbands |
| 26-27 | tension_low/high | [-1, 1] | tanh | Vocal fold tension, 2 subbands |
| 28 | jitter | [0, 0.1] | sigmoid×0.1 | F0 micro-perturbation rate |
| 29 | shimmer | [0, 0.1] | sigmoid×0.1 | Amplitude micro-perturbation |
| 30 | formant_shift | [-1, 1] | tanh | Vocal tract length ratio |
| 31 | roughness | [0, 1] | sigmoid | Subharmonic / roughness |

### Design Rationale

- **2-subband breathiness/tension**: low (< 3kHz) and high (>= 3kHz) capture
  frequency-dependent source characteristics without excessive dimensionality.
- **jitter/shimmer**: capped at 0.1 to reflect physiological ranges.
- **formant_shift**: -1 (lengthened tract, lower formants) to +1 (shortened,
  higher formants) — enables voice feminization/masculinization.
- **roughness**: captures pressed/creaky voice and subharmonics.

## Constants

```yaml
n_voice_source_params: 8    # New voice source parameters (indices 24-31)
n_acoustic_params: 32       # = n_ir_params (24) + n_voice_source_params (8)
```

The original `n_ir_params: 24` is retained for backward compatibility and
to identify the environmental sub-vector.

## Model Changes

### IREstimator → AcousticEstimator

The IREstimator MLP head is extended from 24 to 32 outputs. The range
constraint logic applies existing activations to indices 0-23 and new
activations to indices 24-31.

### Converter

The FiLM conditioning dimension changes from `d_speaker + n_ir_params` (216)
to `d_speaker + n_acoustic_params` (224). This affects all four variants:
ConverterStudent, ConverterStudentGTM, ConverterStudentHQ.

### Teacher U-Net

`film_ir` is renamed to `film_acoustic` with input dimension 32.

## Training Strategy

### Environmental Parameters (0-23)

Training is unchanged: zero-room target for distillation, IR augmentation
in Phase 2.

### Voice Source Parameters (24-31)

Training uses **three** complementary objectives (2026-02 更新):

1. **External Distillation (Phase 2, 新規追加):**
   事前学習済みの外部 voice source 推定器から知識蒸留する。
   これにより、voice source params が物理的に意味のある値を持つことを保証する。
   
   ```
   # 外部推定器 (凍結、学習済み)
   # 例: NKF-stack based breathiness/tension 推定、またはカスタム CNN
   external_voice_estimator = VoiceSourceEstimator.from_pretrained("voice_src_v1")
   
   # 学習時
   with torch.no_grad():
       voice_gt = external_voice_estimator(audio)  # [B, 8]
   voice_pred = acoustic_estimator(audio)[:, 24:32]
   loss_voice_distill = MSE(voice_pred, voice_gt)
   ```
   
   **外部推定器の選択肢:**
   | 推定器 | パラメータ | 学習データ | 推奨 |
   |---|---|---|---|
   | NKF-stack based | ~2M | VCTK + 自作アノテーション | 検討中 |
   | Custom CNN (self-supervised) | ~1M | Reconstruction loss | 実装必要 |
   | Rule-based (openSMILE) | — | — | ベースライン |
   
   **蒸留の利点:**
   - 明示的な教師信号 → 学習が安定
   - 物理的に解釈可能なパラメータ値
   - プリセットブレンド時の整合性が向上

2. **Regularization (Phase A):** L2 loss toward zero (`lambda_ir` weight),
   preventing the estimator from outputting arbitrary large values before
   the converter learns to use them:
   ```
   loss_voice_reg = MSE(voice_source_pred, zeros)
   ```

3. **Implicit supervision via converter gradient:** The voice source
   parameters feed into the converter's FiLM conditioning. As the converter
   is trained end-to-end (flow matching loss + STFT loss), gradients
   propagate back through the FiLM layer into the estimator, encouraging
   voice source parameters to encode information that improves conversion
   quality.

4. **Phase progression (更新):**
   - Phase A: `loss_total = loss_flow + lambda_ir * loss_voice_reg`
     — voice source regularized toward zero, converter learns basic mapping.
   - Phase B/B2: Voice source params participate in end-to-end training
     via converter forward pass; external distillation loss added:
     `loss_total += lambda_voice * loss_voice_distill`
   - Phase C: Metric optimization で voice source も最適化

The `lambda_ir` weight (default 0.1) applies to regularization.
The `lambda_voice` weight (default 0.2) applies to external distillation.

## LoRA Impact

- `_FILM_D_IN`: 216 → 224
- `_FILM_LAYER_PARAM_SIZE`: 224×4 + 4×768 = 896 + 3072 = 3968
- 4 layers × 3968 = **15872** = `lora_delta_size`

## ONNX Contract

- `ir_estimator.onnx` output: `"acoustic_params"` shape `[1, 32]`
- `converter.onnx` / `converter_hq.onnx` input: `"acoustic_params"` shape `[1, 32]`
- Backward-incompatible: old ONNX models are not loadable (dimension mismatch).

## Rust Engine

- `TensorPool`: `OFF_ACOUSTIC_PARAMS` replaces `OFF_IR_PARAMS`, size 24 → 32
- `OrtBundle`: I/O tensor names updated, shapes use `N_ACOUSTIC_PARAMS`
- `StreamingEngine`: `ir_params_cached` → `acoustic_params_cached: [f32; N_ACOUSTIC_PARAMS]`

## Voice Source Presets

### Overview

Voice source presets enable "萌え寄せ" — blending the IR estimator's predicted
voice source params with a pre-computed target profile at inference time.

### Preset Computation (Training Time)

`VoiceSourceStatsTracker` (in `tmrvc_train/voice_source_stats.py`) accumulates
per-speaker running means of voice source params (indices 24-31) during
distillation. After training:

```python
compute_group_preset(stats_path="stats.json", patterns=["moe/*"])
# → {"preset": [8 floats], "matched_speakers": [...], "n_speakers": int}
```

The resulting preset can be embedded in `.tmrvc_speaker` metadata:
```json
{"voice_source_preset": [0.5, 0.6, 0.1, -0.2, 0.01, 0.02, 0.3, 0.7]}
```

### Blending (Inference Time)

The blend formula is:

```
blended[24+i] = lerp(estimated[24+i], preset[i], alpha)
```

- `alpha = 0`: estimated values (no change, backward-compatible)
- `alpha = 1`: full preset application
- Alpha is user-controllable via GUI slider ("Voice preset" 0-100%)

### Implementation

| Component | File | Mechanism |
|---|---|---|
| Stats tracking | `voice_source_stats.py` | `VoiceSourceStatsTracker.update()` in `DistillationTrainer` |
| Preset storage | `speaker_file.py` | `voice_source_preset` field in JSON metadata |
| Python blend | `audio_engine.py` | `_blend_voice_source()` creates local copy before converter |
| Rust blend | `processor.rs` | Stack copy of `acoustic_params_cached`, 8 lerps, zero allocation |
| GUI | `realtime_demo.py` | "Voice preset" alpha slider (0-100%) |

### RT-Safety (Rust Engine)

The blend is RT-safe: `[f32; 32]` stack copy + 8 multiply-add operations.
No heap allocation. `acoustic_params_cached` is not modified, preventing
double-blend on subsequent frames.
