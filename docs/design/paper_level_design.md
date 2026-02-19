# Paper-Level Design: Target-Utterance Controlled Few-Shot Real-Time VC

Created: 2026-02-18
Scope: TMRVC (Rust real-time engine + Python training stack)

## 1. Problem Definition

Given:
- source streaming audio `x_{1:T}` (live input)
- additional target audio set `A = {a_i}` (arbitrary user-provided references)
- optional target utterance reference `u` (for intonation/articulation guidance)

Generate output `y_{1:T}` such that:
1. speaker identity matches target speaker from `A`
2. linguistic content follows source `x`
3. prosody/articulation can be steered toward `u`
4. real-time constraints are satisfied under a latency-quality control `q in [0,1]`

This is a constrained conditional generation problem:

`y_t = G(x_{<=t}, c_spk(A), c_utt(u), q, theta)`

where `c_spk` is speaker conditioning, `c_utt` is utterance-style conditioning, and `theta` is the student model.

## 2. Core Contributions

1. Dual conditioning:
- Speaker conditioning `c_spk` for timbre identity
- Utterance conditioning `c_utt` for intonation/articulation control

2. Few-shot adaptation:
- Fast personalization with low-rank adapters (LoRA) using a small `A`

3. Real-time controllability:
- Continuous latency-quality spectrum `q` with explicit runtime budget tracking

4. Distillation-compatible architecture:
- Teacher-to-student training with streaming-safe student inference

## 3. Model Architecture

## 3.1 Encoder/Conditioning Blocks

- `E_content`: causal content encoder from source frames
- `E_spk`: speaker encoder producing `e_spk`
- `E_utt`: utterance-style encoder producing:
  - prosody tokens `Z_prosody`
  - pitch template `p_ref` (F0 contour summary)
  - articulation template `a_ref` (spectral/phonetic clarity proxy)

`E_utt` runs offline or background, never on the RT callback.

## 3.2 Converter

Converter receives:
- content `h_t`
- speaker embedding `e_spk`
- IR/environment params `r_t`
- utterance conditioning memory `Z_prosody`
- streaming state

It outputs acoustic features for vocoder:
- live path: causal converter (low latency)
- quality path: semi-causal converter with lookahead

Both paths share weights where possible; quality path adds right context and larger receptive field.

## 3.3 Vocoder

- lightweight streaming vocoder predicting `mag/phase` or equivalent compact representation
- iSTFT + OLA reconstruction

## 4. Control Variables

Expose orthogonal user controls:
- `alpha_timbre in [0,1]`: target speaker strength
- `beta_prosody in [0,1]`: target utterance intonation/timing strength
- `gamma_articulation in [0,1]`: articulation clarity/style strength
- `q in [0,1]`: latency-quality spectrum

Runtime conditioning:

`c_t = concat(alpha_timbre * e_spk, beta_prosody * z_t, gamma_articulation * a_ref, r_t)`

where `z_t` is attention-pooled from `Z_prosody`.

## 5. Training Objective

Total loss:

`L = lambda_rec L_rec + lambda_stft L_stft + lambda_spk L_spk + lambda_pros L_pros + lambda_art L_art + lambda_dist L_dist + lambda_rt L_rt`

Definitions:
- `L_rec`: frame-level reconstruction (mel/feature L1 or Huber)
- `L_stft`: multi-resolution STFT loss
- `L_spk`: speaker consistency (`1 - cos(E_spk(y), E_spk(target))`)
- `L_pros`: prosody alignment (F0 RMSE + energy + duration alignment)
- `L_art`: articulation consistency (phoneme posterior or ASR-feature alignment)
- `L_dist`: teacher-student distillation (trajectory/feature/logit matching)
- `L_rt`: real-time regularizer penalizing expensive configurations:
  - expected frame compute `tau_hat(q)` beyond hop budget

Prosody mixing target:

`p_target = (1 - beta_prosody) * p_src + beta_prosody * p_ref`

and similarly for articulation templates.

## 6. Few-Shot Adaptation Protocol

Input: additional audio set `A` (seconds to minutes).

Two-stage adaptation:
1. Fast stage (seconds-minutes):
- estimate `e_spk`
- optimize LoRA adapter `Delta_lora` on converter conditioning layers

2. Optional deep stage (offline):
- distill updated teacher behavior into personalized student checkpoint

Output artifacts:
- `*.tmrvc_speaker` (identity + adapter)
- `*.tmrvc_style` (utterance/prosody memory bank)

## 7. Real-Time Inference Algorithm

Per hop:
1. read source hop
2. update streaming features (content/F0/IR)
3. fetch conditioning (`e_spk`, `Z_prosody`, controls)
4. select path by `q` (live/hybrid/quality)
5. run converter + vocoder
6. overlap-add and output
7. update telemetry (`p50/p95`, overruns, applied profile)

Hard constraints:
- no blocking I/O on RT thread
- no allocation spikes in RT path
- model/profile swapping only through lock-free staging

## 8. Evaluation Protocol (Paper-Level)

## 8.1 Objective Metrics

- Speaker: SECS / cosine similarity
- Naturalness: UTMOS, DNSMOS (optional)
- Intelligibility/articulation: WER/CER via frozen ASR
- Prosody: F0 RMSE, F0 correlation, V/UV error, duration deviation
- Spectral: MCD, LSD
- Real-time: RTF, frame p50/p95, overrun rate, end-to-end latency

## 8.2 Subjective Evaluation

- MOS (naturalness)
- CMOS/ABX for target-utterance match
- side-by-side for low-latency vs high-quality settings

## 8.3 Ablations

- remove `E_utt` (no target utterance control)
- disable LoRA (embedding only)
- fixed `q` vs adaptive `q`
- no articulation loss
- no prosody loss

## 9. Gap-to-Implementation Mapping (Current Repo)

Current critical gaps and required implementation:

1. LoRA not applied in Rust inference:
- parse `lora_delta` and apply in converter path (or pre-merge to ONNX)

2. No target-utterance conditioning in Rust RT:
- add style artifact loader (`*.tmrvc_style`)
- add converter inputs for prosody/articulation conditioning

3. No F0-driven control in Rust engine:
- replace placeholder F0 with streaming F0 estimator
- expose `beta_prosody` control

4. Standalone (Rust) lacks adaptation workflow:
- add background job pipeline:
  - ingest reference audio
  - run few-shot finetune
  - write artifacts
  - hot-swap profile

5. Control surface is insufficient:
- extend UI with `alpha_timbre`, `beta_prosody`, `gamma_articulation`, `q`

## 10. Reproducibility Plan

- fix dataset splits and publish manifests
- deterministic seeds for adaptation/eval
- export exact model metadata and artifact schema version
- report confidence intervals and statistical tests (paired tests)

## 11. Minimal Acceptance Criteria

A design is accepted when:
1. target-utterance control improves prosody/articulation metrics over baseline
2. speaker similarity does not regress beyond predefined tolerance
3. live mode maintains low overrun rate
4. quality mode improves subjective/objective quality
5. control knobs are monotonic and interpretable
