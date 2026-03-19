# Track: v4 Training Pipeline Cutover

## Scope

This track owns the `v4` training pipeline: model architecture changes, loss composition, and dataset integration.
This is not a request to keep extending the current `v3` `8-D` training contract.

The critical-path `v4` slice is:

- migrate voice state supervision from `8-D` to `12-D`
- add acting texture latent learning path
- implement supervision tier-aware loss weighting
- implement biological constraint regularization
- integrate `IntentCompilerOutputV4` in training
- update loss composition for all `v4` loss terms
- update dataset loading for `V4BootstrapCacheEntry`

## Primary Files

- `dev.py` (v4 用に書き換え — v3 互換は不要)
- `tmrvc-train/src/tmrvc_train/trainer.py`
- `tmrvc-train/src/tmrvc_train/models/uclm_model.py`
- `tmrvc-train/src/tmrvc_train/models/voice_state_encoder.py`
- `tmrvc-train/src/tmrvc_train/models/uclm_transformer.py`
- `tmrvc-train/src/tmrvc_train/dataset/uclm_dataset.py`
- `configs/train_uclm.yaml.example`

## Open Tasks

### 1. Migrate voice state training from `8-D` to `12-D`

`VoiceStateEncoder` must consume `VoiceStateSupervisionV4` (`12-D`).

Required changes:

- dataset must load `12-D` physical control targets
- loss must handle `12-D` supervision
- encoder input dimension must match the frozen physical control registry

### 2. Add acting texture latent learning

Required components:

- new acting latent encoder/decoder path
- acting latent regularization loss (prevent collapse)
- residual usage penalty to ensure latent captures only what physical cannot
- latent dimension: `24-D` (`D_ACTING_LATENT` from constants; master plan range was `16-D` to `32-D`, `24-D` is the frozen `v4.0` value)

Rules:

- latent must not degenerate to zero usage
- latent must not duplicate what the physical path already covers

### 3. Implement supervision tier-aware loss weighting

Required behavior:

- `SupervisionTier` (`A`/`B`/`C`/`D`) classification from `V4BootstrapCacheEntry`
- tier-based loss weighting: Tier A full weight, Tier D auxiliary only
- low-confidence pseudo-labels must be masked or downweighted
- unknown dimensions must NOT be treated as dense zero

### 4. Implement biological constraint regularization

Required constraints:

- low-rank covariance prior (`BIO_COVARIANCE_RANK=8`)
- intent-conditioned parameter prior
- frame-to-frame transition prior (`BIO_TRANSITION_PENALTY_WEIGHT=0.1`)
- physically implausible combination penalty

Rules:

- no future-frame smoothing (causal only)
- constraints must produce non-zero gradients

### 5. Integrate `IntentCompilerOutputV4` in training

Required behavior:

- training must consume compiled physical targets + acting latent prior
- teacher forcing from ground-truth physical trajectories

### 6. Update loss composition for `v4`

Based on master plan section 8.6, the following loss terms must be active:

- codec token prediction loss (8 codebooks × 2048 vocab at 12.5 Hz)
- control token prediction loss
- pointer progression loss
- explicit physical supervision loss (`12-D`)
- acting latent regularization loss
- disentanglement loss
- speaker consistency loss
- prosody prediction loss
- semantic alignment loss

### 7. Update dataset loading for `V4BootstrapCacheEntry`

Required behavior:

- load all fields from `v4` train-ready cache
- support supervision tier filtering
- load acting semantic annotations

### 8. Rewrite `dev.py` for v4

`dev.py` (1080 lines) is the current v3 development menu CLI. It must be rewritten for `v4` in-place.
v3 backward compatibility is not required (master plan §3.1). v3 dev.py is recoverable from git history.

Required v4 menu structure:

- bootstrap: raw corpus → train-ready cache (delegates to `tmrvc_data.cli.bootstrap`)
- training: v4 supervised training with 12-D physical + 24-D latent + enriched transcripts
- RL fine-tuning: instruction-following RL phase (post-supervised)
- dataset management: v4 corpus listing, supervision tier summary, cache regeneration
- curation: v4 curation pipeline (ingest → score → export → validate)
- finalize: checkpoint promotion with v4 quality gates
- character management: few-shot enrollment via backend API
- serve: v4 inference server startup
- integrity check: v4 contract validation across core/train/export/serve/rust

Required changes from v3:

- replace `8-D voice_state` config validation with `12-D` physical + `24-D` latent
- replace v3 dataset management with v4 bootstrap cache management
- add supervision tier summary display
- add enriched transcript preview
- add RL phase menu entry
- remove legacy MFA path entirely
- update `REQUIRED_TRAINING_FIELDS` for v4 loss composition

### 9. Add enriched transcript training path

> Depends on: `track_data_bootstrap` §5 (enriched transcript generation with inline acting tags)

The text encoder must learn to condition on inline acting tags from enriched transcripts.

Required changes:

- extend text encoder vocabulary to include acting tag tokens from the frozen tag vocabulary (defined in track_architecture §8)
- during training, randomly alternate between:
  - enriched transcript (with inline tags) as text input
  - plain transcript (without tags) as text input
  - this prevents the model from becoming dependent on tags and maintains plain-text TTS capability
- inline acting tags must contribute to the codec token prediction loss
- the model must learn correlations between inline tags and physical control targets:
  - `[angry]` → high tension, high energy
  - `[whisper]` → high breathiness, low energy
  - these correlations emerge from data, not hardcoded rules

### 10. Add RL fine-tuning phase for instruction following

> Depends on: `track_data_bootstrap` §5 (enriched transcripts), §10 (DSP/SSL extractors), §11 (rich-transcription ASR for reward)

After supervised pre-training, add a reinforcement learning phase inspired by Fish Audio S2.

RL reward composition:

1. **Instruction-following reward**: generate audio, re-transcribe with rich-transcription ASR, compare inline tags in output vs input
   - `[angry]` in input but not detected in output → penalty
   - detected vocal event not requested → minor penalty
2. **Physical control compliance reward**: measure physical features of generated audio, compare to explicit physical targets
   - pitch_level target 0.8 but measured 0.3 → penalty
   - this is unique to v4 — Fish S2 cannot enforce physical-level compliance
3. **Intelligibility reward**: plain transcript WER/CER between input and re-transcribed output
4. **Naturalness guard**: penalize degenerate outputs (silence, noise, repetition)

Implementation:

- use the same rich-transcription ASR from bootstrap (Qwen3-ASR or successor) as the reward model
- use DSP/SSL extractors from bootstrap §10 for physical compliance measurement
- RL algorithm: PPO or REINFORCE with baseline, applied to the UCLM codec token policy
- RL phase runs AFTER supervised training converges, not concurrently

Rules:

- RL must not degrade plain-text (no-tag) TTS quality beyond the threshold in `docs/design/acceptance-thresholds.md` §V4-8
- RL must not collapse physical control editability below the threshold in `docs/design/acceptance-thresholds.md` §V4-8
- RL reward weights must be tunable and logged

### 11. Adapt UCLM for dual-rate operation with Mimi codec

The UCLM transformer must handle the dual-rate time axis:

Required changes:

- codec token prediction heads: output vocabulary 2048 (was 1024)
- sequence length: ~12.5 frames/sec (was 100)
- context window at codec rate (200 frames = 16 seconds, was 200 frames = 2 seconds)
- voice state encoder: accepts control-rate input, downsamples to codec rate for transformer
- pointer head: one advance/hold decision per codec frame (80 ms granularity)
- physical supervision loss: computed at control rate, aggregated to codec frames for gradient

Mimi codec weights:

- encoder and decoder are frozen (pre-trained, not fine-tuned)
- only the UCLM transformer, voice state encoder, text encoder, and acting latent modules are trained
- codec encode runs at cache generation time; codec decode runs at inference time

## Required Tests

- `12-D` voice state training convergence test
- acting latent collapse detection test
- supervision tier weighting correctness test
- biological constraint penalty gradient test
- loss composition completeness test
- `V4BootstrapCacheEntry` dataset loading test
- enriched transcript training: model generates different audio for same text with vs without inline tags
- RL instruction-following: model scores higher on tag compliance after RL than before
- RL physical compliance: physical control response monotonicity preserved after RL (> 0.8)
- RL naturalness guard: plain-text TTS quality does not degrade more than 5%

## Out Of Scope

Do not reopen:

- preserving `v3` `8-D` training as mainline
- `v3` checkpoint compatibility
- flow-matching prosody predictor (post-`v4.0` upgrade path)

## Exit Criteria

- `trainer.py` consumes `12-D` `VoiceStateSupervisionV4` and all `v4` loss terms are active
- acting latent path is trained with measurable residual usage (see `docs/design/acceptance-thresholds.md` §V4-2)
- supervision tier weighting is verified (see `docs/design/acceptance-thresholds.md` §V4-3)
- biological constraint regularization meets threshold (see `docs/design/acceptance-thresholds.md` §V4-4)
- all `v4` loss terms produce non-zero gradients in a smoke test
- enriched transcript path produces measurably different outputs for tagged vs untagged input (A/B divergence test)
- RL phase meets all thresholds in `docs/design/acceptance-thresholds.md` §V4-8
