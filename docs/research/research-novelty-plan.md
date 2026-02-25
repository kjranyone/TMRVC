# Research Novelty Plan (Dialogue Acting + Real-Time TTS)

Last updated: 2026-02-25
Owner: TMRVC research track
Status: active

## 1. Objective

Create publication-grade novelty beyond "good engineering integration".
The target is a single coherent contribution:

- Dialogue-only acting without mandatory hint/stage text.
- Long breath and pause behavior is explicitly modeled, not treated as noise.
- Real-time controllability is trained, not only tuned at runtime.

## 2. Primary Claims

Claim C1:
Given only dialogue turns, the model can preserve turn-to-turn acting consistency.

Claim C2:
Breath and pause events are explicitly generated and improve expressive realism.

Claim C3:
A single student model can follow latency budget control `q` with monotonic quality/latency behavior.

These three claims form one algorithmic contribution set.

## 3. Proposed Algorithm

### 3.1 Scene State Latent (SSL)

Maintain a latent acting state across turns:

`z_t = F_state(z_{t-1}, u_t, h_t, s)`

Where:

- `u_t`: current utterance encoding.
- `h_t`: dialogue history summary.
- `s`: speaker/character embedding.

Minimal implementation:

- `F_state`: GRU or gated MLP with residual connection.
- State reset at scene boundary only.
- Optional stage direction becomes a soft additive bias, not a required condition.

Training losses:

- `L_state_recon`: reconstruct state-relevant prosody statistics.
- `L_state_cons`: force adjacent turns to be smooth unless text indicates large emotional shift.

### 3.2 Breath-Pause Event Head (BPEH)

Predict explicit event sequence per utterance:

`E_t = {onset_i, dur_i, amp_i, pause_i}`

Output heads:

- Breath onset logits (frame-level).
- Breath duration (ms).
- Breath intensity (0..1).
- Pause duration (ms).

Training losses:

- `L_onset`: BCE/Focal for breath onsets.
- `L_dur`: L1 for breath and pause duration.
- `L_amp`: L1 for intensity.
- `L_evt_f1`: differentiable surrogate for event F1 (optional).

The generated event stream is injected into F0/content/style controls before vocoder.

### 3.3 Latency-Conditioned Distillation (LCD)

Train student with explicit latency budget condition `q in [0, 1]`.

Inference path:

`y = G(content, style, z_t, E_t, q)`

Losses:

- `L_distill(q)`: teacher-student distillation under sampled `q`.
- `L_latency`: penalty when estimated runtime exceeds `budget(q)`.
- `L_mono`: enforce quality monotonicity:
  for `q_low < q_high`, quality proxy should satisfy
  `Q(y(q_high)) >= Q(y(q_low)) + margin`.

Quality proxy `Q` can be a weighted objective bundle from STFT/SV/ASR proxy metrics.

## 4. Full Training Objective

`L_total =`
`  lambda_main * (L_duration + L_f0 + L_content + L_voiced)`
`+ lambda_state * (L_state_recon + L_state_cons)`
`+ lambda_event * (L_onset + L_dur + L_amp + L_evt_f1)`
`+ lambda_dist * L_distill`
`+ lambda_rt * (L_latency + L_mono)`

No backward compatibility constraints are applied before release.

## 5. Data and Labels

### 5.1 Required label additions

Add event labels per utterance:

- breath onset timestamps
- breath duration
- breath intensity
- pause intent/duration

Recommended artifact:

`data/cache/<dataset>/<speaker>/<utt>/events.json`

Example:

```json
{
  "events": [
    {"type": "breath", "start_ms": 340, "dur_ms": 420, "intensity": 0.71},
    {"type": "pause", "start_ms": 980, "dur_ms": 180}
  ]
}
```

### 5.2 Language coverage

Primary: ja/en.
Target extension: zh/ko with the same event schema and evaluation protocol.

## 6. Experimental Matrix

Define strict variants:

- B0: current baseline (no SSL, no BPEH, no LCD).
- B1: B0 + SSL.
- B2: B0 + BPEH.
- B3: B0 + SSL + BPEH.
- B4: B3 + LCD (full proposal).

Required ablations:

- remove history input from SSL.
- remove event intensity (onset/duration only).
- remove `L_mono`.
- fixed `q` vs sampled `q`.

## 7. Evaluation Protocol

### 7.1 Objective metrics

- Breath event F1 (onset tolerance in ms).
- Pause timing MAE.
- Turn coherence score (adjacent-turn style embedding consistency).
- ASR WER/CER.
- SECS.
- Latency p50/p95 and overrun rate.
- Quality-vs-latency monotonicity pass rate.

### 7.2 Subjective metrics

- MOS naturalness.
- MOS acting coherence across multi-turn scripts.
- A/B preference against baseline for dialogue-only scripts.

### 7.3 Statistics

- Fixed test script set and speaker split.
- 95% CI by bootstrap.
- Paired significance test for B0 vs B4.

## 8. Implementation Plan (Code-Level)

### 8.1 New modules

- `tmrvc-train/src/tmrvc_train/models/scene_state.py`
- `tmrvc-train/src/tmrvc_train/models/breath_event_head.py`

### 8.2 Core modifications

- `tmrvc-train/src/tmrvc_train/tts_trainer.py`
  - add SSL/BPEH forward and losses.
- `tmrvc-train/src/tmrvc_train/distillation.py`
  - add latency-conditioned sampling and monotonic loss.
- `tmrvc-train/src/tmrvc_train/models/converter.py`
  - accept extra conditioning channels from SSL/BPEH and `q`.
- `tmrvc-core/configs/constants.yaml`
  - add dimensions and lambda defaults for SSL/BPEH/LCD.

### 8.3 Serve/runtime integration

- `tmrvc-serve/src/tmrvc_serve/app.py`
  - maintain per-session scene state cache in websocket flow.
  - keep `hint` optional and soft-bias only.

## 9. Milestones and Gates

M1: Baseline freeze and reproducibility lock.

- exact checkpoint and test list frozen.
- deterministic eval run script created.

M2: SSL + BPEH integration complete.

- B3 trains stably.
- objective improvements on event and coherence metrics.

M3: LCD integration complete.

- B4 meets latency budget control.
- monotonicity pass rate reaches target threshold.

M4: Paper package.

- tables, ablation, scripts, and seed configs finalized.

## 10. Publication Readiness Criteria

Ready only when all are true:

- B4 statistically beats B0 on acting coherence and breath metrics.
- B4 keeps intelligibility and speaker similarity within acceptable bounds.
- Real-time behavior remains stable under target deployment constraints.
- Results reproduce across at least two random seeds.

