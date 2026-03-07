# UCLM v3 Evaluation Protocol

## 1. Automated Metrics

### 1.1 Training Quality Gates

| Metric | Threshold | Source |
|--------|-----------|--------|
| Text supervision coverage | > 80% per dataset | `supervision_report()` |
| Pointer-state sanity | advance_prob in (0,1) | `test_uclm_v3_pointer.py` |
| Checkpoint schema validity | All pointer_head keys present | `strict=False` load test |
| Token range validity | `0 <= A_t < 1024`, `0 <= B_t < 64` | Quality gate |

### 1.2 Latency

| Metric | Target | Tool |
|--------|--------|------|
| Per-step inference (CPU) | < 10 ms / frame | `scripts/benchmark_latency.py` |
| Steady-state VRAM (GPU) | < 2 GB | Manual measurement |
| RTF (real-time factor) | < 0.5 on GPU | Engine metrics `rtf` |

### 1.3 Pointer Diagnostics

| Metric | Description |
|--------|-------------|
| Completion ratio | Fraction of phonemes consumed before max_frames |
| Average frames per phoneme | Should correlate with reference speech rate |
| Advance probability histogram | Should be bimodal (hold vs advance) |

## 2. Integration Matrix

| Config | Train | Serve | Status |
|--------|-------|-------|--------|
| v3 pointer, no durations | `--tts-mode pointer` | `tts_mode="pointer"` | Primary |
| v3 pointer, with durations | `--tts-mode pointer` (uses dur as targets) | `tts_mode="pointer"` | Supported |
| v2 legacy | `--tts-mode legacy_duration` | `tts_mode="legacy_duration"` | Legacy |

## 3. TTS Quality Evaluation

### 3.1 Naturalness (MOS)

- 20+ sentences, 3+ listeners
- Compare: v3 pointer vs v2 legacy vs ground truth
- Report: mean MOS with 95% CI

### 3.2 Intelligibility

- CER/WER on ASR re-transcription of synthesized audio
- Target: CER < 5% (Japanese), WER < 10% (English)

### 3.3 Speaker Similarity

- Cosine similarity between synthesized and reference speaker embeddings
- Target: > 0.85

## 4. VC Quality Evaluation

- Speaker similarity: cosine > 0.80
- Intelligibility: CER degradation < 2% vs source
- Latency: RTF < 1.0 for streaming

## 5. Pacing Control Responsiveness

For each control parameter (`pace`, `hold_bias`, `boundary_bias`):

1. Generate same text with control = {-1.0, -0.5, 0.0, 0.5, 1.0}
2. Measure output duration
3. Verify monotonic relationship (higher pace = shorter output)
4. Report Spearman correlation

## 6. Drama-Acting Evaluation (Future)

### 6.1 Context Sensitivity

- Same text, different dialogue context
- Measure: F0 variance, duration variance, pause placement difference
- Pass: statistically significant difference (p < 0.05)

### 6.2 Human Preference

- Paired comparison: dramatic vs neutral delivery
- 10+ dialogue excerpts, 5+ raters
- Report: preference rate with binomial CI

### 6.3 Collapse Detection

- Same text, 5 different contexts
- Measure: pairwise cosine distance of mel spectrograms
- Fail: all pairs < 0.1 distance (collapsed)

## 7. Pseudo-Annotation Audit Checklist

- [ ] ASR spot-check: 50 random utterances, manual CER < 10%
- [ ] Text normalization: no systematic errors in numbers/dates/abbreviations
- [ ] Speaker clustering: purity > 0.9 on sampled clusters
- [ ] Confidence calibration: low-confidence items are genuinely worse
- [ ] Quality score threshold: selected threshold rejects < 20% of data

## 8. Few-Shot Speaker Adaptation Evaluation

### 8.1 Speaker Similarity Under Short References

- `few_shot_speaker_score`: combine speaker-similarity (cosine of speaker
  embeddings) and intelligibility (CER) under fixed short-reference conditions.
- Reference audio lengths: **3 s**, **5 s**, **10 s**.
- For each reference length, synthesise the same set of evaluation prompts and
  compute:
  - Cosine similarity between synthesised and reference speaker embeddings.
  - CER via ASR re-transcription of synthesised audio.
- Pass criteria:
  - Speaker similarity >= 0.80 at 3 s reference, >= 0.85 at 10 s reference.
  - CER degradation < 3% compared to text-prompted (no speaker reference).

### 8.2 Timbre-Prosody Disentanglement

- `timbre_prosody_disentanglement_score`: for the same speaker-prompt, vary the
  dialogue context and measure F0 and duration variance across contexts.
- High variance indicates good disentanglement (the model adapts prosody to
  context while preserving speaker timbre).
- Extract F0 with CREPE or WORLD; compute coefficient of variation (CV) across
  contexts.
- Pass: F0 CV >= 0.08 across contexts with the same speaker prompt.

### 8.3 Protocol

- Fixed reference audio lengths (3 s, 5 s, 10 s) trimmed from the same source
  recording.
- Matched evaluation prompts across all conditions.
- Comparison against external baseline (see Section 9) on the same protocol.

## 9. External Baseline Comparison Protocol

### 9.1 Frozen Baseline

- Frozen baseline system: **Qwen3-TTS** (or stronger public successor at time
  of evaluation).
- Both TMRVC and the baseline must be evaluated on the identical prompt set,
  reference audio set, and inference configuration.

### 9.2 Metrics

- `external_baseline_delta`: directional gap between TMRVC and the baseline on
  each metric (speaker similarity, CER, MOS, preference rate).
- Positive delta = TMRVC is better; negative delta = TMRVC is worse.

### 9.3 Fixed Evaluation Settings

All of the following must be recorded and held constant across systems:

- Checkpoint version / model identifier.
- Reference audio prompt length.
- Inference parameters (temperature, cfg_scale, top-k, etc.).
- Decoding strategy (greedy / sampling / beam).

### 9.4 Mandatory Blind A/B Preference Test

- Minimum 30 raters, randomised presentation order.
- Forced-choice preference with optional "no preference".
- Both **naturalness** and **controllability/acting** must be evaluated as
  separate dimensions.

### 9.5 Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| Naturalness | Overall speech quality and human-likeness |
| Controllability / Acting | Ability to express different emotions, pacing, and dramatic delivery via dialogue context and voice_state controls |

## 10. CFG and Guidance Stability

### 10.1 Controllability Under Guidance

- `cfg_stability_score`: measure the controllability gain under classifier-free
  guidance (CFG) sweeps, penalised by pointer/EOS failures.
- Sweep cfg_scale over the range [1.0, 1.5, 2.0, 2.5, 3.0].

### 10.2 Safe Bounds

- Identify cfg_scale safe bounds: the range of cfg_scale values where pointer
  monotonicity is maintained and EOS is reached within max_frames.

### 10.3 Pointer Monotonicity Check Under Guidance

- For each cfg_scale in the sweep, verify that the pointer index is
  non-decreasing across frames.
- Any monotonicity violation is a hard failure for that cfg_scale value.

### 10.4 Cache Consistency Check Under Guidance

- Compare KV-cache state at pointer-boundary transitions across different
  cfg_scale values.
- Verify that cache entries remain valid and do not produce divergent outputs
  when recomputed without caching.

## 11. Cache and Waveform Quality

### 11.1 Cache Synchronisation

- `cache_sync_error_rate`: compare cached vs recomputed outputs across
  pointer-boundary transitions.
- Run 100+ utterances; flag any output mismatch exceeding a tolerance of 1e-5
  in logit space.
- Target: < 0.1% mismatch rate.

### 11.2 Waveform Artifact Detection

- `waveform_artifact_score`: combine automatic artifact detectors (energy
  spikes, spectral discontinuities, click detection) with human artifact tags.
- Automatic pass: < 2% of frames flagged by energy-spike detector (threshold:
  > 6 dB jump between adjacent frames).
- Human audit: spot-check 50 utterances; < 5% contain audible artifacts.

### 11.3 Code-Switch Intelligibility

- `code_switch_intelligibility_score`: evaluate intelligibility and
  language-boundary preservation on mixed-language utterances (e.g.
  Japanese-English code-switching).
- Measure CER independently for each language segment.
- Verify that language-boundary transitions do not produce audible glitches or
  intelligibility drops.
- Pass: per-segment CER within 2% of monolingual baseline.

## 12. Pointer Fallback Evaluation

| Counter | Description | Warning Threshold |
|---------|-------------|-------------------|
| `forced_advance_count` | 評価実行中の forced advance 発動回数 | 全 advance の 5% 超で警告 |
| `skip_protection_count` | 評価実行中の skip protection 発動回数 | 全 hold の 10% 超で警告 |

- 各評価実行ごとに `forced_advance_count` と `skip_protection_count` を報告すること
- "clean" 評価の基準: 両カウンタがゼロであること

## 13. Rater QC Protocol

| Parameter | Value | Description |
|-----------|-------|-------------|
| Duplicate sample ratio | 10-15% of total | 評価セットに挿入する重複サンプルの割合 |
| Self-consistency metric | Cohen's kappa or agreement rate | rater の自己一貫性の測定指標 |
| Consistency threshold | >= 0.7 | この閾値を下回る rater はフラグする |

- 重複サンプル (全体の 10-15%) を評価セットに挿入し、rater の自己一貫性を測定する
- Cohen's kappa または一致率で rater 自己一貫性を計測する
- 一貫性が 0.7 を下回る rater をフラグする
- director の定性的ノートとブラインド rater スコアは分離して管理する
