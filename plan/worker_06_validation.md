# Worker 06: Validation, Metrics, and Integration

## Scope

Define and implement the test and evaluation matrix that decides whether v3 is acceptable.


## Primary Files

- `tests/train/`
- `tests/serve/`
- `tests/data/`
- `tests/scripts/`
- `tmrvc-engine-rs/tests/`
- `tmrvc-export/src/tmrvc_export/export_uclm.py`
- `tmrvc-train/src/tmrvc_train/cli/train_codec.py`
- `tmrvc-train/src/tmrvc_train/pipeline.py`
- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/external-baseline-registry.md`


## Required Outcomes

- passing automated tests for v3 contracts
- latency checks for causal runtime
- numerical parity checks across Python / Rust / ONNX runtime paths
- quality protocol for TTS and VC comparison
- explicit pass/fail criteria
- **Explicit proof protocol for drama-grade acting (Acting Alignment Score)**
- **Objective diversity and controllability metrics**
- **Explicit proof protocol for few-shot adaptation against a fixed external baseline artifact**
- explicit validation protocol for pseudo-annotated corpora
- baseline artifact/version/settings and statistical test procedure are frozen before sign-off
- explicit validation of `voice_state` pseudo-label usefulness, calibration, and masking semantics
- explicit validation of suprasegmental text-feature integrity and usefulness
- explicit blocking policy for every declared public claim (`MFA-free`, `10 ms causal`, `few-shot`, `drama-grade`, `SOTA-competitive`)
- frozen primary and secondary external baselines before large-scale training
- frozen hardware classes for all runtime and latency claims


## Concrete Tasks

1. Add integration matrix:
   - v2 legacy train
   - v3 pointer train
   - v2 legacy serve
   - v3 pointer serve
   - Rust streaming runtime
   - ONNX export parity
   - UI control plane (Worker 12)
2. Add quality gate updates:
   - text supervision coverage
   - pointer-state sanity
   - checkpoint schema validity
   - **Human Sign-off:** mandatory preference/MOS audit via UI
   - **External Parity:** mandatory blind A/B preference test against the fixed pinned baseline artifact
3. Add latency benchmarks:
   - per-step runtime
   - steady-state streaming memory
4. Add runtime/export parity benchmarks:
   - PyTorch batch vs PyTorch streaming
   - PyTorch vs ONNX
   - Python serve vs Rust runtime on the same checkpoint
   - suprasegmental text-feature parity across Python / export / Rust paths
5. Add evaluation protocol document:
   - TTS naturalness
   - turn-taking tempo/context sensitivity
   - VC similarity and intelligibility
   - **Acting Alignment Score (Correlation with scene/dialogue intent)**
   - **Few-shot speaker similarity and disentanglement (Timbre vs Prosody) against the fixed pinned baseline artifact**
6. Define acceptance thresholds for merging v3 mainline.
7. Add drama-acting evaluation protocol:
   - same text, different context responsiveness
   - same context, different control input responsiveness
   - **CFG Scale Responsiveness (measure acting intensity vs guidance scale)**
   - explicit 8-D `voice_state` responsiveness
   - pause / hesitation / release realism
   - human preference against v2 legacy baseline
   - **Human preference against the fixed pinned external baseline**
8. Add disentanglement leakage checks:
   - `prosody_transfer_leakage_score` comparing prompt-side pitch/duration contours against generated contours when text/context differ
   - fail if generated prosody tracks reference prosody too strongly under cross-context prompting
9. Add pseudo-annotation validation protocol:
   - ASR spot-check accuracy
   - text normalization audit
   - speaker clustering purity spot-check
   - pseudo-label confidence calibration
   - quality-score threshold selection
   - `voice_state` pseudo-label calibration and coverage audit
10. Add separation-aided annotation validation:
   - compare raw vs separated ASR quality
   - compare diarization purity before/after separation
   - compare artifact rate on sampled clips
   - reject separation front-end if annotation benefit is smaller than waveform damage
11. Add multilingual regression protocol:
   - after adding or reweighting a language, re-run frozen held-out tests for existing languages
   - block promotion if existing-language intelligibility or speaker similarity regresses beyond threshold
12. Add explicit regression checks for known open issues:
   - batch vs frame-by-frame CausalConv1d numerical drift
   - `tmrvc-train-codec` `collate_fn` contract verification
   - frame-index parity of `bootstrap_alignment.json` against `tmrvc-core`
   - `text_suprasegmentals.npy` index parity against `phoneme_ids.npy`
13. Freeze external baseline policy:
   - freeze one `primary` public baseline and optionally one `secondary` baseline before large-scale Stage B training
   - baseline name, exact artifact or checkpoint, tokenizer or text-normalization settings, prompt rule, reference lengths, language set, inference settings, and hardware class must be fixed in the evaluation spec
   - baseline changes require an explicit version bump to the evaluation protocol; "or newer successor" is forbidden
   - no release-signoff run may start while the active registry entry is still placeholder or partially unspecified
   - recommended public candidates to evaluate before freezing the registry (must match `docs/design/external-baseline-registry.md`):
     - **CosyVoice 2/3** (arXiv:2412.10117, arXiv:2505.17589): streaming, LLM + CFM, strong zero-shot, multilingual
     - **F5-TTS** (arXiv:2410.06885): flow-matching, non-AR, strong naturalness
     - **MaskGCT**: fully non-AR, no alignment needed, strong long-prompt performance
     - **Qwen3-TTS** (arXiv:2601.15621): 2-stage AR + flow matching, block-wise streaming, multilingual, open-source weights available
     - any additional candidate recorded and justified in `plan/arxiv_survey_2026_03.md`
   - the primary baseline must be reproducible with public artifacts; proprietary-only systems are not acceptable as the primary sign-off target
14. Freeze human-evaluation statistics policy:
   - pre-register sample count, rater count, duplicate-rate QC, and hypothesis test
   - report confidence intervals in addition to `p` values
   - name the exact statistical test used for merge sign-off


## Acceptance Criteria

Release sign-off for v3 mainline requires all of the following:

- v3 TTS training runs without MFA artifacts in the mainline path
- streaming TTS remains causal under the frozen runtime budget on the frozen hardware class
- Python / Rust / ONNX parity gates pass within predefined tolerance
- no regression in VC stability on the frozen smoke corpus
- measurable context sensitivity on a held-out dialogue set
- measurable controllability for `pace`, `hold_bias`, and `boundary_bias`
- measurable controllability for explicit 8-D `voice_state`
- few-shot speaker adaptation passes the frozen protocol with fixed prompt lengths and prompt-budget limits
- human evaluation against `v2 legacy` passes only as an internal regression guard, not as evidence for SOTA claims
- direct evaluation against the pinned external baseline is mandatory for any public quality claim

SOTA-competitive claim policy:

- if TMRVC claims overall SOTA competitiveness, it must match or exceed the pinned `primary` baseline on all declared primary claim axes
- if TMRVC is below the pinned `primary` baseline on any declared primary claim axis, the broad SOTA claim is blocked
- an explicit narrowed claim is allowed only if the weaker axes are named and excluded from the public claim

Primary claim axes must include at minimum:

- zero-shot / few-shot speaker similarity
- intelligibility
- dialogue-context responsiveness
- controllability
- streaming latency
- runtime memory ceiling
- prompt-prosody disentanglement leakage


## Minimum Quantitative Thresholds

Before sign-off, Worker 06 must replace all placeholders below with exact numeric thresholds and frozen hardware identifiers.
Undefined thresholds are not acceptable for release.

- causal runtime:
  - p50, p95, and p99 per-step latency must satisfy the frozen 10 ms streaming budget on the frozen hardware class
  - no full-text recomputation per 10 ms step in the steady-state path
  - batch vs frame-by-frame runtime parity must stay within the predefined numerical tolerance
  - if `full` CFG exceeds the frozen real-time budget, real-time CFG claim is blocked
  - if prompt conditioning exceeds the frozen prompt-budget limits, real-time few-shot claim is blocked
- context sensitivity:
  - same-text different-context evaluation must show separable delivery metrics on held-out dialogue subsets
  - use at least one frozen metric from Worker 02, such as:
    - `context_separation_score`
    - `prosody_collapse_score`
- control responsiveness:
  - `pace`, `hold_bias`, and `boundary_bias` must each produce measurable monotonic movement in at least one registered runtime metric
  - explicit 8-D `voice_state` perturbations must each produce measurable directional movement in at least one registered runtime or acoustic metric
  - suggested runtime metrics:
    - output duration
    - pause count or pause duration
    - boundary hold frequency
- parity:
  - PyTorch vs ONNX and Python vs Rust paths must remain within the predefined tolerance on pointer outputs and state transitions
- suprasegmental parity:
  - exported `text_suprasegmentals` must survive roundtrip without length drift, reordering, or schema-version ambiguity
- frame-alignment parity:
  - all exported `start_frame` / `end_frame` artifacts must match `sample_rate=24000`, `hop_length=240`, `T = ceil(N / 240)` exactly
- `voice_state` supervision:
  - reported coverage, observed ratio, and confidence summaries must exist for every promoted training subset
  - masked training with partial `voice_state` supervision must outperform or equal the no-physical-target ablation on registered controllability metrics
- force-advance parity:
  - Python and Rust must agree on forced-advance trigger timing and post-trigger state updates within the predefined tolerance
- few-shot speaker adaptation:
  - evaluate with fixed short reference durations and matched prompts
  - enforce the same prompt-budget limits used by serving/runtime during evaluation
  - TMRVC must match or exceed the pinned `primary` baseline on the frozen few-shot score bundle, unless the public claim is explicitly narrowed
  - "competitive with trade-offs" is not sufficient for an unqualified SOTA claim
- human evaluation:
  - pairwise A/B against v2 legacy
  - target threshold:
    - win rate `>= 0.55`, or
    - `p < 0.05` under the predefined test procedure
- pseudo-annotation audit:
  - must satisfy the bucket thresholds defined in `plan/ai_curation_system.md`
- separation adoption:
  - allowed only when annotation uplift is larger than measured artifact cost on the validation sample
- multilingual regression:
  - adding a new language or code-switch mapping must not push existing held-out languages below the documented regression threshold


## Metric Definition Requirements

Worker 06 must freeze explicit formulas or scripts for the following before final sign-off:

- `context_separation_score`
  - initial recommendation:
    - for same-text items across different contexts, compute pairwise distance in prosody/style embedding space
    - normalize by within-context variance
- `prosody_collapse_score`
  - initial recommendation:
    - `between_context_variance / total_variance` on same-text grouped samples
- `control_response_score`
  - initial recommendation:
    - monotonic correlation between requested control sweep and measured runtime/audio metric
- `voice_state_response_score`
  - initial recommendation:
    - monotonic or directionally consistent response between explicit physical-control sweeps and measured acoustic/runtime metrics
- `few_shot_speaker_score`
  - initial recommendation:
    - combine speaker-similarity and intelligibility metrics under fixed short-reference conditions
- **`acting_alignment_score`**
  - measure the semantic correlation between dialogue context embeddings and generated acoustic features (prosody embedding).
- **`cfg_responsiveness_score`**
  - measure the change in acting intensity (e.g., F0 variance) relative to `cfg_scale` sweeps.
- **`timbre_prosody_disentanglement_score`**
  - measure the variance of prosody metrics (F0, duration) for the same speaker-prompt under different dialogue-contexts.
  - target: High prosody variance despite identical timbre-prompt.
- `prosody_transfer_leakage_score`
  - measure prompt-prosody leakage by correlating reference-prompt pitch/duration contours with generated contours when target text/context differ.
  - target: low leakage under cross-context prompting while retaining speaker similarity.
- `voice_state_label_utility_score`
  - measure the controllability uplift attributable to curated `voice_state` supervision versus the ablation without those targets.
- `voice_state_calibration_error`
  - compare pseudo-label confidence against downstream residual error or audit buckets.
- `suprasegmental_integrity_score`
  - measure how often exported accent / tone / phrase-boundary features remain aligned and non-empty for languages that declare support
- `external_baseline_delta`
  - initial recommendation:
    - report the directional gap between TMRVC and the fixed public baseline on the same protocol

The exact metric implementation may mature, but sign-off cannot rely on undefined names.


## Guardrails

- do not use only unit tests; include at least one end-to-end smoke per mode
- do not accept quality claims without reproducible scripts or checklists
- **do not claim SOTA status based only on improvement over v2 legacy; external baseline comparison is mandatory.**
- do not leave the external baseline movable at sign-off time; the artifact/version must be pinned before the run starts
- do not allow placeholder baseline entries, placeholder hardware classes, or placeholder thresholds to survive into Stage B large-scale runs
- do not sign off if v3 still requires hidden legacy duration artifacts
- do not sign off if Python, Rust, or ONNX paths disagree on the runtime contract
- do not sign off if expressive claims are based only on anecdotal samples
- do not sign off on pseudo-annotations without manual spot-audit
- do not sign off if `voice_state` controls exist only in the UI/runtime but lack validated training supervision quality
- do not adopt a separator because demos sound impressive; require measured uplift
- do not sign off on SOTA-style claims using only internal baselines
- do not use undefined human-evaluation wording in the final acceptance report; threshold and test procedure must be named
- do not sign off on Japanese/tonal naturalness claims if suprasegmental features were dropped or never parity-tested
- do not accept "competitive", "close", or "promising" as sign-off language for public SOTA claims
- do not allow a broad SOTA claim if TMRVC wins only on one axis while losing on other declared primary axes


## Handoff Contract

- final report names remaining blockers
- acceptance criteria are explicit
- worker outputs can be merged in dependency order without guesswork


## Deliverables

- updated automated tests
- benchmark commands
- evaluation checklist
- final integration report template
- reproducible expressive-evaluation script or score sheet
- external-baseline registry entry used for release sign-off
- pseudo-annotation audit checklist
- separation-front-end comparison checklist
- parity test report for Python / Rust / ONNX paths
- `tmrvc-train-codec` data-loader validation report
- fixed external-baseline evaluation report with exact model/version/settings
