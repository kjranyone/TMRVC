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
- `tmrvc-train/src/tmrvc_train/cli/train_codec.py` (codec trainer; v2-era module retained for codec model training/validation only, not part of the v3 UCLM pointer-TTS mainline)
- `tmrvc-train/src/tmrvc_train/pipeline.py`
- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/external-baseline-registry.md`
- `docs/design/provider-registry.md`


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
- explicit runtime-class claim stratification so Python serve, Rust, VST, and ONNX are not evaluated against claims they do not ship in v3.0


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
   - Python text frontend vs Rust text frontend exact parity on frozen golden text fixtures:
     - identical normalized text routing
     - identical `phoneme_ids`
     - identical `text_suprasegmentals`
     - identical fallback / reject metadata
5. Add evaluation protocol document:
   - TTS naturalness
   - turn-taking tempo/context sensitivity
   - VC similarity and intelligibility
   - **Acting Alignment Score (Correlation with scene/dialogue intent)**
   - **Few-shot speaker similarity and disentanglement (Timbre vs Prosody) against the fixed pinned baseline artifact**
6. Define acceptance thresholds for merging v3 mainline.
   - separate:
     - shared contract / parity gates
     - Python-serve drama-grade gates
     - Rust/VST real-time gates
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
12. Add ReferenceEncoder validation:
   - verify speaker-agnosticism: same-text, different-speaker inputs must produce prosody latents with cosine similarity above threshold (confirming speaker identity is not encoded)
   - verify prosody discriminability: same-speaker, different-prosody inputs must produce latents with cosine distance above threshold
   - verify output shape `[B, d_prosody]` and gradient flow to Prosody Predictor during joint training
13. Add explicit regression checks for known open issues:
   - batch vs frame-by-frame CausalConv1d numerical drift
   - `tmrvc-train-codec` `collate_fn` contract verification (codec model data loader; this is a codec-level regression check, not a v3 UCLM pointer-TTS contract)
   - frame-index parity of `bootstrap_alignment.json` against `tmrvc-core`
   - `text_suprasegmentals.npy` index parity against `phoneme_ids.npy`
   - Python G2P vs Rust G2P golden-fixture parity for `phoneme_ids` and `text_suprasegmentals`
14. Freeze external baseline policy:
   - freeze one `primary` public baseline and optionally one `secondary` baseline before large-scale Stage B training
   - baseline name, exact artifact or checkpoint, tokenizer or text-normalization settings, prompt rule, reference lengths, language set, inference settings, and hardware class must be fixed in the evaluation spec
   - baseline changes require an explicit version bump to the evaluation protocol; "or newer successor" is forbidden
   - no release-signoff run may start while the active registry entry is still placeholder or partially unspecified
   - recommended public candidates to evaluate before freezing the registry (must match `docs/design/external-baseline-registry.md`):
     - **CosyVoice 3 (0.5B)**: Frozen as the `primary` baseline due to scale alignment (0.5B vs TMRVC's ~300M target), streaming support, and strong multilingual zero-shot capability.
     - **Qwen3-TTS (1.7B)**: Frozen as the `secondary` ceiling baseline. SOTA comparison against a 1.7B model with a 300M single-stage AR model carries immense risk. Scale-sensitive quality gaps must be handled via explicit narrow claims rather than considered a test failure.
     - **F5-TTS** (arXiv:2410.06885): flow-matching, non-AR, strong naturalness
     - **MaskGCT**: fully non-AR, no alignment needed, strong long-prompt performance
     - any additional candidate recorded and justified in `plan/arxiv_survey_2026_03.md`
   - the primary baseline must be reproducible with public artifacts; proprietary-only systems are not acceptable as the primary sign-off target
15. Freeze human-evaluation statistics policy:
   - pre-register sample count, rater count, duplicate-rate QC, and hypothesis test
   - report confidence intervals in addition to `p` values
   - name the exact statistical test used for merge sign-off
16. Freeze the threshold schedule as a 3-tier policy rather than one monolithic late-stage freeze:
   - `Tier 0` (must freeze before Stage B):
     - runtime budgets
     - parity tolerances
     - frame/alignment conventions
     - prompt-target evaluation pairings
     - language set and code-switch pairs
     - provider registry entries and hardware classes
   - `Tier 1` (freeze after pilot runs but before large-scale claim-making):
     - few-shot score bundle thresholds
     - context-separation thresholds
     - control-response thresholds
     - leakage/disentanglement thresholds
   - `Tier 2` (protocol frozen early, cutoff finalized from pilot distribution):
     - MOS/preference acceptance cutoffs
     - rater QC cutoffs
     - duplicate-consistency cutoffs
17. Freeze the multilingual evaluation language set before Stage B:
   - the frozen external-comparison set must match the active baseline registry exactly until that registry is version-bumped
   - for `v1_2026_03_08`, the frozen external-comparison set is:
     - English, Mandarin Chinese, Japanese
     - Korean, Spanish, French, German
     - Italian, Russian
   - Vietnamese/Thai or other lower-resource stress languages may run as supplementary internal architecture suites, but they must not be mixed into the frozen external sign-off protocol unless the baseline registry and evaluation set are bumped together
   - language additions after freeze require an explicit protocol version bump
   - code-switch evaluation pairs must be pre-registered (e.g., EN-JA, EN-ZH, ZH-JA)
18. Freeze the v3.1 acoustic-refinement trigger diagnostic:
   - define how to attribute a quality gap to "fine-grained acoustic detail" versus prosody/alignment:
     - token-level alignment and prosody metrics are within acceptable range, AND
     - waveform artifact rate or spectral detail metrics are measurably worse than the baseline, OR
     - ablating the vocoder/codec-decoder with a higher-fidelity alternative closes a significant portion of the MOS gap
   - this diagnostic must be frozen before Stage B
19. Freeze prompt-target evaluation pairing:
   - the exact prompt-target pairings used in holdout few-shot evaluation must be frozen at Stage A
   - these pairings must be exported by Worker 10 and consumed by the evaluation arena (Worker 12) without runtime re-sampling
   - the same frozen pairings must be used for external-baseline comparison to ensure reproducibility


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

Runtime-class claim policy:

- Python serve is the only v3.0 runtime eligible for CFG-enhanced drama-grade acting claims.
- Rust / VST / strict ONNX real-time paths must pass shared pointer/control parity and latency gates, but they are evaluated as control-faithful real-time runtimes rather than full-CFG drama runtimes.
- no report may silently transfer a Python-only drama result onto Rust/VST without a dedicated validated fast-CFG path.

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

Scale-sensitivity classification (must be pre-registered before Stage B):

- **scale-sensitive axes** (quality may be fundamentally limited by model/data scale; narrowed claims are acceptable if gap is attributable to scale):
  - raw MOS on unseen languages
  - zero-shot speaker similarity at very short references (3 s)
- **architecture-sensitive axes** (TMRVC's architectural advantages should show regardless of scale; no scale excuse is accepted):
  - controllability (8-D voice_state, pacing controls)
  - dialogue-context responsiveness
  - streaming latency
  - prompt-prosody disentanglement leakage
  - runtime memory ceiling


## Minimum Quantitative Thresholds

Threshold freeze follows the tier policy above:

- Tier 0 thresholds and protocol constants must be frozen before Stage B.
- Tier 1 thresholds may be estimated from pilot runs, but must be frozen before large-scale claim-making or release-candidate training.
- Tier 2 human-evaluation procedures must be frozen before pilot collection, and their final numeric cutoffs must be frozen before release sign-off.

Undefined thresholds are not acceptable once their tier freeze point is reached.

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
- text frontend parity:
  - for the frozen golden corpus, Python and Rust must match exactly on:
    - normalized text
    - `phoneme_ids`
    - `text_suprasegmentals`
    - fallback / reject classification
  - any mismatch blocks the Rust/VST TTS claim because it invalidates the train/serve contract
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
  - prompt selection must come from the canonical exported eligibility contract rather than ad hoc runtime sampling
  - evaluation protocol must state whether same-file, same-conversation, or cross-lingual prompts are allowed
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
  - the context encoder and prosody/style extractor used by this metric must be frozen, versioned, and external to the evaluated checkpoint's mutable training state
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

Metric hygiene rules:

- metrics must not depend on unfrozen internal embeddings from the exact checkpoint under evaluation unless the metric explicitly tests that internal space
- when an external or auxiliary encoder is used for evaluation, its artifact/version must be pinned in the protocol
- if a metric extractor changes, historical scores must be version-separated rather than compared directly

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
- `tmrvc-train-codec` data-loader validation report (codec regression only)
- fixed external-baseline evaluation report with exact model/version/settings
