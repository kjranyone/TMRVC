# TMRVC Evaluation Set Specification

This document freezes the contents of the public release evaluation set used by
UCLM v3 sign-off.

It must remain consistent with:

- `docs/design/evaluation-protocol.md`
- `docs/design/external-baseline-registry.md`
- `docs/design/acceptance-thresholds.md`
- `plan/worker_06_validation.md`

---

## 1. Active Frozen Set

- `evaluation_set_version`: `tmrvc_eval_public_v1_2026_03_08`
- freeze date: `2026-03-08`
- storage root: `eval/sets/tmrvc_eval_public_v1_2026_03_08/`

Required files:

- `manifest.jsonl`
- `subset_index.json`
- `reference_audio/`
- `reference_text/`
- `target_text/`
- `rating_forms/`
- `README.md`

The set is append-forbidden after freeze.
If any item is replaced, removed, or relabeled, the version must change.

---

## 2. Language Scope

### 2.1 Release Sign-Off Languages

The release sign-off set uses the shared language intersection supported by the
active pinned baselines:

- `zh`
- `en`
- `ja`
- `ko`
- `de`
- `fr`
- `ru`
- `es`
- `it`

This is the authoritative language list for
`tmrvc_eval_public_v1_2026_03_08`.

### 2.2 Excluded From This Frozen Set

- `pt`

`pt` is excluded from this frozen release set because it is not in the shared
intersection of the active `primary` and `secondary` baseline entries.
If Portuguese becomes a declared release language, freeze a new evaluation-set
version instead of mutating this one.

---

## 3. Frozen Subsets

The set is divided into frozen subsets. Counts below are exact.

### 3.1 `read_core`

Purpose:

- naturalness
- intelligibility
- multilingual read-speech quality

Composition:

- `3` prompts per language
- `9` languages
- total base items: `27`

Rules:

- neutral declarative or short explanatory text
- no dialogue context required
- no target text overlap with any few-shot reference transcript used for the
  same speaker

### 3.2 `dialogue_context_pairs`

Purpose:

- same-text different-context evaluation
- dramatic appropriateness
- context sensitivity

Composition:

- `2` text/context pairs per language
- each pair contains the same target line under `2` distinct dialogue contexts
- `9` languages
- total pairs: `18`
- total render targets: `36`

Rules:

- the target line text must be identical within a pair
- contexts must differ in intent, tension, or conversational function
- contexts must be text-side only
- pairs must be balanced so neither context is systematically longer

### 3.3 `control_sweeps`

Purpose:

- monotonic control validation
- exposed runtime control responsiveness

Composition:

- `3` base prompts per language:
  - one for `pace`
  - one for `hold_bias`
  - one for `boundary_bias`
- `9` languages
- total base prompts: `27`

Frozen sweep levels:

- `pace`: `0.85`, `0.95`, `1.00`, `1.05`, `1.15`
- `hold_bias`: `-0.5`, `-0.25`, `0.0`, `0.25`, `0.5`
- `boundary_bias`: `-0.5`, `-0.25`, `0.0`, `0.25`, `0.5`

Total automated render count:

- `27 * 5 = 135`

### 3.4 `few_shot_same_language`

Purpose:

- speaker similarity
- intelligibility under short references
- prompt leakage resistance

Composition:

- `2` speakers per language
- `2` target texts per speaker
- `3` reference lengths per speaker: `3 s`, `5 s`, `10 s`
- `9` languages
- total trials: `2 * 2 * 3 * 9 = 108`

Rules:

- reference utterance and target text must come from different utterances
- target text must not appear verbatim in the reference transcript
- the same speaker identity is used across all three reference lengths

### 3.5 `few_shot_leakage_pairs`

Purpose:

- timbre/prosody disentanglement
- reference-prosody leakage audit

Composition:

- `1` speaker per language selected from `few_shot_same_language`
- `2` target texts per selected speaker
- `3` reference lengths
- total trials: `1 * 2 * 3 * 9 = 54`

Rules:

- reference clip should be neutral or context-mismatched relative to the target
- target is rendered under a dialogue context different from the reference scene
- this subset is scored primarily by leakage and disentanglement metrics, not
  by human preference alone

### 3.6 `code_switch_probe`

Purpose:

- code-switch intelligibility regression

Composition:

- `12` prompts total
- fixed language-pair mix:
  - `ja-en`: `3`
  - `zh-en`: `3`
  - `de-en`: `2`
  - `es-en`: `2`
  - `fr-en`: `2`

Rules:

- each prompt includes at least one switch boundary
- segment boundaries are annotated in `manifest.jsonl`

---

## 4. Few-Shot Prompt Construction

These rules are mandatory for all systems, including TMRVC and pinned external
baselines.

### 4.1 Reference Audio Selection

- single speaker only
- no overlapping speech
- no clipping or obvious corruption
- matched transcript available
- same speaker as the target speaker
- selected from a different utterance than the target text

### 4.2 Reference Trimming

Reference lengths are frozen at:

- `3 s`
- `5 s`
- `10 s`

Trimming rule:

- choose the highest-SNR voiced span satisfying the target duration
- prefer one continuous voiced segment
- if multiple spans qualify, choose the earliest qualifying span
- if no qualifying span exists, exclude the speaker from the set rather than
  inventing a fallback

### 4.3 Reference Text Contract

- the matched reference transcript is required for systems that use it
- the transcript must be verbatim at the lexical level
- punctuation may be normalized, but wording must not be paraphrased
- target text must not appear verbatim in the reference transcript

### 4.4 Cross-System Equality

- the same reference audio file is used for every compared system
- the same reference transcript is used for every compared system
- the same target text and target context are used for every compared system
- no system may receive hidden prompt engineering beyond the frozen
  baseline-specific `prompt_rule` in `docs/design/external-baseline-registry.md`

---

## 5. Human Evaluation Construction

### 5.1 Pairwise A/B Arms

Release sign-off uses three pairwise arms:

- `ab_primary_quality`: `54` items
- `ab_secondary_streaming`: `36` items
- `ab_v2_regression`: `27` items

#### `ab_primary_quality`

Composition:

- `18` items from `read_core`
- `18` items from `dialogue_context_pairs`
- `18` items from `few_shot_same_language`

Use:

- broad public quality claims
- naturalness
- dramatic appropriateness
- few-shot similarity

#### `ab_secondary_streaming`

Composition:

- `18` items from `dialogue_context_pairs`
- `18` items from `few_shot_same_language`

Use:

- streaming-capable external comparison
- causal path competitiveness

#### `ab_v2_regression`

Composition:

- `9` items from `read_core`
- `9` items from `dialogue_context_pairs`
- `9` items from `few_shot_same_language`

Use:

- internal regression guard only

### 5.2 MOS Arm

- `mos_primary`: `36` items total
- composition:
  - `12` from `read_core`
  - `12` from `dialogue_context_pairs`
  - `12` from `few_shot_same_language`

### 5.3 Rating Coverage

- minimum unique raters: `30`
- minimum valid ratings per pairwise item: `6`
- minimum valid ratings per MOS item: `8`
- duplicate-item ratio: `12.5%`

### 5.4 Assignment Policy

The assignment engine must:

- balance languages across raters
- balance systems across presentation order
- prevent the same rater from seeing the same content in more than one arm
- keep raters blinded to model identity, provenance, and internal notes

Target per-rater workload:

- `32` to `40` total judgments, including duplicates

---

## 6. Manifest Contract

Each `manifest.jsonl` row must include:

- `item_id`
- `subset`
- `language_id`
- `speaker_id` when applicable
- `target_text`
- `target_text_norm`
- `dialogue_context_id` when applicable
- `dialogue_context_text`
- `reference_audio_id` when applicable
- `reference_text`
- `reference_length_sec` when applicable
- `code_switch_spans` when applicable
- `human_eval_eligible`
- `automated_eval_eligible`
- `notes`

For paired dialogue items, add:

- `pair_id`
- `context_variant_id`

For few-shot items, add:

- `reference_session_id`
- `target_session_id`
- `cross_utterance_verified`

---

## 7. Exclusion Rules

Exclude an item from the frozen set if any of the following hold:

- transcript uncertainty remains unresolved
- overlapping or mixed speech remains
- speaker identity cannot be verified
- reference clip cannot satisfy the exact `3 s`, `5 s`, or `10 s` trimming rule
- target text appears in the reference transcript
- dialogue context is missing for a context-sensitive item

An excluded item must be replaced only by creating a new set version.

---

## 8. Reporting Requirements

Every release-signoff report that cites
`tmrvc_eval_public_v1_2026_03_08` must report at minimum:

- language-level breakdown
- subset-level breakdown
- pairwise A/B arm breakdown
- few-shot reference-length breakdown
- duplicate-item QC outcome
- item exclusions, if any

---

## 9. Change Policy

Any change to:

- language scope
- item count
- prompt construction
- reference trimming
- A/B arm composition
- MOS arm composition
- rating coverage minimums

requires a new `evaluation_set_version`.
