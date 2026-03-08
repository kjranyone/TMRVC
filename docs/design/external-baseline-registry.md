# External Baseline Registry

This file pins the exact external baseline artifacts allowed for release-signoff
comparisons.

Do not use phrases like "or newer successor" in release reports.
If the baseline changes, add a new entry and bump the evaluation protocol
version.

## Freeze Policy

The active `primary` baseline entry must be fully specified before any large-scale Stage B training or any public quality claim.
A placeholder entry is permitted only during pre-freeze drafting and must block release-signoff and SOTA-language usage.

---

## Entry Template

| Field | Description |
|------|-------------|
| `baseline_id` | Stable identifier used in reports |
| `model_name` | Public model / system name |
| `artifact_id` | Exact checkpoint, commit, or release artifact |
| `tokenizer_version` | Tokenizer / normalization version |
| `prompt_rule` | Prompt trimming and text normalization rule |
| `reference_lengths_sec` | Frozen reference lengths used in few-shot evaluation |
| `inference_settings` | Temperature, top-k, cfg, decoding mode, etc. |
| `evaluation_set_version` | Frozen prompt-set identifier |
| `date_frozen` | Date the baseline entry was frozen |
| `notes` | Any caveats needed for reproducibility |

---

## Active Entries

### `baseline_pending_freeze`

| Field | Value |
|------|-------|
| `baseline_id` | `baseline_pending_freeze` |
| `model_name` | `TBD` |
| `artifact_id` | `TBD` |
| `tokenizer_version` | `TBD` |
| `prompt_rule` | `TBD` |
| `reference_lengths_sec` | `3, 5, 10` |
| `inference_settings` | `TBD` |
| `evaluation_set_version` | `TBD` |
| `date_frozen` | `TBD` |
| `notes` | `Replace this placeholder before any release-signoff comparison.` |

---

## Candidate Baselines (to be frozen before Stage D)

The following systems are recommended candidates based on the 2026-03 arxiv survey. At least one must be selected and frozen with full artifact details before any release-signoff comparison begins.

### CosyVoice 2 / CosyVoice 3

- **Paper:** arXiv:2412.10117 (v2), arXiv:2505.17589 (v3)
- **Architecture:** LLM (0.5B-1.5B) + Conditional Flow Matching (DiT)
- **Strengths:** Streaming support, strong zero-shot speaker similarity, multilingual (9 languages + 18 Chinese dialects), x-vector based timbre-prosody separation
- **Reproducibility:** Open-source model weights available
- **Rationale:** Most directly comparable to TMRVC's streaming + zero-shot + expressive goals

### F5-TTS

- **Paper:** arXiv:2410.06885
- **Architecture:** Non-autoregressive flow matching
- **Strengths:** Strong naturalness, open-source, simple architecture
- **Reproducibility:** Model weights and inference code publicly available
- **Rationale:** Represents the non-AR quality ceiling; useful to measure the AR vs non-AR quality gap

### MaskGCT

- **Architecture:** Fully non-autoregressive masked generative codec transformer
- **Strengths:** No explicit alignment or duration prediction needed, strong long-prompt performance
- **Reproducibility:** Open-source
- **Rationale:** Alternative non-AR paradigm; performs better with longer prompts

### Qwen3-TTS

- **Paper:** arXiv:2601.15621
- **Architecture:** 2-stage AR codec LM (Qwen2.5-based) + Conditional Flow Matching, block-wise streaming
- **Strengths:** Multilingual (>30 languages), strong zero-shot similarity, block-wise refinement preserves streaming, large-scale training (1M+ hours)
- **Reproducibility:** Open-source model weights and inference code available via Hugging Face / ModelScope
- **Rationale:** Represents the 2-stage AR + Non-AR SOTA quality ceiling; directly validates whether TMRVC's v3.1 refinement upgrade path is competitive. The block-wise streaming design is architecturally comparable to TMRVC's planned refinement integration.

### Selection criteria

- the chosen baseline must have publicly available model weights or a reproducible inference endpoint
- proprietary-only systems (Seed-TTS, MiniMax-Speech) are acceptable as secondary comparison points but not as the primary pinned baseline
- at least one baseline must support the same reference lengths (3s, 5s, 10s) and language coverage as the TMRVC evaluation set
