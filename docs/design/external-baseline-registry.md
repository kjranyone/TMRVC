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
| `language_set` | Frozen language coverage used for comparison |
| `prompt_rule` | Prompt trimming and text normalization rule |
| `reference_lengths_sec` | Frozen reference lengths used in few-shot evaluation |
| `inference_settings` | Temperature, top-k, cfg, decoding mode, etc. |
| `hardware_class` | Frozen hardware class used for runtime and latency claims |
| `task_scope` | Which public claim axes this baseline blocks or informs |
| `evaluation_set_version` | Frozen prompt-set identifier defined in `docs/design/evaluation-set-spec.md` |
| `evaluation_protocol_version` | Version of the evaluation protocol under which this entry was frozen |
| `source_refs` | Official source URLs used to justify the entry |
| `date_frozen` | Date the baseline entry was frozen |
| `notes` | Any caveats needed for reproducibility |

---

## Active Entries

### `primary_fun_cosyvoice3_0p5b_2512_hf_29e01c4`

| Field | Value |
|------|-------|
| `baseline_id` | `primary_fun_cosyvoice3_0p5b_2512_hf_29e01c4` |
| `model_name` | `Fun-CosyVoice3-0.5B-2512` |
| `artifact_id` | `hf:FunAudioLLM/Fun-CosyVoice3-0.5B-2512@29e01c4` |
| `tokenizer_version` | `bundled CosyVoice3 tokenizer in artifact @29e01c4; text normalization path = default WeText; optional ttsfrd disabled for sign-off reproducibility` |
| `language_set` | `Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian` |
| `prompt_rule` | `Zero-shot mode only. Reference audio is trimmed to a voiced 3 s / 5 s / 10 s span with matched verbatim reference transcript. Prefix the reference transcript as 'You are a helpful assistant.<|endofprompt|>{reference_text}' to match the official CosyVoice3 zero-shot example contract. Pass target language/content explicitly. If Japanese is evaluated, transliterate prompts to katakana to match the official usage guidance.` |
| `reference_lengths_sec` | `3, 5, 10` |
| `inference_settings` | `Use the official CosyVoice3 inference_zero_shot path; stream=False for offline quality comparison and stream=True only in the streaming-latency suite; default released checkpoint config; no undocumented RAS/decoder overrides; no ttsfrd dependency in the sign-off path.` |
| `hardware_class` | `single_nvidia_rtx_2080ti_22gb_cuda12_sdpa` |
| `task_scope` | `Primary blocker for broad public quality claims because its 0.5B scale is closer to TMRVC's target regime while still covering streaming, multilingual zero-shot quality, few-shot speaker similarity, intelligibility, and overall naturalness.` |
| `evaluation_set_version` | `tmrvc_eval_public_v1_2026_03_08` |
| `evaluation_protocol_version` | `v1_2026_03_08` |
| `source_refs` | `https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/tree/main ; https://github.com/FunAudioLLM/CosyVoice ; https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/example.py` |
| `date_frozen` | `2026-03-08` |
| `notes` | `Chosen as primary because it is an official open-weight streaming-native baseline with published multilingual and zero-shot claims, direct Hugging Face artifact availability, and model scale closer to TMRVC's planned regime.` |

### `secondary_qwen3_tts_12hz_1p7b_base_hf_fd4b254`

| Field | Value |
|------|-------|
| `baseline_id` | `secondary_qwen3_tts_12hz_1p7b_base_hf_fd4b254` |
| `model_name` | `Qwen3-TTS-12Hz-1.7B-Base` |
| `artifact_id` | `hf:Qwen/Qwen3-TTS-12Hz-1.7B-Base@fd4b254` |
| `tokenizer_version` | `hf:Qwen/Qwen3-TTS-Tokenizer-12Hz@7dd38ad` |
| `language_set` | `Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian` |
| `prompt_rule` | `Voice-clone mode only. Reference audio is trimmed to a voiced 3 s / 5 s / 10 s span with matched verbatim reference transcript. Reusable prompt path uses create_voice_clone_prompt. Pass explicit target language for the main evaluation set; allow language=auto only for subsets explicitly marked auto-language stress.` |
| `reference_lengths_sec` | `3, 5, 10` |
| `inference_settings` | `Use the official qwen-tts package in voice-clone mode; dtype=torch.float16; attn_implementation=sdpa; max_new_tokens=2048; all other sampling parameters from the checkpoint generate_config.json; no manual prompt embellishment or hidden decoding retuning.` |
| `hardware_class` | `single_nvidia_rtx_2080ti_22gb_cuda12_sdpa` |
| `task_scope` | `Secondary ceiling baseline for broad multilingual quality, few-shot speaker similarity, overall naturalness, and high-capacity public-baseline comparison. Deficits on scale-sensitive axes require narrowed claims rather than silent failure relabeling.` |
| `evaluation_set_version` | `tmrvc_eval_public_v1_2026_03_08` |
| `evaluation_protocol_version` | `v1_2026_03_08` |
| `source_refs` | `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/main ; https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/tree/main ; https://github.com/QwenLM/Qwen3-TTS` |
| `date_frozen` | `2026-03-08` |
| `notes` | `Chosen as secondary ceiling baseline because it is an official open-weight model with strong multilingual and short-reference voice-cloning claims, but its 1.7B scale is materially larger than TMRVC's target regime.` |

---

## Other Candidate Baselines

The following systems remain relevant comparison candidates from the 2026-03 arxiv survey, but they are not the active pinned sign-off entries unless promoted by an explicit protocol version bump.

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

### Selection criteria

- the chosen baseline must have publicly available model weights or a reproducible inference endpoint
- proprietary-only systems (Seed-TTS, MiniMax-Speech) are acceptable as secondary comparison points but not as the primary pinned baseline
- at least one baseline must support the same reference lengths (3s, 5s, 10s) and language coverage as the TMRVC evaluation set
