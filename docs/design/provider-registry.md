# Curation Provider Registry

この文書は、mainline curation で使用を許可する provider artifact を固定する正本である。
`docs/design/external-baseline-registry.md` と同じ思想で、provider の再現性を artifact / provider_revision / runtime / parameter 単位で凍結する。

## 1. Freeze Policy

- active provider entry は Stage B 大規模 curation の前に fully specified でなければならない
- `or newer`, `latest`, `main branch`, `default provider_revision` のような曖昧表現は禁止
- provider を変更する場合は registry entry を追加し、protocol / calibration version を bump する
- confidence 値は raw のまま scoring に流してはならず、Worker 11 が監査 subset で calibration version を発行してから Worker 09 policy に入れる

## 2. Required Fields

各 entry は最低限以下を持つ。

- `provider_id`
- `stage`
- `model_name`
- `artifact_id`
- `provider_revision`
- `runtime_backend`
- `package_version`
- `dtype`
- `decode_or_inference_params`
- `supported_languages`
- `output_schema_version`
- `confidence_semantics`
- `calibration_status`
- `calibration_version`
- `fallback_policy`
- `license_access`
- `source_refs`
- `date_frozen`
- `notes`

## 3. Active Entries

### `diarization_pyannote_community1_hf_mainline_2026_03_08`

- `provider_id`: `diarization_pyannote_community1_hf_mainline_2026_03_08`
- `stage`: `speaker_structure_recovery`
- `model_name`: `pyannote/speaker-diarization-community-1`
- `artifact_id`: `hf:pyannote/speaker-diarization-community-1`
- `provider_revision`: `2025-09-11 announcement baseline`
- `runtime_backend`: `pyannote.audio pipeline`
- `package_version`: `pyannote.audio pinned in lockfile / deployment env`
- `dtype`: `float32`
- `decode_or_inference_params`: `exclusive diarization enabled when available; default segmentation/clustering params unless explicitly version-bumped`
- `supported_languages`: `language-agnostic diarization`
- `output_schema_version`: `curation_diarization_v1`
- `confidence_semantics`: `pipeline confidence and cluster-purity proxies require downstream calibration`
- `calibration_status`: `required_before_policy_use`
- `calibration_version`: `pending_worker_11`
- `fallback_policy`: `if unavailable or gated, fall back to secondary diarization provider and mark records with provider downgrade provenance`
- `license_access`: `Hugging Face gated access may be required`
- `source_refs`: `https://huggingface.co/pyannote/speaker-diarization-community-1 ; https://www.pyannote.ai/blog/community-1`
- `date_frozen`: `2026-03-08`
- `notes`: `Preferred primary diarization provider because exclusive diarization simplifies alignment with ASR timestamps and multi-speaker structure recovery`

### `asr_qwen3_asr_1p7b_hf_mainline_2026_03_08`

- `provider_id`: `asr_qwen3_asr_1p7b_hf_mainline_2026_03_08`
- `stage`: `transcript_recovery`
- `model_name`: `Qwen3-ASR-1.7B`
- `artifact_id`: `hf:Qwen/Qwen3-ASR-1.7B`
- `provider_revision`: `registry_freeze_2026_03_08`
- `runtime_backend`: `transformers / official Qwen ASR inference path`
- `package_version`: `transformers + official qwen audio stack pinned in lockfile / deployment env`
- `dtype`: `float16`
- `decode_or_inference_params`: `official long-form inference defaults; no undocumented decoder retuning`
- `supported_languages`: `52-language ASR coverage declared by provider`
- `output_schema_version`: `curation_asr_v1`
- `confidence_semantics`: `token/utterance confidence requires Worker 11 calibration before thresholding`
- `calibration_status`: `required_before_policy_use`
- `calibration_version`: `pending_worker_11`
- `fallback_policy`: `if GPU/runtime unavailable, use throughput fallback provider and mark downgrade provenance`
- `license_access`: `public`
- `source_refs`: `https://huggingface.co/Qwen/Qwen3-ASR-1.7B ; https://github.com/QwenLM/Qwen3-ASR`
- `date_frozen`: `2026-03-08`
- `notes`: `Chosen as primary ASR because it aligns with the Qwen forced-aligner family and reduces cross-provider timestamp drift`

### `align_qwen3_forced_aligner_0p6b_hf_mainline_2026_03_08`

- `provider_id`: `align_qwen3_forced_aligner_0p6b_hf_mainline_2026_03_08`
- `stage`: `bootstrap_alignment_projection`
- `model_name`: `Qwen3-ForcedAligner-0.6B`
- `artifact_id`: `hf:Qwen/Qwen3-ForcedAligner-0.6B`
- `provider_revision`: `registry_freeze_2026_03_08`
- `runtime_backend`: `official Qwen forced aligner path`
- `package_version`: `official qwen audio stack pinned in lockfile / deployment env`
- `dtype`: `float16`
- `decode_or_inference_params`: `official alignment defaults; no heuristic retiming outside the frozen projection recipe`
- `supported_languages`: `11-language forced-alignment coverage declared by provider`
- `output_schema_version`: `bootstrap_alignment_v1`
- `confidence_semantics`: `alignment confidence is transitional supervision only and requires calibration`
- `calibration_status`: `required_before_policy_use`
- `calibration_version`: `pending_worker_11`
- `fallback_policy`: `if target language is unsupported, export no aligner-derived labels and route record to fallback / review policy instead of synthesizing pseudo-truth`
- `license_access`: `public`
- `source_refs`: `https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B ; https://github.com/QwenLM/Qwen3-ASR`
- `date_frozen`: `2026-03-08`
- `notes`: `Primary bootstrap-alignment provider because it shares a family with the chosen ASR provider and reduces timestamp-contract drift`

### `asr_faster_whisper_fallback_mainline_2026_03_08`

- `provider_id`: `asr_faster_whisper_fallback_mainline_2026_03_08`
- `stage`: `transcript_recovery_fallback`
- `model_name`: `faster-whisper`
- `artifact_id`: `github:SYSTRAN/faster-whisper`
- `provider_revision`: `release-pinned-by-lockfile`
- `runtime_backend`: `ctranslate2`
- `package_version`: `release-pinned in lockfile / deployment env`
- `dtype`: `float16` or `int8_float16` as explicitly pinned per deployment profile
- `decode_or_inference_params`: `beam size / language detection / vad filter must be pinned in deployment config`
- `supported_languages`: `inherits Whisper model coverage used at deployment`
- `output_schema_version`: `curation_asr_v1`
- `confidence_semantics`: `Whisper-style log-prob confidence requires separate calibration from Qwen-family scores`
- `calibration_status`: `required_before_policy_use`
- `calibration_version`: `pending_worker_11`
- `fallback_policy`: `used only when primary ASR is unavailable, too slow for backlog-clearing, or explicitly selected for throughput passes`
- `license_access`: `public`
- `source_refs`: `https://github.com/SYSTRAN/faster-whisper/releases`
- `date_frozen`: `2026-03-08`
- `notes`: `Throughput fallback, not the mainline truth source`

## 4. Policy Rules

- unsupported-language cases must not be silently mapped onto a provider that lacks declared support
- when a provider downgrade occurs, `provider_id`, `reason`, and `fallback_class` must be written into provenance
- mixed-provider confidence scores must not share one threshold until Worker 11 demonstrates cross-provider calibration parity or installs provider-specific thresholds
- any score entering `promote/review/reject` must carry both `provider_id` and `calibration_version`

## 5. Non-Goals

- this registry does not freeze score thresholds; Worker 06 owns evaluation thresholds and Worker 09 owns promotion policy thresholds
- this registry does not replace the canonical manifest or export contract; it complements them by pinning inference dependencies
