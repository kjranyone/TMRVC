# Track: v4 Data Bootstrap

## Scope

This track owns the raw-audio bootstrap pipeline for `v4`.
It covers everything between "raw unlabeled audio files" and "train-ready cache".

This is the critical-path prerequisite for all other `v4` tracks.
Without it, the repository cannot start from unlabeled corpora.

## Primary Files

- `tmrvc-data/src/tmrvc_data/`
- `docs/design/dataset-preparation-flow.md`
- `docs/design/curation-contract.md`
- `docs/design/v4-master-plan.md` (Section 7)

## Open Tasks

### 1. Freeze the `v4` raw-audio bootstrap contract

Define the canonical pipeline stages from raw corpus to train-ready cache:

1. ingest
2. audio normalization
3. VAD segmentation
4. overlap / music / noise rejection
5. diarization or speaker clustering
6. pseudo speaker assignment
7. speaker embedding extraction
8. Whisper transcription
9. text normalization and G2P
10. DSP / SSL physical feature extraction
11. LLM semantic / acting annotation
12. confidence scoring and artifact masking
13. train-ready cache export

### 2. Freeze the `v4` train-ready cache contract

Minimum contents per utterance:

- `acoustic tokens`
- `control tokens`
- `pseudo speaker_id`
- `speaker_embed`
- `text transcript`
- `enriched_transcript` (transcript with inline acting tags)
- `phoneme_ids`
- `language metadata`
- `physical control targets`
- `acting semantic annotations`
- `quality/confidence metadata`

This contract supersedes the `v3` dataset contract.

Codec tokenization uses **Mimi** (`kyutai/mimi`), frozen pre-trained encoder:
- acoustic tokens: `[8, T_codec]` at 12.5 Hz (T_codec = duration_sec × 12.5)
- physical control targets remain at higher rate: `[T_control, 12]` at 100 Hz

### 3. Implement supervision tier classification

Each utterance must be classified:

- Tier A: speaker / transcript / physical / semantic all high-confidence
- Tier B: transcript and speaker high-confidence, physical or semantic partly pseudo
- Tier C: transcript and basic speaker anchor present, physical supervision sparse
- Tier D: reference-only or auxiliary-only

Low-confidence pseudo-labels must not be treated as dense ground truth.

### 4. Enforce DSP/SSL and LLM role separation

DSP / SSL / audio-derived estimators own:

- physical voice control targets
- confidence
- observed mask
- speaker timbre anchor

Whisper + LLM own:

- transcript
- punctuation recovery
- scene summary
- dialogue intent
- emotion description
- acting hint

Whisper + LLM must not replace physical supervision.

DSP / vocal event detector owns (§5a, added 2026-03-18):

- breathing detection (inhale/exhale)
- laughter detection
- sobbing / crying detection
- voice break detection
- pause detection

These events are inserted into enriched transcripts as inline tags.
Whisper alone cannot detect non-linguistic vocal events (sobs, breaths, laughs).
A dedicated DSP-based vocal event detector (`vocal_event_detector.py`) must run
as part of the annotation pipeline.

### 5. Generate enriched transcripts with inline acting tags

Inspired by Fish Audio S2's rich-transcription approach, the bootstrap pipeline must produce
enriched transcripts that embed vocal events and acting directives inline:

```text
# Plain transcript (§8 output)
本当にありがとう

# Enriched transcript (§4a output)
[inhale] 本当に [emphasis] ありがとう [prolonged laugh]
```

The enriched transcript is produced in two possible ways:

1. Rich-transcription ASR: if the ASR model (Qwen3-ASR or successor) natively outputs vocal events, use those directly
2. LLM enrichment (§11): the semantic annotation LLM (Qwen3.5-9B) reads the plain transcript + detected audio events and injects inline tags

Tag vocabulary must be frozen as part of the v4 dataset contract (track_architecture §8 defines the text encoder contract; this track defines the generation-side vocabulary). The initial freeze covers these minimum categories:

- vocal events: `[inhale]`, `[exhale]`, `[laugh]`, `[sigh]`, `[cough]`, `[click]`
- prosodic markers: `[emphasis]`, `[prolonged]`, `[pause]`
- acting directives: `[angry]`, `[whisper]`, `[calm]`, `[excited]`, `[tender]`, `[professional]`
- free-form acting instructions: `[in a hurry]`, `[with a slight smile]`, etc.

Rules:

- enriched transcript is an ADDITIONAL field alongside plain transcript, not a replacement
- physical control targets from DSP/SSL remain the primary editable supervision
- inline tags are a complementary text-conditioned acting path
- tag positions must be aligned to word/phoneme boundaries
- free-form tags must be normalized to a canonical surface form by the annotation LLM

### 6. Implement bootstrap quality gates

Required metrics (track_data_bootstrap implements the measurement code; track_validation defines thresholds and sign-off criteria):

- diarization purity
- speaker-cluster consistency
- overlap rejection precision
- transcript WER or CER proxy
- physical-label coverage
- physical-label confidence calibration
- language coverage

### 7. Support raw corpus input formats

The pipeline must accept at minimum:

```text
data/raw_corpus/<corpus_id>/**/*.wav
data/raw_corpus/<corpus_id>/**/*.flac
data/raw_corpus/<corpus_id>/**/*.mp3
```

No assumption of pre-existing speaker separation or transcripts.

## Model Selection

### ASR (据え置き)

- Mainline: `Qwen/Qwen3-ASR-1.7B`
- Fallback: `faster-whisper/large-v3`

### Forced Alignment (据え置き)

- `Qwen/Qwen3-ForcedAligner-0.6B`

### Bootstrap Semantic Annotation (§11)

- `Qwen/Qwen3.5-9B`
- 用途: scene summary, dialogue intent, emotion description, acting hint
- 選定理由:
  - 201 言語/方言対応で v4 の言語カバレッジ要件を満たす
  - Gated DeltaNet hybrid で長文コンテキストの annotation 効率が高い
  - 9B dense は batch offline 処理で GPU 1 枚に収まる
  - ASR/alignment と同一 Qwen ファミリーで tokenizer/言語能力が揃う

### Audio Codec (v4.0 新規)

- `kyutai/mimi` (Mimi, Kyutai 2024)
- 用途: audio tokenization (encode/decode)
- エンコーダは frozen（事前学習済み重みをそのまま使用、fine-tune しない）
- 選定理由:
  - 24kHz、8 RVQ × 2048、12.5 Hz — UCLM の系列長を 1/8 に削減
  - fully streaming / causal — v4 runtime contract と互換
  - CC-BY-4.0 ライセンス
  - Moshi で実戦採用済み
- 代替候補: SNAC (24kHz speech, multi-scale), X-Codec 2 (single VQ, 50 TPS)

### SSL Feature Extraction (据え置き)

- `microsoft/wavlm-base-plus` (768-dim)
- `microsoft/wavlm-large` (1024-dim) — v4 の 12-D〜16-D physical 拡張に対応する場合

### Speaker Diarization (据え置き、要検討)

- `pyannote/speaker-diarization-community-1`
- 注意: pyannote 3.x 系への更新を検討すること

## Out Of Scope

Do not reopen:

- direct training on raw unsegmented audio
- using `v3` dataset contracts for `v4` training
- replacing DSP/SSL physical supervision with Whisper + LLM only

## Exit Criteria

- raw-audio bootstrap pipeline is implemented end-to-end (all 13 stages run without manual intervention)
- train-ready cache contract is frozen and documented (merged to main, roundtrip test passes)
- supervision tier classification is functional (each utterance receives a tier label)
- bootstrap quality gates produce measurable reports (thresholds are defined by track_validation and enforced at export time)
- enriched transcripts are generated with inline acting tags aligned to word boundaries
- the pipeline can regenerate train-ready cache from raw corpus deterministically
