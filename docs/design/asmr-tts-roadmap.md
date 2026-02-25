# ASMR TTS Roadmap (Pre-release)

Created: 2026-02-25 (Asia/Tokyo)

## Goal
- Build a top-tier ASMR reading stack on top of current TMRVC-TTS.
- Keep runtime real-time friendly while improving whisper/breath realism.

## Current Runtime Baseline
- `style_preset` is available in:
  - `POST /tts`
  - `POST /tts/stream`
  - `WS /ws/chat` (`speak` + `configure`)
- Supported presets:
  - `default`
  - `asmr_soft`
  - `asmr_intimate`
- Presets control:
  - style bias (`whisper`, lower arousal/energy/rate)
  - speed multiplier
  - sentence pause length
  - per-sentence auto-style behavior

## Dialogue-Only Acting Baseline
- Acting does not depend on `hint`.
- Runtime style inference path:
  - explicit `emotion` override (if provided)
  - context inference from `history + situation + current text`
  - rule-based fallback when external context API is unavailable
- `hint` and `situation` are soft-bias inputs (blended), not required control channels.
- `WS /ws/chat` keeps rolling dialogue history and applies turn-transition cues
  (for example, question/answer tension carry-over) so scripts without narration can still produce acting variation.
- Supported fields:
  - REST: `hint`, `context`, `situation`
  - WS speak: `hint`, `situation`
  - WS configure: `situation`

## Why This Is Not Yet SOTA
- No ASMR-specialized corpus fine-tuning yet.
- No dedicated breath event supervision.
- No external benchmark protocol for ASMR naturalness/intimacy.

## SOTA Track (Execution Order)
1. Data
- Build multi-lingual ASMR corpus (ja/en/zh/ko) with licensing clarity.
- Segment to sentence/phrase level and annotate:
  - whisper intensity
  - breath events
  - pause intent
  - intimacy level
- Target: >=200h clean ASMR, >=20 speakers, >=10h/speaker upper cap.

2. Modeling
- Add explicit breath/pause control tokens in TTS front-end.
- Train ASMR style head (continuous intensity) on top of `StyleEncoder`.
- Fine-tune TTS front-end + style-sensitive converter blocks.
- Keep vocoder freeze/unfreeze as ablation axis.

3. Objective/Eval
- Objective:
  - style classification accuracy
  - pause timing error
  - F0 variance profile vs reference
  - breath event F1
- Subjective:
  - MOS naturalness
  - MOS intimacy
  - preference A/B vs baseline
- Report per language and cross-language transfer.

4. Runtime Productization
- Keep `style_preset` as stable API.
- Add optional fine-grained controls after benchmark success:
  - `breathiness` scalar
  - `intimacy` scalar
  - `pause_profile`

## Release Gate (ASMR)
- A/B preference > 65% against current baseline on blind test.
- No regression in non-ASMR default preset quality.
- Streaming latency and glitch rate unchanged in live mode.
