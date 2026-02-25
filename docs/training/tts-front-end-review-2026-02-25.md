# TTS Front-end Review (2026-02-25)

## Conclusion
- Pre-release方針として tokenizer-first を正式採用した。
- 旧TTSチェックポイント互換は維持しない（意図的に破棄）。
- `hint` 依存だけではなく、台本中の地の文/ト書きを直接演技制御へ変換する経路を追加した。

## Findings (ordered by severity)
1. Runtime fragility from hard G2P backend dependency
- Japanese path previously required `pyopenjtalk` at runtime.
- Missing dependency caused immediate synthesis failure.

2. Front-end scalability bottleneck
- Multi-language maintenance cost grows with language-specific G2P rules.
- Hard to iterate style/acting behavior quickly when front-end blocks inference.

3. Model/checkpoint coupling risk
- Converter conditioning width mismatch (`216/224/256`) causes hard incompatibility.
- This blocks rapid experimentation unless checkpoint metadata and validation are stricter.

## Decisions
- Tokenizer-firstをデフォルト運用する。
- 旧TTS checkpoint（`text_frontend`/`text_vocab_size`欠落）はロード拒否する。
- チェックポイントのfrontend/vocab不整合は即時エラーにする。
- 地の文/ト書きは発話テキストから分離し、style/speed/pause/silenceへ変換して合成制御に使う。

## Immediate changes applied
- Japanese G2P now has fallback chain:
  - `pyopenjtalk` -> `phonemizer` (`ja`/`ja-jp`) -> grapheme fallback.
- Warmup no longer blocks startup if front-end dependencies are missing.
- Added tokenizer-first frontend module:
  - byte-level tokenizer (`token_ids`) for `ja/en/zh/ko`
  - no external G2P dependency required for tokenizer path
- Added strict checkpoint validation in serve engine:
  - required: `text_frontend`, `text_vocab_size`
  - reject legacy checkpoints without metadata
  - reject frontend mismatch between runtime config and checkpoint
- Default frontend alignment:
  - `tmrvc-serve`: tokenizer default
  - `TTSEngine`: tokenizer default
  - TTS trainer config: tokenizer default (`vocab=262`)
  - forced-alignment script default: tokenizer
- Added inline stage-direction analysis (`tmrvc_core.text_utils`):
  - supports `(...)` / `（...）` / `[...]` / `【...】` / `<...>` / `＜...＞`
  - extracts non-spoken directions and applies soft style overlay
  - outputs speed scale, inter-sentence pause delta, and leading/trailing silence controls
  - integrated into `/tts`, `/tts/stream`, and websocket synthesis path

## Evaluation entrypoint
- Use `scripts/evaluate_tts_frontends.py` for tokenizer/phoneme A/B evaluation.
- Input: YAML script (same format as `tmrvc_data.script_parser`), checkpoints, device.
- Output:
  - per-frontend wavs
  - `rows.csv` (per-entry latency/RTF metrics)
  - `summary.json` (frontend-level aggregate)
  - optional `ab_blind/` package (`--create-ab`) for blind listening.

Example:

```bash
uv run python scripts/evaluate_tts_frontends.py \
  scripts/scene.yaml \
  --tts-checkpoint checkpoints/tts/tts_step200000.pt \
  --vc-checkpoint checkpoints/vc/vc_step200000.pt \
  --frontends tokenizer phoneme \
  --create-ab \
  --output-dir eval/tts_frontend_ab
```
