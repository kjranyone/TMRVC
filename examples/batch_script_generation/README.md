# Batch Script Generation Example

YAML台本ファイルからバッチでTTS音声を生成するスタンドアロンスクリプト。

## Usage

```bash
# Basic usage (from repository root)
uv run python examples/batch_script_generation/generate.py examples/batch_script_generation/example_script.yaml

# With checkpoints and output directory
uv run python examples/batch_script_generation/generate.py script.yaml \
  --output-dir output/ \
  --tts-checkpoint checkpoints/tts.pt \
  --vc-checkpoint checkpoints/distill/best.pt \
  --device xpu

# Options
uv run python examples/batch_script_generation/generate.py script.yaml \
  --speed 1.2 \
  --format flac \
  --sample-rate 48000 \
  -v
```

## Script YAML Format

```yaml
title: "Script Title"
situation: "Scene description for context-aware emotion."

characters:
  character_id:
    name: "Display Name"
    personality: "Character description"
    voice_description: "Voice characteristics"
    language: ja  # or en
    speaker_file: path/to/speaker.tmrvc_speaker  # optional

entries:
  - speaker: character_id
    text: "Dialogue text"
    hint: "emotion hint for style prediction"
    speed: 1.0  # optional per-entry speed
```

## Output

Generated audio files are saved as `0001_speaker.wav`, `0002_speaker.wav`, etc.
in the output directory (default: `{script_name}_output/`).
