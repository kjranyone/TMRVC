#!/usr/bin/env python3
"""Batch TTS generation from YAML scripts.

Standalone example script — reads a YAML script file with character profiles
and dialogue entries, then generates audio files for each entry.

Usage:
    python generate.py script.yaml --output-dir output/
    python generate.py script.yaml --tts-checkpoint ckpt/tts.pt --vc-checkpoint ckpt/vc.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate audio from a YAML script file.",
    )
    p.add_argument("script", type=Path, help="YAML script file path")
    p.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory (default: script_name_output/)",
    )
    p.add_argument("--tts-checkpoint", type=Path, default=None, help="TTS checkpoint")
    p.add_argument("--vc-checkpoint", type=Path, default=None, help="VC checkpoint")
    p.add_argument("--device", default="cpu", help="Torch device")
    p.add_argument("--speed", type=float, default=1.0, help="Speed factor")
    p.add_argument("--format", choices=["wav", "flac"], default="wav", help="Audio format")
    p.add_argument("--sample-rate", type=int, default=None, help="Output sample rate (None = native)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import soundfile as sf
    import torch

    from tmrvc_core.constants import SAMPLE_RATE
    from tmrvc_core.dialogue_types import StyleParams
    from tmrvc_data.script_parser import load_script
    from tmrvc_serve.tts_engine import TTSEngine

    script_path: Path = args.script
    if not script_path.exists():
        logger.error("Script file not found: %s", script_path)
        sys.exit(1)

    output_dir: Path = args.output_dir or script_path.with_suffix("") .parent / f"{script_path.stem}_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading script: %s", script_path)
    script = load_script(script_path)
    logger.info(
        "Script '%s': %d characters, %d entries",
        script.title or "(untitled)", len(script.characters), len(script.entries),
    )

    if not script.entries:
        logger.warning("No dialogue entries in script. Nothing to generate.")
        return

    # Load TTS engine
    engine = TTSEngine(
        tts_checkpoint=args.tts_checkpoint,
        vc_checkpoint=args.vc_checkpoint,
        device=args.device,
    )
    engine.load_models()

    # Load speaker embeddings per character
    from tmrvc_export.speaker_file import read_speaker_file

    speaker_embeds: dict[str, torch.Tensor] = {}
    for char_id, char_profile in script.characters.items():
        if char_profile.speaker_file and char_profile.speaker_file.exists():
            spk_embed, _lora, _meta, _thumb = read_speaker_file(char_profile.speaker_file)
            speaker_embeds[char_id] = torch.from_numpy(spk_embed).float()
            logger.info("Loaded speaker: %s from %s", char_id, char_profile.speaker_file)
        else:
            speaker_embeds[char_id] = torch.zeros(192)
            logger.warning("No speaker file for '%s', using zero embedding", char_id)

    out_sr = args.sample_rate or SAMPLE_RATE
    total_duration = 0.0
    t0 = time.perf_counter()

    for i, entry in enumerate(script.entries):
        # Determine style
        style = entry.style_override
        if style is None:
            char = script.characters.get(entry.speaker)
            if char and char.default_style:
                style = char.default_style
            else:
                style = StyleParams.neutral()

        # Determine language
        char = script.characters.get(entry.speaker)
        language = char.language if char else "ja"

        # Get speaker embedding
        spk_embed = speaker_embeds.get(entry.speaker, torch.zeros(192))

        logger.info("[%d/%d] %s: %s", i + 1, len(script.entries), entry.speaker, entry.text[:40])

        # Per-entry speed (from script) * global speed multiplier (from CLI)
        entry_speed = entry.speed * args.speed

        audio, duration_sec = engine.synthesize(
            text=entry.text,
            language=language,
            spk_embed=spk_embed,
            style=style,
            speed=entry_speed,
        )
        total_duration += duration_sec

        # Write audio file
        filename = f"{i + 1:04d}_{entry.speaker}.{args.format}"
        out_path = output_dir / filename
        sf.write(str(out_path), audio, out_sr, format=args.format.upper())
        logger.info("  -> %s (%.2fs)", out_path.name, duration_sec)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: %d files, %.1fs audio in %.1fs (RTF=%.2fx)",
        len(script.entries), total_duration, elapsed,
        elapsed / max(total_duration, 0.01),
    )


if __name__ == "__main__":
    main()
