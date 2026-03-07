#!/usr/bin/env python3
"""Batch TTS generation from YAML scripts using UCLM.

Reads a YAML script file with character profiles and dialogue entries,
then generates audio files for each entry using the unified UCLM pipeline.

Usage:
    python generate.py script.yaml --output-dir output/
    python generate.py script.yaml --uclm-checkpoint ckpt/uclm.pt --codec-checkpoint ckpt/codec.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch generate audio using UCLM.",
    )
    p.add_argument("script", type=Path, help="YAML script file path")
    p.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory (default: script_name_output/)",
    )
    p.add_argument("--uclm-checkpoint", type=Path, default="checkpoints/uclm/uclm_latest.pt", help="UCLM checkpoint")
    p.add_argument("--codec-checkpoint", type=Path, default="checkpoints/codec/codec_latest.pt", help="Codec checkpoint")
    p.add_argument("--device", default="cpu", help="Torch device")
    p.add_argument("--format", choices=["wav", "flac"], default="wav", help="Audio format")
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
    from tmrvc_core.constants import SAMPLE_RATE
    from tmrvc_core.dialogue_types import StyleParams
    from tmrvc_data.g2p import text_to_phonemes
    from tmrvc_data.script_parser import load_script
    from tmrvc_serve.uclm_engine import UCLMEngine
    from tmrvc_export.speaker_file import read_speaker_file

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

    # Load UCLM engine
    if not args.uclm_checkpoint.exists() or not args.codec_checkpoint.exists():
        logger.error("UCLM/Codec checkpoints not found.")
        sys.exit(1)

    engine = UCLMEngine(
        uclm_checkpoint=args.uclm_checkpoint,
        codec_checkpoint=args.codec_checkpoint,
        device=args.device,
    )
    engine.load_models()

    # Load speaker embeddings per character
    speaker_embeds: dict[str, torch.Tensor] = {}
    for char_id, char_profile in script.characters.items():
        if char_profile.speaker_file and char_profile.speaker_file.exists():
            speaker = read_speaker_file(char_profile.speaker_file)
            speaker_embeds[char_id] = torch.from_numpy(speaker.spk_embed).float().unsqueeze(0)
            logger.info("Loaded speaker: %s from %s", char_id, char_profile.speaker_file)
        else:
            from tmrvc_core.constants import D_SPEAKER
            speaker_embeds[char_id] = torch.zeros(1, D_SPEAKER)
            logger.warning("No speaker file for '%s', using zero embedding", char_id)

    total_duration = 0.0
    t_start = time.perf_counter()

    for i, entry in enumerate(script.entries):
        char = script.characters.get(entry.speaker)
        language = char.language if char else "ja"
        spk_t = speaker_embeds.get(entry.speaker, torch.zeros(1, 192))

        # Determine style
        style = entry.style_override
        if style is None:
            if char and char.default_style:
                style = char.default_style
            else:
                style = StyleParams.neutral()

        logger.info("[%d/%d] %s: %s", i + 1, len(script.entries), entry.speaker, entry.text[:40])

        # G2P
        phoneme_ids = text_to_phonemes(entry.text, language=language)
        phonemes_t = torch.tensor(phoneme_ids).long().unsqueeze(0)

        # Synthesis
        audio_t, metrics = engine.tts(
            phonemes=phonemes_t,
            speaker_embed=spk_t,
            style=style
        )
        audio = audio_t.cpu().numpy()
        total_duration += metrics.output_duration_ms / 1000

        # Write audio file
        filename = f"{i + 1:04d}_{entry.speaker}.{args.format}"
        out_path = output_dir / filename
        sf.write(str(out_path), audio, SAMPLE_RATE, format=args.format.upper())
        logger.info("  -> %s (%.2fs)", out_path.name, metrics.output_duration_ms/1000)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done: %d files, %.1fs audio in %.1fs (RTF=%.2fx)",
        len(script.entries), total_duration, elapsed,
        elapsed / max(total_duration, 0.01),
    )


if __name__ == "__main__":
    main()
