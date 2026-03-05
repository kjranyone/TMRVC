#!/usr/bin/env python3
"""Evaluate UCLM v2 TTS engine performance and quality.

Generates audio and metrics from a YAML script using the unified UCLMEngine.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_core.text_utils import analyze_inline_stage_directions, text_to_phonemes
from tmrvc_data.script_parser import load_script
from tmrvc_serve.uclm_engine import UCLMEngine

logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    entry_index: int
    speaker: str
    text: str
    audio_path: str
    duration_sec: float
    uclm_ms: float
    decoder_ms: float
    total_ms: float
    rtf: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate UCLM v2 performance.")
    parser.add_argument("script", type=Path, help="YAML script path.")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_uclm"), help="Output directory.")
    parser.add_argument("--uclm-checkpoint", type=Path, default="checkpoints/uclm/uclm_latest.pt")
    parser.add_argument("--codec-checkpoint", type=Path, default="checkpoints/codec/codec_latest.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import soundfile as sf
    from tmrvc_export.speaker_file import read_speaker_file

    script_obj = load_script(args.script)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Engine
    engine = UCLMEngine(args.uclm_checkpoint, args.codec_checkpoint, device=args.device)
    engine.load_models()

    # Load Speaker Embeddings
    speaker_embeds = {}
    for char_id, profile in script_obj.characters.items():
        if profile.speaker_file and Path(profile.speaker_file).exists():
            spk_embed, _, _, _ = read_speaker_file(profile.speaker_file)
            speaker_embeds[char_id] = torch.from_numpy(spk_embed).float().unsqueeze(0)
        else:
            speaker_embeds[char_id] = torch.zeros(1, 192)

    rows: list[EvalRow] = []
    for idx, entry in enumerate(script_obj.entries, start=1):
        char = script_obj.characters.get(entry.speaker)
        language = char.language if char else "ja"
        spk_t = speaker_embeds.get(entry.speaker, torch.zeros(1, 192))
        
        style = entry.style_override or StyleParams.neutral()
        
        # Synthesis
        phoneme_ids = text_to_phonemes(entry.text, language=language)
        phonemes_t = torch.tensor(phoneme_ids).long().unsqueeze(0)
        
        audio_t, metrics = engine.tts(phonemes_t, spk_t, style)
        audio = audio_t.cpu().numpy()
        
        out_wav = output_dir / f"{idx:04d}_{entry.speaker}.wav"
        sf.write(out_wav, audio, SAMPLE_RATE)
        
        row = EvalRow(
            entry_index=idx,
            speaker=entry.speaker,
            text=entry.text,
            audio_path=str(out_wav),
            duration_sec=len(audio) / SAMPLE_RATE,
            uclm_ms=float(metrics.get("gen_time_ms", 0.0)),
            decoder_ms=0.0,
            total_ms=float(metrics.get("gen_time_ms", 0.0)),
            rtf=float(metrics.get("rtf", 0.0)),
        )
        rows.append(row)
        logger.info("[%04d] RTF=%.3f, Total=%.1fms", idx, row.rtf, row.total_ms)

    # Write results
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows: writer.writerow(asdict(row))

    logger.info("Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
