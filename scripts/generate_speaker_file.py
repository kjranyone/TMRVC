#!/usr/bin/env python3
"""Generate a .tmrvc_speaker file from reference audio.

Usage::

    uv run python scripts/generate_speaker_file.py \
        --audio ref1.wav ref2.wav \
        --name Speaker_A \
        --output Speaker_A.tmrvc_speaker
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tmrvc_core.constants import LORA_DELTA_SIZE
from tmrvc_data.speaker import SpeakerEncoder
from tmrvc_export.speaker_file import write_speaker_file

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a .tmrvc_speaker file from reference audio.",
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="One or more reference audio files (.wav, .flac, .mp3).",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Speaker name (used in filename if --output is omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <name>.tmrvc_speaker).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_path = args.output or Path(f"{args.name}.tmrvc_speaker")

    encoder = SpeakerEncoder()
    embeddings = []

    for path in args.audio:
        logger.info("Extracting speaker embedding from %s", path)
        emb = encoder.extract_from_file(path)
        embeddings.append(emb)

    # Average and re-normalise
    avg_embed = torch.stack(embeddings).mean(dim=0)
    avg_embed = F.normalize(avg_embed, p=2, dim=-1)

    spk_embed_np = avg_embed.numpy().astype(np.float32)
    lora_delta_np = np.zeros(LORA_DELTA_SIZE, dtype=np.float32)

    write_speaker_file(output_path, spk_embed_np, lora_delta_np)
    logger.info("Speaker file written to %s", output_path)


if __name__ == "__main__":
    main()
