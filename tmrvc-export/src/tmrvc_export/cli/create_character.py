"""``tmrvc-create-character`` — Create .tmrvc_character from .tmrvc_speaker.

Usage::

    tmrvc-create-character models/speaker.tmrvc_speaker -o models/character.tmrvc_character \\
        --name "桜" --personality "明るく元気" --language ja
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-create-character",
        description="Create a .tmrvc_character file from a .tmrvc_speaker file.",
    )
    parser.add_argument(
        "speaker_file", type=Path,
        help="Input .tmrvc_speaker file.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .tmrvc_character file (default: same name with .tmrvc_character extension).",
    )
    parser.add_argument("--name", default="", help="Character name.")
    parser.add_argument("--personality", default="", help="Character personality description.")
    parser.add_argument("--voice-description", default="", help="Voice description.")
    parser.add_argument(
        "--language",
        default="ja",
        choices=["ja", "en", "zh", "ko"],
        help="Language.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


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

    speaker_path: Path = args.speaker_file
    if not speaker_path.exists():
        logger.error("Speaker file not found: %s", speaker_path)
        sys.exit(1)

    output_path = args.output or speaker_path.with_suffix(".tmrvc_character")

    profile = {
        "name": args.name,
        "personality": args.personality,
        "voice_description": args.voice_description,
        "language": args.language,
    }

    from tmrvc_export.character_file import from_speaker_file

    result = from_speaker_file(
        speaker_path=speaker_path,
        output_path=output_path,
        profile=profile,
    )

    logger.info("Created character file: %s (%d bytes)", result, result.stat().st_size)


if __name__ == "__main__":
    main()
