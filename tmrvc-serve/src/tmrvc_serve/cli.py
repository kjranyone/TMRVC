"""``tmrvc-serve`` — Start the Unified UCLM v2 API server.

Usage::

    tmrvc-serve --uclm-checkpoint checkpoints/uclm/latest.pt --codec-checkpoint checkpoints/codec/latest.pt
    tmrvc-serve --port 8000 --api-key $ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-serve",
        description="Start the TMRVC Unified (TTS/VC) API server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000).",
    )
    parser.add_argument(
        "--uclm-checkpoint",
        type=Path,
        default=None,
        help="Path to UCLM model checkpoint (.pt).",
    )
    parser.add_argument(
        "--codec-checkpoint",
        type=Path,
        default=None,
        help="Path to Emotion-Aware Codec checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key for context prediction (default: $ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    from tmrvc_serve.app import init_app

    init_app(
        uclm_checkpoint=args.uclm_checkpoint,
        codec_checkpoint=args.codec_checkpoint,
        device=args.device,
        api_key=api_key,
    )

    import uvicorn

    uvicorn.run(
        "tmrvc_serve.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
