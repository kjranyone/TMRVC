"""``tmrvc-serve`` — Start the TTS API server.

Usage::

    tmrvc-serve --tts-checkpoint checkpoints/tts/tts_step200000.pt --device xpu
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
        description="Start the TMRVC TTS API server.",
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
        "--tts-checkpoint",
        type=Path,
        default=None,
        help="Path to TTS checkpoint (.pt).",
    )
    parser.add_argument(
        "--vc-checkpoint",
        type=Path,
        default=None,
        help="Path to VC/distill checkpoint (.pt) for Converter+Vocoder.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--text-frontend",
        choices=["phoneme", "tokenizer", "auto"],
        default="tokenizer",
        help="Text frontend mode (default: tokenizer).",
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
        tts_checkpoint=args.tts_checkpoint,
        vc_checkpoint=args.vc_checkpoint,
        device=args.device,
        api_key=api_key,
        text_frontend=args.text_frontend,
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
