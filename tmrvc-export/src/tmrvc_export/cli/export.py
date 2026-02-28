"""``tmrvc-export`` — Export Codec-Latent models to ONNX.

Usage::

    tmrvc-export --codec checkpoints/codec/best.pt --token checkpoints/token/best.pt \\
        --output-dir models/fp32 --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-export",
        description="Export Codec-Latent models to ONNX.",
    )
    parser.add_argument(
        "--codec",
        type=Path,
        required=True,
        help="Codec checkpoint path (StreamingCodec).",
    )
    parser.add_argument(
        "--token",
        type=Path,
        required=True,
        help="Token model checkpoint path (TokenModel).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for ONNX files.",
    )
    parser.add_argument(
        "--speaker-encoder",
        type=Path,
        default=None,
        help="Speaker encoder checkpoint (optional).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX export by running inference.",
    )
    parser.add_argument("--device", default="cpu", help="Device (cuda/cpu/xpu).")
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

    import torch

    from tmrvc_train.models.streaming_codec import StreamingCodec, CodecConfig
    from tmrvc_train.models.token_model import TokenModel, TokenModelConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading codec checkpoint: %s", args.codec)
    codec_ckpt = torch.load(args.codec, map_location=args.device, weights_only=False)

    logger.info("Loading token checkpoint: %s", args.token)
    token_ckpt = torch.load(args.token, map_location=args.device, weights_only=False)

    codec_config = CodecConfig()
    token_config = TokenModelConfig()

    codec = StreamingCodec(codec_config)
    token_model = TokenModel(token_config)

    if "state_dict" in codec_ckpt:
        codec.load_state_dict(codec_ckpt["state_dict"])
    else:
        codec.load_state_dict(codec_ckpt)

    if "model_state_dict" in token_ckpt:
        token_model.load_state_dict(token_ckpt["model_state_dict"])
    elif "state_dict" in token_ckpt:
        token_model.load_state_dict(token_ckpt["state_dict"])
    else:
        token_model.load_state_dict(token_ckpt)

    codec = codec.to(args.device).eval()
    token_model = token_model.to(args.device).eval()

    speaker_encoder = None
    if args.speaker_encoder:
        logger.info("Loading speaker encoder checkpoint: %s", args.speaker_encoder)
        from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA

        spk_ckpt = torch.load(
            args.speaker_encoder, map_location=args.device, weights_only=False
        )
        speaker_encoder = SpeakerEncoderWithLoRA()
        if "state_dict" in spk_ckpt:
            speaker_encoder.load_state_dict(spk_ckpt["state_dict"])
        else:
            speaker_encoder.load_state_dict(spk_ckpt)
        speaker_encoder = speaker_encoder.to(args.device).eval()

    from tmrvc_export.export_onnx import (
        export_codec_encoder,
        export_codec_decoder,
        export_token_model,
        export_speaker_encoder,
    )

    logger.info("Exporting codec_encoder...")
    export_codec_encoder(codec, args.output_dir / "codec_encoder.onnx", codec_config)

    logger.info("Exporting codec_decoder...")
    export_codec_decoder(codec, args.output_dir / "codec_decoder.onnx", codec_config)

    logger.info("Exporting token_model...")
    export_token_model(token_model, args.output_dir / "token_model.onnx", token_config)

    if speaker_encoder:
        logger.info("Exporting speaker_encoder...")
        export_speaker_encoder(
            speaker_encoder, args.output_dir / "speaker_encoder.onnx"
        )

    if args.verify:
        logger.info("Verifying ONNX exports...")
        import onnxruntime as ort

        for onnx_file in [
            "codec_encoder.onnx",
            "codec_decoder.onnx",
            "token_model.onnx",
        ]:
            path = args.output_dir / onnx_file
            if not path.exists():
                continue
            try:
                sess = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                logger.info(
                    "  %s: OK (inputs: %s)",
                    onnx_file,
                    [i.name for i in sess.get_inputs()],
                )
            except Exception as e:
                logger.error("  %s: FAILED - %s", onnx_file, e)
                sys.exit(1)

    logger.info("Export complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
