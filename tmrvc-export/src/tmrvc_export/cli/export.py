"""``tmrvc-export`` — Export, quantize, and verify ONNX models.

Usage::

    tmrvc-export --checkpoint distill.pt --output-dir models/fp32
    tmrvc-export --checkpoint distill.pt --output-dir models/ --quantize --verify
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-export",
        description="Export student models to ONNX, optionally quantize and verify.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to distillation checkpoint (contains content_encoder, converter, vocoder).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for ONNX models.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce INT8 quantized models.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run parity verification after export.",
    )
    parser.add_argument(
        "--speaker-encoder-ckpt",
        type=Path,
        default=None,
        help="Optional: path to speaker encoder checkpoint.",
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

    from tmrvc_export.export_onnx import (
        export_content_encoder,
        export_converter,
        export_ir_estimator,
        export_speaker_encoder,
        export_vocoder,
    )
    from tmrvc_train.models.content_encoder import ContentEncoderStudent
    from tmrvc_train.models.converter import ConverterStudent
    from tmrvc_train.models.ir_estimator import IREstimator
    from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
    from tmrvc_train.models.vocoder import VocoderStudent

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Create models
    content_encoder = ContentEncoderStudent()
    converter = ConverterStudent()
    vocoder = VocoderStudent()
    ir_estimator = IREstimator()

    # Load state dicts
    if "content_encoder" in ckpt:
        content_encoder.load_state_dict(ckpt["content_encoder"])
    if "converter" in ckpt:
        converter.load_state_dict(ckpt["converter"])
    if "vocoder" in ckpt:
        vocoder.load_state_dict(ckpt["vocoder"])
    if "ir_estimator" in ckpt:
        ir_estimator.load_state_dict(ckpt["ir_estimator"])

    fp32_dir = args.output_dir / "fp32"
    fp32_dir.mkdir(parents=True, exist_ok=True)

    # Export all
    export_content_encoder(content_encoder, fp32_dir / "content_encoder.onnx")
    export_converter(converter, fp32_dir / "converter.onnx")
    export_vocoder(vocoder, fp32_dir / "vocoder.onnx")
    export_ir_estimator(ir_estimator, fp32_dir / "ir_estimator.onnx")

    # Speaker encoder (optional)
    if args.speaker_encoder_ckpt:
        spk_encoder = SpeakerEncoderWithLoRA()
        spk_ckpt = torch.load(args.speaker_encoder_ckpt, map_location="cpu", weights_only=False)
        spk_encoder.load_state_dict(spk_ckpt)
        export_speaker_encoder(spk_encoder, fp32_dir / "speaker_encoder.onnx")

    # Quantize
    if args.quantize:
        from tmrvc_export.quantize import quantize_all

        int8_dir = args.output_dir / "int8"
        quantize_all(fp32_dir, int8_dir)

    # Verify parity
    if args.verify:
        from tmrvc_export.verify_parity import verify_all

        models = {
            "content_encoder": content_encoder,
            "converter": converter,
            "vocoder": vocoder,
            "ir_estimator": ir_estimator,
        }
        results = verify_all(models, fp32_dir)
        all_passed = all(r.passed for r in results)
        if not all_passed:
            logger.error("Parity verification FAILED!")
            raise SystemExit(1)

    logger.info("Export complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
