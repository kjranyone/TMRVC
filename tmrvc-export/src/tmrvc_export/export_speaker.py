"""Export Speaker Encoder to ONNX for offline speaker embedding extraction.

Exports:
    - speaker_encoder.onnx: mel_ref -> spk_embed, lora_delta

Usage:
    uv run tmrvc-export-speaker \\
        --checkpoint checkpoints/speaker/best.pt \\
        --output-dir models/fp32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from tmrvc_core.constants import D_SPEAKER, LORA_DELTA_SIZE
from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA

logger = logging.getLogger(__name__)

OPSET_VERSION = 18
PARITY_THRESHOLD = 1e-4
N_MELS = 80


class SpeakerEncoderExportWrapper(nn.Module):
    """Wrapper for SpeakerEncoder ONNX export.

    Returns only spk_embed (lora_delta is optional for inference).
    """

    def __init__(self, model: SpeakerEncoderWithLoRA, include_lora: bool = True):
        super().__init__()
        self.model = model
        self.include_lora = include_lora

    def forward(self, mel_ref: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram.

        Args:
            mel_ref: [B, 80, T] log-mel spectrogram

        Returns:
            spk_embed: [B, 192] L2-normalized speaker embedding
        """
        spk_embed, lora_delta = self.model(mel_ref)
        if self.include_lora:
            return spk_embed, lora_delta
        return spk_embed


def export_speaker_encoder(
    model: SpeakerEncoderWithLoRA,
    output_dir: Path,
    device: str,
    opset_version: int,
    include_lora: bool = True,
) -> Path:
    """Export speaker encoder to ONNX."""
    wrapper = SpeakerEncoderExportWrapper(model, include_lora).eval()
    onnx_path = output_dir / "speaker_encoder.onnx"

    T_ref = 100
    dummy_mel = torch.zeros(1, N_MELS, T_ref, device=device)

    if include_lora:
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            onnx_path,
            input_names=["mel_ref"],
            output_names=["spk_embed", "lora_delta"],
            dynamic_axes={
                "mel_ref": {0: "batch", 2: "T"},
                "spk_embed": {0: "batch"},
                "lora_delta": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            onnx_path,
            input_names=["mel_ref"],
            output_names=["spk_embed"],
            dynamic_axes={
                "mel_ref": {0: "batch", 2: "T"},
                "spk_embed": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    logger.info("Exported speaker_encoder to %s", onnx_path)
    return onnx_path


def verify_speaker_parity(
    model: SpeakerEncoderWithLoRA,
    onnx_path: Path,
    device: str,
    include_lora: bool = True,
) -> None:
    """Verify ONNX outputs match PyTorch outputs."""
    model.eval()

    with torch.no_grad():
        dummy_mel = torch.randn(1, N_MELS, 100, device=device)
        pt_spk, pt_lora = model(dummy_mel)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if include_lora:
        onnx_spk, onnx_lora = sess.run(
            None,
            {"mel_ref": dummy_mel.cpu().numpy()},
        )
        lora_err = np.max(np.abs(pt_lora.cpu().numpy() - onnx_lora))
    else:
        onnx_spk = sess.run(
            None,
            {"mel_ref": dummy_mel.cpu().numpy()},
        )[0]
        lora_err = 0.0

    spk_err = np.max(np.abs(pt_spk.cpu().numpy() - onnx_spk))

    pt_norm = np.linalg.norm(pt_spk.cpu().numpy(), axis=-1)
    onnx_norm = np.linalg.norm(onnx_spk, axis=-1)

    logger.info("Parity check results:")
    logger.info("  spk_embed L_inf = %.6e", spk_err)
    logger.info("  spk_embed L2 norm (PyTorch) = %.6f", pt_norm[0])
    logger.info("  spk_embed L2 norm (ONNX) = %.6f", onnx_norm[0])
    if include_lora:
        logger.info("  lora_delta L_inf = %.6e", lora_err)

    if spk_err > PARITY_THRESHOLD:
        raise AssertionError(f"spk_embed parity failed: {spk_err:.6e}")

    logger.info("Parity check passed!")


def export_speaker(
    checkpoint_path: Optional[str | Path] = None,
    output_dir: str | Path = "models/fp32",
    device: str = "cpu",
    opset_version: int = OPSET_VERSION,
    include_lora: bool = True,
    verify: bool = True,
) -> Path:
    """Export Speaker Encoder to ONNX.

    Args:
        checkpoint_path: Path to speaker encoder checkpoint (optional).
        output_dir: Directory to save ONNX files.
        device: Device for export.
        opset_version: ONNX opset version.
        include_lora: Whether to include lora_delta output.
        verify: Whether to run parity verification.

    Returns:
        Path to exported ONNX file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating SpeakerEncoderWithLoRA model")
    model = SpeakerEncoderWithLoRA(
        d_speaker=D_SPEAKER,
        lora_delta_size=LORA_DELTA_SIZE,
    ).to(device)

    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.eval()

    onnx_path = export_speaker_encoder(
        model, output_dir, device, opset_version, include_lora
    )

    if verify:
        logger.info("Running parity verification...")
        verify_speaker_parity(model, onnx_path, device, include_lora)

    logger.info("Export complete!")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export Speaker Encoder to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/fp32", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for export")
    parser.add_argument(
        "--no-lora", action="store_true", help="Exclude lora_delta output"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip parity verification"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    export_speaker(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        include_lora=not args.no_lora,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
