"""Export UCLM model to ONNX format for real-time inference.

Exports a single-frame inference wrapper for streaming operation.

Usage:
    uv run python -m tmrvc_export.export_uclm \\
        --checkpoint checkpoints/uclm/uclm_step50000.pt \\
        --output-dir models/fp32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_MODEL,
    D_SPEAKER,
    N_CODEBOOKS,
    VOCAB_SIZE,
)
from tmrvc_train.models.uclm import UCLM, UCLMConfig

logger = logging.getLogger(__name__)

OPSET_VERSION = 18


class UCLMFrameWrapper(nn.Module):
    """Wraps UCLM for single-frame ONNX export.

    Inputs:
        - source_tokens: [B, n_codebooks, context_len] - past tokens + current
        - voice_state: [B, context_len, 8]
        - speaker_embed: [B, d_speaker]

    Outputs:
        - logits_ar: [B, vocab_size] - AR logits for next token (first codebook)
        - logits_parallel: [B, n_codebooks-1, vocab_size] - parallel logits
    """

    def __init__(self, model: UCLM):
        super().__init__()
        self.model = model

    def forward(
        self,
        source_tokens: torch.Tensor,
        voice_state: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-frame inference."""
        output = self.model.forward(
            text_features=None,
            source_tokens=source_tokens,
            voice_state=voice_state,
            speaker_embed=speaker_embed,
            past_tokens=None,
            target_tokens=source_tokens,
            mode="vc",
        )

        # Return only last frame logits
        logits_ar = output["logits_ar"][:, -1, :]  # [B, vocab_size]
        logits_parallel = output["logits_parallel"][:, :, -1, :]  # [B, n_cb-1, vocab]

        return logits_ar, logits_parallel


class UCLMSampler(nn.Module):
    """Wraps UCLM with top-k sampling for ONNX export.

    Outputs sampled tokens directly for use in streaming pipeline.
    Uses argmax (greedy) for ONNX compatibility.
    """

    def __init__(self, model: UCLM, temperature: float = 1.0, top_k: int = 50):
        super().__init__()
        self.wrapper = UCLMFrameWrapper(model)
        self.temperature = temperature
        self.top_k = top_k
        self.n_codebooks = model.config.n_codebooks
        self.vocab_size = model.config.vocab_size

    def forward(
        self,
        source_tokens: torch.Tensor,
        voice_state: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Sample tokens from UCLM output.

        Uses argmax (greedy decoding) for ONNX compatibility.
        multinomial is not well-supported in ONNX Runtime.

        Returns:
            tokens: [B, n_codebooks] sampled tokens
        """
        logits_ar, logits_parallel = self.wrapper(
            source_tokens, voice_state, speaker_embed
        )

        # AR decoding (first codebook) - use argmax for ONNX compatibility
        logits_ar = logits_ar / self.temperature
        if self.top_k < self.vocab_size:
            top_k_logits, top_k_indices = torch.topk(logits_ar, self.top_k, dim=-1)
            token_0 = top_k_indices[:, 0]
        else:
            token_0 = torch.argmax(logits_ar, dim=-1)

        # Parallel decoding (remaining codebooks)
        tokens = torch.zeros(
            source_tokens.shape[0],
            self.n_codebooks,
            dtype=torch.long,
            device=source_tokens.device,
        )
        tokens[:, 0] = token_0

        for i in range(self.n_codebooks - 1):
            logits = logits_parallel[:, i, :] / self.temperature
            if self.top_k < self.vocab_size:
                top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
                tokens[:, i + 1] = top_k_indices[:, 0]
            else:
                tokens[:, i + 1] = torch.argmax(logits, dim=-1)

        return tokens


def export_uclm(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    context_frames: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
    opset_version: int = OPSET_VERSION,
) -> dict[str, Path]:
    """Export UCLM model to ONNX.

    Args:
        checkpoint_path: Path to UCLM checkpoint.
        output_dir: Directory to save ONNX files.
        context_frames: Number of context frames for streaming.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        device: Device for export.
        opset_version: ONNX opset version.

    Returns:
        Dict with paths to exported ONNX files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading UCLM from %s", checkpoint_path)
    config = UCLMConfig(
        vocab_size=VOCAB_SIZE,
        n_codebooks=N_CODEBOOKS,
        d_model=D_MODEL,
        d_speaker=D_SPEAKER,
    )
    model = UCLM(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Create wrapper with sampling
    sampler = UCLMSampler(model, temperature=temperature, top_k=top_k)
    sampler.eval()

    # Create dummy inputs
    B = 1
    dummy_source_tokens = torch.zeros(
        B, N_CODEBOOKS, context_frames, dtype=torch.long, device=device
    )
    dummy_voice_state = torch.zeros(B, context_frames, 8, device=device)
    dummy_speaker_embed = torch.zeros(B, D_SPEAKER, device=device)

    # Export
    onnx_path = output_dir / "uclm_vc.onnx"
    logger.info("Exporting to %s", onnx_path)

    torch.onnx.export(
        sampler,
        (dummy_source_tokens, dummy_voice_state, dummy_speaker_embed),
        onnx_path,
        input_names=[
            "source_tokens",
            "voice_state",
            "speaker_embed",
        ],
        output_names=["tokens"],
        dynamic_axes={
            "source_tokens": {0: "batch", 2: "context_len"},
            "voice_state": {0: "batch", 1: "context_len"},
            "speaker_embed": {0: "batch"},
            "tokens": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    logger.info("Exported UCLM to %s", onnx_path)

    return {"uclm_vc": onnx_path}


def main():
    parser = argparse.ArgumentParser(description="Export UCLM to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models/fp32")
    parser.add_argument("--context-frames", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    export_uclm(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        context_frames=args.context_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
