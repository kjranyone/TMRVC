"""Export Emotion-Aware Codec to ONNX for real-time streaming inference.

Exports two ONNX models:
    - codec_encoder.onnx: audio_frame + state_in -> acoustic_tokens + state_out
    - codec_decoder.onnx: acoustic_tokens + control_tokens + voice_state + event_trace_in + state_in -> audio_frame + event_trace_out + state_out

Usage:
    uv run tmrvc-export-codec \\
        --checkpoint checkpoints/codec/best.pt \\
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
import torch.nn.functional as F

from tmrvc_core.constants import (
    CONTROL_VOCAB_SIZE,
    D_VOICE_STATE,
    N_CODEBOOKS,
    RVQ_VOCAB_SIZE,
)
from tmrvc_train.models.emotion_codec import EmotionAwareCodec
from tmrvc_train.models.voice_state_film import VoiceStateFiLM
from tmrvc_train.models.control_encoder import ControlEncoder

logger = logging.getLogger(__name__)

OPSET_VERSION = 18
PARITY_THRESHOLD = 1e-4

FRAME_SIZE = 240
ENC_STATE_DIM = 512
ENC_STATE_FRAMES = 16
DEC_STATE_DIM = 512
DEC_STATE_FRAMES = 16
EVENT_TRACE_DIM = 128
CODEBOOK_DIM = 64


class StreamingCodecEncoder(nn.Module):
    """Streaming codec encoder with explicit state management.

    Single-frame processing version for ONNX export.
    """

    def __init__(self, codec: EmotionAwareCodec):
        super().__init__()
        self.encoder = codec.encoder
        self.d_model = codec.encoder.d_model
        self.n_codebooks = codec.encoder.n_codebooks

    def forward(
        self,
        audio_frame: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process single 10ms frame.

        Args:
            audio_frame: [B, 1, 240] single frame
            state_in: [B, ENC_STATE_DIM, ENC_STATE_FRAMES] encoder state

        Returns:
            acoustic_tokens: [B, 8] RVQ indices
            state_out: [B, ENC_STATE_DIM, ENC_STATE_FRAMES] updated state
        """
        x = audio_frame

        new_state = torch.roll(state_in, -1, dims=2)
        new_state[:, :, -1] = 0

        for layer in self.encoder.encoder:
            if hasattr(layer, "padding"):
                x = F.pad(x, (layer.padding, 0))
                x = layer.conv(x)
            elif isinstance(layer, nn.Conv1d):
                x = layer(x)
            else:
                x = layer(x)

        if x.shape[-1] > 0:
            new_state[:, :, -1] = x[:, :, -1]

        z = self.encoder.rvq(x.transpose(1, 2))[1]
        acoustic_tokens = z[:, :, -1]

        return acoustic_tokens, new_state


class StreamingCodecDecoder(nn.Module):
    """Streaming codec decoder for single-frame processing.

    Uses the original EmotionAwareDecoder backbone (ConvTranspose1d layers).
    Projects control encoder output to match acoustic embedding dimension.
    """

    def __init__(self, codec: EmotionAwareCodec):
        super().__init__()
        self.n_codebooks = codec.decoder.n_codebooks
        self.d_model = codec.decoder.d_model

        self.codebook_embeds = codec.decoder.codebook_embeds
        self.control_encoder = codec.decoder.control_encoder
        self.film = codec.decoder.film
        self.decoder = codec.decoder.decoder

        ctrl_dim = codec.decoder.control_encoder.d_model
        self.ctrl_proj = nn.Linear(ctrl_dim, self.d_model)

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        control_tokens: torch.Tensor,
        voice_state: torch.Tensor,
        event_trace_in: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process single frame.

        Args:
            acoustic_tokens: [B, 8] A_t indices
            control_tokens: [B, 4] B_t [op, type, dur, int]
            voice_state: [B, 8] voice state parameters
            event_trace_in: [B, EVENT_TRACE_DIM] event hysteresis
            state_in: [B, DEC_STATE_DIM, DEC_STATE_FRAMES] decoder state (unused in conv)

        Returns:
            audio_frame: [B, 1, 240] decoded audio
            event_trace_out: [B, EVENT_TRACE_DIM] updated trace
            state_out: [B, DEC_STATE_DIM, DEC_STATE_FRAMES] updated state
        """
        embeds = []
        for i, emb in enumerate(self.codebook_embeds):
            embeds.append(emb(acoustic_tokens[:, i]))
        z_a = torch.cat(embeds, dim=-1)

        z_b = self.control_encoder(control_tokens).squeeze(-1)
        z_b = self.ctrl_proj(z_b)

        z = z_a + z_b

        z = self.film(z.unsqueeze(1), voice_state)

        audio = self.decoder(z.transpose(1, 2))

        if audio.shape[-1] >= FRAME_SIZE:
            audio_frame = audio[:, :, :FRAME_SIZE]
        else:
            audio_frame = F.pad(audio, (0, FRAME_SIZE - audio.shape[-1]))

        audio_frame = torch.tanh(audio_frame)

        event_trace_out = 0.9 * event_trace_in + 0.1 * z[:, 0, :EVENT_TRACE_DIM]

        return audio_frame, event_trace_out, state_in.clone()


def export_codec_encoder(
    codec: EmotionAwareCodec,
    output_dir: Path,
    device: str,
    opset_version: int,
) -> Path:
    """Export streaming codec encoder to ONNX."""
    wrapper = StreamingCodecEncoder(codec).eval()
    onnx_path = output_dir / "codec_encoder.onnx"

    dummy_audio = torch.zeros(1, 1, FRAME_SIZE, device=device)
    dummy_state = torch.zeros(1, ENC_STATE_DIM, ENC_STATE_FRAMES, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_state),
        onnx_path,
        input_names=["audio_frame", "state_in"],
        output_names=["acoustic_tokens", "state_out"],
        dynamic_axes={
            "audio_frame": {0: "batch"},
            "state_in": {0: "batch"},
            "acoustic_tokens": {0: "batch"},
            "state_out": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported codec_encoder to %s", onnx_path)
    return onnx_path


def export_codec_decoder(
    codec: EmotionAwareCodec,
    output_dir: Path,
    device: str,
    opset_version: int,
) -> Path:
    """Export streaming codec decoder to ONNX."""
    wrapper = StreamingCodecDecoder(codec).eval()
    onnx_path = output_dir / "codec_decoder.onnx"

    dummy_acoustic = torch.zeros(1, N_CODEBOOKS, dtype=torch.long, device=device)
    dummy_control = torch.zeros(1, 4, dtype=torch.long, device=device)
    dummy_voice = torch.zeros(1, D_VOICE_STATE, device=device)
    dummy_event = torch.zeros(1, EVENT_TRACE_DIM, device=device)
    dummy_state = torch.zeros(1, DEC_STATE_DIM, DEC_STATE_FRAMES, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_acoustic, dummy_control, dummy_voice, dummy_event, dummy_state),
        onnx_path,
        input_names=[
            "acoustic_tokens",
            "control_tokens",
            "voice_state",
            "event_trace_in",
            "state_in",
        ],
        output_names=["audio_frame", "event_trace_out", "state_out"],
        dynamic_axes={
            "acoustic_tokens": {0: "batch"},
            "control_tokens": {0: "batch"},
            "voice_state": {0: "batch"},
            "event_trace_in": {0: "batch"},
            "state_in": {0: "batch"},
            "audio_frame": {0: "batch"},
            "event_trace_out": {0: "batch"},
            "state_out": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported codec_decoder to %s", onnx_path)
    return onnx_path


def verify_codec_parity(
    codec: EmotionAwareCodec,
    onnx_paths: dict[str, Path],
    device: str,
) -> None:
    """Verify ONNX outputs match PyTorch outputs."""
    codec.eval()

    with torch.no_grad():
        dummy_audio = torch.zeros(1, 1, FRAME_SIZE, device=device)
        dummy_state_enc = torch.zeros(1, ENC_STATE_DIM, ENC_STATE_FRAMES, device=device)

        enc_wrapper = StreamingCodecEncoder(codec)
        pt_tokens, pt_state_enc = enc_wrapper(dummy_audio, dummy_state_enc)

        dummy_acoustic = pt_tokens
        dummy_control = torch.zeros(1, 4, dtype=torch.long, device=device)
        dummy_voice = torch.zeros(1, D_VOICE_STATE, device=device)
        dummy_event = torch.zeros(1, EVENT_TRACE_DIM, device=device)
        dummy_state_dec = torch.zeros(1, DEC_STATE_DIM, DEC_STATE_FRAMES, device=device)

        dec_wrapper = StreamingCodecDecoder(codec)
        pt_audio, pt_event, pt_state_dec = dec_wrapper(
            dummy_acoustic, dummy_control, dummy_voice, dummy_event, dummy_state_dec
        )

    sess_enc = ort.InferenceSession(
        str(onnx_paths["codec_encoder"]), providers=["CPUExecutionProvider"]
    )
    sess_dec = ort.InferenceSession(
        str(onnx_paths["codec_decoder"]), providers=["CPUExecutionProvider"]
    )

    onnx_tokens, onnx_state_enc = sess_enc.run(
        None,
        {
            "audio_frame": dummy_audio.cpu().numpy(),
            "state_in": dummy_state_enc.cpu().numpy(),
        },
    )

    onnx_audio, onnx_event, onnx_state_dec = sess_dec.run(
        None,
        {
            "acoustic_tokens": onnx_tokens,
            "control_tokens": dummy_control.cpu().numpy(),
            "voice_state": dummy_voice.cpu().numpy(),
            "event_trace_in": dummy_event.cpu().numpy(),
            "state_in": dummy_state_dec.cpu().numpy(),
        },
    )

    tokens_match = np.all(pt_tokens.cpu().numpy() == onnx_tokens)
    state_enc_err = np.max(np.abs(pt_state_enc.cpu().numpy() - onnx_state_enc))
    audio_err = np.max(np.abs(pt_audio.cpu().numpy() - onnx_audio))
    event_err = np.max(np.abs(pt_event.cpu().numpy() - onnx_event))
    state_dec_err = np.max(np.abs(pt_state_dec.cpu().numpy() - onnx_state_dec))

    logger.info("Parity check results:")
    logger.info("  codec_encoder (acoustic_tokens): exact_match = %s", tokens_match)
    logger.info("  codec_encoder (state_out): L_inf = %.6e", state_enc_err)
    logger.info("  codec_decoder (audio_frame): L_inf = %.6e", audio_err)
    logger.info("  codec_decoder (event_trace): L_inf = %.6e", event_err)
    logger.info("  codec_decoder (state_out): L_inf = %.6e", state_dec_err)

    if not tokens_match:
        raise AssertionError("acoustic_tokens mismatch")
    if state_enc_err > PARITY_THRESHOLD:
        raise AssertionError(f"encoder state parity failed: {state_enc_err:.6e}")
    if audio_err > PARITY_THRESHOLD:
        raise AssertionError(f"audio_frame parity failed: {audio_err:.6e}")

    logger.info("All parity checks passed!")


def export_codec(
    checkpoint_path: Optional[str | Path] = None,
    output_dir: str | Path = "models/fp32",
    device: str = "cpu",
    opset_version: int = OPSET_VERSION,
    verify: bool = True,
) -> dict[str, Path]:
    """Export Emotion-Aware Codec to ONNX.

    Args:
        checkpoint_path: Path to codec checkpoint (optional, creates new model if None).
        output_dir: Directory to save ONNX files.
        device: Device for export.
        opset_version: ONNX opset version.
        verify: Whether to run parity verification.

    Returns:
        Dict with paths to exported ONNX files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating EmotionAwareCodec model")
    codec = EmotionAwareCodec(
        d_model=512,
        n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE,
        control_vocab_size=CONTROL_VOCAB_SIZE,
        d_voice_state=D_VOICE_STATE,
    ).to(device)

    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            codec.load_state_dict(ckpt["model_state_dict"])
        else:
            codec.load_state_dict(ckpt)

    codec.eval()

    onnx_paths: dict[str, Path] = {}

    onnx_paths["codec_encoder"] = export_codec_encoder(
        codec, output_dir, device, opset_version
    )

    onnx_paths["codec_decoder"] = export_codec_decoder(
        codec, output_dir, device, opset_version
    )

    if verify:
        logger.info("Running parity verification...")
        verify_codec_parity(codec, onnx_paths, device)

    logger.info("Export complete!")
    return onnx_paths


def main():
    parser = argparse.ArgumentParser(description="Export Emotion-Aware Codec to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/fp32", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for export")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip parity verification"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    export_codec(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
