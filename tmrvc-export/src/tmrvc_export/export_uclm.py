"""Export Disentangled UCLM model to ONNX format for real-time inference.

Exports three ONNX models for the streaming inference pipeline:
    - vc_encoder.onnx: source_A_t -> content_features
    - voice_state_enc.onnx: explicit_state, ssl_state, delta_state -> state_cond
    - uclm_core.onnx: content_features, b_ctx, spk_embed, state_cond, cfg_scale, kv_cache_in -> logits_a, logits_b, kv_cache_out

Usage:
    uv run tmrvc-export-uclm \\
        --checkpoint checkpoints/uclm/best.pt \\
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

from tmrvc_core.constants import (
    CONTROL_VOCAB_SIZE,
    D_ACTING_LATENT,
    D_MODEL,
    D_SPEAKER,
    D_VOICE_STATE_EXPLICIT,
    D_VOICE_STATE_SSL,
    N_CODEBOOKS,
    RVQ_VOCAB_SIZE,
    UCLM_N_HEADS,
    UCLM_N_LAYERS,
    UCLM_VQ_BINS,
    UCLM_TEXT_VOCAB_SIZE,
)
from tmrvc_train.models.uclm_model import DisentangledUCLM, PointerHead, ProsodyPredictor

logger = logging.getLogger(__name__)

OPSET_VERSION = 18
PARITY_THRESHOLD = 1e-4


class VCEncoderExportWrapper(nn.Module):
    """Wrapper for VCEncoder ONNX export.

    Inputs:
        source_A_t: [B, 8, L] - source acoustic tokens (8 codebooks)

    Outputs:
        vq_content_features: [B, d_model, L] - VQ-bottlenecked content features
    """

    def __init__(self, model: DisentangledUCLM):
        super().__init__()
        self.vc_encoder = model.vc_encoder

    def forward(self, source_A_t: torch.Tensor) -> torch.Tensor:
        content_features, _ = self.vc_encoder(source_A_t)
        return content_features.transpose(1, 2)


class VoiceStateEncExportWrapper(nn.Module):
    """Wrapper for VoiceStateEncoder ONNX export.

    Per-frame version: takes single-frame inputs including delta_state.

    Inputs:
        explicit_state: [B, 12] - heuristic voice parameters (12-D physical controls)
        ssl_state: [B, 128] - WavLM latent style features
        delta_state: [B, 12] - voice_state_t - voice_state_{t-1}

    Outputs:
        state_cond: [B, d_model] - fused style condition
    """

    def __init__(self, model: DisentangledUCLM):
        super().__init__()
        self.voice_state_enc = model.voice_state_enc

    def forward(
        self,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> torch.Tensor:
        explicit_state = explicit_state.unsqueeze(1)
        ssl_state = ssl_state.unsqueeze(1)
        delta_state = delta_state.unsqueeze(1)
        state_cond = self.voice_state_enc(explicit_state, ssl_state, delta_state)
        return state_cond.squeeze(1)


class UCLMCoreExportWrapper(nn.Module):
    """Wrapper for UCLM Core Transformer ONNX export with KV Cache and CFG.

    Single-frame output version: returns only the last frame logits.

    Inputs:
        content_features: [B, d_model, L] - VQ content features or text features
        b_ctx: [B, 4, L] - control token context [op, type, dur, int]
        spk_embed: [B, 192] - speaker embedding
        state_cond: [B, d_model] - voice state condition (single frame)
        acting_texture_latent: [B, D_ACTING_LATENT] - 24-D acting texture latent (zeros if unused)
        cfg_scale: [1] - CFG amplification scale
        kv_cache_in: [B, kv_cache_size] - flattened KV cache (zeros for first frame)

    Outputs:
        logits_a: [B, 8, 1024] - next acoustic token logits
        logits_b: [B, 4, 64] - next control token logits
        kv_cache_out: [B, kv_cache_size] - updated KV cache
    """

    def __init__(self, model: DisentangledUCLM, max_seq_len: int = 200):
        super().__init__()
        self.uclm_core = model.uclm_core
        self.acting_latent_conditioner = model.acting_latent_conditioner
        self.max_seq_len = max_seq_len
        self.n_heads = model.uclm_core.n_heads
        self.n_layers = model.uclm_core.n_layers
        self.d_model = model.uclm_core.d_model
        self.head_dim = self.d_model // self.n_heads

        self.kv_cache_size = (
            2 * self.n_layers * self.n_heads * self.head_dim * max_seq_len
        )

        # Precompute conditioner output for zero input (bias baseline).
        # Subtracted in forward() so that zeros truly produce zero conditioning,
        # which is required for correct CFG unconditional semantics in the
        # Rust runtime where None is not representable.
        with torch.no_grad():
            zero_latent = torch.zeros(1, D_ACTING_LATENT)
            self.register_buffer(
                "_act_zero_baseline",
                self.acting_latent_conditioner(zero_latent),
            )

    def forward(
        self,
        content_features: torch.Tensor,
        b_ctx: torch.Tensor,
        spk_embed: torch.Tensor,
        state_cond: torch.Tensor,
        acting_texture_latent: torch.Tensor,
        cfg_scale: torch.Tensor,
        kv_cache_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ONNX forward pass.

        Args:
            content_features: [B, d_model, L]
            b_ctx: [B, 4, L]
            spk_embed: [B, 192]
            state_cond: [B, d_model]
            acting_texture_latent: [B, D_ACTING_LATENT] - 24-D acting texture latent (zeros = no acting)
            cfg_scale: [1]
            kv_cache_in: [B, kv_cache_size] - Flattened KV cache
        """
        B, D, L = content_features.shape

        # v4: Apply acting texture latent conditioning unconditionally.
        # Subtract the zero-input baseline so that zeros truly produce zero
        # conditioning (the MLP has biased Linear layers whose f(0) ≠ 0).
        # This makes the ONNX "zeros = disabled" contract match the Python
        # "None = disabled" contract used by CFG unconditional passes.
        act_cond = self.acting_latent_conditioner(acting_texture_latent) - self._act_zero_baseline  # [B, d_model]
        content_features = content_features + act_cond.unsqueeze(2)

        # Now uses the improved CodecTransformer which supports flattened tensor cache
        logits_a_full, logits_b_full, kv_cache_out = self.uclm_core(
            content_features.transpose(1, 2),
            b_ctx,
            spk_embed,
            state_cond,
            cfg_scale.item(),
            kv_cache_in,
            self.max_seq_len
        )

        logits_a = logits_a_full[:, :, -1, :]
        logits_b = logits_b_full[:, :, -1, :]

        return logits_a, logits_b, kv_cache_out


class PointerHeadExportWrapper(nn.Module):
    """Wraps PointerHead for ONNX export.

    Inputs:
        hidden_states: [B, T, d_model] - transformer hidden states

    Outputs:
        advance_logit: [B, T, 1] - advance vs hold logit
        progress_delta: [B, T, 1] - pointer progress (sigmoid, 0-1)
        boundary_confidence: [B, T, 1] - boundary confidence (sigmoid, 0-1)
    """

    def __init__(self, pointer_head: PointerHead):
        super().__init__()
        self.pointer_head = pointer_head

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pointer_head(hidden_states)


class ProsodyPredictorExportWrapper(nn.Module):
    """Wraps ProsodyPredictor for ONNX export (inference mode only).

    The ODE integration loop is unrolled at export time with a fixed number
    of Euler steps so that the entire computation graph is captured by the
    ONNX tracer without dynamic control flow.

    Inputs:
        text_features: [B, L, d_model] - text encoder output
        dialogue_context: [B, d_model] - optional dialogue context (zeros if unused)
        speaker_embed: [B, d_model] - optional speaker embedding (zeros if unused)

    Outputs:
        prosody_latent: [B, d_prosody] - predicted prosody latent
    """

    def __init__(self, predictor: ProsodyPredictor, num_steps: int = 4):
        super().__init__()
        self.num_steps = num_steps
        self.d_prosody = predictor.d_prosody
        # Copy sub-modules so the wrapper owns them for tracing
        self.time_embed = predictor.time_embed
        self.velocity_net = predictor.velocity_net
        self.context_proj = predictor.context_proj

    def forward(
        self,
        text_features: torch.Tensor,
        dialogue_context: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        # Build conditioning (mirrors ProsodyPredictor._build_condition)
        h = text_features.mean(dim=1)  # [B, d_model]
        # dialogue_context: [B, D] -- add if non-zero
        if dialogue_context.shape[-1] < h.shape[-1]:
            dc = torch.nn.functional.pad(
                dialogue_context, (0, h.shape[-1] - dialogue_context.shape[-1])
            )
        else:
            dc = dialogue_context[:, : h.shape[-1]]
        h = h + self.context_proj(dc)
        # speaker_embed
        if speaker_embed.shape[-1] != h.shape[-1]:
            se = torch.nn.functional.pad(
                speaker_embed, (0, h.shape[-1] - speaker_embed.shape[-1])
            )
        else:
            se = speaker_embed
        h = h + se

        B = h.shape[0]
        # Euler ODE integration (unrolled for ONNX tracing)
        x = torch.zeros(B, self.d_prosody, device=h.device)
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=h.device)
            t_emb = self.time_embed(t)
            h_cond = h + t_emb
            inp = torch.cat([x, h_cond], dim=-1)
            v = self.velocity_net(inp)
            x = x + dt * v
        return x


def export_pointer_head(
    model: DisentangledUCLM,
    output_dir: Path,
    device: str,
    opset_version: int,
) -> Path:
    """Export PointerHead to ONNX."""
    wrapper = PointerHeadExportWrapper(model.pointer_head).eval()
    onnx_path = output_dir / "pointer_head.onnx"

    d_model = model.d_model
    dummy_hidden = torch.zeros(1, 10, d_model, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_hidden,),
        onnx_path,
        input_names=["hidden_states"],
        output_names=["advance_logit", "progress_delta", "boundary_confidence"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "seq_len"},
            "advance_logit": {0: "batch", 1: "seq_len"},
            "progress_delta": {0: "batch", 1: "seq_len"},
            "boundary_confidence": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported pointer_head to %s", onnx_path)
    return onnx_path


def export_prosody_predictor(
    model: DisentangledUCLM,
    output_dir: Path,
    device: str,
    opset_version: int,
    num_ode_steps: int = 4,
) -> Path:
    """Export ProsodyPredictor to ONNX (inference mode, unrolled ODE)."""
    wrapper = ProsodyPredictorExportWrapper(
        model.prosody_predictor, num_steps=num_ode_steps
    ).eval()
    onnx_path = output_dir / "prosody_predictor.onnx"

    d_model = model.d_model
    d_prosody = model.prosody_predictor.d_prosody

    dummy_text_features = torch.zeros(1, 20, d_model, device=device)
    dummy_dialogue_ctx = torch.zeros(1, d_model, device=device)
    dummy_speaker_embed = torch.zeros(1, d_model, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_text_features, dummy_dialogue_ctx, dummy_speaker_embed),
        onnx_path,
        input_names=["text_features", "dialogue_context", "speaker_embed"],
        output_names=["prosody_latent"],
        dynamic_axes={
            "text_features": {0: "batch", 1: "seq_len"},
            "dialogue_context": {0: "batch"},
            "speaker_embed": {0: "batch"},
            "prosody_latent": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported prosody_predictor to %s (ode_steps=%d)", onnx_path, num_ode_steps)
    return onnx_path


def export_vc_encoder(
    model: DisentangledUCLM,
    output_dir: Path,
    context_frames: int,
    device: str,
    opset_version: int,
) -> Path:
    """Export VCEncoder to ONNX."""
    wrapper = VCEncoderExportWrapper(model).eval()
    onnx_path = output_dir / "vc_encoder.onnx"

    dummy_source_A_t = torch.zeros(
        1, N_CODEBOOKS, context_frames, dtype=torch.long, device=device
    )

    torch.onnx.export(
        wrapper,
        (dummy_source_A_t,),
        onnx_path,
        input_names=["source_A_t"],
        output_names=["vq_content_features"],
        dynamic_axes={
            "source_A_t": {0: "batch", 2: "L"},
            "vq_content_features": {0: "batch", 2: "L"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported vc_encoder to %s", onnx_path)
    return onnx_path


def export_voice_state_enc(
    model: DisentangledUCLM,
    output_dir: Path,
    device: str,
    opset_version: int,
) -> Path:
    """Export VoiceStateEncoder to ONNX with delta_state input."""
    wrapper = VoiceStateEncExportWrapper(model).eval()
    onnx_path = output_dir / "voice_state_enc.onnx"

    dummy_explicit = torch.zeros(1, D_VOICE_STATE_EXPLICIT, device=device)
    dummy_ssl = torch.zeros(1, D_VOICE_STATE_SSL, device=device)
    dummy_delta = torch.zeros(1, D_VOICE_STATE_EXPLICIT, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_explicit, dummy_ssl, dummy_delta),
        onnx_path,
        input_names=["explicit_state", "ssl_state", "delta_state"],
        output_names=["state_cond"],
        dynamic_axes={
            "explicit_state": {0: "batch"},
            "ssl_state": {0: "batch"},
            "delta_state": {0: "batch"},
            "state_cond": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported voice_state_enc to %s", onnx_path)
    return onnx_path


def export_uclm_core(
    model: DisentangledUCLM,
    output_dir: Path,
    context_frames: int,
    device: str,
    opset_version: int,
) -> tuple[Path, int]:
    """Export UCLM Core Transformer to ONNX with KV Cache and CFG."""
    wrapper = UCLMCoreExportWrapper(model, max_seq_len=context_frames).eval()
    onnx_path = output_dir / "uclm_core.onnx"

    kv_cache_size = wrapper.kv_cache_size

    dummy_content = torch.zeros(1, D_MODEL, context_frames, device=device)
    dummy_b_ctx = torch.zeros(1, 4, context_frames, dtype=torch.long, device=device)
    dummy_spk = torch.zeros(1, D_SPEAKER, device=device)
    dummy_state_cond = torch.zeros(1, D_MODEL, device=device)
    dummy_acting = torch.zeros(1, D_ACTING_LATENT, device=device)
    dummy_cfg_scale = torch.tensor([1.5], device=device)
    dummy_kv_cache = torch.zeros(1, kv_cache_size, device=device)

    torch.onnx.export(
        wrapper,
        (
            dummy_content,
            dummy_b_ctx,
            dummy_spk,
            dummy_state_cond,
            dummy_acting,
            dummy_cfg_scale,
            dummy_kv_cache,
        ),
        onnx_path,
        input_names=[
            "content_features",
            "b_ctx",
            "spk_embed",
            "state_cond",
            "acting_texture_latent",
            "cfg_scale",
            "kv_cache_in",
        ],
        output_names=["logits_a", "logits_b", "kv_cache_out"],
        dynamic_axes={
            "content_features": {0: "batch", 2: "L"},
            "b_ctx": {0: "batch", 2: "L"},
            "spk_embed": {0: "batch"},
            "state_cond": {0: "batch"},
            "acting_texture_latent": {0: "batch"},
            "cfg_scale": {},
            "kv_cache_in": {0: "batch"},
            "logits_a": {0: "batch"},
            "logits_b": {0: "batch"},
            "kv_cache_out": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info("Exported uclm_core to %s (kv_cache_size=%d)", onnx_path, kv_cache_size)
    return onnx_path, kv_cache_size


def verify_parity(
    model: DisentangledUCLM,
    onnx_paths: dict[str, Path],
    context_frames: int,
    kv_cache_size: int,
    device: str,
) -> None:
    """Verify ONNX outputs match PyTorch outputs within tolerance."""
    model.eval()

    with torch.no_grad():
        dummy_source_A_t = torch.zeros(
            1, N_CODEBOOKS, context_frames, dtype=torch.long, device=device
        )
        dummy_explicit = torch.zeros(1, D_VOICE_STATE_EXPLICIT, device=device)
        dummy_ssl = torch.zeros(1, D_VOICE_STATE_SSL, device=device)
        dummy_delta = torch.zeros(1, D_VOICE_STATE_EXPLICIT, device=device)
        dummy_spk = torch.zeros(1, D_SPEAKER, device=device)
        dummy_cfg_scale = torch.tensor([1.5], device=device)
        dummy_kv_cache = torch.zeros(1, kv_cache_size, device=device)

        pt_content, _ = model.vc_encoder(dummy_source_A_t)
        pt_content = pt_content.transpose(1, 2)

        pt_explicit = dummy_explicit.unsqueeze(1)
        pt_ssl = dummy_ssl.unsqueeze(1)
        pt_delta = dummy_delta.unsqueeze(1)
        pt_state_cond = model.voice_state_enc(pt_explicit, pt_ssl, pt_delta)
        pt_state_cond_frame = pt_state_cond.squeeze(1)

        pt_state_cond_expanded = pt_state_cond_frame.unsqueeze(1).expand(
            -1, context_frames, -1
        )
        pt_logits_a, pt_logits_b, _ = model.uclm_core(
            pt_content.transpose(1, 2),
            pt_state_cond_expanded,
            dummy_spk,
            1.5,
            None,
            context_frames,
        )
        pt_logits_a = pt_logits_a[:, :, -1, :]
        pt_logits_b = pt_logits_b[:, :, -1, :]

    sess_vc = ort.InferenceSession(
        str(onnx_paths["vc_encoder"]), providers=["CPUExecutionProvider"]
    )
    sess_vs = ort.InferenceSession(
        str(onnx_paths["voice_state_enc"]), providers=["CPUExecutionProvider"]
    )
    sess_core = ort.InferenceSession(
        str(onnx_paths["uclm_core"]), providers=["CPUExecutionProvider"]
    )

    onnx_content = sess_vc.run(
        None,
        {"source_A_t": dummy_source_A_t.cpu().numpy()},
    )[0]

    onnx_state_cond = sess_vs.run(
        None,
        {
            "explicit_state": dummy_explicit.cpu().numpy(),
            "ssl_state": dummy_ssl.cpu().numpy(),
            "delta_state": dummy_delta.cpu().numpy(),
        },
    )[0]

    dummy_b_ctx = torch.zeros(1, 4, context_frames, dtype=torch.long, device=device)
    onnx_logits_a, onnx_logits_b, onnx_kv_cache = sess_core.run(
        None,
        {
            "content_features": onnx_content,
            "b_ctx": dummy_b_ctx.cpu().numpy(),
            "spk_embed": dummy_spk.cpu().numpy(),
            "state_cond": onnx_state_cond,
            "cfg_scale": dummy_cfg_scale.cpu().numpy(),
            "kv_cache_in": dummy_kv_cache.cpu().numpy(),
        },
    )

    content_err = np.max(np.abs(pt_content.cpu().numpy() - onnx_content))
    state_cond_err = np.max(np.abs(pt_state_cond_frame.cpu().numpy() - onnx_state_cond))
    logits_a_err = np.max(np.abs(pt_logits_a.cpu().numpy() - onnx_logits_a))
    logits_b_err = np.max(np.abs(pt_logits_b.cpu().numpy() - onnx_logits_b))

    logger.info("Parity check results:")
    logger.info("  vc_encoder (content_features): L_inf = %.6e", content_err)
    logger.info("  voice_state_enc (state_cond): L_inf = %.6e", state_cond_err)
    logger.info("  uclm_core (logits_a): L_inf = %.6e", logits_a_err)
    logger.info("  uclm_core (logits_b): L_inf = %.6e", logits_b_err)
    logger.info("  uclm_core (kv_cache shape): %s", onnx_kv_cache.shape)
    logger.info(
        "  KV cache size: %d floats (%.2f KB)", kv_cache_size, kv_cache_size * 4 / 1024
    )

    if content_err > PARITY_THRESHOLD:
        raise AssertionError(
            f"vc_encoder parity failed: L_inf={content_err:.6e} > {PARITY_THRESHOLD}"
        )
    if state_cond_err > PARITY_THRESHOLD:
        raise AssertionError(
            f"voice_state_enc parity failed: L_inf={state_cond_err:.6e} > {PARITY_THRESHOLD}"
        )
    if logits_a_err > PARITY_THRESHOLD:
        raise AssertionError(
            f"uclm_core (logits_a) parity failed: L_inf={logits_a_err:.6e} > {PARITY_THRESHOLD}"
        )
    if logits_b_err > PARITY_THRESHOLD:
        raise AssertionError(
            f"uclm_core (logits_b) parity failed: L_inf={logits_b_err:.6e} > {PARITY_THRESHOLD}"
        )

    logger.info("All parity checks passed!")


def export_uclm(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    context_frames: int = 200,
    device: str = "cpu",
    opset_version: int = OPSET_VERSION,
    verify: bool = True,
) -> dict[str, Path]:
    """Export Disentangled UCLM model to ONNX.

    Args:
        checkpoint_path: Path to UCLM checkpoint.
        output_dir: Directory to save ONNX files.
        context_frames: Number of context frames (default: 200 = 2 sec).
        device: Device for export.
        opset_version: ONNX opset version.
        verify: Whether to run parity verification.

    Returns:
        Dict with paths to exported ONNX files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DisentangledUCLM from %s", checkpoint_path)
    model = DisentangledUCLM(
        d_model=D_MODEL,
        n_heads=UCLM_N_HEADS,
        n_layers=UCLM_N_LAYERS,
        rvq_vocab_size=RVQ_VOCAB_SIZE,
        n_codebooks=N_CODEBOOKS,
        control_vocab_size=CONTROL_VOCAB_SIZE,
        d_explicit=D_VOICE_STATE_EXPLICIT,
        d_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER,
        vq_bins=UCLM_VQ_BINS,
        vocab_size=UCLM_TEXT_VOCAB_SIZE,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    onnx_paths: dict[str, Path] = {}

    onnx_paths["vc_encoder"] = export_vc_encoder(
        model, output_dir, context_frames, device, opset_version
    )

    onnx_paths["voice_state_enc"] = export_voice_state_enc(
        model, output_dir, device, opset_version
    )

    onnx_paths["uclm_core"], kv_cache_size = export_uclm_core(
        model, output_dir, context_frames, device, opset_version
    )

    onnx_paths["pointer_head"] = export_pointer_head(
        model, output_dir, device, opset_version
    )

    onnx_paths["prosody_predictor"] = export_prosody_predictor(
        model, output_dir, device, opset_version
    )

    if verify:
        logger.info("Running parity verification...")
        verify_parity(model, onnx_paths, context_frames, kv_cache_size, device)

    logger.info("Export complete!")
    return onnx_paths


def main():
    parser = argparse.ArgumentParser(description="Export Disentangled UCLM to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/fp32", help="Output directory"
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=200,
        help="Context frames (default: 200 = 2 sec)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for export")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip parity verification"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    export_uclm(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        context_frames=args.context_frames,
        device=args.device,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
