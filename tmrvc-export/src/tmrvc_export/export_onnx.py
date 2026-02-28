"""Export Codec-Latent models to ONNX format (streaming mode)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from tmrvc_train.models.streaming_codec import (
    CodecConfig,
    StreamingCodec,
)
from tmrvc_train.models.token_model import (
    TokenModelConfig,
    TokenModel,
)
from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA

logger = logging.getLogger(__name__)

OPSET_VERSION = 18

ENCODER_STATE_DIM = 512
ENCODER_STATE_FRAMES = 32
DECODER_STATE_DIM = 256
DECODER_STATE_FRAMES = 32


class CodecEncoderWrapper(nn.Module):
    """Wraps encoder + RVQ for ONNX export with flat state tensor.

    Input: audio_frame [B, 1, 480], state_in [B, 512, 32]
    Output: tokens [B, 4] (int64), state_out [B, 512, 32]
    """

    def __init__(self, codec: StreamingCodec, config: CodecConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def forward(
        self, audio_frame: torch.Tensor, state_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, quantized, state_out = self.codec.encode(audio_frame, state_in)
        tokens = indices[:, :, -1].to(torch.int64)
        return tokens, state_out


class CodecDecoderWrapper(nn.Module):
    """Wraps RVQ decode + decoder for ONNX export with flat state tensor.

    Input: tokens [B, 4] (int64), state_in [B, 256, 32]
    Output: audio_frame [B, 1, 480], state_out [B, 256, 32]
    """

    def __init__(self, codec: StreamingCodec, config: CodecConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def forward(
        self, tokens: torch.Tensor, state_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        indices = tokens.unsqueeze(-1).expand(-1, -1, self.config.frame_size)
        audio, state_out = self.codec.decode(indices, state_in)
        return audio, state_out


class TokenModelWrapper(nn.Module):
    """Wraps Transformer token model for ONNX export with KV-cache.

    Input: tokens [B, K, L], spk_embed [B, 192], f0_condition [B, L, 2], kv_cache_flat [n_layers*2, B, n_heads, ctx, head_dim]
    Output: logits [B, K, vocab], kv_cache_flat_out [n_layers*2, B, n_heads, ctx, head_dim]
    """

    def __init__(self, model: TokenModel):
        super().__init__()
        self.model = model
        self.config = model.config
        self.head_dim = self.config.d_model // self.config.n_heads
        self.cache_shape = (
            self.config.n_layers * 2,
            1,
            self.config.n_heads,
            self.config.context_length,
            self.head_dim,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        spk_embed: torch.Tensor,
        f0_condition: torch.Tensor,
        kv_cache_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        device = tokens.device
        dtype = kv_cache_flat.dtype

        kv_caches = []
        for i in range(self.config.n_layers):
            k = kv_cache_flat[i * 2]
            v = kv_cache_flat[i * 2 + 1]
            if k.shape[2] > 0:
                kv_caches.append((k, v))
            else:
                kv_caches.append(None)

        logits, new_kv_caches = self.model(tokens, spk_embed, f0_condition, kv_caches)

        new_cache_list = []
        for kv in new_kv_caches:
            if kv is not None:
                k, v = kv
                new_cache_list.append(k)
                new_cache_list.append(v)
            else:
                new_cache_list.append(
                    torch.zeros(
                        B,
                        self.config.n_heads,
                        0,
                        self.head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                new_cache_list.append(
                    torch.zeros(
                        B,
                        self.config.n_heads,
                        0,
                        self.head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )

        max_len = self.config.context_length
        padded_cache = []
        for c in new_cache_list:
            if c.shape[2] < max_len:
                pad_len = max_len - c.shape[2]
                c = torch.cat(
                    [
                        c,
                        torch.zeros(
                            B,
                            self.config.n_heads,
                            pad_len,
                            self.head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                    ],
                    dim=2,
                )
            elif c.shape[2] > max_len:
                c = c[:, :, -max_len:]
            padded_cache.append(c)

        kv_cache_out = torch.stack(padded_cache, dim=0)

        return logits, kv_cache_out


class SpeakerEncoderWrapper(nn.Module):
    """Wraps speaker encoder for ONNX export.

    Input: mel_ref [B, 80, T]
    Output: spk_embed [B, 192]
    """

    def __init__(self, model: SpeakerEncoderWithLoRA):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        spk_embed, _ = self.model(mel)
        return spk_embed


def export_codec_encoder(
    codec: StreamingCodec,
    output_path: Path,
    config: CodecConfig,
) -> None:
    """Export codec encoder (with RVQ) to ONNX."""
    wrapper = CodecEncoderWrapper(codec, config)
    wrapper.eval()

    dummy_audio = torch.randn(1, 1, config.frame_size)
    dummy_state = torch.zeros(1, ENCODER_STATE_DIM, ENCODER_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_state),
        output_path,
        input_names=["audio_frame", "state_in"],
        output_names=["tokens", "state_out"],
        dynamic_axes={
            "audio_frame": {0: "batch", 2: "samples"},
            "tokens": {0: "batch"},
            "state_in": {0: "batch"},
            "state_out": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    logger.info(f"Exported codec_encoder to {output_path}")


def export_codec_decoder(
    codec: StreamingCodec,
    output_path: Path,
    config: CodecConfig,
) -> None:
    """Export codec decoder (with RVQ) to ONNX."""
    wrapper = CodecDecoderWrapper(codec, config)
    wrapper.eval()

    dummy_tokens = torch.randint(0, config.codebook_size, (1, config.n_codebooks))
    dummy_state = torch.zeros(1, DECODER_STATE_DIM, DECODER_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_state),
        output_path,
        input_names=["tokens", "state_in"],
        output_names=["audio_frame", "state_out"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "audio_frame": {0: "batch", 2: "samples"},
            "state_in": {0: "batch"},
            "state_out": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    logger.info(f"Exported codec_decoder to {output_path}")


def export_token_model(
    model: TokenModel,
    output_path: Path,
    config: TokenModelConfig,
) -> None:
    """Export Transformer token model to ONNX."""
    wrapper = TokenModelWrapper(model)
    wrapper.eval()

    dummy_tokens = torch.randint(
        0, config.codebook_size, (1, config.n_codebooks, config.context_length)
    )
    dummy_spk = torch.randn(1, config.d_spk)
    dummy_f0 = torch.zeros(1, config.context_length, config.d_f0)
    head_dim = config.d_model // config.n_heads
    dummy_kv = torch.zeros(
        config.n_layers * 2, 1, config.n_heads, config.context_length, head_dim
    )

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_spk, dummy_f0, dummy_kv),
        output_path,
        input_names=["tokens_in", "spk_embed", "f0_condition", "kv_cache_in"],
        output_names=["logits", "kv_cache_out"],
        dynamic_axes={
            "tokens_in": {0: "batch", 2: "context"},
            "spk_embed": {0: "batch"},
            "f0_condition": {0: "batch", 1: "context"},
            "kv_cache_in": {1: "batch"},
            "logits": {0: "batch"},
            "kv_cache_out": {1: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    logger.info(f"Exported token_model to {output_path}")


def export_speaker_encoder(
    model: SpeakerEncoderWithLoRA,
    output_path: Path,
) -> None:
    """Export speaker encoder to ONNX."""
    wrapper = SpeakerEncoderWrapper(model)
    wrapper.eval()

    dummy_mel = torch.randn(1, 80, 100)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        output_path,
        input_names=["mel_ref"],
        output_names=["spk_embed"],
        dynamic_axes={
            "mel_ref": {0: "batch", 2: "time"},
            "spk_embed": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    logger.info(f"Exported speaker_encoder to {output_path}")


def export_all(
    codec: StreamingCodec,
    token_model: TokenModel,
    speaker_encoder: SpeakerEncoderWithLoRA,
    output_dir: Path,
    codec_config: CodecConfig,
    token_config: TokenModelConfig,
) -> None:
    """Export all models to ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)

    export_codec_encoder(codec, output_dir / "codec_encoder.onnx", codec_config)
    export_codec_decoder(codec, output_dir / "codec_decoder.onnx", codec_config)
    export_token_model(token_model, output_dir / "token_model.onnx", token_config)
    export_speaker_encoder(speaker_encoder, output_dir / "speaker_encoder.onnx")

    logger.info(f"All models exported to {output_dir}")
