"""Streaming inference for UCLM.

Provides frame-by-frame generation for real-time voice conversion.
Maintains context buffer for temporal coherence.

Usage:
    from tmrvc_train.models.streaming_uclm import StreamingUCLM

    streamer = StreamingUCLM(model, speaker_embed, context_frames=10)

    # Process frame by frame
    for source_frame in source_frames:
        output_tokens = streamer.process_frame(source_tokens, voice_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import N_CODEBOOKS, VOCAB_SIZE

from .uclm import UCLM, UCLMConfig


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""

    context_frames: int = 10  # Number of past frames to keep
    temperature: float = 1.0
    top_k: int = 50
    stream_mode: str = "vc"  # "vc" or "tts"


class StreamingUCLM:
    """Frame-by-frame streaming inference for UCLM.

    Maintains context buffer for temporal coherence across frames.
    Optimized for real-time operation.

    Args:
        model: Trained UCLM model.
        speaker_embed: Target speaker embedding [d_speaker].
        config: Streaming configuration.
    """

    def __init__(
        self,
        model: UCLM,
        speaker_embed: torch.Tensor,
        config: StreamingConfig | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.config = config or StreamingConfig()
        c = self.config

        self.speaker_embed = speaker_embed.unsqueeze(0)  # [1, d_speaker]

        # Context buffer for past tokens
        self.token_buffer: torch.Tensor | None = (
            None  # [1, n_codebooks, context_frames]
        )
        self.voice_state_buffer: torch.Tensor | None = None  # [1, context_frames, 8]

        self.device = next(model.parameters()).device

    def reset(self) -> None:
        """Reset internal state for new utterance."""
        self.token_buffer = None
        self.voice_state_buffer = None

    def process_frame(
        self,
        source_tokens: torch.Tensor | None = None,
        voice_state: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a single frame and generate output tokens.

        Args:
            source_tokens: [1, n_codebooks, 1] source tokens for VC mode.
            voice_state: [1, 1, 8] voice state for current frame.
            text_features: [1, L, d_text] text features for TTS mode.

        Returns:
            output_tokens: [1, n_codebooks, 1] generated tokens.
        """
        c = self.config

        # Ensure inputs are on correct device
        if source_tokens is not None:
            source_tokens = source_tokens.to(self.device)
        if voice_state is not None:
            voice_state = voice_state.to(self.device)
        if text_features is not None:
            text_features = text_features.to(self.device)

        # Update context buffers
        self._update_buffers(source_tokens, voice_state)

        # Build full context for generation
        if self.token_buffer is not None:
            past_tokens = self.token_buffer
            past_voice_state = self.voice_state_buffer
        else:
            past_tokens = source_tokens
            past_voice_state = voice_state

        with torch.no_grad():
            # Single token generation
            output = self.model.forward(
                text_features=text_features,
                source_tokens=source_tokens,
                voice_state=past_voice_state,
                speaker_embed=self.speaker_embed.to(self.device),
                past_tokens=None,  # Already in context
                target_tokens=past_tokens,
                mode=c.stream_mode,
            )

            # Sample from first codebook (AR)
            logits_ar = output["logits_ar"][:, -1, :] / c.temperature  # [1, vocab]
            top_k_logits, top_k_indices = torch.topk(logits_ar, c.top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            token_0 = top_k_indices.gather(-1, sampled).squeeze(-1)  # [1]

            # Sample from remaining codebooks (parallel)
            output_tokens = torch.zeros(
                1,
                self.model.config.n_codebooks,
                1,
                dtype=torch.long,
                device=self.device,
            )
            output_tokens[:, 0, 0] = token_0

            for i in range(self.model.config.n_codebooks - 1):
                logits = output["logits_parallel"][:, i, -1, :] / c.temperature
                top_k_logits, top_k_indices = torch.topk(logits, c.top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                sampled = torch.multinomial(probs, 1)
                token_i = top_k_indices.gather(-1, sampled).squeeze(-1)
                output_tokens[:, i + 1, 0] = token_i

        return output_tokens

    def _update_buffers(
        self,
        source_tokens: torch.Tensor | None,
        voice_state: torch.Tensor | None,
    ) -> None:
        """Update context buffers with new frame."""
        c = self.config

        if source_tokens is None:
            return

        if self.token_buffer is None:
            self.token_buffer = source_tokens.clone()
            if voice_state is not None:
                self.voice_state_buffer = voice_state.clone()
        else:
            self.token_buffer = torch.cat([self.token_buffer, source_tokens], dim=-1)
            if voice_state is not None and self.voice_state_buffer is not None:
                self.voice_state_buffer = torch.cat(
                    [self.voice_state_buffer, voice_state], dim=1
                )

            # Trim to context length
            if self.token_buffer.shape[-1] > c.context_frames:
                self.token_buffer = self.token_buffer[:, :, -c.context_frames :]
                if self.voice_state_buffer is not None:
                    self.voice_state_buffer = self.voice_state_buffer[
                        :, -c.context_frames :, :
                    ]

    def process_sequence(
        self,
        source_tokens: torch.Tensor,
        voice_state: torch.Tensor,
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process full sequence frame by frame.

        Args:
            source_tokens: [1, n_codebooks, T] source tokens.
            voice_state: [1, T, 8] voice state.
            text_features: [1, L, d_text] text features (TTS mode).

        Returns:
            output_tokens: [1, n_codebooks, T] generated tokens.
        """
        self.reset()

        n_frames = source_tokens.shape[-1]

        output_tokens = []

        for t in range(n_frames):
            src_t = source_tokens[:, :, t : t + 1]
            vs_t = voice_state[:, t : t + 1, :]

            out_t = self.process_frame(src_t, vs_t, text_features)
            output_tokens.append(out_t)

        return torch.cat(output_tokens, dim=-1)
