"""ConverterStudent: causal CNN with FiLM conditioning, distilled from Teacher."""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    CONVERTER_HQ_STATE_FRAMES,
    CONVERTER_STATE_FRAMES,
    D_CONTENT,
    D_CONVERTER_HIDDEN,
    D_SPEAKER,
    D_VOCODER_FEATURES,
    GTM_D_ENTRY,
    GTM_N_ENTRIES,
    GTM_N_HEADS,
    MAX_LOOKAHEAD_HOPS,
    N_ACOUSTIC_PARAMS,
    N_STYLE_PARAMS,
)
from tmrvc_train.modules import (
    CausalConvNeXtBlock,
    FiLMConditioner,
    GlobalTimbreMemory,
    SemiCausalConvNeXtBlock,
    TimbreCrossAttention,
)


class ConverterBlock(nn.Module):
    """CausalConvNeXtBlock + FiLM conditioning."""

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        kernel_size: int = 7,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv_block = CausalConvNeXtBlock(
            d_model, kernel_size=kernel_size, dilation=dilation,
        )
        self.film = FiLMConditioner(d_cond, d_model)

    @property
    def context_size(self) -> int:
        return self.conv_block.context_size

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, state_out = self.conv_block(x, state_in)
        x = self.film(x, cond)
        return x, state_out


class ConverterStudent(nn.Module):
    """1-step converter distilled from Teacher U-Net.

    8 CausalConvNeXt blocks with FiLM, d=384, k=3, dilation=[1,1,2,2,4,4,6,6].
    State context: (k-1)*d per block = 2+2+4+4+8+8+12+12 = 52 frames.
    Input: content[B, 256, T] + conditioning (spk_embed + acoustic_params).
    Output: pred_features[B, 513, T].
    """

    def __init__(
        self,
        d_input: int = D_CONTENT,
        d_model: int = D_CONVERTER_HIDDEN,
        d_output: int = D_VOCODER_FEATURES,
        d_speaker: int = D_SPEAKER,
        n_acoustic_params: int = N_ACOUSTIC_PARAMS,
        n_blocks: int = 8,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        d_cond = d_speaker + n_acoustic_params  # 192 + 32 = 224
        dilations = dilations or [1, 1, 2, 2, 4, 4, 6, 6]
        assert len(dilations) == n_blocks

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.SiLU(),
        )

        # Converter blocks with FiLM
        self.blocks = nn.ModuleList([
            ConverterBlock(d_model, d_cond, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

        # Output projection → STFT features
        self.output_proj = nn.Conv1d(d_model, d_output, kernel_size=1)

        # Precompute state slicing
        self._state_sizes = [blk.context_size for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == CONVERTER_STATE_FRAMES, (
            f"Expected total state {CONVERTER_STATE_FRAMES}, got {self._total_state}"
        )

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            content: ``[B, 256, T]`` content features.
            spk_embed: ``[B, 192]`` speaker embedding.
            acoustic_params: ``[B, 32]`` acoustic condition parameters.
            state_in: ``[B, d_model, total_state]`` streaming state or None.

        Returns:
            Tuple of ``(pred_features [B, 513, T], state_out or None)``.
        """
        cond = torch.cat([spk_embed, acoustic_params], dim=-1)  # [B, 224]

        x = self.input_proj(content)  # [B, d_model, T]

        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states = [None] * len(self.blocks)

        new_states = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, cond, s_in)
            new_states.append(s_out)

        x = self.output_proj(x)  # [B, 513, T]

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return x, state_out
        return x, None

    def _split_state(self, state: torch.Tensor) -> list[torch.Tensor]:
        states = []
        offset = 0
        for size in self._state_sizes:
            if size > 0:
                states.append(state[:, :, offset:offset + size])
            else:
                states.append(state[:, :, :0])
            offset += size
        return states

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self._total_state, device=device)


class ConverterBlockGTM(nn.Module):
    """CausalConvNeXtBlock + TimbreCrossAttention(speaker) + FiLM(acoustic)."""

    def __init__(
        self,
        d_model: int,
        n_acoustic_params: int,
        d_entry: int = GTM_D_ENTRY,
        n_heads: int = GTM_N_HEADS,
        kernel_size: int = 7,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv_block = CausalConvNeXtBlock(
            d_model, kernel_size=kernel_size, dilation=dilation,
        )
        self.timbre_attn = TimbreCrossAttention(d_model, d_entry, n_heads)
        self.film_acoustic = FiLMConditioner(n_acoustic_params, d_model)

    @property
    def context_size(self) -> int:
        return self.conv_block.context_size

    def forward(
        self,
        x: torch.Tensor,
        timbre_memory: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, state_out = self.conv_block(x, state_in)
        x = x + self.timbre_attn(x, timbre_memory)
        x = self.film_acoustic(x, acoustic_params)
        return x, state_out


class ConverterStudentGTM(nn.Module):
    """Converter with Global Timbre Memory (cross-attention on speaker).

    Same ONNX I/O contract as ConverterStudent: the GTM expansion happens
    internally from spk_embed, so the external interface is unchanged.

    8 CausalConvNeXt blocks with TimbreCrossAttention + FiLM(acoustic),
    d=384, k=3, dilation=[1,1,2,2,4,4,6,6].
    State: same 52-frame budget as ConverterStudent.
    """

    def __init__(
        self,
        d_input: int = D_CONTENT,
        d_model: int = D_CONVERTER_HIDDEN,
        d_output: int = D_VOCODER_FEATURES,
        d_speaker: int = D_SPEAKER,
        n_acoustic_params: int = N_ACOUSTIC_PARAMS,
        n_blocks: int = 8,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        gtm_n_entries: int = GTM_N_ENTRIES,
        gtm_d_entry: int = GTM_D_ENTRY,
        gtm_n_heads: int = GTM_N_HEADS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        dilations = dilations or [1, 1, 2, 2, 4, 4, 6, 6]
        assert len(dilations) == n_blocks

        # Global Timbre Memory: spk_embed → memory bank
        self.gtm = GlobalTimbreMemory(d_speaker, gtm_n_entries, gtm_d_entry)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.SiLU(),
        )

        # GTM converter blocks
        self.blocks = nn.ModuleList([
            ConverterBlockGTM(
                d_model, n_acoustic_params,
                d_entry=gtm_d_entry, n_heads=gtm_n_heads,
                kernel_size=kernel_size, dilation=d,
            )
            for d in dilations
        ])

        # Output projection
        self.output_proj = nn.Conv1d(d_model, d_output, kernel_size=1)

        # State size accounting
        self._state_sizes = [blk.context_size for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == CONVERTER_STATE_FRAMES, (
            f"Expected total state {CONVERTER_STATE_FRAMES}, got {self._total_state}"
        )

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass (same signature as ConverterStudent).

        Args:
            content: ``[B, 256, T]`` content features.
            spk_embed: ``[B, 192]`` speaker embedding.
            acoustic_params: ``[B, 32]`` acoustic condition parameters.
            state_in: ``[B, d_model, total_state]`` streaming state or None.

        Returns:
            Tuple of ``(pred_features [B, 513, T], state_out or None)``.
        """
        # Expand speaker embedding to memory bank
        timbre_memory = self.gtm(spk_embed)  # [B, N, d_entry]

        x = self.input_proj(content)  # [B, d_model, T]

        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states = [None] * len(self.blocks)

        new_states = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, timbre_memory, acoustic_params, s_in)
            new_states.append(s_out)

        x = self.output_proj(x)  # [B, 513, T]

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return x, state_out
        return x, None

    def _split_state(self, state: torch.Tensor) -> list[torch.Tensor]:
        states = []
        offset = 0
        for size in self._state_sizes:
            if size > 0:
                states.append(state[:, :, offset:offset + size])
            else:
                states.append(state[:, :, :0])
            offset += size
        return states

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self._total_state, device=device)


# ---------------------------------------------------------------------------
# HQ (Semi-Causal) Converter
# ---------------------------------------------------------------------------


def _compute_right_contexts(dilations: list[int], max_lookahead: int) -> list[int]:
    """Greedy allocation of right_context to blocks.

    Distributes ``max_lookahead`` frames across blocks using
    ``min(dilation, remaining)`` per block.
    """
    right_ctxs: list[int] = []
    remaining = max_lookahead
    for d in dilations:
        rc = min(d, remaining)
        right_ctxs.append(rc)
        remaining -= rc
    return right_ctxs


class ConverterBlockHQ(nn.Module):
    """SemiCausalConvNeXtBlock + FiLM conditioning."""

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        kernel_size: int = 3,
        dilation: int = 1,
        right_context: int = 0,
    ) -> None:
        super().__init__()
        self.conv_block = SemiCausalConvNeXtBlock(
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            right_context=right_context,
        )
        self.film = FiLMConditioner(d_cond, d_model)

    @property
    def left_context(self) -> int:
        return self.conv_block.left_context

    @property
    def right_context(self) -> int:
        return self.conv_block.right_context

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, state_out = self.conv_block(x, state_in)
        x = self.film(x, cond)
        return x, state_out


class ConverterStudentHQ(nn.Module):
    """Semi-causal converter for HQ mode.

    Streaming: T_in = 1 + max_lookahead, T_out = 1.
    Training: T_in = T, T_out = T (padding preserves length).

    8 SemiCausalConvNeXt blocks, d=384, k=3, dilation=[1,1,2,2,4,4,6,6].
    right_context greedy allocation: [1,1,2,2,0,0,0,0] for L=6.
    State: sum(left_ctx) = 46 frames.

    Weight structure is identical to ConverterStudent — can init from
    causal weights via :meth:`from_causal`.
    """

    def __init__(
        self,
        d_input: int = D_CONTENT,
        d_model: int = D_CONVERTER_HIDDEN,
        d_output: int = D_VOCODER_FEATURES,
        d_speaker: int = D_SPEAKER,
        n_acoustic_params: int = N_ACOUSTIC_PARAMS,
        n_blocks: int = 8,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        max_lookahead: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_lookahead = max_lookahead
        d_cond = d_speaker + n_acoustic_params  # 224
        dilations = dilations or [1, 1, 2, 2, 4, 4, 6, 6]
        assert len(dilations) == n_blocks

        right_ctxs = _compute_right_contexts(dilations, max_lookahead)

        # Input projection (same structure as ConverterStudent)
        self.input_proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.SiLU(),
        )

        # HQ converter blocks with semi-causal padding
        self.blocks = nn.ModuleList([
            ConverterBlockHQ(
                d_model, d_cond,
                kernel_size=kernel_size, dilation=d,
                right_context=rc,
            )
            for d, rc in zip(dilations, right_ctxs)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(d_model, d_output, kernel_size=1)

        # Precompute state slicing (left_context per block)
        self._state_sizes = [blk.left_context for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == CONVERTER_HQ_STATE_FRAMES, (
            f"Expected total HQ state {CONVERTER_HQ_STATE_FRAMES}, got {self._total_state}"
        )

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            content: ``[B, 256, T]`` content features.
                Training: any T. Streaming: T = 1 + max_lookahead.
            spk_embed: ``[B, 192]`` speaker embedding.
            acoustic_params: ``[B, 32]`` acoustic condition parameters.
            state_in: ``[B, d_model, total_state]`` streaming state or None.

        Returns:
            Training: ``(pred_features [B, 513, T], None)``.
            Streaming: ``(pred_features [B, 513, 1], state_out)``.
        """
        cond = torch.cat([spk_embed, acoustic_params], dim=-1)  # [B, 224]

        x = self.input_proj(content)  # [B, d_model, T]

        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states = [None] * len(self.blocks)

        new_states = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, cond, s_in)
            new_states.append(s_out)

        x = self.output_proj(x)  # [B, 513, T_out]

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return x, state_out
        return x, None

    def _split_state(self, state: torch.Tensor) -> list[torch.Tensor]:
        states = []
        offset = 0
        for size in self._state_sizes:
            if size > 0:
                states.append(state[:, :, offset:offset + size])
            else:
                states.append(state[:, :, :0])
            offset += size
        return states

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self._total_state, device=device)

    @classmethod
    def from_causal(cls, causal_model: ConverterStudent) -> ConverterStudentHQ:
        """Initialize HQ model from trained causal ConverterStudent weights.

        The conv weights, norm, and pointwise layers have identical structure,
        so they can be directly copied. Fine-tuning adapts the model to
        semi-causal padding.
        """
        # Infer conditioning dimensions from causal model's FiLM layer
        d_cond = causal_model.blocks[0].film.proj.in_features  # d_speaker + n_acoustic_params
        d_speaker_inferred = d_cond - N_ACOUSTIC_PARAMS
        hq = cls(
            d_input=causal_model.input_proj[0].in_channels,
            d_model=causal_model.d_model,
            d_output=causal_model.output_proj.out_channels,
            d_speaker=d_speaker_inferred,
            n_acoustic_params=N_ACOUSTIC_PARAMS,
            n_blocks=len(causal_model.blocks),
            kernel_size=causal_model.blocks[0].conv_block.kernel_size,
            dilations=[blk.conv_block.dilation for blk in causal_model.blocks],
            max_lookahead=MAX_LOOKAHEAD_HOPS,
        )
        # Copy input/output projections
        hq.input_proj.load_state_dict(causal_model.input_proj.state_dict())
        hq.output_proj.load_state_dict(causal_model.output_proj.state_dict())

        # Copy block weights (CausalConvNeXtBlock → SemiCausalConvNeXtBlock)
        for hq_block, causal_block in zip(hq.blocks, causal_model.blocks):
            hq_block.conv_block.load_state_dict(
                causal_block.conv_block.state_dict(),
            )
            hq_block.film.load_state_dict(causal_block.film.state_dict())

        return hq


# ---------------------------------------------------------------------------
# TTS-mode Converter (extended FiLM conditioning)
# ---------------------------------------------------------------------------


def migrate_film_weights(
    src_film: FiLMConditioner,
    dst_film: FiLMConditioner,
) -> None:
    """Copy FiLM weights from smaller d_cond to larger d_cond.

    The extra conditioning dimensions are zero-initialized so they
    produce identity modulation (gamma=1, beta=0) until fine-tuned.

    Args:
        src_film: Source FiLM with d_cond_old (e.g., 224).
        dst_film: Destination FiLM with d_cond_new (e.g., 256).
    """
    d_old = src_film.proj.in_features
    d_new = dst_film.proj.in_features
    assert d_new >= d_old, f"dst d_cond ({d_new}) must be >= src d_cond ({d_old})"

    with torch.no_grad():
        # Zero out all weights first, then copy existing columns
        dst_film.proj.weight.zero_()
        dst_film.proj.weight[:, :d_old] = src_film.proj.weight
        # Copy bias as-is (same output dimension)
        dst_film.proj.bias.copy_(src_film.proj.bias)


def converter_from_vc_checkpoint(
    vc_model: ConverterStudent,
    n_style_params: int = N_STYLE_PARAMS,
) -> ConverterStudent:
    """Create a TTS-mode ConverterStudent from a VC checkpoint.

    Extends FiLM conditioning from d_speaker + n_acoustic_params (224)
    to d_speaker + n_style_params (256). Existing weights are preserved;
    new style dimensions are zero-initialized for identity modulation.

    Args:
        vc_model: Trained VC ConverterStudent.
        n_style_params: New conditioning dimension (default: 64).

    Returns:
        New ConverterStudent with extended FiLM conditioning.
    """
    tts_model = ConverterStudent(
        d_input=vc_model.input_proj[0].in_channels,
        d_model=vc_model.d_model,
        d_output=vc_model.output_proj.out_channels,
        d_speaker=D_SPEAKER,
        n_acoustic_params=n_style_params,
        n_blocks=len(vc_model.blocks),
        kernel_size=vc_model.blocks[0].conv_block.kernel_size,
        dilations=[blk.conv_block.dilation for blk in vc_model.blocks],
    )

    # Copy input/output projections
    tts_model.input_proj.load_state_dict(vc_model.input_proj.state_dict())
    tts_model.output_proj.load_state_dict(vc_model.output_proj.state_dict())

    # Copy block weights with FiLM migration
    for tts_block, vc_block in zip(tts_model.blocks, vc_model.blocks):
        tts_block.conv_block.load_state_dict(vc_block.conv_block.state_dict())
        migrate_film_weights(vc_block.film, tts_block.film)

    return tts_model
