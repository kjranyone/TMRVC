"""Export PyTorch student models to ONNX format (streaming mode)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from tmrvc_core.constants import (
    CONTENT_ENCODER_STATE_FRAMES,
    CONVERTER_HQ_STATE_FRAMES,
    CONVERTER_STATE_FRAMES,
    D_CONTENT,
    D_CONVERTER_HIDDEN,
    D_SPEAKER,
    D_VOCODER_FEATURES,
    IR_ESTIMATOR_STATE_FRAMES,
    LORA_ALPHA,
    LORA_DELTA_SIZE,
    LORA_RANK,
    MAX_LOOKAHEAD_HOPS,
    N_ACOUSTIC_PARAMS,
    N_LORA_LAYERS,
    N_MELS,
    VOCODER_STATE_FRAMES,
)
from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.converter import (
    ConverterStudent,
    ConverterStudentGTM,
    ConverterStudentHQ,
)
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
from tmrvc_train.models.vocoder import VocoderStudent

logger = logging.getLogger(__name__)

_OPSET_VERSION = 18
_LORA_SCALE = float(LORA_ALPHA) / float(LORA_RANK)
_FILM_D_IN = D_SPEAKER + N_ACOUSTIC_PARAMS
_FILM_D_OUT = D_CONVERTER_HIDDEN * 2
_FILM_LAYER_PARAM_SIZE = _FILM_D_IN * LORA_RANK + LORA_RANK * _FILM_D_OUT


class _ContentEncoderWrapper(torch.nn.Module):
    """Wrapper for ONNX export with flat tuple output."""

    def __init__(self, model: ContentEncoderStudent) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, mel_frame: torch.Tensor, f0: torch.Tensor, state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content, state_out = self.model(mel_frame, f0, state_in)
        return content, state_out


class _ConverterWrapper(torch.nn.Module):
    """Converter export wrapper with explicit LoRA delta input."""

    def __init__(self, model: ConverterStudent) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _film_lora_delta(
        cond: torch.Tensor,
        lora_delta: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        bsz = cond.shape[0]
        layer_start = layer_idx * _FILM_LAYER_PARAM_SIZE

        a_start = layer_start
        a_end = a_start + _FILM_D_IN * LORA_RANK
        b_end = a_end + LORA_RANK * _FILM_D_OUT

        a = lora_delta[:, a_start:a_end].reshape(bsz, _FILM_D_IN, LORA_RANK)
        b = lora_delta[:, a_end:b_end].reshape(bsz, LORA_RANK, _FILM_D_OUT)

        low_rank = torch.bmm(cond.unsqueeze(1), a).squeeze(1)
        delta = torch.bmm(low_rank.unsqueeze(1), b).squeeze(1)
        return delta * _LORA_SCALE

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        lora_delta: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = torch.cat([spk_embed, acoustic_params], dim=-1)  # [B, 224]

        x = self.model.input_proj(content)

        states = self.model._split_state(state_in)
        new_states = []

        for i, (block, s_in) in enumerate(zip(self.model.blocks, states)):
            x, s_out = block.conv_block(x, s_in)

            gamma_beta = block.film.proj(cond)
            if i < N_LORA_LAYERS:
                gamma_beta = gamma_beta + self._film_lora_delta(cond, lora_delta, i)

            gamma, beta = gamma_beta.chunk(2, dim=-1)
            x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
            new_states.append(s_out)

        pred_features = self.model.output_proj(x)
        state_out = torch.cat(new_states, dim=-1)
        return pred_features, state_out


class _ConverterGTMWrapper(torch.nn.Module):
    """GTM converter export wrapper with explicit LoRA delta input."""

    def __init__(self, model: ConverterStudentGTM) -> None:
        super().__init__()
        self.model = model

        self._gtm_d_in = model.gtm.proj.in_features
        self._gtm_d_out = model.gtm.proj.out_features
        self._gtm_layer_param_size = (
            self._gtm_d_in * LORA_RANK + LORA_RANK * self._gtm_d_out
        )

    def _gtm_lora_delta(
        self,
        spk_embed: torch.Tensor,
        lora_delta: torch.Tensor,
    ) -> torch.Tensor:
        bsz = spk_embed.shape[0]
        used = lora_delta[:, :self._gtm_layer_param_size]

        a_end = self._gtm_d_in * LORA_RANK
        a = used[:, :a_end].reshape(bsz, self._gtm_d_in, LORA_RANK)
        b = used[:, a_end:].reshape(bsz, LORA_RANK, self._gtm_d_out)

        low_rank = torch.bmm(spk_embed.unsqueeze(1), a).squeeze(1)
        delta = torch.bmm(low_rank.unsqueeze(1), b).squeeze(1)
        return delta * _LORA_SCALE

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        lora_delta: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_memory = self.model.gtm.proj(spk_embed)
        delta_memory = self._gtm_lora_delta(spk_embed, lora_delta)
        timbre_memory = (base_memory + delta_memory).reshape(
            -1,
            self.model.gtm.n_entries,
            self.model.gtm.d_entry,
        )

        x = self.model.input_proj(content)

        states = self.model._split_state(state_in)
        new_states = []
        for block, s_in in zip(self.model.blocks, states):
            x, s_out = block.conv_block(x, s_in)
            x = x + block.timbre_attn(x, timbre_memory)
            x = block.film_acoustic(x, acoustic_params)
            new_states.append(s_out)

        pred_features = self.model.output_proj(x)
        state_out = torch.cat(new_states, dim=-1)
        return pred_features, state_out


class _VocoderWrapper(torch.nn.Module):
    def __init__(self, model: VocoderStudent) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, features: torch.Tensor, state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag, phase, state_out = self.model(features, state_in)
        return mag, phase, state_out


class _IREstimatorWrapper(torch.nn.Module):
    def __init__(self, model: IREstimator) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, mel_chunk: torch.Tensor, state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acoustic_params, state_out = self.model(mel_chunk, state_in)
        return acoustic_params, state_out


class _ConverterHQWrapper(torch.nn.Module):
    """HQ converter export wrapper with explicit LoRA delta input."""

    def __init__(self, model: ConverterStudentHQ) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _film_lora_delta(
        cond: torch.Tensor,
        lora_delta: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        bsz = cond.shape[0]
        layer_start = layer_idx * _FILM_LAYER_PARAM_SIZE

        a_start = layer_start
        a_end = a_start + _FILM_D_IN * LORA_RANK
        b_end = a_end + LORA_RANK * _FILM_D_OUT

        a = lora_delta[:, a_start:a_end].reshape(bsz, _FILM_D_IN, LORA_RANK)
        b = lora_delta[:, a_end:b_end].reshape(bsz, LORA_RANK, _FILM_D_OUT)

        low_rank = torch.bmm(cond.unsqueeze(1), a).squeeze(1)
        delta = torch.bmm(low_rank.unsqueeze(1), b).squeeze(1)
        return delta * _LORA_SCALE

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        lora_delta: torch.Tensor,
        acoustic_params: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = torch.cat([spk_embed, acoustic_params], dim=-1)  # [B, 224]

        x = self.model.input_proj(content)

        states = self.model._split_state(state_in)
        new_states = []

        for i, (block, s_in) in enumerate(zip(self.model.blocks, states)):
            x, s_out = block.conv_block(x, s_in)

            gamma_beta = block.film.proj(cond)
            if i < N_LORA_LAYERS:
                gamma_beta = gamma_beta + self._film_lora_delta(cond, lora_delta, i)

            gamma, beta = gamma_beta.chunk(2, dim=-1)
            x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
            new_states.append(s_out)

        pred_features = self.model.output_proj(x)
        state_out = torch.cat(new_states, dim=-1)
        return pred_features, state_out


def _prepare_output_path(output_path: Path) -> None:
    """Remove stale ONNX/external-data files before export."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (output_path, Path(f"{output_path}.data")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

def export_content_encoder(
    model: ContentEncoderStudent, output_path: str | Path,
) -> Path:
    """Export content encoder to ONNX (streaming mode, T=1)."""
    output_path = Path(output_path)
    _prepare_output_path(output_path)
    model.eval()

    wrapper = _ContentEncoderWrapper(model).eval()
    dummy_mel = torch.zeros(1, N_MELS, 1)
    dummy_f0 = torch.zeros(1, 1, 1)
    dummy_state = torch.zeros(1, D_CONTENT, CONTENT_ENCODER_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_mel, dummy_f0, dummy_state),
        str(output_path),
        input_names=["mel_frame", "f0", "state_in"],
        output_names=["content", "state_out"],
        dynamic_axes=None,
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported content_encoder to %s", output_path)
    return output_path


def export_converter(
    model: ConverterStudent | ConverterStudentGTM, output_path: str | Path,
) -> Path:
    """Export converter to ONNX (streaming mode, T=1) with LoRA input."""
    output_path = Path(output_path)
    _prepare_output_path(output_path)
    model.eval()

    if isinstance(model, ConverterStudentGTM):
        wrapper = _ConverterGTMWrapper(model).eval()
    else:
        wrapper = _ConverterWrapper(model).eval()

    dummy_content = torch.zeros(1, D_CONTENT, 1)
    dummy_spk = torch.zeros(1, D_SPEAKER)
    dummy_lora = torch.zeros(1, LORA_DELTA_SIZE)
    dummy_acoustic = torch.zeros(1, N_ACOUSTIC_PARAMS)
    dummy_state = torch.zeros(1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_content, dummy_spk, dummy_lora, dummy_acoustic, dummy_state),
        str(output_path),
        input_names=["content", "spk_embed", "lora_delta", "acoustic_params", "state_in"],
        output_names=["pred_features", "state_out"],
        dynamic_axes=None,
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported converter to %s", output_path)
    return output_path


def export_converter_hq(
    model: ConverterStudentHQ, output_path: str | Path,
) -> Path:
    """Export HQ converter to ONNX (streaming mode, T_in=1+L, T_out=1)."""
    output_path = Path(output_path)
    _prepare_output_path(output_path)
    model.eval()

    wrapper = _ConverterHQWrapper(model).eval()
    dummy_content = torch.zeros(1, D_CONTENT, 1 + MAX_LOOKAHEAD_HOPS)  # [1, 256, 7]
    dummy_spk = torch.zeros(1, D_SPEAKER)
    dummy_lora = torch.zeros(1, LORA_DELTA_SIZE)
    dummy_acoustic = torch.zeros(1, N_ACOUSTIC_PARAMS)
    dummy_state = torch.zeros(1, D_CONVERTER_HIDDEN, CONVERTER_HQ_STATE_FRAMES)  # [1, 384, 46]

    torch.onnx.export(
        wrapper,
        (dummy_content, dummy_spk, dummy_lora, dummy_acoustic, dummy_state),
        str(output_path),
        input_names=["content", "spk_embed", "lora_delta", "acoustic_params", "state_in"],
        output_names=["pred_features", "state_out"],
        dynamic_axes=None,
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported converter_hq to %s", output_path)
    return output_path


def export_vocoder(
    model: VocoderStudent, output_path: str | Path,
) -> Path:
    """Export vocoder to ONNX (streaming mode, T=1)."""
    output_path = Path(output_path)
    _prepare_output_path(output_path)
    model.eval()

    wrapper = _VocoderWrapper(model).eval()
    dummy_features = torch.zeros(1, D_VOCODER_FEATURES, 1)
    dummy_state = torch.zeros(1, D_CONTENT, VOCODER_STATE_FRAMES)  # d_model=256

    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_state),
        str(output_path),
        input_names=["features", "state_in"],
        output_names=["stft_mag", "stft_phase", "state_out"],
        dynamic_axes=None,
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported vocoder to %s", output_path)
    return output_path


def export_ir_estimator(
    model: IREstimator, output_path: str | Path,
) -> Path:
    """Export IR estimator to ONNX (chunk mode, N=10)."""
    output_path = Path(output_path)
    _prepare_output_path(output_path)
    model.eval()

    wrapper = _IREstimatorWrapper(model).eval()
    dummy_mel = torch.zeros(1, N_MELS, 10)  # ir_update_interval
    dummy_state = torch.zeros(1, 128, IR_ESTIMATOR_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_mel, dummy_state),
        str(output_path),
        input_names=["mel_chunk", "state_in"],
        output_names=["acoustic_params", "state_out"],
        dynamic_axes=None,
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported ir_estimator to %s", output_path)
    return output_path


def export_speaker_encoder(
    model: SpeakerEncoderWithLoRA, output_path: str | Path,
) -> Path:
    """Export speaker encoder to ONNX (offline, variable-length input)."""
    output_path = Path(output_path)
    _prepare_output_path(output_path)
    model.eval()

    # T_ref is variable (3-15 seconds, 300-1500 frames)
    dummy_mel = torch.zeros(1, N_MELS, 500)

    torch.onnx.export(
        model,
        (dummy_mel,),
        str(output_path),
        input_names=["mel_ref"],
        output_names=["spk_embed", "lora_delta"],
        dynamic_axes={"mel_ref": {2: "T_ref"}},
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported speaker_encoder to %s", output_path)
    return output_path


def export_all(
    content_encoder: ContentEncoderStudent,
    converter: ConverterStudent | ConverterStudentGTM,
    vocoder: VocoderStudent,
    ir_estimator: IREstimator,
    speaker_encoder: SpeakerEncoderWithLoRA,
    output_dir: str | Path,
    converter_hq: ConverterStudentHQ | None = None,
) -> dict[str, Path]:
    """Export all models to ONNX.

    Args:
        converter_hq: Optional HQ converter. If provided, exports as
            ``converter_hq.onnx`` alongside the live converter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["content_encoder"] = export_content_encoder(
        content_encoder, output_dir / "content_encoder.onnx",
    )
    paths["converter"] = export_converter(
        converter, output_dir / "converter.onnx",
    )
    if converter_hq is not None:
        paths["converter_hq"] = export_converter_hq(
            converter_hq, output_dir / "converter_hq.onnx",
        )
    paths["vocoder"] = export_vocoder(
        vocoder, output_dir / "vocoder.onnx",
    )
    paths["ir_estimator"] = export_ir_estimator(
        ir_estimator, output_dir / "ir_estimator.onnx",
    )
    paths["speaker_encoder"] = export_speaker_encoder(
        speaker_encoder, output_dir / "speaker_encoder.onnx",
    )

    n_models = 5 + (1 if converter_hq is not None else 0)
    logger.info("All %d models exported to %s", n_models, output_dir)
    return paths
