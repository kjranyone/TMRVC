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
    MAX_LOOKAHEAD_HOPS,
    N_IR_PARAMS,
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
    def __init__(self, model: ConverterStudent | ConverterStudentGTM) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        ir_params: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_features, state_out = self.model(content, spk_embed, ir_params, state_in)
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
        ir_params, state_out = self.model(mel_chunk, state_in)
        return ir_params, state_out


def export_content_encoder(
    model: ContentEncoderStudent, output_path: str | Path,
) -> Path:
    """Export content encoder to ONNX (streaming mode, T=1)."""
    output_path = Path(output_path)
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


class _ConverterHQWrapper(torch.nn.Module):
    def __init__(self, model: ConverterStudentHQ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        ir_params: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_features, state_out = self.model(content, spk_embed, ir_params, state_in)
        return pred_features, state_out


def export_converter(
    model: ConverterStudent | ConverterStudentGTM, output_path: str | Path,
) -> Path:
    """Export converter to ONNX (streaming mode, T=1)."""
    output_path = Path(output_path)
    model.eval()

    wrapper = _ConverterWrapper(model).eval()
    dummy_content = torch.zeros(1, D_CONTENT, 1)
    dummy_spk = torch.zeros(1, D_SPEAKER)
    dummy_ir = torch.zeros(1, N_IR_PARAMS)
    dummy_state = torch.zeros(1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_content, dummy_spk, dummy_ir, dummy_state),
        str(output_path),
        input_names=["content", "spk_embed", "ir_params", "state_in"],
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
    model.eval()

    wrapper = _ConverterHQWrapper(model).eval()
    dummy_content = torch.zeros(1, D_CONTENT, 1 + MAX_LOOKAHEAD_HOPS)  # [1, 256, 7]
    dummy_spk = torch.zeros(1, D_SPEAKER)
    dummy_ir = torch.zeros(1, N_IR_PARAMS)
    dummy_state = torch.zeros(1, D_CONVERTER_HIDDEN, CONVERTER_HQ_STATE_FRAMES)  # [1, 384, 46]

    torch.onnx.export(
        wrapper,
        (dummy_content, dummy_spk, dummy_ir, dummy_state),
        str(output_path),
        input_names=["content", "spk_embed", "ir_params", "state_in"],
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
    model.eval()

    wrapper = _IREstimatorWrapper(model).eval()
    dummy_mel = torch.zeros(1, N_MELS, 10)  # ir_update_interval
    dummy_state = torch.zeros(1, 128, IR_ESTIMATOR_STATE_FRAMES)

    torch.onnx.export(
        wrapper,
        (dummy_mel, dummy_state),
        str(output_path),
        input_names=["mel_chunk", "state_in"],
        output_names=["ir_params", "state_out"],
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
    converter: ConverterStudent,
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
