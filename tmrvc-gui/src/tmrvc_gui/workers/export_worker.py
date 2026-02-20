"""ONNX export and quantization worker for TMRVC.

Converts PyTorch checkpoints to the 5 ONNX models required by the
streaming engine and optionally applies INT8 dynamic quantisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .base_worker import BaseWorker

logger = logging.getLogger(__name__)

# The ONNX model names that compose the TMRVC inference pipeline.
# converter_hq is optional (HQ mode only).
MODEL_NAMES: list[str] = [
    "content_encoder",
    "converter",
    "converter_hq",
    "vocoder",
    "ir_estimator",
    "speaker_encoder",
]

# Export function mapping: model_name → (export_func_name, model_class)
_EXPORT_MAP = {
    "content_encoder": "export_content_encoder",
    "converter": "export_converter",
    "converter_hq": "export_converter_hq",
    "vocoder": "export_vocoder",
    "ir_estimator": "export_ir_estimator",
    "speaker_encoder": "export_speaker_encoder",
}


def _load_models_from_checkpoint(
    checkpoint_path: Path,
    model_names: list[str],
) -> dict[str, torch.nn.Module]:
    """Load student models from a training checkpoint.

    The checkpoint is expected to contain model state dicts keyed by
    model name (e.g. ``content_encoder``, ``converter``, etc.).

    Returns:
        Dict of model_name → instantiated PyTorch model in eval mode.
    """
    from tmrvc_train.models.content_encoder import ContentEncoderStudent
    from tmrvc_train.models.converter import ConverterStudent, ConverterStudentHQ
    from tmrvc_train.models.ir_estimator import IREstimator
    from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
    from tmrvc_train.models.vocoder import VocoderStudent

    _MODEL_CLASSES: dict[str, type] = {
        "content_encoder": ContentEncoderStudent,
        "converter": ConverterStudent,
        "converter_hq": ConverterStudentHQ,
        "vocoder": VocoderStudent,
        "ir_estimator": IREstimator,
        "speaker_encoder": SpeakerEncoderWithLoRA,
    }

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    models = {}
    for name in model_names:
        cls = _MODEL_CLASSES.get(name)
        if cls is None:
            raise ValueError(f"Unknown model: {name}")

        model = cls()

        if name == "converter_hq" and name not in ckpt:
            # Initialize from causal converter weights if no dedicated weights
            if "converter" in models:
                model = ConverterStudentHQ.from_causal(models["converter"])
            elif "converter" in ckpt:
                causal = ConverterStudent()
                causal.load_state_dict(ckpt["converter"])
                model = ConverterStudentHQ.from_causal(causal)
            else:
                logger.warning("No converter weights found for converter_hq init")
                continue
        else:
            if name not in ckpt:
                logger.warning("Key '%s' not in checkpoint, skipping", name)
                continue
            model.load_state_dict(ckpt[name])

        model.eval()
        models[name] = model

    return models


class ExportWorker(BaseWorker):
    """Background worker for ONNX export and quantization.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the PyTorch checkpoint file (``.pt`` or ``.ckpt``).
    output_dir : Path
        Directory where exported ONNX files will be written.
    models : list[str]
        Subset of :data:`MODEL_NAMES` to export.  Pass all five for a
        full export.
    quantize : bool
        If *True*, apply INT8 dynamic quantisation to eligible models
        after export.  ``speaker_encoder`` is always excluded from
        quantisation (it runs offline only).
    parent : QObject, optional
        Parent Qt object.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        models: list[str],
        quantize: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.models = models
        self.quantize = quantize

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Export ONNX models and optionally quantise."""
        from tmrvc_export import export_onnx
        from tmrvc_export.quantize import quantize_model

        total_steps = len(self.models)
        if self.quantize:
            quantizable = [m for m in self.models if m != "speaker_encoder"]
            total_steps += len(quantizable)

        current_step = 0

        self.log_message.emit(
            f"[ExportWorker] Starting export from {self.checkpoint_path} "
            f"to {self.output_dir}  models={self.models}  "
            f"quantize={self.quantize}"
        )

        try:
            # Load models from checkpoint
            self.log_message.emit("[ExportWorker] Loading models from checkpoint...")
            pytorch_models = _load_models_from_checkpoint(
                self.checkpoint_path, self.models,
            )

            # ----------------------------------------------------------
            # Phase 1: FP32 ONNX export
            # ----------------------------------------------------------
            fp32_dir = self.output_dir / "fp32"
            fp32_dir.mkdir(parents=True, exist_ok=True)

            for model_name in self.models:
                if self.is_cancelled:
                    self.log_message.emit("[ExportWorker] Cancelled by user.")
                    self.finished.emit(False, "Cancelled")
                    return

                onnx_path = fp32_dir / f"{model_name}.onnx"
                self.log_message.emit(
                    f"[ExportWorker] Exporting {model_name} -> {onnx_path}"
                )

                export_fn = getattr(export_onnx, _EXPORT_MAP[model_name])
                export_fn(pytorch_models[model_name], onnx_path)

                current_step += 1
                self.progress.emit(current_step, total_steps)
                self.log_message.emit(
                    f"[ExportWorker] Exported {model_name} (FP32)"
                )

            # ----------------------------------------------------------
            # Phase 2: INT8 dynamic quantization (optional)
            # ----------------------------------------------------------
            if self.quantize:
                int8_dir = self.output_dir / "int8"
                int8_dir.mkdir(parents=True, exist_ok=True)

                for model_name in self.models:
                    if model_name == "speaker_encoder":
                        continue

                    if self.is_cancelled:
                        self.log_message.emit("[ExportWorker] Cancelled by user.")
                        self.finished.emit(False, "Cancelled")
                        return

                    src = fp32_dir / f"{model_name}.onnx"
                    dst = int8_dir / f"{model_name}_int8.onnx"
                    self.log_message.emit(
                        f"[ExportWorker] Quantizing {model_name} -> {dst}"
                    )

                    quantize_model(str(src), str(dst))

                    current_step += 1
                    self.progress.emit(current_step, total_steps)
                    self.log_message.emit(
                        f"[ExportWorker] Quantized {model_name} (INT8)"
                    )

            self.log_message.emit("[ExportWorker] Export complete.")
            self.finished.emit(True, "Export completed successfully")

        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))
