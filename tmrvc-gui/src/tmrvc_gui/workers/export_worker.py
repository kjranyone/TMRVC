"""ONNX export and quantization worker for TMRVC UCLM.

Converts UCLM and Codec checkpoints to ONNX for the unified engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .base_worker import BaseWorker

logger = logging.getLogger(__name__)

# UCLM Unified ONNX components
MODEL_NAMES: list[str] = [
    "uclm",
    "codec",
    "speaker_encoder",
]

_EXPORT_MAP = {
    "uclm": "export_uclm",
    "codec": "export_codec",
    "speaker_encoder": "export_speaker_encoder",
}


def _load_model_from_path(
    name: str,
    path: Path,
) -> torch.nn.Module:
    """Load a specific model from a checkpoint path."""
    from tmrvc_train.models import DisentangledUCLM, EmotionAwareCodec, SpeakerEncoderWithLoRA

    _CLASSES = {
        "uclm": DisentangledUCLM,
        "codec": EmotionAwareCodec,
        "speaker_encoder": SpeakerEncoderWithLoRA,
    }

    cls = _CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")

    model = cls()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Handle both full checkpoints and state_dicts
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class ExportWorker(BaseWorker):
    """Background worker for ONNX export and quantization (UCLM)."""

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

    def run(self) -> None:
        from tmrvc_export import export_onnx
        from tmrvc_export.quantize import quantize_model

        total_steps = len(self.models)
        if self.quantize:
            total_steps += len([m for m in self.models if m != "speaker_encoder"])

        current_step = 0

        try:
            fp32_dir = self.output_dir / "fp32"
            fp32_dir.mkdir(parents=True, exist_ok=True)

            for model_name in self.models:
                if self.is_cancelled:
                    self.finished.emit(False, "Cancelled")
                    return

                self.log_message.emit(f"Loading and exporting {model_name}...")
                
                # In UCLM, each component might be in its own file or shared
                # For simplicity, we assume the provided path contains the target model
                pytorch_model = _load_model_from_path(model_name, self.checkpoint_path)
                
                onnx_path = fp32_dir / f"{model_name}.onnx"
                export_fn = getattr(export_onnx, _EXPORT_MAP[model_name])
                export_fn(pytorch_model, onnx_path)

                current_step += 1
                self.progress.emit(current_step, total_steps)

                if self.quantize and model_name != "speaker_encoder":
                    int8_dir = self.output_dir / "int8"
                    int8_dir.mkdir(parents=True, exist_ok=True)
                    dst = int8_dir / f"{model_name}_int8.onnx"
                    
                    self.log_message.emit(f"Quantizing {model_name}...")
                    quantize_model(str(onnx_path), str(dst))
                    
                    current_step += 1
                    self.progress.emit(current_step, total_steps)

            self.log_message.emit("Export complete.")
            self.finished.emit(True, "Export completed successfully")

        except Exception as exc:
            logger.exception("Export failed")
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))
