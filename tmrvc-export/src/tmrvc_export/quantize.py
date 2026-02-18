"""INT8 dynamic quantization for ONNX models."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Models that should be quantized (per-frame real-time models)
QUANTIZE_TARGETS = [
    "content_encoder",
    "converter",
    "vocoder",
    "ir_estimator",
]

# Models to skip (offline execution, no speed requirement)
SKIP_QUANTIZE = ["speaker_encoder"]


def quantize_model(input_path: str | Path, output_path: str | Path) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model.

    State tensors are excluded from quantization to preserve precision.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path to write quantized model.

    Returns:
        Path to the quantized model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=[],  # Node-level exclusion if needed
    )

    logger.info("Quantized %s → %s", input_path.name, output_path.name)
    return output_path


def quantize_all(fp32_dir: str | Path, int8_dir: str | Path) -> dict[str, Path]:
    """Quantize all target models from fp32_dir to int8_dir.

    Args:
        fp32_dir: Directory containing FP32 ONNX models.
        int8_dir: Output directory for INT8 models.

    Returns:
        Dict of model_name → quantized path.
    """
    fp32_dir = Path(fp32_dir)
    int8_dir = Path(int8_dir)
    int8_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for model_name in QUANTIZE_TARGETS:
        fp32_path = fp32_dir / f"{model_name}.onnx"
        if not fp32_path.exists():
            logger.warning("Skipping %s (not found: %s)", model_name, fp32_path)
            continue

        int8_path = int8_dir / f"{model_name}_int8.onnx"
        paths[model_name] = quantize_model(fp32_path, int8_path)

    logger.info("Quantized %d models to %s", len(paths), int8_dir)
    return paths
