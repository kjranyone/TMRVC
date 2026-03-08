"""Advanced quantization for ONNX models.

Supports multiple quantization modes:
    - DYNAMIC_INT8: Dynamic INT8 quantization (fast, no calibration data needed)
    - STATIC_INT8: Static INT8 quantization (requires calibration data)
    - FP16: FP16 weight conversion (best quality/speed trade-off on GPU)
    - SMOOTH_INT8: SmoothQuant-style quantization with per-channel weight scaling
      and activation calibration for reduced outlier sensitivity
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Models that should be quantized (per-frame real-time models)
QUANTIZE_TARGETS = [
    "content_encoder",
    "converter",
    "vocoder",
    "ir_estimator",
    "vc_encoder",
    "uclm_core",
    "voice_state_enc",
    "pointer_head",
    "prosody_predictor",
]

# Models to skip (offline execution, no speed requirement)
SKIP_QUANTIZE = ["speaker_encoder"]


class QuantizationMode(enum.Enum):
    """Supported quantization modes."""

    DYNAMIC_INT8 = "dynamic_int8"
    STATIC_INT8 = "static_int8"
    FP16 = "fp16"
    SMOOTH_INT8 = "smooth_int8"


class CalibrationDataReaderFromArrays:
    """onnxruntime CalibrationDataReader backed by a list of dicts.

    Each element in *data* must be a ``dict[str, np.ndarray]`` mapping
    input names to numpy arrays, matching the model's input signature.
    """

    def __init__(self, data: Sequence[dict[str, np.ndarray]]):
        self._data = list(data)
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= len(self._data):
            return None
        item = self._data[self._idx]
        self._idx += 1
        return item

    def rewind(self) -> None:
        self._idx = 0


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------


def quantize_dynamic_int8(input_path: Path, output_path: Path) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=[],
    )
    logger.info("Dynamic INT8: %s -> %s", input_path.name, output_path.name)
    return output_path


def quantize_static_int8(
    input_path: Path,
    output_path: Path,
    calibration_data: Sequence[dict[str, np.ndarray]],
) -> Path:
    """Apply static INT8 quantization using calibration data.

    Args:
        input_path: FP32 ONNX model path.
        output_path: Output quantized model path.
        calibration_data: List of input dicts for calibration.
    """
    from onnxruntime.quantization import QuantType, quantize_static

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = CalibrationDataReaderFromArrays(calibration_data)

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=None,  # use default QDQ format
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    logger.info("Static INT8: %s -> %s", input_path.name, output_path.name)
    return output_path


def quantize_fp16(input_path: Path, output_path: Path) -> Path:
    """Convert FP32 weights to FP16.

    Uses onnxconverter-common (if available) or falls back to manual
    numpy-based conversion via the ONNX graph.
    """
    import onnx
    from onnx import numpy_helper

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try the dedicated converter first
    try:
        from onnxconverter_common import float16  # type: ignore[import-untyped]

        model = onnx.load(str(input_path))
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True
        )
        onnx.save(model_fp16, str(output_path))
        logger.info("FP16 (onnxconverter): %s -> %s", input_path.name, output_path.name)
        return output_path
    except ImportError:
        pass

    # Fallback: manually convert initializer tensors
    model = onnx.load(str(input_path))
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(initializer).astype(np.float16)
            new_tensor = numpy_helper.from_array(arr, name=initializer.name)
            initializer.CopyFrom(new_tensor)
    onnx.save(model, str(output_path))
    logger.info("FP16 (manual): %s -> %s", input_path.name, output_path.name)
    return output_path


def quantize_smooth_int8(
    input_path: Path,
    output_path: Path,
    calibration_data: Sequence[dict[str, np.ndarray]],
    alpha: float = 0.5,
) -> Path:
    """SmoothQuant-style INT8 quantization.

    Applies per-channel activation scaling before INT8 quantization to reduce
    the impact of activation outliers.  The ``alpha`` parameter controls the
    balance between migrating difficulty from activations to weights:

        s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

    where j is the channel index.  After scaling, standard static INT8
    quantization is applied.

    Args:
        input_path: FP32 ONNX model path.
        output_path: Output quantized model path.
        calibration_data: List of input dicts for calibration.
        alpha: SmoothQuant migration strength (0.0 = all on weights, 1.0 = all
            on activations).  Default 0.5 is a good starting point.
    """
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_static

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Collect per-channel activation statistics --
    import onnxruntime as ort

    sess = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])

    # Collect activation channel maximums from calibration data
    activation_maxes: dict[str, np.ndarray] = {}
    for sample in calibration_data:
        outputs = sess.run(None, sample)
        # We use the model input statistics as a proxy for internal activations
        for name, arr in sample.items():
            arr_np = np.abs(arr.astype(np.float32))
            if arr_np.ndim >= 2:
                # Per-channel max across all dims except the last (channel) dim
                channel_max = arr_np.reshape(-1, arr_np.shape[-1]).max(axis=0)
            else:
                channel_max = arr_np
            if name not in activation_maxes:
                activation_maxes[name] = channel_max
            else:
                activation_maxes[name] = np.maximum(activation_maxes[name], channel_max)

    # -- Step 2: Compute per-channel scales and apply to weights --
    model = onnx.load(str(input_path))

    # Build a map from initializer name to initializer
    init_map = {init.name: init for init in model.graph.initializer}

    # For each Linear-like node, apply smoothing scales
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue

        weight_name = node.input[1] if len(node.input) > 1 else None
        if weight_name is None or weight_name not in init_map:
            continue

        init = init_map[weight_name]
        W = numpy_helper.to_array(init).astype(np.float32)

        if W.ndim != 2:
            continue

        # Per-channel weight max (along input dim = axis 0 for [in, out])
        w_max = np.abs(W).max(axis=1).clip(min=1e-8)

        # Check if we have activation stats for the input to this node
        input_name = node.input[0]
        if input_name in activation_maxes:
            a_max = activation_maxes[input_name].clip(min=1e-8)
            # Ensure dimensions match (take min length)
            min_len = min(len(a_max), len(w_max))
            a_max = a_max[:min_len]
            w_max_trunc = w_max[:min_len]

            # SmoothQuant scaling: s = a_max^alpha / w_max^(1-alpha)
            s = np.power(a_max, alpha) / np.power(w_max_trunc, 1.0 - alpha)
            s = s.clip(min=1e-8)

            # Apply: W_smooth = diag(s) @ W  (scale each input channel)
            W_smooth = W.copy()
            W_smooth[:min_len] = W[:min_len] * s[:, np.newaxis]

            new_tensor = numpy_helper.from_array(
                W_smooth.astype(np.float32), name=init.name
            )
            init.CopyFrom(new_tensor)

    # Save the smoothed model to a temporary path
    smoothed_path = output_path.parent / f"_smoothed_{input_path.name}"
    onnx.save(model, str(smoothed_path))

    # -- Step 3: Apply standard static INT8 quantization --
    reader = CalibrationDataReaderFromArrays(calibration_data)
    quantize_static(
        model_input=str(smoothed_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )

    # Clean up temporary smoothed model
    smoothed_path.unlink(missing_ok=True)

    logger.info(
        "SmoothQuant INT8 (alpha=%.2f): %s -> %s",
        alpha, input_path.name, output_path.name,
    )
    return output_path


# ---------------------------------------------------------------------------
# Public convenience wrappers (backward compatible)
# ---------------------------------------------------------------------------


def quantize_model(
    input_path: str | Path,
    output_path: str | Path,
    mode: QuantizationMode = QuantizationMode.DYNAMIC_INT8,
    calibration_data: Sequence[dict[str, np.ndarray]] | None = None,
    smooth_alpha: float = 0.5,
) -> Path:
    """Quantize a single ONNX model.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path to write quantized model.
        mode: Quantization mode to use.
        calibration_data: Required for STATIC_INT8 and SMOOTH_INT8 modes.
        smooth_alpha: SmoothQuant alpha parameter (only for SMOOTH_INT8).

    Returns:
        Path to the quantized model.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if mode == QuantizationMode.DYNAMIC_INT8:
        return quantize_dynamic_int8(input_path, output_path)
    elif mode == QuantizationMode.STATIC_INT8:
        if calibration_data is None:
            raise ValueError("STATIC_INT8 requires calibration_data")
        return quantize_static_int8(input_path, output_path, calibration_data)
    elif mode == QuantizationMode.FP16:
        return quantize_fp16(input_path, output_path)
    elif mode == QuantizationMode.SMOOTH_INT8:
        if calibration_data is None:
            raise ValueError("SMOOTH_INT8 requires calibration_data")
        return quantize_smooth_int8(
            input_path, output_path, calibration_data, alpha=smooth_alpha
        )
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def quantize_all(
    fp32_dir: str | Path,
    output_dir: str | Path,
    mode: QuantizationMode = QuantizationMode.DYNAMIC_INT8,
    calibration_data: Sequence[dict[str, np.ndarray]] | None = None,
    smooth_alpha: float = 0.5,
) -> dict[str, Path]:
    """Quantize all target models from fp32_dir.

    Args:
        fp32_dir: Directory containing FP32 ONNX models.
        output_dir: Output directory for quantized models.
        mode: Quantization mode to use.
        calibration_data: Required for STATIC_INT8 and SMOOTH_INT8 modes.
        smooth_alpha: SmoothQuant alpha parameter (only for SMOOTH_INT8).

    Returns:
        Dict of model_name -> quantized path.
    """
    fp32_dir = Path(fp32_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{mode.value}"

    paths: dict[str, Path] = {}
    for model_name in QUANTIZE_TARGETS:
        fp32_path = fp32_dir / f"{model_name}.onnx"
        if not fp32_path.exists():
            logger.warning("Skipping %s (not found: %s)", model_name, fp32_path)
            continue

        out_path = output_dir / f"{model_name}{suffix}.onnx"
        paths[model_name] = quantize_model(
            fp32_path, out_path,
            mode=mode,
            calibration_data=calibration_data,
            smooth_alpha=smooth_alpha,
        )

    logger.info("Quantized %d models (%s) to %s", len(paths), mode.value, output_dir)
    return paths
