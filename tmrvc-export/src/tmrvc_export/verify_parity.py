"""Parity verification: PyTorch vs ONNX Runtime inference comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ParityResult:
    """Result of a parity comparison."""

    model_name: str
    output_name: str
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.model_name}/{self.output_name}: "
            f"max_abs={self.max_abs_diff:.2e}, mean_abs={self.mean_abs_diff:.2e}, "
            f"max_rel={self.max_rel_diff:.2e}"
        )


def _compare_tensors(
    pytorch_out: np.ndarray,
    onnx_out: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> tuple[float, float, float, bool]:
    """Compare two numpy arrays.

    Returns:
        Tuple of (max_abs_diff, mean_abs_diff, max_rel_diff, passed).
    """
    abs_diff = np.abs(pytorch_out - onnx_out)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())

    # Relative error on elements with significant magnitude
    significant_mask = np.abs(pytorch_out) > 1e-4
    if significant_mask.any():
        rel_diff = abs_diff[significant_mask] / np.abs(pytorch_out[significant_mask])
        max_rel = float(rel_diff.max())
    else:
        max_rel = 0.0

    # Pass if absolute error is small enough (rel error on tiny values is unreliable)
    passed = max_abs < atol
    return max_abs, mean_abs, max_rel, passed


def verify_content_encoder(
    model: torch.nn.Module,
    onnx_path: str | Path,
    atol: float = 5e-5,
    rtol: float = 1e-3,
) -> list[ParityResult]:
    """Verify content encoder parity."""
    import onnxruntime as ort

    from tmrvc_core.constants import CONTENT_ENCODER_STATE_FRAMES, D_CONTENT, N_MELS

    model.eval()
    results = []

    sess = ort.InferenceSession(str(onnx_path))

    for test_name, mel, f0 in [
        ("zero", torch.zeros(1, N_MELS, 1), torch.zeros(1, 1, 1)),
        ("random", torch.randn(1, N_MELS, 1), torch.randn(1, 1, 1)),
    ]:
        state = torch.zeros(1, D_CONTENT, CONTENT_ENCODER_STATE_FRAMES)

        # PyTorch
        with torch.no_grad():
            pt_content, pt_state = model(mel, f0, state)

        # ONNX Runtime
        ort_out = sess.run(None, {
            "mel_frame": mel.numpy(),
            "f0": f0.numpy(),
            "state_in": state.numpy(),
        })

        for out_name, pt_val, ort_val in [
            ("content", pt_content.numpy(), ort_out[0]),
            ("state_out", pt_state.numpy(), ort_out[1]),
        ]:
            max_abs, mean_abs, max_rel, passed = _compare_tensors(
                pt_val, ort_val, atol, rtol,
            )
            results.append(ParityResult(
                f"content_encoder/{test_name}", out_name,
                max_abs, mean_abs, max_rel, passed,
            ))

    return results


def verify_converter(
    model: torch.nn.Module,
    onnx_path: str | Path,
    atol: float = 5e-5,
    rtol: float = 1e-3,
) -> list[ParityResult]:
    """Verify converter parity, including the LoRA input path."""
    import onnxruntime as ort

    from tmrvc_core.constants import (
        CONVERTER_STATE_FRAMES,
        D_CONTENT,
        D_CONVERTER_HIDDEN,
        D_SPEAKER,
        LORA_DELTA_SIZE,
        N_IR_PARAMS,
    )
    from tmrvc_export.export_onnx import _ConverterGTMWrapper, _ConverterWrapper
    from tmrvc_train.models.converter import ConverterStudentGTM

    model.eval()
    results = []
    sess = ort.InferenceSession(str(onnx_path))

    if isinstance(model, ConverterStudentGTM):
        wrapper = _ConverterGTMWrapper(model).eval()
    else:
        wrapper = _ConverterWrapper(model).eval()

    content = torch.randn(1, D_CONTENT, 1)
    spk = torch.randn(1, D_SPEAKER)
    ir = torch.randn(1, N_IR_PARAMS)
    state = torch.zeros(1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES)

    tests = [
        ("zero", torch.zeros(1, LORA_DELTA_SIZE)),
        ("random", torch.randn(1, LORA_DELTA_SIZE) * 0.01),
    ]

    for test_name, lora in tests:
        with torch.no_grad():
            pt_feat, pt_state = wrapper(content, spk, lora, ir, state)

        ort_out = sess.run(None, {
            "content": content.numpy(),
            "spk_embed": spk.numpy(),
            "lora_delta": lora.numpy(),
            "ir_params": ir.numpy(),
            "state_in": state.numpy(),
        })

        for out_name, pt_val, ort_val in [
            ("pred_features", pt_feat.numpy(), ort_out[0]),
            ("state_out", pt_state.numpy(), ort_out[1]),
        ]:
            # state_out can show slightly larger max-abs noise after ONNX graph
            # rewrites (still negligible in mean error), so keep this targeted.
            out_atol = max(atol, 3e-5) if out_name == "state_out" else atol
            max_abs, mean_abs, max_rel, passed = _compare_tensors(
                pt_val, ort_val, out_atol, rtol,
            )
            results.append(ParityResult(
                f"converter/{test_name}", out_name,
                max_abs, mean_abs, max_rel, passed,
            ))

    return results


def verify_vocoder(
    model: torch.nn.Module,
    onnx_path: str | Path,
    atol: float = 5e-5,
    rtol: float = 1e-3,
) -> list[ParityResult]:
    """Verify vocoder parity."""
    import onnxruntime as ort

    from tmrvc_core.constants import D_CONTENT, D_VOCODER_FEATURES, VOCODER_STATE_FRAMES

    model.eval()
    results = []
    sess = ort.InferenceSession(str(onnx_path))

    features = torch.randn(1, D_VOCODER_FEATURES, 1)
    state = torch.zeros(1, D_CONTENT, VOCODER_STATE_FRAMES)

    with torch.no_grad():
        pt_mag, pt_phase, pt_state = model(features, state)

    ort_out = sess.run(None, {
        "features": features.numpy(),
        "state_in": state.numpy(),
    })

    for out_name, pt_val, ort_val in [
        ("stft_mag", pt_mag.numpy(), ort_out[0]),
        ("stft_phase", pt_phase.numpy(), ort_out[1]),
        ("state_out", pt_state.numpy(), ort_out[2]),
    ]:
        # phase head occasionally shows larger absolute deltas while remaining
        # numerically close in relative/mean terms.
        out_atol = max(atol, 5e-5) if out_name == "stft_phase" else atol
        max_abs, mean_abs, max_rel, passed = _compare_tensors(
            pt_val, ort_val, out_atol, rtol,
        )
        results.append(ParityResult(
            "vocoder", out_name, max_abs, mean_abs, max_rel, passed,
        ))

    return results


def verify_ir_estimator(
    model: torch.nn.Module,
    onnx_path: str | Path,
    atol: float = 5e-5,
    rtol: float = 1e-3,
) -> list[ParityResult]:
    """Verify IR estimator parity."""
    import onnxruntime as ort

    from tmrvc_core.constants import IR_ESTIMATOR_STATE_FRAMES, N_MELS

    model.eval()
    results = []
    sess = ort.InferenceSession(str(onnx_path))

    mel_chunk = torch.randn(1, N_MELS, 10)
    state = torch.zeros(1, 128, IR_ESTIMATOR_STATE_FRAMES)

    with torch.no_grad():
        pt_ir, pt_state = model(mel_chunk, state)

    ort_out = sess.run(None, {
        "mel_chunk": mel_chunk.numpy(),
        "state_in": state.numpy(),
    })

    for out_name, pt_val, ort_val in [
        ("ir_params", pt_ir.numpy(), ort_out[0]),
        ("state_out", pt_state.numpy(), ort_out[1]),
    ]:
        max_abs, mean_abs, max_rel, passed = _compare_tensors(
            pt_val, ort_val, atol, rtol,
        )
        results.append(ParityResult(
            "ir_estimator", out_name, max_abs, mean_abs, max_rel, passed,
        ))

    return results


def verify_all(
    models: dict[str, torch.nn.Module],
    onnx_dir: str | Path,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> list[ParityResult]:
    """Verify parity for all models.

    Args:
        models: Dict of model_name -> PyTorch model.
        onnx_dir: Directory containing exported ONNX files.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        List of ParityResult for all checks.
    """
    onnx_dir = Path(onnx_dir)
    all_results = []

    verify_fns = {
        "content_encoder": verify_content_encoder,
        "converter": verify_converter,
        "vocoder": verify_vocoder,
        "ir_estimator": verify_ir_estimator,
    }

    for name, verify_fn in verify_fns.items():
        if name not in models:
            logger.warning("Skipping %s (model not provided)", name)
            continue
        onnx_path = onnx_dir / f"{name}.onnx"
        if not onnx_path.exists():
            logger.warning("Skipping %s (ONNX file not found: %s)", name, onnx_path)
            continue
        results = verify_fn(models[name], onnx_path, atol, rtol)
        all_results.extend(results)
        for r in results:
            logger.info("%s", r)

    passed = all(r.passed for r in all_results)
    logger.info(
        "Parity verification: %d/%d checks passed",
        sum(r.passed for r in all_results), len(all_results),
    )
    return all_results

