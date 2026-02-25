"""Export TTS front-end models to ONNX format (batch/offline mode).

Unlike VC models (streaming, state_in/state_out), TTS front-end models
are stateless and operate on variable-length sequences:

- TextEncoder: phoneme_ids[B,L] → text_features[B,256,L]
- DurationPredictor: text_features[B,256,L] + style[B,32] → durations[B,L]
- F0Predictor: text_features[B,256,T] + style[B,32] → f0[B,1,T], voiced[B,1,T]
- ContentSynthesizer: text_features[B,256,T] → content[B,256,T]

Usage::

    from tmrvc_export.export_tts import export_tts_all
    paths = export_tts_all(text_enc, dur_pred, f0_pred, content_synth, "models/tts/")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import (
    D_CONTENT,
    D_STYLE,
    D_TEXT_ENCODER,
    N_MELS,
    PHONEME_VOCAB_SIZE,
)
from tmrvc_export._utils import prepare_output_path

logger = logging.getLogger(__name__)

_OPSET_VERSION = 18


# --- Wrappers ---


class _TextEncoderWrapper(torch.nn.Module):
    """Wrapper for TextEncoder ONNX export.

    Fixes language_ids to a constant input and drops phoneme_lengths
    (ONNX inference uses full sequences without padding mask).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(phoneme_ids, language_ids, phoneme_lengths=None)


class _DurationPredictorWrapper(torch.nn.Module):
    """Wrapper for DurationPredictor ONNX export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        text_features: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(text_features, style)


class _F0PredictorWrapper(torch.nn.Module):
    """Wrapper for F0Predictor ONNX export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        text_features: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(text_features, style)


class _ContentSynthesizerWrapper(torch.nn.Module):
    """Wrapper for ContentSynthesizer ONNX export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.model(text_features)


# --- Export functions ---


def export_text_encoder(
    model: torch.nn.Module, output_path: str | Path,
) -> Path:
    """Export TextEncoder to ONNX (variable-length L)."""
    output_path = Path(output_path)
    prepare_output_path(output_path)
    model.eval()

    wrapper = _TextEncoderWrapper(model).eval()
    L = 50  # representative phoneme sequence length
    dummy_phonemes = torch.zeros(1, L, dtype=torch.long)
    dummy_lang = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_phonemes, dummy_lang),
        str(output_path),
        input_names=["phoneme_ids", "language_ids"],
        output_names=["text_features"],
        dynamic_axes={
            "phoneme_ids": {1: "L"},
            "text_features": {2: "L"},
        },
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported text_encoder to %s", output_path)
    return output_path


def export_duration_predictor(
    model: torch.nn.Module, output_path: str | Path,
) -> Path:
    """Export DurationPredictor to ONNX (variable-length L)."""
    output_path = Path(output_path)
    prepare_output_path(output_path)
    model.eval()

    wrapper = _DurationPredictorWrapper(model).eval()
    L = 50
    dummy_features = torch.randn(1, D_TEXT_ENCODER, L)
    dummy_style = torch.zeros(1, D_STYLE)

    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_style),
        str(output_path),
        input_names=["text_features", "style"],
        output_names=["durations"],
        dynamic_axes={
            "text_features": {2: "L"},
            "durations": {1: "L"},
        },
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported duration_predictor to %s", output_path)
    return output_path


def export_f0_predictor(
    model: torch.nn.Module, output_path: str | Path,
) -> Path:
    """Export F0Predictor to ONNX (variable-length T)."""
    output_path = Path(output_path)
    prepare_output_path(output_path)
    model.eval()

    wrapper = _F0PredictorWrapper(model).eval()
    T = 200
    dummy_features = torch.randn(1, D_TEXT_ENCODER, T)
    dummy_style = torch.zeros(1, D_STYLE)

    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_style),
        str(output_path),
        input_names=["text_features", "style"],
        output_names=["f0", "voiced_prob"],
        dynamic_axes={
            "text_features": {2: "T"},
            "f0": {2: "T"},
            "voiced_prob": {2: "T"},
        },
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported f0_predictor to %s", output_path)
    return output_path


def export_content_synthesizer(
    model: torch.nn.Module, output_path: str | Path,
) -> Path:
    """Export ContentSynthesizer to ONNX (variable-length T)."""
    output_path = Path(output_path)
    prepare_output_path(output_path)
    model.eval()

    wrapper = _ContentSynthesizerWrapper(model).eval()
    T = 200
    dummy_features = torch.randn(1, D_TEXT_ENCODER, T)

    torch.onnx.export(
        wrapper,
        (dummy_features,),
        str(output_path),
        input_names=["text_features"],
        output_names=["content"],
        dynamic_axes={
            "text_features": {2: "T"},
            "content": {2: "T"},
        },
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported content_synthesizer to %s", output_path)
    return output_path


def export_tts_all(
    text_encoder: torch.nn.Module,
    duration_predictor: torch.nn.Module,
    f0_predictor: torch.nn.Module,
    content_synthesizer: torch.nn.Module,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export all TTS front-end models to ONNX.

    Args:
        text_encoder: TextEncoder model.
        duration_predictor: DurationPredictor model.
        f0_predictor: F0Predictor model.
        content_synthesizer: ContentSynthesizer model.
        output_dir: Output directory for ONNX files.

    Returns:
        Dict mapping model name to ONNX file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["text_encoder"] = export_text_encoder(
        text_encoder, output_dir / "text_encoder.onnx",
    )
    paths["duration_predictor"] = export_duration_predictor(
        duration_predictor, output_dir / "duration_predictor.onnx",
    )
    paths["f0_predictor"] = export_f0_predictor(
        f0_predictor, output_dir / "f0_predictor.onnx",
    )
    paths["content_synthesizer"] = export_content_synthesizer(
        content_synthesizer, output_dir / "content_synthesizer.onnx",
    )

    logger.info("All 4 TTS models exported to %s", output_dir)
    return paths


# --- StyleEncoder export ---


class _StyleEncoderWrapper(torch.nn.Module):
    """Wrapper for AudioStyleEncoder ONNX export.

    Exports only the audio encoder path: mel[B,80,T] → style[B,32].
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.model(mel)


def export_style_encoder(
    model: torch.nn.Module, output_path: str | Path,
) -> Path:
    """Export StyleEncoder (audio mode) to ONNX.

    Input: mel[B, 80, T] → Output: style[B, 32]
    """
    output_path = Path(output_path)
    prepare_output_path(output_path)
    model.eval()

    wrapper = _StyleEncoderWrapper(model).eval()
    T = 200
    dummy_mel = torch.randn(1, N_MELS, T)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(output_path),
        input_names=["mel"],
        output_names=["style"],
        dynamic_axes={
            "mel": {2: "T"},
        },
        opset_version=_OPSET_VERSION,
    )
    logger.info("Exported style_encoder to %s", output_path)
    return output_path


# --- Parity verification ---


def verify_text_encoder(
    model: torch.nn.Module, onnx_path: str | Path,
    atol: float = 5e-5,
) -> list[dict]:
    """Verify TextEncoder ONNX parity."""
    import onnxruntime as ort

    model.eval()
    onnx_path = str(onnx_path)
    session = ort.InferenceSession(onnx_path)

    results = []
    for name, L in [("short", 10), ("medium", 50), ("long", 150)]:
        phonemes = torch.randint(1, PHONEME_VOCAB_SIZE, (1, L))
        lang = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            pt_out = model(phonemes, lang).numpy()

        ort_out = session.run(None, {
            "phoneme_ids": phonemes.numpy(),
            "language_ids": lang.numpy(),
        })[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        results.append({
            "model": "text_encoder",
            "test": name,
            "max_abs_diff": max_diff,
            "passed": max_diff < atol,
        })
    return results


def verify_duration_predictor(
    model: torch.nn.Module, onnx_path: str | Path,
    atol: float = 5e-5,
) -> list[dict]:
    """Verify DurationPredictor ONNX parity."""
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(onnx_path))

    results = []
    for name, L in [("short", 10), ("long", 80)]:
        features = torch.randn(1, D_TEXT_ENCODER, L)
        style = torch.randn(1, D_STYLE) * 0.5

        with torch.no_grad():
            pt_out = model(features, style).numpy()

        ort_out = session.run(None, {
            "text_features": features.numpy(),
            "style": style.numpy(),
        })[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        results.append({
            "model": "duration_predictor",
            "test": name,
            "max_abs_diff": max_diff,
            "passed": max_diff < atol,
        })
    return results


def verify_f0_predictor(
    model: torch.nn.Module, onnx_path: str | Path,
    atol: float = 5e-5,
) -> list[dict]:
    """Verify F0Predictor ONNX parity."""
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(onnx_path))

    results = []
    for name, T in [("short", 50), ("long", 300)]:
        features = torch.randn(1, D_TEXT_ENCODER, T)
        style = torch.randn(1, D_STYLE) * 0.5

        with torch.no_grad():
            f0_pt, voiced_pt = model(features, style)
            f0_pt = f0_pt.numpy()
            voiced_pt = voiced_pt.numpy()

        ort_outs = session.run(None, {
            "text_features": features.numpy(),
            "style": style.numpy(),
        })
        f0_ort, voiced_ort = ort_outs[0], ort_outs[1]

        for out_name, pt, ort_val in [("f0", f0_pt, f0_ort), ("voiced", voiced_pt, voiced_ort)]:
            max_diff = float(np.abs(pt - ort_val).max())
            results.append({
                "model": "f0_predictor",
                "test": f"{name}_{out_name}",
                "max_abs_diff": max_diff,
                "passed": max_diff < atol,
            })
    return results


def verify_content_synthesizer(
    model: torch.nn.Module, onnx_path: str | Path,
    atol: float = 5e-5,
) -> list[dict]:
    """Verify ContentSynthesizer ONNX parity."""
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(onnx_path))

    results = []
    for name, T in [("short", 50), ("long", 300)]:
        features = torch.randn(1, D_TEXT_ENCODER, T)

        with torch.no_grad():
            pt_out = model(features).numpy()

        ort_out = session.run(None, {
            "text_features": features.numpy(),
        })[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        results.append({
            "model": "content_synthesizer",
            "test": name,
            "max_abs_diff": max_diff,
            "passed": max_diff < atol,
        })
    return results


def verify_style_encoder(
    model: torch.nn.Module, onnx_path: str | Path,
    atol: float = 5e-5,
) -> list[dict]:
    """Verify StyleEncoder ONNX parity."""
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(onnx_path))

    results = []
    for name, T in [("short", 50), ("medium", 200), ("long", 500)]:
        mel = torch.randn(1, N_MELS, T)

        with torch.no_grad():
            pt_out = model(mel).numpy()

        ort_out = session.run(None, {"mel": mel.numpy()})[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        results.append({
            "model": "style_encoder",
            "test": name,
            "max_abs_diff": max_diff,
            "passed": max_diff < atol,
        })
    return results
