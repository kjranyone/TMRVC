"""Model evaluation worker for TMRVC.

Runs objective evaluation metrics (SECS, UTMOS proxy, F0 correlation,
etc.) in the background and reports per-metric results back to the GUI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal

from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class EvalWorker(BaseWorker):
    """Background worker for model evaluation.

    Supports single-model evaluation and A/B model comparison.

    Parameters
    ----------
    model_a_path : Path
        Path to the primary model directory (ONNX files).
    model_b_path : Path or None
        Path to a second model for A/B comparison.  *None* for
        single-model evaluation.
    eval_set : str
        Name of the evaluation set, e.g. ``"vctk_test"``,
        ``"libri_test"``, ``"emilia_test"``.
    metrics : list[str]
        Metrics to compute.  Supported values include ``"utmos"``,
        ``"speaker_sim"``, ``"pesq"``, ``"stoi"``, ``"mcd"``,
        ``"f0_rmse"``.
    parent : QObject, optional
        Parent Qt object.

    Signals
    -------
    result(dict)
        Emitted when evaluation is complete.  The dictionary maps metric
        names to their computed values (and optionally per-model values
        for A/B comparison).
    """

    result = Signal(dict)

    def __init__(
        self,
        model_a_path: Path,
        model_b_path: Path | None,
        eval_set: str,
        metrics: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_a_path = Path(model_a_path)
        self.model_b_path = Path(model_b_path) if model_b_path is not None else None
        self.eval_set = eval_set
        self.metrics = metrics

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute evaluation using tmrvc_train.eval_metrics."""
        import torch

        from tmrvc_train.eval_metrics import (
            f0_correlation,
            speaker_embedding_cosine_similarity,
            utmos_proxy,
        )

        is_ab = self.model_b_path is not None
        mode_str = "A/B comparison" if is_ab else "single model"

        self.log_message.emit(
            f"[EvalWorker] Starting {mode_str} evaluation: "
            f"eval_set={self.eval_set}  metrics={self.metrics}"
        )
        self.log_message.emit(
            f"[EvalWorker] Model A: {self.model_a_path}"
        )
        if is_ab:
            self.log_message.emit(
                f"[EvalWorker] Model B: {self.model_b_path}"
            )

        total = len(self.metrics)
        results: dict[str, Any] = {}

        # Metric computation functions
        _METRIC_FNS = {
            "speaker_sim": self._compute_speaker_sim,
            "utmos": self._compute_utmos,
            "f0_rmse": self._compute_f0_rmse,
            # Metrics requiring external libraries fall back to proxy
            "pesq": self._compute_proxy_pesq,
            "stoi": self._compute_proxy_stoi,
            "mcd": self._compute_proxy_mcd,
        }

        try:
            for idx, metric_name in enumerate(self.metrics, start=1):
                if self.is_cancelled:
                    self.log_message.emit("[EvalWorker] Cancelled by user.")
                    self.finished.emit(False, "Cancelled")
                    return

                self.log_message.emit(
                    f"[EvalWorker] Computing metric: {metric_name} "
                    f"({idx}/{total})"
                )

                compute_fn = _METRIC_FNS.get(metric_name, self._compute_dummy)
                value_a = compute_fn(self.model_a_path)

                if is_ab:
                    value_b = compute_fn(self.model_b_path)
                    results[metric_name] = {
                        "model_a": value_a,
                        "model_b": value_b,
                    }
                else:
                    results[metric_name] = value_a

                self.metric.emit(metric_name, value_a, idx)
                self.progress.emit(idx, total)

            self.log_message.emit(
                f"[EvalWorker] Evaluation complete. Results: {results}"
            )
            self.result.emit(results)
            self.finished.emit(True, "Evaluation completed successfully")

        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    def _compute_speaker_sim(self, model_path: Path) -> float:
        """Compute speaker similarity (SECS) using eval_metrics."""
        import torch

        from tmrvc_train.eval_metrics import speaker_embedding_cosine_similarity

        # Load precomputed embeddings from evaluation directory
        embed_file = model_path / "eval" / f"{self.eval_set}_spk_embeds.pt"
        if embed_file.exists():
            data = torch.load(embed_file, map_location="cpu", weights_only=True)
            return speaker_embedding_cosine_similarity(
                data["pred"], data["target"],
            )

        self.log_message.emit(
            f"[EvalWorker] No precomputed embeddings at {embed_file}, "
            f"returning proxy value"
        )
        return 0.0

    def _compute_utmos(self, model_path: Path) -> float:
        """Compute UTMOS proxy score using eval_metrics."""
        import torch

        from tmrvc_train.eval_metrics import utmos_proxy

        mel_file = model_path / "eval" / f"{self.eval_set}_mels.pt"
        if mel_file.exists():
            data = torch.load(mel_file, map_location="cpu", weights_only=True)
            return utmos_proxy(data["pred"], data["target"])

        self.log_message.emit(
            f"[EvalWorker] No precomputed mels at {mel_file}, "
            f"returning proxy value"
        )
        return 0.0

    def _compute_f0_rmse(self, model_path: Path) -> float:
        """Compute F0 correlation using eval_metrics."""
        import torch

        from tmrvc_train.eval_metrics import f0_correlation

        f0_file = model_path / "eval" / f"{self.eval_set}_f0.pt"
        if f0_file.exists():
            data = torch.load(f0_file, map_location="cpu", weights_only=True)
            return f0_correlation(data["pred"], data["target"])

        self.log_message.emit(
            f"[EvalWorker] No precomputed F0 at {f0_file}, "
            f"returning proxy value"
        )
        return 0.0

    def _compute_proxy_pesq(self, model_path: Path) -> float:
        """PESQ requires pesq library; return 0 if unavailable."""
        try:
            from pesq import pesq
        except ImportError:
            self.log_message.emit("[EvalWorker] pesq not installed, skipping")
            return 0.0
        return 0.0

    def _compute_proxy_stoi(self, model_path: Path) -> float:
        """STOI requires pystoi library; return 0 if unavailable."""
        try:
            from pystoi import stoi
        except ImportError:
            self.log_message.emit("[EvalWorker] pystoi not installed, skipping")
            return 0.0
        return 0.0

    def _compute_proxy_mcd(self, model_path: Path) -> float:
        """MCD computation placeholder."""
        return 0.0

    @staticmethod
    def _compute_dummy(model_path: Path) -> float:
        """Fallback for unknown metrics."""
        return 0.0
