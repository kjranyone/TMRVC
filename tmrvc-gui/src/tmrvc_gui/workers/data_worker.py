"""Data preprocessing worker for TMRVC.

Runs the data preparation pipeline (resampling, normalisation, VAD,
segmentation, feature extraction) in a background thread so the GUI
remains responsive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .base_worker import BaseWorker

logger = logging.getLogger(__name__)

# Ordered list of valid preprocessing steps.
VALID_STEPS: list[str] = [
    "resample",
    "normalize",
    "vad_trim",
    "segment",
    "features",
]


class DataWorker(BaseWorker):
    """Background worker for data preprocessing.

    Parameters
    ----------
    config : dict
        Preprocessing configuration with the following keys:

        * **corpus_paths** (*list[Path]*) -- Paths to raw corpus
          directories (e.g. VCTK, JVS, LibriTTS-R, Emilia).
        * **dataset_names** (*list[str]*) -- Dataset name per corpus
          (``"vctk"``, ``"jvs"``, ``"libritts_r"``).
        * **cache_dir** (*Path*) -- Output directory for feature cache.
        * **steps** (*list[str]*) -- Ordered list of preprocessing
          steps to execute.
        * **n_workers** (*int*) -- Number of parallel worker
          processes for CPU-bound stages.
        * **device** (*str*) -- Device for feature extraction models.
    parent : QObject, optional
        Parent Qt object.
    """

    def __init__(self, config: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.corpus_paths: list[Path] = [Path(p) for p in config["corpus_paths"]]
        self.dataset_names: list[str] = config.get("dataset_names", ["vctk"] * len(self.corpus_paths))
        self.cache_dir: Path = Path(config.get("cache_dir", "data/cache"))
        self.steps: list[str] = config.get(
            "steps", ["resample", "normalize", "vad_trim", "segment", "features"]
        )
        self.n_workers: int = config.get("n_workers", 4)
        self.device: str = config.get("device", "cpu")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the preprocessing pipeline.

        Uses tmrvc_data's preprocessing and feature extraction functions
        to process each corpus.  The pipeline follows the same flow as
        the ``tmrvc-preprocess`` CLI.
        """
        from tmrvc_core.audio import compute_mel
        from tmrvc_core.types import FeatureSet
        from tmrvc_data.cache import FeatureCache
        from tmrvc_data.dataset_adapters import get_adapter
        from tmrvc_data.preprocessing import (
            normalize_loudness,
            preprocess_audio,
            segment_utterance,
        )

        do_features = "features" in self.steps
        cache = FeatureCache(self.cache_dir)

        # Lazy-load extractors only when needed
        content_extractor = None
        f0_extractor = None
        spk_encoder = None
        if do_features:
            from tmrvc_data.features import ContentVecExtractor, create_f0_extractor
            from tmrvc_data.speaker import SpeakerEncoder

            self.log_message.emit("[DataWorker] Loading feature extractors...")
            content_extractor = ContentVecExtractor(device=self.device)
            f0_extractor = create_f0_extractor("torchcrepe", device=self.device)
            spk_encoder = SpeakerEncoder(device=self.device)

        self.log_message.emit(
            f"[DataWorker] Starting preprocessing: "
            f"{len(self.corpus_paths)} corpora, "
            f"steps={self.steps}, n_workers={self.n_workers}"
        )

        total_processed = 0
        total_errors = 0

        try:
            for corpus_path, dataset_name in zip(self.corpus_paths, self.dataset_names):
                if self.is_cancelled:
                    self.log_message.emit("[DataWorker] Cancelled by user.")
                    self.finished.emit(False, "Cancelled")
                    return

                self.log_message.emit(
                    f"[DataWorker] Processing corpus: {dataset_name} ({corpus_path})"
                )

                adapter = get_adapter(dataset_name)
                utterances = list(adapter.iter_utterances(corpus_path))
                total_utts = len(utterances)

                self.log_message.emit(
                    f"[DataWorker] Found {total_utts} utterances in {dataset_name}"
                )

                for utt_idx, utt in enumerate(utterances):
                    if self.is_cancelled:
                        self.log_message.emit("[DataWorker] Cancelled by user.")
                        self.finished.emit(False, "Cancelled")
                        return

                    try:
                        # 1. Load, resample, normalize, trim
                        waveform, sr = preprocess_audio(str(utt.audio_path))

                        # 2. Segment
                        for seg_idx, segment in enumerate(segment_utterance(waveform)):
                            seg_id = (
                                f"{utt.utterance_id}_seg{seg_idx}" if seg_idx > 0
                                else utt.utterance_id
                            )

                            # 3. Extract features (if requested)
                            if do_features:
                                mel = compute_mel(segment).squeeze(0)  # [80, T]
                                n_frames = mel.shape[1]

                                content = content_extractor.extract(segment, sr)
                                if content.shape[1] != n_frames:
                                    content = torch.nn.functional.interpolate(
                                        content.unsqueeze(0), size=n_frames,
                                        mode="linear", align_corners=False,
                                    ).squeeze(0)

                                f0 = f0_extractor.extract(segment, sr)
                                if f0.shape[1] != n_frames:
                                    f0 = torch.nn.functional.interpolate(
                                        f0.unsqueeze(0), size=n_frames,
                                        mode="linear", align_corners=False,
                                    ).squeeze(0)

                                spk_embed = spk_encoder.extract(segment, sr)

                                features = FeatureSet(
                                    mel=mel,
                                    content=content,
                                    f0=f0,
                                    spk_embed=spk_embed,
                                    utterance_id=seg_id,
                                    speaker_id=utt.speaker_id,
                                    n_frames=n_frames,
                                )
                                cache.save(features, dataset_name, "train")

                            total_processed += 1

                    except Exception:
                        logger.error(
                            "Failed to process %s", utt.utterance_id, exc_info=True,
                        )
                        self.log_message.emit(
                            f"[DataWorker] Error processing {utt.utterance_id}"
                        )
                        total_errors += 1

                    # Report progress per utterance
                    self.progress.emit(utt_idx + 1, total_utts)

                    if (utt_idx + 1) % 100 == 0:
                        self.log_message.emit(
                            f"[DataWorker] {dataset_name}: {utt_idx + 1}/{total_utts} "
                            f"(processed={total_processed}, errors={total_errors})"
                        )

            self.log_message.emit(
                f"[DataWorker] Preprocessing complete. "
                f"Processed={total_processed}, Errors={total_errors}"
            )
            self.finished.emit(True, "Preprocessing completed successfully")

        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))
