"""Reproducible training pipeline for UCLM v2."""

from __future__ import annotations

import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tmrvc_train.experiment import ExperimentManager

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline with reproducibility guarantees."""

    def __init__(
        self,
        experiment_dir: Path,
        dataset: str,
        raw_dir: Path,
        cache_dir: Path,
        config: dict[str, Any],
        workers: int = 1,
        seed: int = 42,
        skip_preprocess: bool = False,
    ):
        self.experiment_dir = experiment_dir
        self.dataset = dataset
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.config = config
        self.workers = workers
        self.seed = seed
        self.skip_preprocess = skip_preprocess

        self.experiment_manager = ExperimentManager(experiment_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> bool:
        """Execute full training pipeline."""

        experiment_id = self.experiment_dir.name

        metadata = self.experiment_manager.create_experiment(
            experiment_id=experiment_id,
            dataset=self.dataset,
            config=self.config,
            seed=self.seed,
            workers=self.workers,
        )

        logger.info("Experiment: %s", experiment_id)
        logger.info("Dataset: %s", self.dataset)
        logger.info("Workers: %d", self.workers)
        logger.info("Seed: %d", self.seed)
        logger.info("Git hash: %s", metadata.git_hash)

        self._set_seed(self.seed)

        try:
            if not self.skip_preprocess:
                self.experiment_manager.update_status("preprocessing")
                success = self._run_preprocessing()
                if not success:
                    self.experiment_manager.update_status("preprocessing_failed")
                    return False

            self.experiment_manager.update_status("training")
            success = self._run_training()

            if success:
                self.experiment_manager.update_status("completed")
            else:
                self.experiment_manager.update_status("training_failed")

            return success

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            self.experiment_manager.update_status("failed")
            return False

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_preprocessing(self) -> bool:
        """Run parallel preprocessing."""

        from tmrvc_data.dataset_adapters import get_adapter

        adapter_type = self.config.get("adapter_type")
        language = self.config.get("language", "ja")
        speaker_map_path = self.config.get("speaker_map")

        adapter = get_adapter(
            self.dataset,
            adapter_type=adapter_type,
            language=language,
            speaker_map_path=speaker_map_path,
        )

        utterances = list(adapter.iter_utterances(self.raw_dir, "train"))
        logger.info("Found %d utterances in %s", len(utterances), self.dataset)

        if self.workers == 1:
            return self._preprocess_single_worker(utterances)
        else:
            return self._preprocess_multi_worker(utterances)

    def _preprocess_single_worker(self, utterances: list) -> bool:
        """Preprocessing in single process."""

        sys.path.insert(0, str(Path("tmrvc-data/src")))
        sys.path.insert(0, str(Path("tmrvc-core/src")))

        from tmrvc_data.preprocessing import preprocess_single_utterance
        from tmrvc_data.cache import FeatureCache
        from tmrvc_data.codec import UCLMCodecWrapper
        from tmrvc_data.voice_state import SSLVoiceStateEstimator
        from tmrvc_data.speaker import SpeakerEncoder
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"

        codec = UCLMCodecWrapper(None, device=device)
        vs_estimator = SSLVoiceStateEstimator(device=device)
        spk_encoder = SpeakerEncoder(device=device)
        compute_type = "float16" if device == "cuda" else "int8"
        whisper = WhisperModel(
            "large-v3-turbo", device=device, compute_type=compute_type
        )

        models = {
            "codec": codec,
            "vs_estimator": vs_estimator,
            "spk_encoder": spk_encoder,
            "whisper": whisper,
        }

        processed = 0
        errors = 0

        import tqdm

        for utt in tqdm.tqdm(utterances, desc="Preprocessing"):
            try:
                cache_path = (
                    self.cache_dir
                    / self.dataset
                    / "train"
                    / utt.speaker_id
                    / utt.utterance_id
                )

                if (cache_path / "meta.json").exists():
                    processed += 1
                    continue

                preprocess_single_utterance(
                    utt=utt,
                    cache_dir=self.cache_dir,
                    dataset=self.dataset,
                    split="train",
                    device=device,
                    language=self.config.get("language", "ja"),
                    models=models,
                )
                processed += 1

            except Exception as e:
                logger.error("Failed to process %s: %s", utt.utterance_id, e)
                self.experiment_manager.log_error(
                    utterance_id=utt.utterance_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stage="preprocessing",
                )
                errors += 1

        logger.info(
            "Preprocessing complete: %d processed, %d errors", processed, errors
        )
        return errors < len(utterances) * 0.1

    def _preprocess_multi_worker(self, utterances: list) -> bool:
        """Preprocessing with multiple workers."""

        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        chunk_size = (len(utterances) + self.workers - 1) // self.workers
        chunks = [
            utterances[i : i + chunk_size]
            for i in range(0, len(utterances), chunk_size)
        ]

        logger.info(
            "Splitting %d utterances into %d chunks (%d each)",
            len(utterances),
            len(chunks),
            chunk_size,
        )

        processed_total = 0
        errors_total = 0

        with ProcessPoolExecutor(
            max_workers=self.workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(
                    _worker_process_chunk,
                    chunk,
                    self.cache_dir,
                    self.dataset,
                    self.config.get("language", "ja"),
                    worker_id,
                ): worker_id
                for worker_id, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    processed, error_logs = future.result()
                    processed_total += processed
                    errors_total += len(error_logs)
                    all_error_logs.extend(error_logs)
                    logger.info(
                        "Worker %d: %d processed, %d errors",
                        worker_id,
                        processed,
                        len(error_logs),
                    )
                except Exception as e:
                    logger.exception("Worker %d failed: %s", worker_id, e)

        if all_error_logs:
            self.experiment_manager.log_errors(all_error_logs)

        logger.info(
            "Preprocessing complete: %d processed, %d errors",
            processed_total,
            errors_total,
        )

        return errors_total < len(utterances) * 0.1

    def _run_training(self) -> bool:
        """Run UCLM training."""

        from tmrvc_train.cli.train_uclm import main as train_uclm_main

        logger.info("Starting UCLM training...")

        train_args = [
            "--dataset",
            self.dataset,
            "--cache-dir",
            str(self.cache_dir),
            "--output-dir",
            str(self.experiment_dir / "checkpoints"),
            "--config",
            str(self._write_train_config()),
        ]

        if "train_steps" in self.config:
            train_args.extend(["--train-steps", str(self.config["train_steps"])])

        try:
            train_uclm_main(train_args)
            return True
        except Exception as e:
            logger.exception("Training failed: %s", e)
            return False

    def _write_train_config(self) -> Path:
        """Write training configuration file."""
        import yaml

        config_path = self.experiment_dir / "train_config.yaml"

        train_config = {
            "dataset": self.dataset,
            "seed": self.seed,
            **self.config,
        }

        with open(config_path, "w") as f:
            yaml.dump(train_config, f, default_flow_style=False)

        return config_path


def _worker_process_chunk(
    utterances: list,
    cache_dir: Path,
    dataset: str,
    language: str,
    worker_id: int,
) -> tuple[int, list]:
    """Worker function for parallel preprocessing."""

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("tmrvc-core/src")))
    sys.path.insert(0, str(Path("tmrvc-data/src")))

    import torch
    import tqdm
    from tmrvc_data.preprocessing import preprocess_single_utterance
    from tmrvc_data.cache import FeatureCache
    from tmrvc_data.codec import UCLMCodecWrapper
    from tmrvc_data.voice_state import SSLVoiceStateEstimator
    from tmrvc_data.speaker import SpeakerEncoder
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models ONCE per worker (not per utterance!)
    print(f"Worker {worker_id}: Initializing models on {device}...")
    codec = UCLMCodecWrapper(None, device=device)
    vs_estimator = SSLVoiceStateEstimator(device=device)
    spk_encoder = SpeakerEncoder(device=device)
    compute_type = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)

    models = {
        "codec": codec,
        "vs_estimator": vs_estimator,
        "spk_encoder": spk_encoder,
        "whisper": whisper,
    }

    print(f"Worker {worker_id}: Models initialized, starting processing...")

    cache = FeatureCache(cache_dir)
    processed = 0
    error_logs = []

    for utt in tqdm.tqdm(utterances, desc=f"Worker {worker_id}"):
        try:
            # Check if already cached
            if cache.exists(dataset, "train", utt.speaker_id, utt.utterance_id):
                processed += 1
                continue

            success = preprocess_single_utterance(
                utt=utt,
                cache_dir=cache_dir,
                dataset=dataset,
                split="train",
                device=device,
                language=language,
                models=models,
            )

            if success:
                processed += 1
            else:
                error_logs.append(
                    {
                        "utterance_id": utt.utterance_id,
                        "error_type": "Skipped",
                        "error_message": "Duration out of range or other skip reason",
                        "stage": "preprocessing",
                    }
                )

        except Exception as e:
            print(f"Worker {worker_id} error on {utt.utterance_id}: {e}")
            import traceback

            traceback.print_exc()
            error_logs.append(
                {
                    "utterance_id": utt.utterance_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "stage": "preprocessing",
                }
            )

    print(f"Worker {worker_id} done: {processed} processed, {len(error_logs)} errors")
    return processed, error_logs
