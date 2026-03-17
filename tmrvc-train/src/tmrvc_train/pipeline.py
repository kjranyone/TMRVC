"""Reproducible training pipeline for UCLM v3."""

from __future__ import annotations

import logging
import random
import sys
import gc
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tmrvc_train.experiment import ExperimentManager
from tmrvc_core.constants import HOP_LENGTH

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
        run_training: bool = True,
        train_datasets: list[str] | None = None,
        base_checkpoint: Path | None = None,
    ):
        self.experiment_dir = experiment_dir
        self.dataset = dataset
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.config = config
        self.workers = workers
        self.seed = seed
        self.skip_preprocess = skip_preprocess
        self.run_training = run_training
        self.train_datasets = train_datasets
        self.base_checkpoint = base_checkpoint

        self.experiment_manager = ExperimentManager(experiment_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> bool:
        """Execute full training pipeline."""

        experiment_id = self.experiment_dir.name

        metadata_dataset = (
            ",".join(self.train_datasets)
            if self.train_datasets is not None
            else self.dataset
        )
        metadata = self.experiment_manager.create_experiment(
            experiment_id=experiment_id,
            dataset=metadata_dataset,
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

            if not self.run_training:
                self.experiment_manager.update_status("preprocessed")
                return True

            self._cleanup_cuda_memory()
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

    def _cleanup_cuda_memory(self) -> None:
        """Best-effort cleanup between preprocessing and training."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                "CUDA memory after cleanup: allocated=%.2f GB, reserved=%.2f GB",
                allocated,
                reserved,
            )

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
        if not utterances:
            logger.error("No utterances found for dataset=%s in raw_dir=%s", self.dataset, self.raw_dir)
            return False

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
        del models, codec, vs_estimator, spk_encoder, whisper
        if device == "cuda":
            torch.cuda.empty_cache()
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
        all_error_logs = []

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
        datasets_for_training = self.train_datasets or [self.dataset]
        missing_dirs: list[Path] = []
        empty_dirs: list[Path] = []
        for dataset in datasets_for_training:
            cache_train_dir = self.cache_dir / dataset / "train"
            if not cache_train_dir.exists():
                missing_dirs.append(cache_train_dir)
                continue
            if not any(cache_train_dir.rglob("meta.json")):
                empty_dirs.append(cache_train_dir)

        for p in missing_dirs:
            logger.error("Cache directory not found: %s", p)
        for p in empty_dirs:
            logger.error("No cached utterances found under: %s", p)
        if missing_dirs or empty_dirs:
            return False

        if not self._run_cache_quality_gate(datasets_for_training):
            return False

        config_path = self._write_train_config()
        train_args = [
            "--cache-dir",
            str(self.cache_dir),
            "--output-dir",
            str(self.experiment_dir / "checkpoints"),
        ]
        if datasets_for_training:
            train_args.extend(["--datasets", ",".join(datasets_for_training)])
        logger.info("Training config snapshot: %s", config_path)

        train_args.extend(["--seed", str(self.seed)])
        if "train_steps" in self.config:
            train_args.extend(["--train-steps", str(self.config["train_steps"])])
        if "train_batch_size" in self.config:
            train_args.extend(["--batch-size", str(self.config["train_batch_size"])])
        if "train_device" in self.config:
            train_args.extend(["--device", str(self.config["train_device"])])
        if "train_sampling_strategy" in self.config:
            train_args.extend(
                ["--sampling-strategy", str(self.config["train_sampling_strategy"])]
            )
        if bool(self.config.get("train_require_tts_supervision", False)):
            train_args.append("--require-tts-supervision")
        
        if self.base_checkpoint:
            train_args.extend(["--base-checkpoint", str(self.base_checkpoint)])
        tts_mode = self.config.get("tts_mode", "pointer")
        if tts_mode in ("pointer", "legacy_duration"):
            train_args.extend(["--tts-mode", tts_mode])
        if "pointer_loss_weight" in self.config:
            train_args.extend(["--pointer-loss-weight", str(self.config["pointer_loss_weight"])])
        if "progress_loss_weight" in self.config:
            train_args.extend(["--progress-loss-weight", str(self.config["progress_loss_weight"])])
        alignment_loss_type = self.config.get("alignment_loss_type", "none")
        if alignment_loss_type in ("none", "mas", "ctc"):
            train_args.extend(["--alignment-loss-type", alignment_loss_type])
        if "pointer_target_source" in self.config:
            train_args.extend(["--pointer-target-source", str(self.config["pointer_target_source"])])
        if "legacy_duration_loss_weight" in self.config:
            train_args.extend(["--legacy-duration-loss-weight", str(self.config["legacy_duration_loss_weight"])])
        if "voice_state_loss_weight" in self.config:
            train_args.extend(["--voice-state-loss-weight", str(self.config["voice_state_loss_weight"])])
        if "delta_voice_state_loss_weight" in self.config:
            train_args.extend(["--delta-voice-state-loss-weight", str(self.config["delta_voice_state_loss_weight"])])

        try:
            train_uclm_main(train_args)
            return True
        except Exception as e:
            logger.exception("Training failed: %s", e)
            return False

    def _run_cache_quality_gate(self, datasets_for_training: list[str]) -> bool:
        """Validate cache integrity and token ranges before training."""
        if not bool(self.config.get("quality_gate_enabled", True)):
            logger.info("Quality gate disabled by config.")
            return True

        max_invalid_ratio = float(self.config.get("quality_gate_max_invalid_ratio", 0.0))
        min_valid_utterances = int(
            self.config.get("quality_gate_min_utterances_per_dataset", 1)
        )
        min_speakers = int(self.config.get("quality_gate_min_speakers_per_dataset", 1))
        token_samples = int(self.config.get("quality_gate_token_samples", 64))
        rvq_vocab_size = int(self.config.get("rvq_vocab_size", 1024))
        control_vocab_size = int(self.config.get("control_vocab_size", 64))
        allow_waveform_tail_remainder = bool(
            self.config.get("quality_gate_allow_waveform_tail_remainder", True)
        )

        required = (
            "meta.json",
            "codec_tokens.npy",
            "explicit_state.npy",
            "ssl_state.npy",
            "spk_embed.npy",
        )

        report: dict[str, Any] = {"datasets": {}, "status": "ok"}
        ok = True
        rng = random.Random(self.seed)

        for dataset in datasets_for_training:
            base = self.cache_dir / dataset / "train"
            utt_dirs = [p for p in base.glob("*/*") if p.is_dir()]
            total = len(utt_dirs)
            valid_dirs: list[Path] = []
            invalid_missing = 0
            speakers: set[str] = set()

            for utt_dir in utt_dirs:
                if all((utt_dir / f).exists() for f in required):
                    valid_dirs.append(utt_dir)
                    speakers.add(utt_dir.parent.name)
                else:
                    invalid_missing += 1

            valid = len(valid_dirs)
            invalid_ratio = 0.0 if total == 0 else float(invalid_missing) / float(total)

            token_errors = 0
            waveform_length_errors = 0
            waveform_tail_remainders = 0
            waveform_checked = 0
            if valid_dirs and token_samples > 0:
                sample_count = min(token_samples, len(valid_dirs))
                sampled = rng.sample(valid_dirs, sample_count)
                for utt_dir in sampled:
                    try:
                        codec_tokens = np.load(utt_dir / "codec_tokens.npy")
                        if (
                            codec_tokens.size == 0
                            or np.min(codec_tokens) < 0
                            or np.max(codec_tokens) >= rvq_vocab_size
                        ):
                            token_errors += 1
                            continue

                        control_path = utt_dir / "control_tokens.npy"
                        if control_path.exists():
                            control_tokens = np.load(control_path)
                            if (
                                control_tokens.size == 0
                                or np.min(control_tokens) < 0
                                or np.max(control_tokens) >= control_vocab_size
                            ):
                                token_errors += 1
                    except Exception:
                        token_errors += 1

            # Check waveform/sample alignment against n_frames for all entries that include waveform.npy.
            # This is lightweight (header/meta reads only) and catches frame drift early.
            for utt_dir in valid_dirs:
                waveform_path = utt_dir / "waveform.npy"
                if not waveform_path.exists():
                    continue
                waveform_checked += 1
                try:
                    with open(utt_dir / "meta.json", encoding="utf-8") as f:
                        meta = json.load(f)
                    n_frames = int(meta.get("n_frames", -1))
                    if n_frames < 0:
                        waveform_length_errors += 1
                        continue
                    expected_samples = n_frames * int(HOP_LENGTH)
                    waveform = np.load(waveform_path, mmap_mode="r")
                    actual_samples = int(waveform.shape[-1])
                    if actual_samples != expected_samples:
                        diff = abs(actual_samples - expected_samples)
                        floor_frames = actual_samples // int(HOP_LENGTH)
                        ceil_frames = (actual_samples + int(HOP_LENGTH) - 1) // int(HOP_LENGTH)
                        if (
                            allow_waveform_tail_remainder
                            and diff < int(HOP_LENGTH)
                            and n_frames in {floor_frames, ceil_frames}
                        ):
                            waveform_tail_remainders += 1
                        else:
                            waveform_length_errors += 1
                except Exception:
                    waveform_length_errors += 1

            # Text supervision coverage (v3)
            text_supervised = 0
            legacy_duration_supervised = 0
            for utt_dir in valid_dirs:
                has_phonemes = (utt_dir / "phoneme_ids.npy").exists()
                has_durations = (utt_dir / "durations.npy").exists()
                if has_phonemes:
                    text_supervised += 1
                if has_phonemes and has_durations:
                    legacy_duration_supervised += 1

            dataset_report = {
                "total_entries": total,
                "valid_entries": valid,
                "invalid_missing_files": invalid_missing,
                "invalid_ratio": invalid_ratio,
                "unique_speakers": len(speakers),
                "token_errors": token_errors,
                "token_samples": min(token_samples, len(valid_dirs)),
                "waveform_checked": waveform_checked,
                "waveform_tail_remainders": waveform_tail_remainders,
                "waveform_length_errors": waveform_length_errors,
                "text_supervised": text_supervised,
                "legacy_duration_supervised": legacy_duration_supervised,
                "pointer_target_coverage": text_supervised,
                "canonical_text_unit_coverage": text_supervised / max(len(valid_dirs), 1),
            }
            report["datasets"][dataset] = dataset_report

            if valid < min_valid_utterances:
                logger.error(
                    "Quality gate failed [%s]: valid_entries=%d < min=%d",
                    dataset,
                    valid,
                    min_valid_utterances,
                )
                ok = False
            if len(speakers) < min_speakers:
                logger.error(
                    "Quality gate failed [%s]: unique_speakers=%d < min=%d",
                    dataset,
                    len(speakers),
                    min_speakers,
                )
                ok = False
            if invalid_ratio > max_invalid_ratio:
                logger.error(
                    "Quality gate failed [%s]: invalid_ratio=%.4f > max=%.4f",
                    dataset,
                    invalid_ratio,
                    max_invalid_ratio,
                )
                ok = False
            if token_errors > 0:
                logger.error(
                    "Quality gate failed [%s]: token_errors=%d (sampled=%d)",
                    dataset,
                    token_errors,
                    dataset_report["token_samples"],
                )
                ok = False
            if waveform_tail_remainders > 0:
                logger.warning(
                    "Quality gate warning [%s]: waveform_tail_remainders=%d (checked=%d, hop=%d)",
                    dataset,
                    waveform_tail_remainders,
                    waveform_checked,
                    HOP_LENGTH,
                )
            if waveform_length_errors > 0:
                logger.error(
                    "Quality gate failed [%s]: waveform_length_errors=%d (checked=%d, expected=n_frames*%d)",
                    dataset,
                    waveform_length_errors,
                    waveform_checked,
                    HOP_LENGTH,
                )
                ok = False

        report["status"] = "ok" if ok else "failed"
        report_path = self.experiment_dir / "quality_gate_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Quality gate report: %s", report_path)

        return ok

    def _write_train_config(self) -> Path:
        """Write training configuration file."""
        import yaml

        config_path = self.experiment_dir / "train_config.yaml"

        train_config = {
            "dataset": self.dataset,
            "train_datasets": self.train_datasets or [self.dataset],
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
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                print(f"Worker {worker_id}: CUDA OOM detected, clearing cache and continuing...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
    del models, codec, vs_estimator, spk_encoder, whisper
    if device == "cuda":
        torch.cuda.empty_cache()
    return processed, error_logs
