#!/usr/bin/env python3
"""End-to-end training pipeline: Preprocess → Teacher → Distill → Export.

Usage::

    uv run python scripts/run_pipeline.py --vctk-dir data/raw/VCTK-Corpus-0.92

This runs a small-scale validation pipeline (CPU-friendly):
  - Preprocess 50 utterances from VCTK
  - Train Teacher U-Net for 500 steps
  - Distill to Student models for 200 steps
  - Export to ONNX + verify parity

License: VCTK = CC BY 4.0 (safe for commercial use with attribution)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(cmd: list[str], desc: str) -> None:
    """Run a subprocess, streaming output."""
    logger.info("=" * 60)
    logger.info("STEP: %s", desc)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    if result.returncode != 0:
        logger.error("FAILED: %s (exit code %d)", desc, result.returncode)
        sys.exit(result.returncode)
    logger.info("DONE: %s", desc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full training pipeline")
    parser.add_argument("--vctk-dir", type=Path, required=True, help="Path to VCTK root")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Feature cache dir")
    parser.add_argument("--max-utterances", type=int, default=50, help="Utterances to preprocess")
    parser.add_argument("--teacher-steps", type=int, default=500, help="Teacher training steps")
    parser.add_argument("--distill-steps", type=int, default=200, help="Distillation steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (small for CPU)")
    parser.add_argument("--device", default="cpu", help="Training device")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    uv = "uv"

    # 1. Preprocess
    run(
        [uv, "run", "tmrvc-preprocess",
         "--dataset", "vctk",
         "--raw-dir", str(args.vctk_dir),
         "--cache-dir", str(args.cache_dir),
         "--max-utterances", str(args.max_utterances),
         "--skip-existing",
         "-v"],
        "Preprocess VCTK (50 utterances)",
    )

    # 2. Verify cache
    run(
        [uv, "run", "tmrvc-verify-cache",
         "--cache-dir", str(args.cache_dir),
         "--dataset", "vctk"],
        "Verify feature cache",
    )

    # 3. Teacher training
    ckpt_dir = Path("checkpoints")
    run(
        [uv, "run", "tmrvc-train-teacher",
         "--config", "configs/train_teacher.yaml",
         "--cache-dir", str(args.cache_dir),
         "--dataset", "vctk",
         "--phase", "0",
         "--max-steps", str(args.teacher_steps),
         "--batch-size", str(args.batch_size),
         "--checkpoint-dir", str(ckpt_dir),
         "--device", args.device,
         "-v"],
        f"Train Teacher ({args.teacher_steps} steps)",
    )

    # Find latest teacher checkpoint
    teacher_ckpts = sorted(ckpt_dir.glob("teacher_step*.pt"))
    if not teacher_ckpts:
        # Trainer might save as different name, check what's there
        teacher_ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not teacher_ckpts:
        logger.error("No teacher checkpoint found in %s", ckpt_dir)
        sys.exit(1)
    teacher_ckpt = teacher_ckpts[-1]
    logger.info("Using teacher checkpoint: %s", teacher_ckpt)

    # 4. Distillation
    distill_ckpt_dir = Path("checkpoints/distill")
    run(
        [uv, "run", "tmrvc-distill",
         "--config", "configs/train_student.yaml",
         "--cache-dir", str(args.cache_dir),
         "--dataset", "vctk",
         "--teacher-ckpt", str(teacher_ckpt),
         "--phase", "A",
         "--max-steps", str(args.distill_steps),
         "--batch-size", str(args.batch_size),
         "--checkpoint-dir", str(distill_ckpt_dir),
         "--device", args.device,
         "-v"],
        f"Distill Phase A ({args.distill_steps} steps)",
    )

    # Find latest distill checkpoint
    distill_ckpts = sorted(distill_ckpt_dir.glob("distill_step*.pt"))
    if not distill_ckpts:
        distill_ckpts = sorted(distill_ckpt_dir.glob("*.pt"))
    if not distill_ckpts:
        logger.error("No distill checkpoint found in %s", distill_ckpt_dir)
        sys.exit(1)
    distill_ckpt = distill_ckpts[-1]
    logger.info("Using distill checkpoint: %s", distill_ckpt)

    # 5. ONNX export
    run(
        [uv, "run", "tmrvc-export",
         "--checkpoint", str(distill_ckpt),
         "--output-dir", "models",
         "--verify",
         "-v"],
        "Export ONNX + verify parity",
    )

    # 6. Generate test speaker file (from first few VCTK utterances)
    logger.info("=" * 60)
    logger.info("STEP: Generate test speaker file")
    logger.info("=" * 60)

    # Find a few audio files from VCTK for speaker enrollment
    wav_dir = args.vctk_dir / "wav48_silence_trimmed"
    if not wav_dir.exists():
        wav_dir = args.vctk_dir / "wav48"
    test_wavs = []
    for spk_dir in sorted(wav_dir.iterdir()):
        if spk_dir.is_dir():
            wavs = sorted(spk_dir.glob("*_mic1.flac"))[:3]
            test_wavs.extend(str(w) for w in wavs)
            if len(test_wavs) >= 3:
                break

    if test_wavs:
        run(
            [uv, "run", "python", "scripts/generate_speaker_file.py",
             "--audio"] + test_wavs + [
             "--name", "test_speaker",
             "--output", "models/test_speaker.tmrvc_speaker",
             "-v"],
            "Generate test speaker file",
        )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("")
    logger.info("ONNX models:  models/fp32/*.onnx")
    logger.info("Speaker file: models/test_speaker.tmrvc_speaker")
    logger.info("")
    logger.info("Next: Launch GUI with 'uv run tmrvc-gui'")
    logger.info("  -> Go to Realtime Demo page")
    logger.info("  -> Set ONNX model dir to 'models/fp32'")
    logger.info("  -> Set speaker file to 'models/test_speaker.tmrvc_speaker'")
    logger.info("  -> Click Start")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
