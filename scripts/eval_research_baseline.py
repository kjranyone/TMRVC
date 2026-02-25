#!/usr/bin/env python3
"""Deterministic baseline evaluation for research ablation (WP0).

Loads a Teacher U-Net checkpoint, runs flow-matching sampling on a fixed
test split, computes objective metrics, and saves reproducible JSON results.

Two consecutive runs with the same ``--seed`` must produce identical metrics
within floating-point tolerance.

Usage::

    uv run python scripts/eval_research_baseline.py \
        --config configs/research/b0.yaml \
        --checkpoint checkpoints/teacher_step100000.pt \
        --cache-dir data/cache \
        --seed 42 \
        --device cuda \
        --output-dir eval/research/b0

    # Verify reproducibility
    uv run python scripts/eval_research_baseline.py \
        --config configs/research/b0.yaml \
        --checkpoint checkpoints/teacher_step100000.pt \
        --cache-dir data/cache \
        --seed 42 \
        --device cuda \
        --output-dir eval/research/b0_check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from tmrvc_core.constants import (
    D_CONTENT_VEC,
    HOP_LENGTH,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from tmrvc_data.cache import FeatureCache
from tmrvc_data.speaker import SpeakerEncoder
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.eval_metrics import (
    f0_correlation,
    speaker_embedding_cosine_similarity,
    utmos_proxy,
)
from tmrvc_train.models.teacher_unet import TeacherUNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class UtteranceResult:
    speaker_id: str
    utterance_id: str
    n_frames: int
    mel_mse: float
    secs: float
    f0_corr: float
    utmos: float


@dataclass
class EvalResult:
    variant: str
    checkpoint: str
    seed: int
    sampling_steps: int
    sway_coefficient: float
    cfg_scale: float
    n_utterances: int
    elapsed_sec: float
    per_utterance: list[UtteranceResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Griffin-Lim
# ---------------------------------------------------------------------------

def griffin_lim(
    log_mel: torch.Tensor,
    n_iter: int = 60,
) -> torch.Tensor:
    """Convert log-mel spectrogram to waveform via Griffin-Lim.

    Args:
        log_mel: [B, n_mels, T] log-mel spectrogram.
        n_iter: Number of Griffin-Lim iterations.

    Returns:
        [B, T_samples] waveform.
    """
    from tmrvc_core.audio import create_mel_filterbank

    device = log_mel.device
    mel_linear = log_mel.exp()

    mel_basis = create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE, MEL_FMIN, MEL_FMAX)
    mel_pinv = torch.linalg.pinv(mel_basis).to(device=device, dtype=mel_linear.dtype)

    power = torch.matmul(mel_pinv, mel_linear).clamp(min=0)
    magnitude = power.sqrt()

    window = torch.hann_window(WINDOW_LENGTH, periodic=True, device=device)
    n_freq = N_FFT // 2 + 1
    T = magnitude.shape[-1]

    phase = torch.randn(magnitude.shape[0], n_freq, T, device=device) * 2 * torch.pi

    for _ in range(n_iter):
        stft_complex = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(
            stft_complex, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH, window=window, center=True,
        )
        stft_new = torch.stft(
            waveform, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH, window=window, center=True,
            return_complex=True,
        )
        phase = stft_new.angle()

    stft_complex = magnitude * torch.exp(1j * phase)
    waveform = torch.istft(
        stft_complex, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH, window=window, center=True,
    )
    return waveform


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Test set resolver
# ---------------------------------------------------------------------------

def resolve_test_set(
    cache: FeatureCache,
    config: dict,
) -> list[tuple[str, str, str]]:
    """Resolve (dataset, speaker_id, utterance_id) triples from config.

    Returns:
        Sorted list of (dataset, speaker_id, utterance_id).
    """
    test_cfg = config["test_split"]
    datasets = test_cfg["datasets"]
    allowed_speakers = set(test_cfg.get("speakers", []))
    max_per_speaker = test_cfg.get("max_utterances_per_speaker", 10)

    triples: list[tuple[str, str, str]] = []
    for ds in datasets:
        entries = cache.iter_entries(ds, split="train")
        # Group by speaker
        by_speaker: dict[str, list[str]] = {}
        for e in entries:
            sid = e["speaker_id"]
            # Match speaker: prefix with dataset name for cross-dataset uniqueness
            full_sid = f"{ds}_{sid}"
            if allowed_speakers and full_sid not in allowed_speakers:
                continue
            by_speaker.setdefault(sid, []).append(e["utterance_id"])

        for sid, utts in sorted(by_speaker.items()):
            utts_sorted = sorted(utts)[:max_per_speaker]
            for uid in utts_sorted:
                triples.append((ds, sid, uid))

    triples.sort()
    return triples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_teacher(checkpoint_path: Path, device: torch.device) -> tuple[TeacherUNet, int]:
    """Load TeacherUNet from checkpoint.

    Returns:
        (model, training_step)
    """
    teacher = TeacherUNet(d_content=D_CONTENT_VEC).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"Unsupported checkpoint format: {checkpoint_path} "
            "(missing 'model_state_dict'). Legacy checkpoints are not supported."
        )

    state_dict = ckpt["model_state_dict"]
    model_sd = teacher.state_dict()

    missing_keys = sorted(set(model_sd) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(model_sd))
    shape_mismatches = [
        (k, tuple(state_dict[k].shape), tuple(model_sd[k].shape))
        for k in model_sd.keys() & state_dict.keys()
        if state_dict[k].shape != model_sd[k].shape
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing={missing_keys[:5]}")
        if unexpected_keys:
            details.append(f"unexpected={unexpected_keys[:5]}")
        if shape_mismatches:
            k, got, expected = shape_mismatches[0]
            details.append(f"shape_mismatch={k}: {got} != {expected}")
        raise RuntimeError(
            "Checkpoint is incompatible with current TeacherUNet and was rejected. "
            + "; ".join(details)
        )

    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    step = ckpt.get("step")
    if not isinstance(step, int):
        raise RuntimeError(
            f"Unsupported checkpoint metadata: {checkpoint_path} "
            "(missing/invalid 'step')."
        )
    return teacher, step


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    teacher: TeacherUNet,
    cache: FeatureCache,
    test_set: list[tuple[str, str, str]],
    config: dict,
    device: torch.device,
    output_dir: Path | None,
    seed: int,
    speaker_encoder: SpeakerEncoder | None = None,
) -> EvalResult:
    """Run deterministic evaluation on the test set."""
    sampling_cfg = config.get("sampling", {})
    steps = sampling_cfg.get("steps", 32)
    sway = sampling_cfg.get("sway_coefficient", 1.0)
    cfg_scale = sampling_cfg.get("cfg_scale", 1.0)
    save_audio = config.get("output", {}).get("save_audio", False)
    griffin_lim_iters = int(config.get("evaluation", {}).get("griffin_lim_iters", 32))
    if griffin_lim_iters <= 0:
        raise ValueError("evaluation.griffin_lim_iters must be >= 1")

    scheduler = FlowMatchingScheduler()
    results: list[UtteranceResult] = []
    speaker_encoder = speaker_encoder or SpeakerEncoder(device=str(device))

    if output_dir and save_audio:
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()

    for i, (ds, sid, uid) in enumerate(test_set):
        # Re-seed per utterance for exact reproducibility even if order changes
        seed_everything(seed + i)

        # Load features
        fs = cache.load(ds, "train", sid, uid, mmap=False)
        mel_gt = fs.mel.unsqueeze(0).to(device)       # [1, 80, T]
        content = fs.content.unsqueeze(0).to(device)   # [1, C, T]
        f0 = fs.f0.unsqueeze(0).to(device)             # [1, 1, T]
        spk_embed = fs.spk_embed.unsqueeze(0).to(device)  # [1, 192]
        T = mel_gt.shape[-1]

        # Sample
        with torch.no_grad():
            if cfg_scale > 1.0:
                mel_pred = scheduler.sample_cfg(
                    teacher, shape=(1, N_MELS, T), steps=steps,
                    device=str(device), cfg_scale=cfg_scale,
                    sway_coefficient=sway,
                    content=content, f0=f0, spk_embed=spk_embed,
                )
            else:
                mel_pred = scheduler.sample(
                    teacher, shape=(1, N_MELS, T), steps=steps,
                    device=str(device), sway_coefficient=sway,
                    content=content, f0=f0, spk_embed=spk_embed,
                )

        # Metrics
        mel_mse = (mel_pred - mel_gt).pow(2).mean().item()
        wav_pred = griffin_lim(mel_pred.detach().cpu(), n_iter=griffin_lim_iters)
        pred_embed = speaker_encoder.extract(wav_pred, sample_rate=SAMPLE_RATE)
        secs_val = speaker_embedding_cosine_similarity(
            pred_embed,
            fs.spk_embed.cpu(),
        )
        f0_corr_val = f0_correlation(f0.squeeze(), f0.squeeze())
        utmos_val = utmos_proxy(mel_pred.squeeze(0), mel_gt.squeeze(0))

        result = UtteranceResult(
            speaker_id=f"{ds}_{sid}",
            utterance_id=uid,
            n_frames=T,
            mel_mse=round(mel_mse, 6),
            secs=round(secs_val, 4),
            f0_corr=round(f0_corr_val, 4),
            utmos=round(utmos_val, 4),
        )
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_set):
            logger.info(
                "  [%d/%d] %s/%s: mel_mse=%.4f secs=%.4f utmos=%.2f",
                i + 1, len(test_set), result.speaker_id, uid,
                mel_mse, secs_val, utmos_val,
            )

        # Save audio
        if output_dir and save_audio:
            import soundfile as sf

            wav_np = wav_pred.squeeze(0).numpy()
            peak = np.abs(wav_np).max()
            if peak > 0:
                wav_np = wav_np / peak * 0.95
            sf.write(
                str(audio_dir / f"{result.speaker_id}_{uid}.wav"),
                wav_np, SAMPLE_RATE,
            )

    elapsed = time.monotonic() - t_start

    # Aggregate
    aggregate = {}
    for metric in ["mel_mse", "secs", "f0_corr", "utmos"]:
        vals = [getattr(r, metric) for r in results]
        aggregate[metric] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
        }

    return EvalResult(
        variant=config.get("variant", "b0"),
        checkpoint=str(config.get("_checkpoint_path", "")),
        seed=seed,
        sampling_steps=steps,
        sway_coefficient=sway,
        cfg_scale=cfg_scale,
        n_utterances=len(results),
        elapsed_sec=round(elapsed, 2),
        per_utterance=results,
        aggregate=aggregate,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="eval_research_baseline",
        description="Deterministic baseline evaluation for research ablation.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Research config YAML (e.g. configs/research/b0.yaml).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Teacher checkpoint path.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Feature cache directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/xpu).")
    parser.add_argument(
        "--griffin-lim-iters",
        type=int,
        default=32,
        help="Number of Griffin-Lim iterations used for SECS/audio reconstruction.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_checkpoint_path"] = str(args.checkpoint)
    config.setdefault("evaluation", {})["griffin_lim_iters"] = args.griffin_lim_iters

    variant = config.get("variant", "b0")
    output_dir = args.output_dir or Path(f"eval/research/{variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Research Baseline Evaluation ===")
    logger.info("Variant: %s", variant)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Seed: %d", args.seed)
    logger.info("Device: %s", args.device)
    logger.info("Griffin-Lim iters: %d", args.griffin_lim_iters)

    device = torch.device(args.device)

    # Seed
    seed_everything(args.seed)

    # Load model
    logger.info("Loading model...")
    teacher, step = load_teacher(args.checkpoint, device)
    logger.info("Model loaded (step %d, %.1fM params)", step, sum(p.numel() for p in teacher.parameters()) / 1e6)

    # Resolve test set
    cache = FeatureCache(args.cache_dir)
    test_set = resolve_test_set(cache, config)
    logger.info("Test set: %d utterances", len(test_set))

    if not test_set:
        logger.error("No test utterances found. Check cache-dir and config speakers.")
        sys.exit(1)

    # Evaluate
    result = evaluate(
        teacher=teacher,
        cache=cache,
        test_set=test_set,
        config=config,
        device=device,
        output_dir=output_dir,
        seed=args.seed,
    )

    # Save results
    result_path = output_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", result_path)

    # Summary
    logger.info("")
    logger.info("=== Aggregate Results ===")
    for metric, stats in result.aggregate.items():
        logger.info("  %-15s mean=%.4f  std=%.4f  [%.4f, %.4f]",
                     metric, stats["mean"], stats["std"], stats["min"], stats["max"])
    logger.info("")
    logger.info("Elapsed: %.1fs (%d utterances)", result.elapsed_sec, result.n_utterances)

    # Save config snapshot
    config_snapshot_path = output_dir / "config_snapshot.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info("Config snapshot saved to %s", config_snapshot_path)


if __name__ == "__main__":
    main()
