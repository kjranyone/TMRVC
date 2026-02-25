"""Generate audio samples from a trained TeacherUNet to evaluate quality.

Usage:
    uv run python scripts/eval_teacher_sample.py \
        --checkpoint checkpoints/teacher_step20000.pt \
        --cache-dir data/cache --dataset vctk --speaker vctk_p225 \
        --output-dir eval_samples
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tmrvc_core.audio import MelSpectrogram, create_mel_filterbank
from tmrvc_core.constants import (
    D_CONTENT_VEC,
    HOP_LENGTH,
    LOG_FLOOR,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.models.teacher_unet import TeacherUNet

logger = logging.getLogger(__name__)


def griffin_lim(
    log_mel: torch.Tensor,
    n_iter: int = 60,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window_length: int = WINDOW_LENGTH,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Convert log-mel spectrogram to waveform via Griffin-Lim.

    Args:
        log_mel: [B, n_mels, T] log-mel spectrogram.
        n_iter: Number of Griffin-Lim iterations.

    Returns:
        [B, T_samples] waveform.
    """
    device = log_mel.device
    # Invert log → linear mel
    mel_linear = log_mel.exp()  # [B, 80, T]

    # Pseudo-invert mel filterbank: [80, 513] → pinv [513, 80]
    mel_basis = create_mel_filterbank(n_fft, n_mels, sample_rate, MEL_FMIN, MEL_FMAX)
    mel_pinv = torch.linalg.pinv(mel_basis).to(device=device, dtype=mel_linear.dtype)

    # Approximate power spectrogram
    power = torch.matmul(mel_pinv, mel_linear).clamp(min=0)  # [B, 513, T]
    magnitude = power.sqrt()

    # Griffin-Lim
    window = torch.hann_window(window_length, periodic=True, device=device)
    n_freq = n_fft // 2 + 1
    T = magnitude.shape[-1]

    # Random phase initialization
    phase = torch.randn(magnitude.shape[0], n_freq, T, device=device) * 2 * torch.pi

    for _ in range(n_iter):
        stft_complex = magnitude * torch.exp(1j * phase)
        # iSTFT
        waveform = torch.istft(
            stft_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            center=True,
        )
        # Re-STFT to get improved phase
        stft_new = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            center=True,
            return_complex=True,
        )
        phase = stft_new.angle()

    # Final reconstruction
    stft_complex = magnitude * torch.exp(1j * phase)
    waveform = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        center=True,
    )
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Evaluate Teacher U-Net quality")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="vctk")
    parser.add_argument("--speaker", default="vctk_p225")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_samples"))
    parser.add_argument("--steps", type=int, default=32, help="ODE sampling steps")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sway", type=float, default=1.0, help="Sway sampling coefficient")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Find utterances
    speaker_dir = args.cache_dir / args.dataset / "train" / args.speaker
    utts = sorted(speaker_dir.iterdir())
    if not utts:
        logger.error("No utterances found in %s", speaker_dir)
        return
    logger.info("Found %d utterances for %s", len(utts), args.speaker)

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    teacher = TeacherUNet(d_content=D_CONTENT_VEC).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"Unsupported checkpoint format: {args.checkpoint} "
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
            f"Unsupported checkpoint metadata: {args.checkpoint} "
            "(missing/invalid 'step')."
        )
    logger.info("Model loaded (step %d, %.1fM params)", step, sum(p.numel() for p in teacher.parameters()) / 1e6)

    scheduler = FlowMatchingScheduler()

    # Process first 3 utterances
    for utt_dir in utts[:3]:
        utt_id = utt_dir.name
        logger.info("Processing %s", utt_id)

        # Load cached features
        mel_gt = torch.from_numpy(np.load(utt_dir / "mel.npy")).unsqueeze(0)  # [1, 80, T]
        content = torch.from_numpy(np.load(utt_dir / "content.npy")).unsqueeze(0)  # [1, 768, T]
        f0 = torch.from_numpy(np.load(utt_dir / "f0.npy")).unsqueeze(0)  # [1, 1, T]
        spk_embed = torch.from_numpy(np.load(utt_dir / "spk_embed.npy")).unsqueeze(0)  # [1, 192]

        T = mel_gt.shape[-1]
        logger.info("  mel: %s (min=%.2f, max=%.2f, mean=%.2f)", mel_gt.shape, mel_gt.min(), mel_gt.max(), mel_gt.mean())

        mel_gt = mel_gt.to(device)
        content = content.to(device)
        f0 = f0.to(device)
        spk_embed = spk_embed.to(device)

        # Generate mel via flow matching sampling
        with torch.no_grad():
            mel_pred = scheduler.sample(
                teacher,
                shape=(1, N_MELS, T),
                steps=args.steps,
                device=str(device),
                sway_coefficient=args.sway,
                content=content,
                f0=f0,
                spk_embed=spk_embed,
            )

        # Compute metrics
        mse = (mel_pred - mel_gt).pow(2).mean().item()
        logger.info("  Generated mel MSE vs GT: %.4f", mse)
        logger.info("  Generated mel: min=%.2f, max=%.2f, mean=%.2f", mel_pred.min(), mel_pred.max(), mel_pred.mean())

        # Griffin-Lim: convert both GT and predicted mel to audio
        logger.info("  Running Griffin-Lim (60 iterations)...")
        wav_gt = griffin_lim(mel_gt.cpu())
        wav_pred = griffin_lim(mel_pred.cpu())

        # Normalize and save
        for suffix, wav in [("gt", wav_gt), ("pred", wav_pred)]:
            wav_np = wav.squeeze(0).numpy()
            peak = np.abs(wav_np).max()
            if peak > 0:
                wav_np = wav_np / peak * 0.95
            out_path = args.output_dir / f"step{step}_{utt_id}_{suffix}.wav"
            sf.write(str(out_path), wav_np, SAMPLE_RATE)
            logger.info("  Saved: %s (%.1fs)", out_path, len(wav_np) / SAMPLE_RATE)

    logger.info("Done! Samples in %s", args.output_dir)


if __name__ == "__main__":
    main()
