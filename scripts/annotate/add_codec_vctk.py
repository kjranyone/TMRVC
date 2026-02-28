#!/usr/bin/env python3
"""Add codec_tokens and voice_state to VCTK cache.

Usage:
    uv run python scripts/annotate/add_codec_vctk.py \
        --cache-dir data/cache/vctk \
        --audio-dir data/raw/wav48_silence_trimmed
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from tmrvc_core.constants import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from tmrvc_data.codec import EnCodecWrapper
from tmrvc_data.voice_state import VoiceStateEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(torch.from_numpy(audio).float()).numpy()
    return audio


def extract_mel(audio: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    ).to(audio.device)
    mel = mel_transform(audio)
    return torch.log(mel + 1e-8)


def extract_f0(
    audio: torch.Tensor, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH
) -> torch.Tensor:
    import librosa

    audio_np = audio.squeeze().cpu().numpy()
    f0, _, _ = librosa.pyin(audio_np, fmin=50, fmax=500, sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    return torch.from_numpy(f0).float().unsqueeze(0).to(audio.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="data/cache/vctk")
    parser.add_argument(
        "--audio-dir", type=str, default="data/raw/wav48_silence_trimmed"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    audio_dir = Path(args.audio_dir)

    # Load models
    codec = EnCodecWrapper(device=str(device))
    vs_estimator = VoiceStateEstimator(device=str(device))

    # Process cache
    cache_root = cache_dir / "train" if (cache_dir / "train").exists() else cache_dir
    count = 0

    for speaker_dir in sorted(cache_root.glob("vctk_p*")):
        speaker_id = speaker_dir.name  # vctk_p225
        raw_speaker_id = speaker_id.replace("vctk_", "")  # p225

        for utt_dir in sorted(speaker_dir.iterdir()):
            if not utt_dir.is_dir():
                continue

            utt_id = utt_dir.name  # vctk_p225_001
            raw_utt_id = utt_id.replace("vctk_", "")  # p225_001

            codec_path = utt_dir / "codec_tokens.npy"
            vs_path = utt_dir / "voice_state.npy"

            if not args.overwrite and codec_path.exists() and vs_path.exists():
                continue

            # Find audio file (prefer mic1)
            audio_file = audio_dir / raw_speaker_id / f"{raw_utt_id}_mic1.flac"
            if not audio_file.exists():
                audio_file = audio_dir / raw_speaker_id / f"{raw_utt_id}_mic2.flac"
            if not audio_file.exists():
                logger.debug("No audio for %s", utt_id)
                continue

            try:
                # Load audio
                audio = load_audio(str(audio_file))
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

                # Extract codec tokens
                with torch.no_grad():
                    codec_tokens = codec.encode_simple(audio_tensor)  # [1, 8, T]
                n_frames = codec_tokens.shape[-1]

                # Extract voice state
                mel = extract_mel(audio_tensor)
                f0 = extract_f0(audio_tensor)
                with torch.no_grad():
                    voice_state = vs_estimator.estimate(mel, f0)  # [1, T, 8]

                # Resample voice_state to match codec frame rate
                if voice_state.shape[1] != n_frames:
                    voice_state = torch.nn.functional.interpolate(
                        voice_state.transpose(1, 2),
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

                # Save
                np.save(codec_path, codec_tokens.squeeze(0).cpu().numpy())
                np.save(vs_path, voice_state.squeeze(0).cpu().numpy())

                count += 1
                if count % 100 == 0:
                    logger.info("Processed %d utterances", count)

            except Exception as e:
                logger.warning("Failed to process %s: %s", utt_id, e)

    logger.info("Added codec_tokens and voice_state to %d utterances", count)


if __name__ == "__main__":
    main()
