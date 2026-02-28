#!/usr/bin/env python3
"""Generate VC samples for subjective evaluation.

Creates multiple voice conversion samples from different speakers.

Usage:
    uv run python scripts/eval/eval_uclm_vc.py \
        --checkpoint checkpoints/uclm/uclm_step99000.pt \
        --output-dir scratch/eval/vc_samples
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from tmrvc_core.constants import (
    D_MODEL,
    D_SPEAKER,
    HOP_LENGTH,
    N_CODEBOOKS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    VOCAB_SIZE,
)
from tmrvc_data.codec import EnCodecWrapper
from tmrvc_data.speaker import SpeakerEncoder
from tmrvc_data.voice_state import VoiceStateEstimator
from tmrvc_train.models.uclm import UCLM, UCLMConfig

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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="scratch/eval/vc_samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    config = UCLMConfig(
        vocab_size=VOCAB_SIZE,
        n_codebooks=N_CODEBOOKS,
        d_model=D_MODEL,
        d_speaker=D_SPEAKER,
    )
    model = UCLM(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Load codec and estimators
    codec = EnCodecWrapper(device=str(device))
    vs_estimator = VoiceStateEstimator(device=str(device))
    spk_encoder = SpeakerEncoder(device=str(device))

    # Find speaker files
    import json

    speaker_map_path = Path("data/moe_multispeaker_voices/_speaker_map.json")
    with open(speaker_map_path) as f:
        speaker_map = json.load(f)["mapping"]

    # Group files by speaker
    speakers = {}
    for fname, spk in speaker_map.items():
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(fname)

    # Select pairs: source from one speaker, target from another
    speaker_list = list(speakers.keys())
    logger.info(f"Found {len(speaker_list)} speakers")

    import random

    random.seed(42)

    pairs = []
    for _ in range(args.n_samples):
        src_spk = random.choice(speaker_list)
        tgt_spk = random.choice([s for s in speaker_list if s != src_spk])
        src_file = random.choice(speakers[src_spk])
        tgt_file = random.choice(speakers[tgt_spk])
        pairs.append((src_file, tgt_file))

    # Process pairs
    for i, (src_file, tgt_file) in enumerate(pairs):
        logger.info(f"Processing pair {i + 1}/{len(pairs)}: {src_file} -> {tgt_file}")

        src_path = Path(f"data/moe_multispeaker_voices/{src_file}")
        tgt_path = Path(f"data/moe_multispeaker_voices/{tgt_file}")

        if not src_path.exists() or not tgt_path.exists():
            logger.warning(f"Skipping: file not found")
            continue

        # Load audio
        src_audio = load_audio(str(src_path))
        tgt_audio = load_audio(str(tgt_path))

        src_tensor = torch.from_numpy(src_audio).unsqueeze(0).to(device)
        tgt_tensor = torch.from_numpy(tgt_audio).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            src_tokens = codec.encode_simple(src_tensor)
            n_frames = src_tokens.shape[-1]

            mel = extract_mel(src_tensor)
            f0 = extract_f0(src_tensor)
            voice_state = vs_estimator.estimate(mel, f0)

            if voice_state.shape[1] != n_frames:
                voice_state = torch.nn.functional.interpolate(
                    voice_state.transpose(1, 2),
                    size=n_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            tgt_spk_embed = spk_encoder.extract(tgt_tensor).unsqueeze(0).to(device)

            # Generate
            out_tokens = model.generate(
                voice_state=voice_state,
                speaker_embed=tgt_spk_embed,
                source_tokens=src_tokens,
                mode="vc",
                max_length=n_frames,
                temperature=args.temperature,
                top_k=50,
            )

            out_audio = codec.decode(out_tokens).squeeze().cpu().numpy()

        # Save
        out_name = f"vc_{i + 1}_{src_file.replace('.wav', '')}_to_{tgt_file}"
        sf.write(output_dir / f"{out_name}_source.wav", src_audio, SAMPLE_RATE)
        sf.write(output_dir / f"{out_name}_converted.wav", out_audio, SAMPLE_RATE)
        logger.info(f"Saved: {out_name}")

    logger.info(f"Done! Samples saved to {output_dir}")


if __name__ == "__main__":
    main()
