#!/usr/bin/env python3
"""Test UCLM Voice Conversion inference.

Converts source audio to target speaker voice using trained UCLM model.

Usage:
    uv run python scripts/demo/test_uclm_vc.py \
        --checkpoint checkpoints/uclm/uclm_step50000.pt \
        --source data/sample_voice/source.wav \
        --target-speaker data/sample_voice/target_speaker.wav \
        --output vc_output.wav
"""

import argparse
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


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio and resample if needed."""
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(torch.from_numpy(audio).float()).numpy()

    return audio, target_sr


def extract_mel(audio: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Extract mel spectrogram."""
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ).to(audio.device)

    mel = mel_transform(audio)
    mel = torch.log(mel + 1e-8)
    return mel


def extract_f0(
    audio: torch.Tensor, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH
) -> torch.Tensor:
    """Extract F0 using librosa."""
    import librosa

    audio_np = audio.squeeze().cpu().numpy()
    f0, _, _ = librosa.pyin(
        audio_np,
        fmin=50,
        fmax=500,
        sr=sr,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return torch.from_numpy(f0).float().unsqueeze(0).to(audio.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target-speaker", type=str, required=True)
    parser.add_argument("--output", type=str, default="vc_output.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load codec and estimators
    codec = EnCodecWrapper(device=str(device))
    vs_estimator = VoiceStateEstimator(device=str(device))
    spk_encoder = SpeakerEncoder(device=str(device))

    # Load source audio
    print(f"Loading source audio: {args.source}")
    source_audio, sr = load_audio(args.source)
    source_audio_tensor = torch.from_numpy(source_audio).unsqueeze(0).to(device)
    print(f"  Duration: {len(source_audio) / sr:.2f}s")

    # Load target speaker reference
    print(f"Loading target speaker reference: {args.target_speaker}")
    target_audio, _ = load_audio(args.target_speaker)
    target_audio_tensor = torch.from_numpy(target_audio).unsqueeze(0).to(device)

    # Extract source tokens
    print("Extracting source codec tokens...")
    with torch.no_grad():
        source_tokens = codec.encode_simple(source_audio_tensor)
    n_frames = source_tokens.shape[-1]
    print(f"  Frames: {n_frames} ({n_frames / 75:.2f}s at 75fps)")

    # Extract voice state from source
    print("Extracting voice state from source...")
    with torch.no_grad():
        mel = extract_mel(source_audio_tensor)
        f0 = extract_f0(source_audio_tensor)
        voice_state = vs_estimator.estimate(mel, f0)

    # Resample voice_state to match codec frame rate
    if voice_state.shape[1] != n_frames:
        voice_state = torch.nn.functional.interpolate(
            voice_state.transpose(1, 2),
            size=n_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    print(f"  Voice state shape: {voice_state.shape}")

    # Extract target speaker embedding
    print("Extracting target speaker embedding...")
    with torch.no_grad():
        target_spk_embed = spk_encoder.extract(target_audio_tensor)
        target_spk_embed = target_spk_embed.unsqueeze(0).to(device)
    print(f"  Speaker embed shape: {target_spk_embed.shape}")

    # Generate converted tokens
    print("Generating converted tokens (VC mode)...")
    with torch.no_grad():
        target_tokens = model.generate(
            voice_state=voice_state,
            speaker_embed=target_spk_embed,
            source_tokens=source_tokens,
            mode="vc",
            max_length=n_frames,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(f"  Generated tokens shape: {target_tokens.shape}")

    # Decode to audio
    print("Decoding to audio...")
    with torch.no_grad():
        output_audio = codec.decode(target_tokens)

    output_audio = output_audio.squeeze().cpu().numpy()
    print(f"  Output audio shape: {output_audio.shape}")

    # Save
    sf.write(args.output, output_audio, 24000)
    print(f"Saved to {args.output}")

    # Also save source for comparison
    source_out = Path(args.output).with_stem(Path(args.output).stem + "_source")
    sf.write(str(source_out), source_audio, 24000)
    print(f"Saved source to {source_out}")


if __name__ == "__main__":
    main()
