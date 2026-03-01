"""``tmrvc-enroll`` — Create .tmrvc_speaker v3 file from audio.

Usage::

    # Light level (embedding only)
    tmrvc-enroll --audio ref.wav --output models/speaker.tmrvc_speaker --level light

    # Standard level (in-context reference tokens)
    tmrvc-enroll --audio-dir data/voice/ --output models/speaker.tmrvc_speaker \\
        --level standard --codec-checkpoint checkpoints/codec/best.pt

    # Full level (LoRA fine-tuning) - NOT YET IMPLEMENTED
    tmrvc-enroll --audio-dir data/voice/ --output models/speaker.tmrvc_speaker \\
        --level full --token-model checkpoints/token.pt --finetune-steps 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-enroll",
        description="Create a .tmrvc_speaker v3 file from audio.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Single audio file (use --audio-dir for multiple files).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .tmrvc_speaker file.",
    )
    parser.add_argument(
        "--level",
        choices=["light", "standard", "full"],
        default="standard",
        help="Adaptation level: light=spk_embed only, standard=+style+tokens, full=+LoRA",
    )
    parser.add_argument("--name", default="Speaker", help="Speaker name.")
    parser.add_argument(
        "--codec-checkpoint",
        type=Path,
        default=None,
        help="StreamingCodec checkpoint for reference token extraction (standard/full).",
    )
    parser.add_argument(
        "--token-model",
        type=Path,
        default=None,
        help="Token model checkpoint for LoRA fine-tuning (full only).",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=200,
        help="LoRA fine-tuning steps (full only, default: 200).",
    )
    parser.add_argument(
        "--max-ref-frames",
        type=int,
        default=150,
        help="Maximum reference token frames (standard/full, default: 150).",
    )
    parser.add_argument("--device", default="cpu", help="Device (cuda/cpu/xpu).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def collect_audio_files(args: argparse.Namespace) -> list[Path]:
    """Collect audio file paths from arguments."""
    paths = []

    if args.audio:
        if not args.audio.exists():
            logger.error("Audio file not found: %s", args.audio)
            sys.exit(1)
        paths.append(args.audio)

    if args.audio_dir:
        if not args.audio_dir.exists():
            logger.error("Audio directory not found: %s", args.audio_dir)
            sys.exit(1)
        for ext in ("*.wav", "*.flac", "*.mp3"):
            paths.extend(sorted(args.audio_dir.glob(ext)))

    if not paths:
        logger.error("No audio files provided. Use --audio or --audio-dir.")
        sys.exit(1)

    return paths


def load_audio(path: str | Path, target_sr: int = 24000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import soundfile as sf

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def extract_speaker_embedding(audio_paths: list[Path], device: str) -> np.ndarray:
    """Extract speaker embedding from audio files."""
    import torch

    from tmrvc_data.speaker import SpeakerEncoder

    encoder = SpeakerEncoder(device=device)
    embeddings = []

    for path in audio_paths:
        emb = encoder.extract_from_file(str(path))
        embeddings.append(emb)

    avg_embed = torch.stack(embeddings).mean(dim=0)
    avg_embed = torch.nn.functional.normalize(avg_embed, p=2, dim=-1)
    return avg_embed.numpy().astype(np.float32)


def extract_style_embedding(audio_paths: list[Path], device: str) -> np.ndarray:
    """Extract style embedding from audio files."""
    from tmrvc_data.style import compute_style_from_files

    return compute_style_from_files([str(p) for p in audio_paths], device=device)


def extract_ssl_state(audio_paths: list[Path], device: str) -> np.ndarray | None:
    """Extract default SSL state from audio files using WavLM.

    Returns the mean SSL state across all audio files, shape [128].
    Returns None if WavLM is not available.
    """
    try:
        import torch
        from tmrvc_train.models.ssl_extractor import WavLMSSLExtractor
    except ImportError:
        logger.warning("WavLMSSLExtractor not available, skipping ssl_state extraction")
        return None

    logger.info("Extracting SSL state from %d file(s)...", len(audio_paths))

    try:
        extractor = WavLMSSLExtractor(d_ssl=128, cache_dir=None).to(device)
        extractor.eval()
    except Exception as e:
        logger.warning("Failed to load WavLM: %s, using zeros", e)
        return np.zeros(128, dtype=np.float32)

    ssl_states = []

    with torch.no_grad():
        for path in audio_paths:
            try:
                audio = load_audio(path)
                ssl = extractor.extract(audio, sample_rate=24000)
                # Take mean across time dimension: [1, T, 128] -> [128]
                ssl_mean = ssl.mean(dim=1).squeeze(0).cpu().numpy()
                ssl_states.append(ssl_mean)
            except Exception as e:
                logger.warning("Failed to extract SSL from %s: %s", path, e)
                continue

    if not ssl_states:
        logger.warning("No SSL states extracted, using zeros")
        return np.zeros(128, dtype=np.float32)

    # Average across all files
    avg_ssl = np.mean(ssl_states, axis=0).astype(np.float32)
    logger.info("  ssl_state shape: %s", avg_ssl.shape)
    return avg_ssl


def extract_reference_tokens(
    audio_paths: list[Path],
    codec_checkpoint: Path,
    max_frames: int,
    device: str,
) -> np.ndarray:
    """Extract reference tokens using StreamingCodec."""
    import torch

    from tmrvc_train.models.streaming_codec import StreamingCodec, CodecConfig

    ckpt = torch.load(codec_checkpoint, map_location=device, weights_only=False)

    codec = StreamingCodec(CodecConfig())

    if "state_dict" in ckpt:
        codec.load_state_dict(ckpt["state_dict"])
    else:
        codec.load_state_dict(ckpt)

    codec = codec.to(device).eval()

    all_tokens = []

    with torch.no_grad():
        for path in audio_paths:
            audio = load_audio(path)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)

            indices, _, _ = codec.encode(audio_tensor)

            indices_np = (
                indices.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.int32)
            )
            all_tokens.append(indices_np)

            if sum(len(t) for t in all_tokens) >= max_frames:
                break

    if not all_tokens:
        logger.warning("No tokens extracted, returning zeros")
        return np.zeros((max_frames, 4), dtype=np.int32)

    reference_tokens = np.concatenate(all_tokens, axis=0)[:max_frames]

    if reference_tokens.shape[1] < 4:
        padded = np.zeros((len(reference_tokens), 4), dtype=np.int32)
        padded[:, : reference_tokens.shape[1]] = reference_tokens
        reference_tokens = padded

    return reference_tokens


def finetune_lora(
    token_model_path: Path,
    reference_tokens: np.ndarray,
    spk_embed: np.ndarray,
    n_steps: int,
    device: str,
) -> np.ndarray:
    """Fine-tune TokenModel with LoRA and return flattened delta."""
    import torch

    from tmrvc_train.models.token_model import TokenModel, TokenModelConfig
    from tmrvc_train.lora import finetune_token_model_lora

    ckpt = torch.load(token_model_path, map_location=device, weights_only=False)

    model = TokenModel(TokenModelConfig())
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    ref_tokens_tensor = torch.from_numpy(reference_tokens).long()
    spk_embed_tensor = torch.from_numpy(spk_embed).float()

    delta_flat = finetune_token_model_lora(
        model=model,
        reference_tokens=ref_tokens_tensor,
        spk_embed=spk_embed_tensor,
        n_steps=n_steps,
        lr=1e-4,
        device=device,
    )

    return delta_flat.cpu().numpy().astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    audio_paths = collect_audio_files(args)
    logger.info("Processing %d audio file(s)", len(audio_paths))

    if args.level == "full":
        if not args.token_model:
            logger.error("Full level requires --token-model")
            sys.exit(1)
        if not args.token_model.exists():
            logger.error("Token model not found: %s", args.token_model)
            sys.exit(1)

    if args.level in ("standard", "full") and args.codec_checkpoint:
        if not args.codec_checkpoint.exists():
            logger.error("Codec checkpoint not found: %s", args.codec_checkpoint)
            sys.exit(1)

    logger.info("Extracting speaker embedding...")
    spk_embed = extract_speaker_embedding(audio_paths, args.device)
    logger.info("  spk_embed shape: %s", spk_embed.shape)

    style_embed = None
    if args.level in ("standard", "full"):
        logger.info("Extracting style embedding...")
        style_embed = extract_style_embedding(audio_paths, args.device)
        logger.info("  style_embed shape: %s", style_embed.shape)

    # Extract SSL state (always, for voice conditioning)
    ssl_state = extract_ssl_state(audio_paths, args.device)

    reference_tokens = None
    if args.level in ("standard", "full") and args.codec_checkpoint:
        logger.info("Extracting reference tokens...")
        reference_tokens = extract_reference_tokens(
            audio_paths, args.codec_checkpoint, args.max_ref_frames, args.device
        )
        logger.info("  reference_tokens shape: %s", reference_tokens.shape)

    lora_delta = None
    if args.level == "full" and args.token_model and args.finetune_steps > 0:
        if reference_tokens is None:
            logger.error(
                "Full level requires --codec-checkpoint for reference token extraction"
            )
            sys.exit(1)
        logger.info(
            "Fine-tuning TokenModel with LoRA (%d steps)...", args.finetune_steps
        )
        lora_delta = finetune_lora(
            token_model_path=args.token_model,
            reference_tokens=reference_tokens,
            spk_embed=spk_embed,
            n_steps=args.finetune_steps,
            device=args.device,
        )
        logger.info("  lora_delta shape: %s", lora_delta.shape)

    logger.info("Writing speaker file...")
    from tmrvc_export.speaker_file import write_speaker_file

    metadata = {
        "profile_name": args.name,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": f"Created with tmrvc-enroll ({args.level} level)",
        "source_audio_files": [p.name for p in audio_paths],
        "adaptation_level": args.level,
        "checkpoint_name": args.token_model.name if args.token_model else "",
    }

    write_speaker_file(
        output_path=args.output,
        spk_embed=spk_embed,
        style_embed=style_embed,
        reference_tokens=reference_tokens,
        lora_delta=lora_delta,
        ssl_state=ssl_state,
        metadata=metadata,
    )

    logger.info("Created: %s (%d bytes)", args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
