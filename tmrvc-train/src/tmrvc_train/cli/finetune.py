"""`tmrvc-finetune` CLI for few-shot LoRA fine-tuning.

Usage::

    tmrvc-finetune --audio-dir /path/to/speaker/ --checkpoint distill.pt --output out.tmrvc_speaker
    tmrvc-finetune --audio-files a.wav b.wav --checkpoint distill.pt --output out.tmrvc_speaker --steps 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-finetune",
        description="Few-shot LoRA fine-tuning for target speaker.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory containing target speaker audio files.",
    )
    group.add_argument(
        "--audio-files",
        nargs="+",
        type=Path,
        help="Individual audio file paths.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to distillation checkpoint (distill.pt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .tmrvc_speaker file path.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Profile name stored in speaker metadata (default: output stem).",
    )
    parser.add_argument(
        "--author-name",
        type=str,
        default="",
        help="Author name stored in speaker metadata.",
    )
    parser.add_argument(
        "--co-author-name",
        type=str,
        default="",
        help="Co-author name stored in speaker metadata.",
    )
    parser.add_argument(
        "--licence-url",
        type=str,
        default="",
        help="Licence URL stored in speaker metadata.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Fine-tuning steps (default: 200).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--use-gtm",
        action="store_true",
        help="Use GTM adapter instead of FiLM LoRA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto' (XPU>CUDA>CPU), 'xpu', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file for fine-tuning parameters.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def _collect_audio_paths(args: argparse.Namespace) -> list[Path]:
    """Collect audio file paths from CLI arguments."""
    if args.audio_files:
        paths = args.audio_files
    else:
        paths = sorted(
            p for p in args.audio_dir.iterdir()
            if p.suffix.lower() in _AUDIO_EXTENSIONS
        )
    if not paths:
        raise ValueError("No audio files found.")
    return paths


def _load_config(config_path: Path | None) -> dict:
    """Load YAML config if provided."""
    if config_path is None:
        return {}
    import yaml

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import torch

    from tmrvc_core.device import get_device
    from tmrvc_data.speaker import SpeakerEncoder
    from tmrvc_export.speaker_file import write_speaker_file
    from tmrvc_train.fewshot import FewShotConfig, FewShotFinetuner
    from tmrvc_train.models.content_encoder import ContentEncoderStudent
    from tmrvc_train.models.converter import ConverterStudent, ConverterStudentGTM

    # Merge YAML config with CLI overrides
    file_cfg = _load_config(args.config)
    max_steps = args.steps if args.steps is not None else file_cfg.get("steps", 200)
    lr = args.lr if args.lr is not None else file_cfg.get("lr", 1e-3)
    use_gtm = args.use_gtm if args.use_gtm else file_cfg.get("use_gtm", False)

    # Device selection
    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # 1. Collect audio paths
    audio_paths = _collect_audio_paths(args)
    logger.info("Found %d audio files", len(audio_paths))

    # 2. Load student models from checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    content_encoder = ContentEncoderStudent()
    content_encoder.load_state_dict(ckpt["content_encoder"])
    content_encoder.eval()
    content_encoder.to(device)

    if use_gtm:
        converter = ConverterStudentGTM()
        converter.load_state_dict(ckpt["converter_gtm"])
    else:
        converter = ConverterStudent()
        converter.load_state_dict(ckpt["converter"])
    converter.to(device)

    # 3. Extract speaker embedding (average over all audio files)
    logger.info("Extracting speaker embeddings...")
    spk_encoder = SpeakerEncoder(device=str(device))
    embeddings = []
    for path in audio_paths:
        emb = spk_encoder.extract_from_file(str(path))
        embeddings.append(emb)

    spk_embed = torch.stack(embeddings).mean(0)
    spk_embed = torch.nn.functional.normalize(spk_embed, p=2, dim=-1)

    # 4. Fine-tune
    config = FewShotConfig(
        max_steps=max_steps,
        lr=lr,
        use_gtm=use_gtm,
    )
    finetuner = FewShotFinetuner(converter, content_encoder, spk_embed, config)

    logger.info("Preparing data...")
    data = finetuner.prepare_data([str(p) for p in audio_paths])

    logger.info("Starting fine-tuning (steps=%d, lr=%.1e)...", max_steps, lr)
    for step, loss in finetuner.finetune_iter(data):
        if step % config.log_every == 0:
            logger.info("Step %d/%d  loss=%.4f", step, config.max_steps, loss)

    # 5. Save .tmrvc_speaker
    from datetime import datetime, timezone

    lora_delta = finetuner.get_lora_delta()
    metadata = {
        "profile_name": args.profile_name or args.output.stem,
        "author_name": args.author_name,
        "co_author_name": args.co_author_name,
        "licence_url": args.licence_url,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "",
        "source_audio_files": [p.name for p in audio_paths],
        "source_sample_count": 0,
        "training_mode": "finetune",
        "checkpoint_name": args.checkpoint.name,
    }
    write_speaker_file(
        args.output,
        spk_embed.numpy().astype("float32"),
        lora_delta.detach().numpy().astype("float32"),
        metadata=metadata,
    )
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()


