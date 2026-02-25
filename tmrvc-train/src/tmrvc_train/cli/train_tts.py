"""``tmrvc-train-tts`` — TTS front-end training CLI.

Usage::

    tmrvc-train-tts --cache-dir data/cache --device xpu
    tmrvc-train-tts --cache-dir data/cache --config configs/train_tts.yaml --resume checkpoints/tts/tts_step50000.pt
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from tmrvc_core.constants import PHONEME_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor
from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.tts_trainer import TTSTrainer, TTSTrainerConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-train-tts",
        description="Train TTS front-end models (TextEncoder, DurationPredictor, F0Predictor, ContentSynthesizer).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (e.g. configs/train_tts.yaml). CLI flags override.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name(s), comma-separated (e.g. 'jsut,ljspeech,vctk,jvs').",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (default: 200000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from TTS checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device (default: cpu).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0, use 0 on Windows).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Fixed frame count per batch (default: 400).",
    )
    parser.add_argument(
        "--text-frontend",
        choices=["phoneme", "tokenizer", "auto"],
        default=None,
        help="Text frontend for training data/input IDs.",
    )
    parser.add_argument(
        "--enable-ssl",
        action="store_true",
        default=None,
        help="Enable Scene State Latent (SSL) modules.",
    )
    parser.add_argument(
        "--enable-bpeh",
        action="store_true",
        default=None,
        help="Enable Breath-Pause Event Head (BPEH).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def _load_config(config_path: Path | None) -> dict:
    """Load YAML config."""
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

    file_cfg = _load_config(args.config)

    lr = args.lr if args.lr is not None else file_cfg.get("lr", 1e-4)
    max_steps = args.max_steps if args.max_steps is not None else file_cfg.get("max_steps", 200_000)
    batch_size = args.batch_size if args.batch_size is not None else file_cfg.get("batch_size", 32)
    save_every = args.save_every if args.save_every is not None else file_cfg.get("save_every", 10_000)
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else Path(file_cfg.get("checkpoint_dir", "checkpoints/tts"))
    warmup_steps = file_cfg.get("warmup_steps", 5000)
    max_frames = args.max_frames if args.max_frames is not None else file_cfg.get("max_frames", 400)
    text_frontend = (
        args.text_frontend
        if args.text_frontend is not None
        else file_cfg.get("text_frontend", "tokenizer")
    )
    if text_frontend == "auto":
        text_frontend = "tokenizer"
    vocab_size = TOKENIZER_VOCAB_SIZE if text_frontend == "tokenizer" else PHONEME_VOCAB_SIZE

    # Resolve datasets
    if args.dataset:
        datasets = [d.strip() for d in args.dataset.split(",")]
    else:
        datasets = file_cfg.get("datasets")
        if not datasets:
            parser.error("--dataset is required (or specify 'datasets' in YAML config)")
    logger.info("Datasets: %s", datasets)

    device = torch.device(args.device)

    # Create models
    text_encoder = TextEncoder(vocab_size=vocab_size).to(device)
    duration_predictor = DurationPredictor().to(device)
    f0_predictor = F0Predictor().to(device)
    content_synthesizer = ContentSynthesizer().to(device)

    logger.info(
        "Models created: TextEncoder(%.1fM), DurationPredictor(%.1fM), "
        "F0Predictor(%.1fM), ContentSynthesizer(%.1fM)",
        sum(p.numel() for p in text_encoder.parameters()) / 1e6,
        sum(p.numel() for p in duration_predictor.parameters()) / 1e6,
        sum(p.numel() for p in f0_predictor.parameters()) / 1e6,
        sum(p.numel() for p in content_synthesizer.parameters()) / 1e6,
    )

    # Optimizer
    all_params = (
        list(text_encoder.parameters())
        + list(duration_predictor.parameters())
        + list(f0_predictor.parameters())
        + list(content_synthesizer.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    # LR scheduler: linear warmup + cosine decay
    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Create dataloader
    from tmrvc_data.tts_dataset import create_tts_dataloader

    dataloader = create_tts_dataloader(
        cache_dir=args.cache_dir,
        dataset=datasets,
        batch_size=batch_size,
        num_workers=args.num_workers,
        max_frames=max_frames,
        frontend=text_frontend,
    )

    logger.info("TTS dataset: %d utterances", len(dataloader.dataset))

    # Resolve SSL/BPEH from CLI flags or config model section
    model_cfg = file_cfg.get("model", {})
    loss_cfg = file_cfg.get("loss_weights", {})
    enable_ssl = args.enable_ssl if args.enable_ssl else model_cfg.get("ssl_enabled", False)
    enable_bpeh = args.enable_bpeh if args.enable_bpeh else model_cfg.get("bpeh_enabled", False)

    # Trainer config
    config = TTSTrainerConfig(
        lr=lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_every=save_every,
        checkpoint_dir=str(checkpoint_dir),
        grad_clip=file_cfg.get("grad_clip", 1.0),
        lambda_duration=loss_cfg.get("lambda_duration", file_cfg.get("lambda_duration", 1.0)),
        lambda_f0=loss_cfg.get("lambda_f0", file_cfg.get("lambda_f0", 0.5)),
        lambda_content=loss_cfg.get("lambda_content", file_cfg.get("lambda_content", 1.0)),
        lambda_voiced=loss_cfg.get("lambda_voiced", file_cfg.get("lambda_voiced", 0.2)),
        enable_ssl=enable_ssl,
        lambda_state_recon=loss_cfg.get("lambda_state_recon", 0.5),
        lambda_state_cons=loss_cfg.get("lambda_state_cons", 0.3),
        enable_bpeh=enable_bpeh,
        lambda_onset=loss_cfg.get("lambda_onset", 0.5),
        lambda_dur=loss_cfg.get("lambda_dur", 0.3),
        lambda_amp=loss_cfg.get("lambda_amp", 0.2),
        use_wandb=args.wandb,
        text_frontend=text_frontend,
        text_vocab_size=vocab_size,
    )

    trainer = TTSTrainer(
        text_encoder=text_encoder,
        duration_predictor=duration_predictor,
        f0_predictor=f0_predictor,
        content_synthesizer=content_synthesizer,
        optimizer=optimizer,
        dataloader=dataloader,
        config=config,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    # Rebuild optimizer to include SSL/BPEH params
    if enable_ssl or enable_bpeh:
        all_params = list(trainer._trainable_params())
        trainer.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
        trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=_lr_lambda)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(
        "Starting TTS training (frontend=%s, vocab=%d, lr=%.2e, max_steps=%d, batch_size=%d, max_frames=%d)",
        text_frontend, vocab_size, lr, max_steps, batch_size, max_frames,
    )

    try:
        for step, losses in trainer.train_iter():
            pass
    except KeyboardInterrupt:
        logger.info("Interrupted at step %d, saving checkpoint...", trainer.global_step)

    trainer.save_checkpoint()
    logger.info("TTS training complete (step %d).", trainer.global_step)


if __name__ == "__main__":
    main()
