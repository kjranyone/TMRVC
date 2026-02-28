"""
Streaming Codec Training CLI

Usage:
    uv run tmrvc-train-codec --cache-dir data/cache --device cuda
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tmrvc_train.models.streaming_codec import (
    CodecConfig,
    StreamingCodec,
    MultiScaleDiscriminator,
    MultiScaleSTFTLoss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Streaming Codec")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/wav48_silence_trimmed"),
        help="Raw audio directory (VCTK, JVS, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/xpu/cpu)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--lambda-rec", type=float, default=1.0, help="Reconstruction loss weight"
    )
    parser.add_argument(
        "--lambda-adv", type=float, default=1.0, help="Adversarial loss weight"
    )
    parser.add_argument(
        "--lambda-commit", type=float, default=0.25, help="Commitment loss weight"
    )
    parser.add_argument(
        "--lambda-stft", type=float, default=1.0, help="STFT loss weight"
    )
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument(
        "--save-every", type=int, default=5000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )
    return parser.parse_args()


class CodecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dir: Path,
        sample_rate: int = 24000,
        frame_size: int = 480,
        segment_frames: int = 50,
    ):
        self.raw_dir = raw_dir
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.segment_frames = segment_frames
        self.segment_samples = segment_frames * frame_size

        self.audio_files = self._collect_audio_files()
        logger.info(f"Found {len(self.audio_files)} audio segments")

    def _collect_audio_files(self) -> list:
        files = []
        for ext in ["*.flac", "*.wav"]:
            for f in self.raw_dir.rglob(ext):
                if f.is_file():
                    files.append(f)
        return files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        import torchaudio

        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        audio = audio.mean(dim=0, keepdim=True)

        if audio.shape[1] < self.segment_samples:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_samples - audio.shape[1])
            )
        elif audio.shape[1] > self.segment_samples:
            start = torch.randint(0, audio.shape[1] - self.segment_samples, (1,))
            audio = audio[:, start : start + self.segment_samples]

        return audio


class CodecTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        self.config = CodecConfig()
        self.model = StreamingCodec(self.config).to(self.device)
        self.discriminator = MultiScaleDiscriminator().to(self.device)

        self.stft_loss = MultiScaleSTFTLoss().to(self.device)

        self.optimizer_g = AdamW(self.model.parameters(), lr=args.lr)
        self.optimizer_d = AdamW(self.discriminator.parameters(), lr=args.lr)

        self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=args.steps)
        self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=args.steps)

        self.global_step = 0

        if args.resume:
            self._load_checkpoint(args.resume)

    def _load_checkpoint(self, path: Path):
        logger.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.global_step = ckpt["global_step"]

    def _save_checkpoint(self, path: Path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "global_step": self.global_step,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def _discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        real_outs = self.discriminator(real)
        fake_outs = self.discriminator(fake.detach())

        loss = 0.0
        for real_out, fake_out in zip(real_outs, fake_outs):
            loss = loss + torch.mean((real_out - 1) ** 2)
            loss = loss + torch.mean(fake_out**2)
        return loss

    def _generator_adv_loss(self, fake: torch.Tensor) -> torch.Tensor:
        fake_outs = self.discriminator(fake)
        loss = 0.0
        for fake_out in fake_outs:
            loss = loss + torch.mean((fake_out - 1) ** 2)
        return loss

    def _feature_matching_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        real_outs = self.discriminator(real)
        fake_outs = self.discriminator(fake)

        loss = 0.0
        for real_out, fake_out in zip(real_outs, fake_outs):
            loss = loss + F.l1_loss(real_out, fake_out)
        return loss

    def train_step(self, batch: torch.Tensor) -> dict:
        batch = batch.to(self.device)

        audio_rec, indices, commit_loss, _, _ = self.model(batch)

        self.optimizer_d.zero_grad()
        d_loss = self._discriminator_loss(batch, audio_rec)
        d_loss.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        stft_loss = self.stft_loss(audio_rec, batch)

        adv_loss = self._generator_adv_loss(audio_rec)

        fm_loss = self._feature_matching_loss(batch, audio_rec)

        rec_loss = F.l1_loss(audio_rec, batch)

        g_loss = (
            self.args.lambda_rec * rec_loss
            + self.args.lambda_stft * stft_loss
            + self.args.lambda_adv * (adv_loss + 0.5 * fm_loss)
            + self.args.lambda_commit * commit_loss
        )

        g_loss.backward()
        self.optimizer_g.step()

        self.scheduler_g.step()
        self.scheduler_d.step()

        self.global_step += 1

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "rec_loss": rec_loss.item(),
            "stft_loss": stft_loss.item(),
            "adv_loss": adv_loss.item(),
            "commit_loss": commit_loss.item(),
        }

    def train(self):
        dataset = CodecDataset(self.args.raw_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        output_dir = self.args.output_dir / "codec"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training for {self.args.steps} steps")
        logger.info(f"Output directory: {output_dir}")

        data_iter = iter(dataloader)
        pbar = tqdm(total=self.args.steps, initial=self.global_step)

        while self.global_step < self.args.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            metrics = self.train_step(batch)

            pbar.update(1)
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            if self.global_step % self.args.log_every == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                )

            if self.global_step % self.args.save_every == 0:
                self._save_checkpoint(output_dir / f"codec_step{self.global_step}.pt")

        self._save_checkpoint(output_dir / "codec_final.pt")
        logger.info("Training complete!")


def main():
    args = parse_args()
    trainer = CodecTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
