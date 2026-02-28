"""
Token Model Training CLI

Usage:
    uv run tmrvc-train-token --codec-checkpoint checkpoints/codec/codec_final.pt --cache-dir data/cache --device cuda
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

from tmrvc_train.models.streaming_codec import StreamingCodec, CodecConfig
from tmrvc_train.models.token_model import TokenModel, TokenModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Token Model")
    parser.add_argument(
        "--codec-checkpoint",
        type=Path,
        required=True,
        help="Path to trained codec checkpoint",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Feature cache directory",
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--steps", type=int, default=200000, help="Total training steps"
    )
    parser.add_argument(
        "--context-length", type=int, default=10, help="Context length (in frames)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument(
        "--save-every", type=int, default=10000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )
    return parser.parse_args()


class TokenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: Path,
        codec: StreamingCodec,
        context_length: int = 10,
        device: str = "cpu",
    ):
        self.cache_dir = cache_dir
        self.codec = codec
        self.context_length = context_length
        self.device = device

        self.codec.eval()

        self.utterances = self._collect_utterances()
        logger.info(f"Found {len(self.utterances)} utterances")

    def _collect_utterances(self) -> list:
        utterances = []
        for dataset_dir in self.cache_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for speaker_dir in dataset_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                for utt_dir in speaker_dir.iterdir():
                    if not utt_dir.is_dir():
                        continue
                    audio_file = utt_dir / "audio.wav"
                    spk_file = utt_dir / "spk_embed.npy"
                    if audio_file.exists() and spk_file.exists():
                        utterances.append((audio_file, spk_file))
        return utterances

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        import torchaudio
        import numpy as np

        audio_path, spk_path = self.utterances[idx]

        audio, sr = torchaudio.load(audio_path)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        audio = audio.mean(dim=0, keepdim=True)

        spk_embed = np.load(spk_path)
        spk_embed = torch.from_numpy(spk_embed).float()

        with torch.no_grad():
            tokens, _, _ = self.codec.encode(audio)
            tokens = tokens.squeeze(0)

        return {
            "tokens": tokens,
            "spk_embed": spk_embed,
            "n_frames": tokens.shape[1],
        }


def collate_fn(batch):
    max_len = max(item["n_frames"] for item in batch)

    tokens_batch = []
    spk_batch = []
    mask_batch = []

    for item in batch:
        tokens = item["tokens"]
        n_frames = item["n_frames"]

        if n_frames < max_len:
            tokens = F.pad(tokens, (0, max_len - n_frames))
            mask = torch.zeros(max_len)
            mask[:n_frames] = 1
        else:
            mask = torch.ones(max_len)

        tokens_batch.append(tokens)
        spk_batch.append(item["spk_embed"])
        mask_batch.append(mask)

    return {
        "tokens": torch.stack(tokens_batch),
        "spk_embed": torch.stack(spk_batch),
        "mask": torch.stack(mask_batch),
    }


class TokenTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        self.codec = self._load_codec(args.codec_checkpoint)

        self.config = TokenModelConfig(
            context_length=args.context_length,
        )
        self.model = TokenModel(self.config).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.steps)

        self.global_step = 0

        if args.resume:
            self._load_checkpoint(args.resume)

    def _load_codec(self, path: Path) -> StreamingCodec:
        logger.info(f"Loading codec from {path}")
        ckpt = torch.load(path, map_location=self.device)
        config = CodecConfig(**ckpt.get("config", {}))
        codec = StreamingCodec(config).to(self.device)
        codec.load_state_dict(ckpt["model"])
        codec.eval()
        return codec

    def _load_checkpoint(self, path: Path):
        logger.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]

    def _save_checkpoint(self, path: Path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def _compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        B, K, L = target.shape
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)

        loss = F.cross_entropy(logits_flat, target_flat, reduction="none")
        loss = loss.view(B, K, L)

        mask_expanded = mask.unsqueeze(1).expand_as(loss)
        loss = (loss * mask_expanded).sum() / mask_expanded.sum()

        return loss

    def train_step(self, batch: dict) -> dict:
        tokens = batch["tokens"].to(self.device)
        spk_embed = batch["spk_embed"].to(self.device)
        mask = batch["mask"].to(self.device)

        B, K, L = tokens.shape
        context_len = self.args.context_length

        total_loss = 0.0
        n_chunks = 0

        for start in range(0, L - context_len - 1, context_len):
            end = start + context_len

            input_tokens = tokens[:, :, start:end]
            target_token = tokens[:, :, end : end + 1]

            self.optimizer.zero_grad()

            logits, _ = self.model(input_tokens, spk_embed)

            chunk_mask = mask[:, end : end + 1]
            loss = self._compute_loss(logits.unsqueeze(-1), target_token, chunk_mask)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_chunks += 1
            self.global_step += 1

        return {
            "loss": total_loss / max(n_chunks, 1),
        }

    def train(self):
        dataset = TokenDataset(
            self.args.cache_dir,
            self.codec,
            context_length=self.args.context_length,
            device="cpu",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        output_dir = self.args.output_dir / "token_model"
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

            pbar.update(self.args.batch_size)
            pbar.set_postfix(metrics)

            if self.global_step % self.args.log_every == 0:
                logger.info(f"Step {self.global_step}: loss={metrics['loss']:.4f}")

            if self.global_step % self.args.save_every == 0:
                self._save_checkpoint(
                    output_dir / f"token_model_step{self.global_step}.pt"
                )

        self._save_checkpoint(output_dir / "token_model_final.pt")
        logger.info("Training complete!")


def main():
    args = parse_args()
    trainer = TokenTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
