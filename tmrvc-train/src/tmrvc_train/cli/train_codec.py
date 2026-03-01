"""``tmrvc-train-codec`` — Train EmotionAwareCodec for UCLM.

Trains the EmotionAwareCodec model:
- Encoder: audio → A_t (acoustic tokens) + B_t (control logits)
- Decoder: A_t + B_t + voice_state → audio

Uses pre-extracted features from cache (mel, f0, voice_state).

Usage::

    tmrvc-train-codec --cache-dir data/cache/vctk --output-dir checkpoints/codec --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tmrvc_train.models.emotion_codec import EmotionAwareCodec

logger = logging.getLogger(__name__)


class CodecDataset(Dataset):
    """Dataset for EmotionAwareCodec training from cached features."""

    def __init__(
        self,
        cache_dir: str | Path,
        max_frames: int = 400,
        sample_rate: int = 24000,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.utterances = []

        for meta_path in self.cache_dir.rglob("meta.json"):
            if "train" not in str(meta_path):
                continue
            utt_dir = meta_path.parent

            required = ["mel.npy", "f0.npy", "waveform.npy"]
            if not all((utt_dir / f).exists() for f in required):
                continue

            try:
                with open(meta_path, encoding="utf-8") as f:
                    import json

                    meta = json.load(f)

                self.utterances.append(
                    {
                        "path": utt_dir,
                        "n_frames": meta.get("n_frames", 0),
                    }
                )
            except Exception:
                continue

        self.utterances.sort(key=lambda x: x["path"].name)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        utt = self.utterances[idx]
        utt_dir = utt["path"]

        mel = np.load(utt_dir / "mel.npy")
        f0 = np.load(utt_dir / "f0.npy")
        waveform = np.load(utt_dir / "waveform.npy")

        T = mel.shape[1]
        if T > self.max_frames:
            start = np.random.randint(0, T - self.max_frames)
            mel = mel[:, start : start + self.max_frames]
            f0 = f0[:, start : start + self.max_frames]

            wav_start = start * 240
            wav_end = start * 240 + self.max_frames * 240
            if wav_end <= waveform.shape[-1]:
                waveform = waveform[..., wav_start:wav_end]

        explicit_state = np.zeros((mel.shape[1], 8), dtype=np.float32)
        if (utt_dir / "explicit_state.npy").exists():
            es = np.load(utt_dir / "explicit_state.npy")
            if es.shape[0] >= mel.shape[1]:
                explicit_state = es[: mel.shape[1], :]

        return {
            "mel": torch.from_numpy(mel).float(),
            "f0": torch.from_numpy(f0).float(),
            "waveform": torch.from_numpy(waveform).float(),
            "voice_state": torch.from_numpy(explicit_state).float(),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(item["mel"].shape[1] for item in batch)

    mels, f0s, waveforms, voice_states = [], [], [], []

    for item in batch:
        T = item["mel"].shape[1]
        pad_len = max_len - T

        mels.append(nn.functional.pad(item["mel"], (0, pad_len)))
        f0s.append(nn.functional.pad(item["f0"], (0, pad_len)))

        vs = item["voice_state"]
        if vs.dim() == 2:
            vs = nn.functional.pad(vs, (0, 0, 0, pad_len))
        voice_states.append(vs)

        wav = item["waveform"]
        wav_T = wav.shape[-1]
        target_T = max_len * 240
        if wav_T < target_T:
            wav = nn.functional.pad(wav, (0, target_T - wav_T))
        else:
            wav = wav[..., :target_T]
        waveforms.append(wav)

    return {
        "mel": torch.stack(mels),
        "f0": torch.stack(f0s),
        "waveform": torch.stack(waveforms),
        "voice_state": torch.stack(voice_states),
    }


def train_codec(
    cache_dir: Path,
    output_dir: Path,
    batch_size: int,
    max_frames: int,
    max_steps: int,
    device: str,
    lr: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", cache_dir)
    dataset = CodecDataset(cache_dir, max_frames=max_frames)

    if len(dataset) == 0:
        logger.error("No valid utterances found in %s", cache_dir)
        return

    logger.info("Found %d utterances", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    logger.info("Initializing EmotionAwareCodec model")
    model = EmotionAwareCodec().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %d", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.99))

    def mel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(pred, target)

    def stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.squeeze(1)
        target_flat = target.squeeze(1)

        stft_pred = torch.stft(
            pred_flat,
            n_fft=1024,
            hop_length=240,
            win_length=960,
            window=torch.hann_window(960, device=pred.device),
            return_complex=True,
        )
        stft_target = torch.stft(
            target_flat,
            n_fft=1024,
            hop_length=240,
            win_length=960,
            window=torch.hann_window(960, device=target.device),
            return_complex=True,
        )

        mag_pred = torch.abs(stft_pred) + 1e-7
        mag_target = torch.abs(stft_target) + 1e-7

        sc_loss = torch.norm(mag_pred - mag_target, p="fro") / (
            torch.norm(mag_target, p="fro") + 1e-7
        )
        log_loss = nn.functional.l1_loss(torch.log(mag_pred), torch.log(mag_target))

        return sc_loss + log_loss

    step = 0
    epoch = 0
    pbar = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        epoch += 1
        for batch in loader:
            if step >= max_steps:
                break

            waveform = batch["waveform"].to(device)
            voice_state = batch["voice_state"].to(device)

            optimizer.zero_grad()

            a_tokens, b_logits, enc_states = model.encode(waveform)

            B = a_tokens.shape[0]
            T_frames = a_tokens.shape[-1]

            voice_state_frames = voice_state[:, :T_frames, :]

            b_tokens = b_logits.argmax(dim=-1)
            audio_recon, dec_states = model.decode(
                a_tokens,
                b_tokens,
                voice_state_frames,
            )

            min_len = min(waveform.shape[-1], audio_recon.shape[-1])
            if min_len < 512:
                continue
            waveform_crop = waveform[..., :min_len]
            audio_recon_crop = audio_recon[..., :min_len]

            loss_stft = stft_loss(audio_recon_crop, waveform_crop)

            total_loss = loss_stft

            total_loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix(
                {
                    "step": step,
                    "loss": f"{total_loss.item():.4f}",
                    "stft": f"{loss_stft.item():.4f}",
                }
            )

            if step > 0 and step % 1000 == 0:
                ckpt_path = output_dir / f"codec_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

            step += 1

    pbar.close()

    final_path = output_dir / "codec_final.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_path,
    )
    logger.info("Training complete. Final model saved to %s", final_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-train-codec",
        description="Train EmotionAwareCodec for UCLM.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/codec"),
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Max frames per utterance.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Max training steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    train_codec(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        max_steps=args.max_steps,
        device=args.device,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
