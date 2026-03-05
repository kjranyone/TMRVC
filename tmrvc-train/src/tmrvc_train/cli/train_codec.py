"""UCLM v2 Codec Training CLI."""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tmrvc_train.models.emotion_codec import EmotionAwareCodec, CodecLoss


ACOUSTIC_VOCAB_SIZE = 1024
CONTROL_VOCAB_SIZE = 64
HOP_LENGTH = 240

class CodecDataset(Dataset):
    def __init__(self, cache_dir, max_frames=400):
        self.cache_dir = Path(cache_dir); self.max_frames = max_frames; self.utterances = []
        for p in self.cache_dir.rglob("meta.json"):
            if "train" in str(p) and (p.parent / "waveform.npy").exists():
                self.utterances.append(p.parent)
    def __len__(self): return len(self.utterances)
    def __getitem__(self, idx):
        d = self.utterances[idx]
        w = np.load(d / "waveform.npy"); a = np.load(d / "codec_tokens.npy")
        T = a.shape[1]
        b = np.load(d / "control_tokens.npy") if (d / "control_tokens.npy").exists() else np.zeros((4, T), dtype=np.int64)
        v = np.load(d / "explicit_state.npy") if (d / "explicit_state.npy").exists() else np.zeros((T, 8), dtype=np.float32)
        if T > self.max_frames:
            s = np.random.randint(0, T - self.max_frames)
            a = a[:, s:s+self.max_frames]; b = b[:, s:s+self.max_frames]; v = v[s:s+self.max_frames]; w = w[..., s*240:(s+self.max_frames)*240]
        return {"waveform": torch.from_numpy(w).float(), "target_a": torch.from_numpy(a).long(), "target_b": torch.from_numpy(b).long(), "voice_state": torch.from_numpy(v).float()}

def collate_fn(batch):
    max_T = max(item["target_a"].shape[1] for item in batch); B = len(batch)
    ta = torch.full((B, 8, max_T), -1, dtype=torch.long)
    tb = torch.full((B, 4, max_T), -1, dtype=torch.long)
    vs = torch.zeros((B, max_T, 8), dtype=torch.float32)
    wf = torch.zeros((B, 1, max_T * HOP_LENGTH), dtype=torch.float32)
    for i, item in enumerate(batch):
        T = item["target_a"].shape[1]
        ta[i, :, :T] = item["target_a"]; tb[i, :, :T] = item["target_b"]; vs[i, :T, :] = item["voice_state"]
        w = _align_waveform_to_token_frames(item["waveform"], T)
        wf[i, :, :T * HOP_LENGTH] = w
    return {"waveform": wf, "target_a": ta, "target_b": tb, "voice_state": vs}


def _align_waveform_to_token_frames(
    waveform: torch.Tensor, n_frames: int, hop_length: int = HOP_LENGTH
) -> torch.Tensor:
    """Align waveform length to token frame count (trim or zero-pad tail)."""
    target_samples = n_frames * hop_length
    if waveform.dim() == 1:
        wave = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        if waveform.shape[0] == 1:
            wave = waveform
        elif waveform.shape[1] == 1:
            wave = waveform.transpose(0, 1)
        else:
            raise ValueError(
                f"Expected mono waveform with one channel, got shape={tuple(waveform.shape)}."
            )
    else:
        raise ValueError(
            f"Expected waveform rank 1 or 2, got rank={waveform.dim()} shape={tuple(waveform.shape)}."
        )

    current_samples = wave.shape[-1]
    if current_samples == target_samples:
        return wave
    if current_samples > target_samples:
        return wave[..., :target_samples]
    return F.pad(wave, (0, target_samples - current_samples))


def _prepare_decoder_inputs(target_a: torch.Tensor, target_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert loss targets to valid embedding indices for teacher forcing decode."""
    # -1 is padding label for CE(ignore_index=-1); decoder embedding requires >=0 ids.
    return target_a.clamp_min(0), target_b.clamp_min(0)


def _validate_token_ranges(target_a: torch.Tensor, target_b: torch.Tensor) -> None:
    """Fail fast with a clear error before CUDA index assert."""
    invalid_a_low = (target_a < -1).any().item()
    invalid_a_high = (target_a >= ACOUSTIC_VOCAB_SIZE).any().item()
    invalid_b_low = (target_b < -1).any().item()
    invalid_b_high = (target_b >= CONTROL_VOCAB_SIZE).any().item()
    if invalid_a_low or invalid_a_high or invalid_b_low or invalid_b_high:
        a_min = int(target_a.min().item())
        a_max = int(target_a.max().item())
        b_min = int(target_b.min().item())
        b_max = int(target_b.max().item())
        raise ValueError(
            "Invalid token range in codec training batch: "
            f"A[min={a_min}, max={a_max}] expected in [-1, {ACOUSTIC_VOCAB_SIZE - 1}], "
            f"B[min={b_min}, max={b_max}] expected in [-1, {CONTROL_VOCAB_SIZE - 1}]."
        )

def train_codec(cache_dir, output_dir, batch_size, max_steps, device, lr):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CodecDataset(cache_dir)
    if len(dataset) == 0:
        raise ValueError(
            f"No codec training utterances found in cache_dir={cache_dir}. "
            "Expected train entries with meta.json + waveform.npy."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = EmotionAwareCodec().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=lr); crit = CodecLoss().to(device)
    pbar = tqdm(total=max_steps, desc="Training")
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break
            optimizer.zero_grad()
            w, vs, ta, tb = batch["waveform"].to(device), batch["voice_state"].to(device), batch["target_a"].to(device), batch["target_b"].to(device)
            _validate_token_ranges(ta, tb)
            
            # Encoder forward for distillation
            _, pred_b_logits, _, pred_a_logits = model.encode(w)
            
            # Decoder forward with ground truth tokens (Teacher forcing)
            ta_in, tb_in = _prepare_decoder_inputs(ta, tb)
            recon, _ = model.decode(ta_in, tb_in, vs)
            
            losses = crit(recon, w, pred_b_logits, tb, pred_a_logits, ta)
            losses["loss"].backward(); optimizer.step()
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{losses['loss'].item():.3f}",
                    "stft": f"{losses['loss_stft'].item():.3f}",
                    "ctrl": f"{losses['loss_control'].item():.3f}",
                    "distill": f"{losses['loss_distill'].item():.3f}",
                }
            )
            step += 1
    torch.save({"model": model.state_dict()}, output_dir / "codec_final.pt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/codec"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    train_codec(**vars(args))

if __name__ == "__main__":
    main()
