"""UCLM v2 Codec Training CLI."""
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tmrvc_train.models.emotion_codec import EmotionAwareCodec, CodecLoss

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
    wf = torch.zeros((B, 1, max_T * 240), dtype=torch.float32)
    for i, item in enumerate(batch):
        T = item["target_a"].shape[1]
        ta[i, :, :T] = item["target_a"]; tb[i, :, :T] = item["target_b"]; vs[i, :T, :] = item["voice_state"]
        w = item["waveform"]; wf[i, :, :T*240] = w if w.dim() == 2 else w.unsqueeze(0)
    return {"waveform": wf, "target_a": ta, "target_b": tb, "voice_state": vs}

def train_codec(cache_dir, output_dir, batch_size, max_steps, device, lr):
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(CodecDataset(cache_dir), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = EmotionAwareCodec().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=lr); crit = CodecLoss().to(device)
    pbar = tqdm(total=max_steps, desc="Training")
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break
            optimizer.zero_grad()
            w, vs, ta, tb = batch["waveform"].to(device), batch["voice_state"].to(device), batch["target_a"].to(device), batch["target_b"].to(device)
            
            # Encoder forward for distillation
            _, pred_b_logits, _, pred_a_logits = model.encode(w)
            
            # Decoder forward with ground truth tokens (Teacher forcing)
            recon, _ = model.decode(ta, tb, vs)
            
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
