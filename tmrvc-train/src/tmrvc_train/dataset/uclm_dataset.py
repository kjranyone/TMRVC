import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import json


class DisentangledUCLMDataset(Dataset):
    """Dataset for Disentangled UCLM multi-task training (TTS & VC)."""

    def __init__(self, cache_dir: str | Path, max_frames: int = 400):
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        self.utterances = []

        # Scan for utterance meta.json files
        for meta_path in self.cache_dir.rglob("meta.json"):
            if "train" not in str(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Ensure all required SOTA files exist
                utt_dir = meta_path.parent
                required_files = [
                    "codec_tokens.npy",
                    "explicit_state.npy",
                    "ssl_state.npy",
                    "spk_embed.npy",
                ]

                if all((utt_dir / req).exists() for req in required_files):
                    self.utterances.append({"meta": meta, "path": utt_dir})
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        utt = self.utterances[idx]
        utt_dir = utt["path"]
        meta = utt["meta"]

        # Load arrays
        codec_tokens = np.load(utt_dir / "codec_tokens.npy")  # [8, T]
        explicit_state = np.load(utt_dir / "explicit_state.npy")  # [T, 8]
        ssl_state = np.load(utt_dir / "ssl_state.npy")  # [T, 128]
        spk_embed = np.load(utt_dir / "spk_embed.npy")  # [192]

        # Truncate or pad to max_frames for batching simplicity
        # (In a real scenario we'd use a collate_fn to pad dynamically)
        T = codec_tokens.shape[-1]
        if T > self.max_frames:
            start = np.random.randint(0, T - self.max_frames)
            codec_tokens = codec_tokens[:, start : start + self.max_frames]
            explicit_state = explicit_state[start : start + self.max_frames, :]
            ssl_state = ssl_state[start : start + self.max_frames, :]

        # Target tokens
        target_a = torch.from_numpy(codec_tokens).long()

        # Load Control tokens (B_t)
        if (utt_dir / "control_tokens.npy").exists():
            target_b = torch.from_numpy(np.load(utt_dir / "control_tokens.npy")).long()
        else:
            # Fallback for legacy cache without B_t
            target_b = torch.zeros((4, target_a.shape[1]), dtype=torch.long)

        # Load F0 and create f0_condition [T, 2]
        f0_condition = None
        if (utt_dir / "f0.npy").exists():
            f0 = np.load(utt_dir / "f0.npy").squeeze()  # [T]
            f0_mean = meta.get("f0_mean", 150.0)
            # log2(f0 / mean) normalization
            f0_norm = np.log2((f0 + 1e-8) / f0_mean)
            f0_norm = np.clip(f0_norm, -2.0, 2.0)

            # f0_condition: [f0_norm, pitch_shift]
            # pitch_shift is 0 during training
            f0_cond_np = np.stack([f0_norm, np.zeros_like(f0_norm)], axis=-1)
            f0_condition = torch.from_numpy(f0_cond_np).float()

        phoneme_ids = None
        durations = None
        if (utt_dir / "phoneme_ids.npy").exists():
            phoneme_ids = torch.from_numpy(np.load(utt_dir / "phoneme_ids.npy")).long()
            if (utt_dir / "durations.npy").exists():
                durations = torch.from_numpy(np.load(utt_dir / "durations.npy")).long()

        phoneme_lens = None
        if phoneme_ids is not None:
            phoneme_lens = torch.tensor(len(phoneme_ids)).long()

        return {
            "source_a_t": target_a.clone(),  # For VC mode
            "target_a": target_a,
            "target_b": target_b,
            "explicit_state": torch.from_numpy(explicit_state).float(),
            "ssl_state": torch.from_numpy(ssl_state).float(),
            "speaker_embed": torch.from_numpy(spk_embed).float(),
            "speaker_id": torch.tensor(meta.get("speaker_id_int", 0)).long(),
            "phoneme_ids": phoneme_ids,
            "phoneme_lens": phoneme_lens,
            "durations": durations,
            "f0_condition": f0_condition,
            "language_id": torch.tensor(meta.get("language_id", 0)).long(),
            "text": meta.get("text", ""),
        }
