import torch
import torch.nn as nn


class ControlEncoder(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model // 4)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, ctrl_tokens: torch.Tensor) -> torch.Tensor:
        if ctrl_tokens.dim() == 3:
            ctrl_tokens = ctrl_tokens.squeeze(1)
        e = self.embed(ctrl_tokens) # [B, 4, d/4]
        e = e.reshape(e.shape[0], self.d_model) # Force [B, 512]
        return self.proj(e).unsqueeze(-1)


class ControlEncoderTemporal(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model // 4)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, ctrl_tokens: torch.Tensor) -> torch.Tensor:
        # Expected ctrl_tokens: [B, T, 4]
        B, T, _ = ctrl_tokens.shape
        e = self.embed(ctrl_tokens) # [B, T, 4, d/4]
        e = e.reshape(B, T, self.d_model) # Force [B, T, 512]
        return self.proj(e) # Should now correctly apply Linear to last dim


CONTROL_VOCAB = {
    "special": {"<ctrl_none>": 0, "<ctrl_pad>": 1, "<ctrl_bos>": 2, "<ctrl_eos>": 3},
    "op": {"<op_none>": 4, "<op_start>": 5, "<op_hold>": 6, "<op_end>": 7},
    "type": {"<laugh>": 8, "<sob>": 9, "<sigh>": 10, "<breath>": 11, "<moan>": 12, "<silence>": 13},
    "duration_start": 14,
    "duration_end": 53,
    "intensity_start": 54,
    "intensity_end": 61,
    "reserved": [62, 63],
}

def encode_duration_bin(duration_ms: float) -> int:
    bin_idx = int(duration_ms / 50)
    bin_idx = max(1, min(40, bin_idx))
    return CONTROL_VOCAB["duration_start"] + bin_idx - 1

def encode_intensity_bin(intensity: float) -> int:
    bin_idx = int(intensity * 8)
    bin_idx = max(0, min(7, bin_idx))
    return CONTROL_VOCAB["intensity_start"] + bin_idx

def decode_duration_bin(bin_idx: int) -> int:
    return (bin_idx - CONTROL_VOCAB["duration_start"] + 1) * 50

def decode_intensity_bin(bin_idx: int) -> float:
    return (bin_idx - CONTROL_VOCAB["intensity_start"]) / 7.0
