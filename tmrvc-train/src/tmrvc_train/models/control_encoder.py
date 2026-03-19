import torch
import torch.nn as nn

from tmrvc_core.constants import D_MODEL


class ControlEncoder(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = D_MODEL):
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
    def __init__(self, vocab_size: int = 64, d_model: int = D_MODEL):
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

