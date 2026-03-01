import torch
import torch.nn as nn


class ControlEncoder(nn.Module):
    """Encode control tuple tokens for decoder-side conditioning.

    Input B_t: [B, 4] -> output: [B, d_model, 1]

    Converts [op_id, type_id, dur_id, int_id] tuple into a conditioning vector
    for the decoder.
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model * 4, d_model)

    def forward(self, ctrl_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctrl_tokens: [B, 4] with values in [0, vocab_size)

        Returns:
            [B, d_model, 1] conditioning vector
        """
        e = self.embed(ctrl_tokens)
        e = e.flatten(start_dim=1)
        return self.proj(e).unsqueeze(-1)


class ControlEncoderTemporal(nn.Module):
    """Temporal version for [B, T, 4] control sequences."""

    def __init__(self, vocab_size: int = 64, d_model: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model * 4, d_model)

    def forward(self, ctrl_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctrl_tokens: [B, T, 4]

        Returns:
            [B, T, d_model]
        """
        B, T, _ = ctrl_tokens.shape
        e = self.embed(ctrl_tokens)
        e = e.view(B, T, -1)
        return self.proj(e)


CONTROL_VOCAB = {
    "special": {
        "<ctrl_none>": 0,
        "<ctrl_pad>": 1,
        "<ctrl_bos>": 2,
        "<ctrl_eos>": 3,
    },
    "op": {
        "<op_none>": 4,
        "<op_start>": 5,
        "<op_hold>": 6,
        "<op_end>": 7,
    },
    "type": {
        "<laugh>": 8,
        "<sob>": 9,
        "<sigh>": 10,
        "<breath>": 11,
        "<moan>": 12,
        "<silence>": 13,
    },
    "duration_start": 14,
    "duration_end": 53,
    "intensity_start": 54,
    "intensity_end": 61,
    "reserved": [62, 63],
}


def encode_duration_bin(duration_ms: float) -> int:
    """Convert duration in ms to bin index (50ms steps, 1-40 bins)."""
    bin_idx = int(duration_ms / 50)
    bin_idx = max(1, min(40, bin_idx))
    return CONTROL_VOCAB["duration_start"] + bin_idx - 1


def encode_intensity_bin(intensity: float) -> int:
    """Convert intensity [0, 1] to bin index (0-7)."""
    bin_idx = int(intensity * 8)
    bin_idx = max(0, min(7, bin_idx))
    return CONTROL_VOCAB["intensity_start"] + bin_idx


def decode_duration_bin(bin_idx: int) -> int:
    """Convert bin index to duration in ms."""
    return (bin_idx - CONTROL_VOCAB["duration_start"] + 1) * 50


def decode_intensity_bin(bin_idx: int) -> float:
    """Convert bin index to intensity [0, 1]."""
    return (bin_idx - CONTROL_VOCAB["intensity_start"]) / 7.0
