import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ControlEvent:
    op: str
    event_type: str
    duration_ms: int
    intensity: float


def event_to_tuple(event: ControlEvent) -> tuple[int, int, int, int]:
    """Convert ControlEvent to [op_id, type_id, dur_id, int_id] tuple."""
    op_map = {"none": 4, "start": 5, "hold": 6, "end": 7}
    type_map = {
        "laugh": 8,
        "sob": 9,
        "sigh": 10,
        "breath": 11,
        "moan": 12,
        "silence": 13,
    }

    op_id = op_map.get(event.op, 4)
    type_id = type_map.get(event.event_type, 13)
    dur_bin = max(0, min(39, (event.duration_ms // 50) - 1)) + 14
    int_bin = max(0, min(7, int(event.intensity * 8))) + 54

    return (op_id, type_id, dur_bin, int_bin)


def tuple_to_event(op_id: int, type_id: int, dur_id: int, int_id: int) -> ControlEvent:
    """Convert tuple to ControlEvent."""
    op_rev = {4: "none", 5: "start", 6: "hold", 7: "end"}
    type_rev = {
        8: "laugh",
        9: "sob",
        10: "sigh",
        11: "breath",
        12: "moan",
        13: "silence",
    }

    return ControlEvent(
        op=op_rev.get(op_id, "none"),
        event_type=type_rev.get(type_id, "silence"),
        duration_ms=(dur_id - 14 + 1) * 50,
        intensity=(int_id - 54) / 7.0,
    )


class ControlTokenizer:
    """Convert events to token tuples and vice versa."""

    def __init__(self):
        self.op_map = {"none": 4, "start": 5, "hold": 6, "end": 7}
        self.type_map = {
            "laugh": 8,
            "sob": 9,
            "sigh": 10,
            "breath": 11,
            "moan": 12,
            "silence": 13,
        }
        self.op_rev = {v: k for k, v in self.op_map.items()}
        self.type_rev = {v: k for k, v in self.type_map.items()}

    def encode_event(self, event: ControlEvent) -> torch.Tensor:
        """Convert single event to tensor [4]."""
        tup = event_to_tuple(event)
        return torch.tensor(tup, dtype=torch.long)

    def encode_events(self, events: list[ControlEvent]) -> torch.Tensor:
        """Convert list of events to tensor [T, 4]."""
        tuples = [event_to_tuple(e) for e in events]
        return torch.tensor(tuples, dtype=torch.long)

    def decode_tensor(self, tokens: torch.Tensor) -> list[ControlEvent]:
        """Convert tensor [T, 4] to list of events."""
        events = []
        for t in range(tokens.shape[0]):
            op_id, type_id, dur_id, int_id = tokens[t].tolist()
            events.append(tuple_to_event(op_id, type_id, dur_id, int_id))
        return events

    def tokenize_nonverbal_event(
        self, event_type: str, duration_ms: int, intensity: float, frame_ms: int = 10
    ) -> list[ControlEvent]:
        """Expand a non-verbal event into frame-level events.

        Example: 1200ms moan -> 120 frame events with start/hold/end
        """
        n_frames = duration_ms // frame_ms
        events = []

        for i in range(n_frames):
            remaining_ms = (n_frames - i) * frame_ms

            if i == 0:
                op = "start"
            elif i == n_frames - 1:
                op = "end"
            else:
                op = "hold"

            frame_intensity = intensity
            if op == "end":
                frame_intensity = intensity * 0.8

            events.append(
                ControlEvent(
                    op=op,
                    event_type=event_type,
                    duration_ms=remaining_ms,
                    intensity=frame_intensity,
                )
            )

        return events


class EventTrace(nn.Module):
    """Event Hysteresis for non-verbal event decay.

    event_trace_t = alpha * event_trace_{t-1} + event_embed(B_t)

    Used to maintain decay traces for laughs, sobs, breaths etc.
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 64, alpha: float = 0.9):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.alpha = alpha

        self.event_embed = nn.Embedding(vocab_size, d_model)

    def forward(
        self,
        ctrl_tokens: torch.Tensor,
        prev_trace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ctrl_tokens: [B, T, 4] control tokens
            prev_trace: [B, d_model] previous event trace (optional)

        Returns:
            trace: [B, T, d_model] event trace for each frame
        """
        B, T, _ = ctrl_tokens.shape

        type_tokens = ctrl_tokens[:, :, 1]
        event_embeds = self.event_embed(type_tokens)

        if prev_trace is None:
            prev_trace = torch.zeros(
                B, self.d_model, device=ctrl_tokens.device, dtype=event_embeds.dtype
            )

        traces = []
        current_trace = prev_trace

        for t in range(T):
            current_trace = self.alpha * current_trace + event_embeds[:, t, :]
            traces.append(current_trace.clone())

        return torch.stack(traces, dim=1)

    def forward_streaming(
        self,
        ctrl_token: torch.Tensor,
        prev_trace: torch.Tensor,
    ) -> torch.Tensor:
        """Single-frame streaming update.

        Args:
            ctrl_token: [B, 4] single frame control tokens
            prev_trace: [B, d_model] previous trace

        Returns:
            trace: [B, d_model] updated trace
        """
        type_token = ctrl_token[:, 1]
        event_embed = self.event_embed(type_token)
        return self.alpha * prev_trace + event_embed
