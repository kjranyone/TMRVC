"""Scene State Latent (SSL) — maintains acting state across dialogue turns.

The scene state ``z_t`` is a latent vector updated each turn:

    z_t = F_state(z_{t-1}, u_t, h_t, s)

Where:
    - z_{t-1}: previous scene state
    - u_t: current utterance encoding (from TextEncoder or ContentEncoder)
    - h_t: dialogue history summary
    - s: speaker/character embedding

State resets at scene boundaries. Optional stage direction ``hint`` enters as
a soft additive bias, not a required condition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_HISTORY,
    D_SCENE_STATE,
    D_SPEAKER,
    D_TEXT_ENCODER,
    SSL_N_GRU_LAYERS,
    SSL_PROSODY_STATS_DIM,
)


class SceneStateUpdate(nn.Module):
    """GRU-based scene state update: z_t = F_state(z_{t-1}, u_t, h_t, s).

    Concatenates ``[u_t, h_t, s]`` as input to a GRU that evolves the
    hidden state ``z_t``. A residual connection stabilises training.

    Args:
        d_state: Scene state latent dimension.
        d_utterance: Utterance encoding dimension (pooled).
        d_history: Dialogue history summary dimension.
        d_speaker: Speaker/character embedding dimension.
        n_gru_layers: Number of GRU layers.
    """

    def __init__(
        self,
        d_state: int = D_SCENE_STATE,
        d_utterance: int = D_TEXT_ENCODER,
        d_history: int = D_HISTORY,
        d_speaker: int = D_SPEAKER,
        n_gru_layers: int = SSL_N_GRU_LAYERS,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        d_input = d_utterance + d_history + d_speaker
        self.input_proj = nn.Linear(d_input, d_state)
        self.gru = nn.GRU(
            input_size=d_state,
            hidden_size=d_state,
            num_layers=n_gru_layers,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_prev: torch.Tensor,
        u_t: torch.Tensor,
        h_t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Update scene state for one turn.

        Args:
            z_prev: ``[B, D_state]`` previous scene state (zeros for first turn).
            u_t: ``[B, D_utt]`` current utterance encoding (mean-pooled text features).
            h_t: ``[B, D_hist]`` dialogue history summary.
            s: ``[B, D_spk]`` speaker/character embedding.

        Returns:
            ``[B, D_state]`` updated scene state z_t.
        """
        x = torch.cat([u_t, h_t, s], dim=-1)  # [B, D_input]
        x = self.input_proj(x)  # [B, D_state]

        # GRU: treat as single-step sequence
        h_0 = z_prev.unsqueeze(0).expand(self.gru.num_layers, -1, -1).contiguous()
        gru_out, _ = self.gru(x.unsqueeze(1), h_0)  # [B, 1, D_state]
        gru_out = gru_out.squeeze(1)  # [B, D_state]

        # Gated residual: z_t = gate * gru_out + (1 - gate) * z_prev
        gate = self.gate(torch.cat([gru_out, z_prev], dim=-1))
        z_t = gate * gru_out + (1.0 - gate) * z_prev
        return z_t

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zero initial state for scene start.

        Returns:
            ``[B, D_state]`` zero tensor.
        """
        return torch.zeros(batch_size, self.d_state, device=device)


class DialogueHistoryEncoder(nn.Module):
    """Encode dialogue history into a fixed-size summary vector.

    Uses a simple GRU over past utterance encodings.

    Args:
        d_utterance: Per-utterance encoding dimension.
        d_history: Output history summary dimension.
    """

    def __init__(
        self,
        d_utterance: int = D_TEXT_ENCODER,
        d_history: int = D_HISTORY,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_utterance, d_history)
        self.gru = nn.GRU(
            input_size=d_history,
            hidden_size=d_history,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        utterance_history: torch.Tensor,
        history_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a sequence of past utterance encodings.

        Args:
            utterance_history: ``[B, N_turns, D_utt]`` past utterance encodings.
            history_lengths: ``[B]`` number of valid turns per batch item.

        Returns:
            ``[B, D_history]`` summary vector.
        """
        x = self.proj(utterance_history)  # [B, N, D_hist]

        if history_lengths is not None:
            # Pack for efficiency with variable lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                x, history_lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False,
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        return h_n.squeeze(0)  # [B, D_history]


class ProsodyStatsPredictor(nn.Module):
    """Predict prosody statistics from scene state (for L_state_recon).

    Predicts: pitch_mean, pitch_std, energy_mean, energy_std,
              speaking_rate, voiced_ratio, pause_ratio, spectral_tilt.
    """

    def __init__(
        self,
        d_state: int = D_SCENE_STATE,
        d_prosody: int = SSL_PROSODY_STATS_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, d_prosody),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict prosody statistics from scene state.

        Args:
            z: ``[B, D_state]`` scene state.

        Returns:
            ``[B, D_prosody]`` predicted prosody statistics.
        """
        return self.net(z)


class SceneStateLoss(nn.Module):
    """SSL losses: L_state_recon + L_state_cons.

    L_state_recon: MSE between predicted and actual prosody statistics.
    L_state_cons: smoothness penalty on adjacent scene states (cosine distance).
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_cons: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_cons = lambda_cons

    def forward(
        self,
        pred_prosody: torch.Tensor,
        gt_prosody: torch.Tensor,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute SSL losses.

        Args:
            pred_prosody: ``[B, D_prosody]`` predicted prosody stats.
            gt_prosody: ``[B, D_prosody]`` ground-truth prosody stats.
            z_t: ``[B, D_state]`` current scene state.
            z_prev: ``[B, D_state]`` previous scene state.

        Returns:
            Dict with ``state_recon``, ``state_cons``, ``state_total``.
        """
        l_recon = nn.functional.mse_loss(pred_prosody, gt_prosody)

        # Consistency: adjacent states should be similar (cosine distance)
        cos_sim = nn.functional.cosine_similarity(z_t, z_prev, dim=-1)
        l_cons = (1.0 - cos_sim).mean()

        total = self.lambda_recon * l_recon + self.lambda_cons * l_cons
        return {
            "state_recon": l_recon,
            "state_cons": l_cons,
            "state_total": total,
        }
