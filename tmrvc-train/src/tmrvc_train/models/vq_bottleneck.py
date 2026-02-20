"""Factorized VQ Bottleneck for speaker leakage reduction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import (
    D_CONTENT,
    VQ_N_CODEBOOKS,
    VQ_CODEBOOK_SIZE,
    VQ_CODEBOOK_DIM,
    VQ_COMMITMENT_LAMBDA,
)


class FactorizedVQBottleneck(nn.Module):
    """Factorized Vector Quantized bottleneck.

    Splits input into N codebooks, quantizes each independently,
    then concatenates. Uses EMA for codebook updates and
    straight-through estimator for gradients.

    Args:
        d_input: Input dimension (default: D_CONTENT = 256).
        n_codebooks: Number of codebooks (default: 2).
        codebook_size: Entries per codebook (default: 8192).
        codebook_dim: Dimension per codebook entry (default: 128).
        commitment_lambda: Commitment loss weight (default: 0.25).
    """

    def __init__(
        self,
        d_input: int = D_CONTENT,
        n_codebooks: int = VQ_N_CODEBOOKS,
        codebook_size: int = VQ_CODEBOOK_SIZE,
        codebook_dim: int = VQ_CODEBOOK_DIM,
        commitment_lambda: float = VQ_COMMITMENT_LAMBDA,
    ) -> None:
        super().__init__()
        assert d_input == n_codebooks * codebook_dim, (
            f"d_input ({d_input}) must equal n_codebooks ({n_codebooks}) * codebook_dim ({codebook_dim})"
        )

        self.d_input = d_input
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_lambda = commitment_lambda

        # Codebooks: [n_codebooks, codebook_size, codebook_dim]
        # Initialize with uniform distribution
        self.register_buffer(
            "codebooks",
            torch.randn(n_codebooks, codebook_size, codebook_dim) * 0.1,
        )

        # EMA decay for codebook updates
        self.ema_decay = 0.99

        # EMA counts for codebook usage tracking
        self.register_buffer(
            "ema_counts",
            torch.ones(n_codebooks, codebook_size),
        )
        # EMA sums: running weighted sum of assigned vectors
        self.register_buffer(
            "ema_sums",
            self.codebooks.clone(),  # [n_codebooks, codebook_size, codebook_dim]
        )
        # EMA weights: codebook = ema_sums / ema_counts
        self.register_buffer(
            "ema_weights",
            self.codebooks.clone(),
        )

    def forward(
        self,
        x: torch.Tensor,
        use_ema: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with quantization.

        Args:
            x: Input tensor [B, d_input, T] or [B, d_input].
            use_ema: Whether to use EMA codebook weights (default: True).

        Returns:
            Tuple of:
                - quantized: Quantized tensor, same shape as input.
                - indices: Codebook indices [B, n_codebooks, T] or [B, n_codebooks].
                - commitment_loss: Scalar commitment loss.
        """
        # Handle both [B, C, T] and [B, C] inputs
        is_1d = x.dim() == 2
        if is_1d:
            x = x.unsqueeze(-1)  # [B, C] -> [B, C, 1]

        B, C, T = x.shape
        assert C == self.d_input, f"Expected C={self.d_input}, got {C}"

        # Reshape: [B, C, T] -> [B, n_codebooks, codebook_dim, T] -> [B * T, n_codebooks, codebook_dim]
        x_split = x.view(B, self.n_codebooks, self.codebook_dim, T)
        x_split = x_split.permute(0, 3, 1, 2).reshape(
            B * T, self.n_codebooks, self.codebook_dim
        )

        # Select codebook to use
        codebooks = self.ema_weights if use_ema else self.codebooks

        # Quantize each codebook
        all_indices = []
        all_quantized = []
        total_commitment_loss = 0.0

        for i in range(self.n_codebooks):
            # [B*T, codebook_dim] and [codebook_size, codebook_dim]
            x_i = x_split[:, i, :]  # [B*T, codebook_dim]
            cb_i = codebooks[i]  # [codebook_size, codebook_dim]

            # Compute distances: [B*T, codebook_size]
            distances = torch.cdist(x_i, cb_i)

            # Find nearest codebook entry
            indices_i = distances.argmin(dim=-1)  # [B*T]

            # Gather quantized vectors
            quantized_i = cb_i[indices_i]  # [B*T, codebook_dim]

            # Straight-through estimator: copy gradients from input
            quantized_i = x_i + (quantized_i - x_i).detach()

            all_indices.append(indices_i)
            all_quantized.append(quantized_i)

            # Commitment loss (only for training)
            if self.training:
                commitment_loss = F.mse_loss(x_i, quantized_i.detach())
                total_commitment_loss = total_commitment_loss + commitment_loss

        # Stack and reshape
        indices = torch.stack(all_indices, dim=1)  # [B*T, n_codebooks]
        quantized = torch.stack(
            all_quantized, dim=1
        )  # [B*T, n_codebooks, codebook_dim]

        # Reshape back: [B*T, n_codebooks, codebook_dim] -> [B, T, n_codebooks, codebook_dim] -> [B, C, T]
        quantized = quantized.view(B, T, self.n_codebooks, self.codebook_dim)
        quantized = quantized.permute(0, 2, 3, 1).reshape(B, C, T)

        indices = indices.view(B, T, self.n_codebooks).permute(
            0, 2, 1
        )  # [B, n_codebooks, T]

        # Handle 1d case
        if is_1d:
            quantized = quantized.squeeze(-1)
            indices = indices.squeeze(-1)

        return quantized, indices, total_commitment_loss * self.commitment_lambda

    def update_ema(self, x: torch.Tensor, indices: torch.Tensor) -> None:
        """Update EMA codebook weights based on recent encodings.

        Should be called after forward pass during training.

        Args:
            x: Input tensor [B, d_input, T] or [B, d_input].
            indices: Codebook indices from forward pass [B, n_codebooks, T] or [B, n_codebooks].
        """
        if not self.training:
            return

        # Handle both shapes
        is_1d = x.dim() == 2
        if is_1d:
            x = x.unsqueeze(-1)
            indices = indices.unsqueeze(-1)

        B, C, T = x.shape

        # Reshape input
        x_split = x.view(B, self.n_codebooks, self.codebook_dim, T)
        x_split = x_split.permute(0, 3, 1, 2).reshape(
            B * T, self.n_codebooks, self.codebook_dim
        )

        # Reshape indices
        indices = indices.permute(0, 2, 1).reshape(B * T, self.n_codebooks)

        # Update EMA for each codebook
        with torch.no_grad():
            for i in range(self.n_codebooks):
                x_i = x_split[:, i, :]  # [B*T, codebook_dim]
                idx_i = indices[:, i]  # [B*T]

                # Count occurrences
                counts = torch.zeros(self.codebook_size, device=x.device)
                ones = torch.ones_like(idx_i, dtype=torch.float)
                counts.scatter_add_(0, idx_i, ones)

                # Sum of vectors per code
                sums = torch.zeros(
                    self.codebook_size, self.codebook_dim, device=x.device
                )
                sums.scatter_add_(
                    0, idx_i.unsqueeze(1).expand(-1, self.codebook_dim), x_i
                )

                # Update EMA counts
                self.ema_counts[i] = (
                    self.ema_decay * self.ema_counts[i] + (1 - self.ema_decay) * counts
                )

                # Update EMA sums
                self.ema_sums[i] = (
                    self.ema_decay * self.ema_sums[i] + (1 - self.ema_decay) * sums
                )

                # Derive codebook weights: ema_sums / ema_counts
                n = self.ema_counts[i].clamp(min=1e-8).unsqueeze(1)
                self.ema_weights[i] = self.ema_sums[i] / n

    def get_codebook_usage(self) -> torch.Tensor:
        """Return codebook usage statistics.

        Returns:
            [n_codebooks, codebook_size] tensor of usage counts.
        """
        return self.ema_counts.clone()
