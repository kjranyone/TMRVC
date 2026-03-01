import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


from .disentangle_losses import GradientReversalLayer


class VoiceStateEncoder(nn.Module):
    """Encode explicit continuous parameters and SSL latent space,
    with a Gradient Reversal Layer (GRL) for adversarial disentanglement.

    Supports delta_voice_state for temporal dynamics modeling (Section 4.3).
    """

    def __init__(self, d_explicit=8, d_ssl=128, d_model=512, num_speakers=100):
        super().__init__()
        self.explicit_proj = nn.Linear(d_explicit, d_model // 2)
        self.ssl_proj = nn.Linear(d_ssl, d_model // 2)
        self.fusion = nn.Linear(d_model, d_model)

        self.delta_proj = nn.Linear(d_explicit, d_model // 4)

        self.final_proj = nn.Linear(d_model + d_model // 4, d_model)

        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        # GRL Adversarial Classifier (predicts speaker to unlearn it)
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers),
        )

    def forward(
        self,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        delta_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            explicit_state: [B, T, 8] voice state parameters
            ssl_state: [B, T, 128] WavLM SSL features
            delta_state: [B, T, 8] optional voice_state_t - voice_state_{t-1}

        Returns:
            state_cond: [B, T, d_model]
            adv_logits: [B, T, num_speakers] (only if in training and requested)
        """
        x_exp = self.explicit_proj(explicit_state)
        x_ssl = self.ssl_proj(ssl_state)

        x = torch.cat([x_exp, x_ssl], dim=-1)
        x = self.fusion(x)

        if delta_state is None:
            delta_state = torch.zeros_like(explicit_state)
            if explicit_state.shape[1] > 1:
                delta_state[:, 1:, :] = (
                    explicit_state[:, 1:, :] - explicit_state[:, :-1, :]
                )

        x_delta = self.delta_proj(delta_state)

        x = torch.cat([x, x_delta], dim=-1)
        x = self.final_proj(x)

        x = x.transpose(1, 2)
        x = F.relu(self.temporal_conv(x))
        x = x.transpose(1, 2)

        if self.training:
            # Adversarial branch
            reversed_x = self.grl(x)
            adv_logits = self.adversarial_classifier(reversed_x)
            return x, adv_logits

        return x

    def forward_streaming(
        self,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        prev_explicit_state: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming forward with explicit delta computation.

        Args:
            explicit_state: [B, 1, 8] current frame voice state
            ssl_state: [B, 1, 128] current frame SSL features
            prev_explicit_state: [B, 8] previous frame voice state

        Returns:
            state_cond: [B, 1, d_model]
        """
        delta_state = explicit_state.squeeze(1) - prev_explicit_state
        delta_state = delta_state.unsqueeze(1)

        return self.forward(explicit_state, ssl_state, delta_state)


class VectorQuantizer(nn.Module):
    """Information Bottleneck (VQ) for VC Encoder.
    Strips speaker and style info by mapping continuous embeddings to discrete codes.
    """

    def __init__(self, n_bins: int, d_model: int, beta: float = 0.25):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model
        self.beta = beta

        self.embedding = nn.Embedding(n_bins, d_model)
        self.embedding.weight.data.uniform_(-1.0 / n_bins, 1.0 / n_bins)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, T, d_model]
        Returns:
            z_q: [B, T, d_model] quantized vectors
            loss: VQ commitment loss
            indices: [B, T] quantization indices
        """
        # Flatten
        z_flattened = z.reshape(-1, self.d_model)

        # Distances from z to embeddings
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find closest embeddings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_bins, device=z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Loss: commitment loss (pulling encoder output to embeddings) + codebook loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        indices = min_encoding_indices.view(z.shape[:2])
        return z_q, loss, indices


class VCEncoder(nn.Module):
    """Encodes source A_t tokens and applies VQ bottleneck to remove style/speaker info."""

    def __init__(self, n_codebooks=8, vocab_size=1024, d_model=512, vq_bins=128):
        super().__init__()
        # Each codebook gets d_model // n_codebooks dimensions to concat into d_model
        self.codebook_embeds = nn.ModuleList(
            [
                nn.Embedding(vocab_size, d_model // n_codebooks)
                for _ in range(n_codebooks)
            ]
        )

        # Causal convolution instead of full transformer to keep it light and causal
        self.source_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2)

        self.vq_bottleneck = VectorQuantizer(vq_bins, d_model)

    def forward(self, source_a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_a_t: [B, 8, T] from EnCodec
        Returns:
            content_features: [B, T, d_model]
            vq_loss: scalar tensor
        """
        B, n_cb, T = source_a_t.shape

        # Embed and concatenate along feature dim
        embeds = []
        for i, emb_layer in enumerate(self.codebook_embeds):
            embeds.append(emb_layer(source_a_t[:, i, :]))  # [B, T, d_model//8]

        x = torch.cat(embeds, dim=-1)  # [B, T, d_model]

        # Causal conv [B, d_model, T] -> [B, T, d_model]
        x = x.transpose(1, 2)
        x = F.relu(self.source_conv(x))
        x = x[:, :, :-2]  # Remove padding to keep causal
        x = x.transpose(1, 2)

        # Apply Information Bottleneck
        content_features, vq_loss, _ = self.vq_bottleneck(x)

        return content_features, vq_loss
