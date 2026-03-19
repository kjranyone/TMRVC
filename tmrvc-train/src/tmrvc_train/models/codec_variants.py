"""Codec comparison condition B/C/D scaffolding.

Experimental modules for the codec strategy comparison (track_codec_strategy.md):

- Condition B (Mimi AR+NAR): NARRefinementHead
    CB0 generated autoregressively, CB1-7 predicted in parallel via MLP.

- Condition C (Mimi delay-pattern AR): DelayPatternScheduler
    All codebooks generated AR with staggered delays (CB_k delayed by k steps).

- Condition D (single-codebook): SingleCodebookProjector
    Maps between a single large-vocabulary codebook and the multi-codebook space.

These are experimental scaffolds, not production code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import D_MODEL, N_CODEBOOKS, RVQ_VOCAB_SIZE


# ---------------------------------------------------------------------------
# Condition B: Mimi AR + NAR refinement
# ---------------------------------------------------------------------------

class NARRefinementHead(nn.Module):
    """Non-autoregressive head that predicts CB1..CB_{K-1} from CB0 + hidden states.

    Given hidden states from the AR backbone (which produces CB0), this module
    predicts the remaining codebooks in parallel -- one independent linear head
    per codebook.

    Args:
        d_model: Hidden dimension from the AR backbone.
        n_nar_codebooks: Number of codebooks to predict (default: N_CODEBOOKS - 1 = 7).
        codebook_size: Vocabulary size for each codebook (default: RVQ_VOCAB_SIZE).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_nar_codebooks: int = N_CODEBOOKS - 1,
        codebook_size: int = RVQ_VOCAB_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_nar_codebooks = n_nar_codebooks
        self.codebook_size = codebook_size

        # One independent linear projection per NAR codebook
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(n_nar_codebooks)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict NAR codebook logits from hidden states.

        Args:
            hidden_states: [B, T, d_model] from the AR backbone.

        Returns:
            logits: [B, T, n_nar_codebooks, codebook_size]
        """
        # Stack predictions from each head: list of [B, T, codebook_size]
        per_cb = [head(hidden_states) for head in self.heads]
        return torch.stack(per_cb, dim=2)  # [B, T, n_nar_codebooks, codebook_size]


# ---------------------------------------------------------------------------
# Condition C: Mimi delay-pattern AR
# ---------------------------------------------------------------------------

class DelayPatternScheduler:
    """Manages codebook delay offsets for multi-codebook AR generation.

    In the delay pattern, each codebook CB_k is delayed by k steps relative to
    the first codebook CB_0.  During training, the targets are shifted so that
    the model sees at position t:
        CB_0[t], CB_1[t-1], CB_2[t-2], ..., CB_7[t-7]

    This class provides stateless helpers to apply and undo the delay.

    Args:
        n_codebooks: Total number of codebooks.
        pad_token_id: Token used to fill the padding region introduced by delays.
    """

    def __init__(
        self,
        n_codebooks: int = N_CODEBOOKS,
        pad_token_id: int = 0,
    ):
        self.n_codebooks = n_codebooks
        self.pad_token_id = pad_token_id

    def prepare_delayed_input(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply delay offsets to multi-codebook tokens.

        Each codebook k is shifted right by k positions.  The output sequence is
        longer than the input by (n_codebooks - 1) steps to accommodate all delays.

        Args:
            tokens: [B, n_codebooks, T] original tokens.

        Returns:
            delayed: [B, n_codebooks, T + n_codebooks - 1] delayed tokens,
                     padded with pad_token_id on the left per codebook.
        """
        B, K, T = tokens.shape
        T_out = T + K - 1
        delayed = tokens.new_full((B, K, T_out), self.pad_token_id)
        for k in range(K):
            # CB_k starts at offset k
            delayed[:, k, k:k + T] = tokens[:, k, :]
        return delayed

    def undelay_output(self, delayed: torch.Tensor) -> torch.Tensor:
        """Remove delay offsets to recover aligned multi-codebook tokens.

        Inverse of prepare_delayed_input: strips padding so that codebooks are
        re-aligned in time.

        Args:
            delayed: [B, n_codebooks, T_delayed] delayed tokens.

        Returns:
            tokens: [B, n_codebooks, T_original] where
                    T_original = T_delayed - (n_codebooks - 1).
        """
        B, K, T_delayed = delayed.shape
        T_orig = T_delayed - (K - 1)
        if T_orig <= 0:
            raise ValueError(
                f"Delayed sequence too short ({T_delayed}) for {K} codebooks. "
                f"Need at least {K} timesteps."
            )
        tokens = delayed.new_empty((B, K, T_orig))
        for k in range(K):
            tokens[:, k, :] = delayed[:, k, k:k + T_orig]
        return tokens


# ---------------------------------------------------------------------------
# Condition D: Single large-vocabulary codebook
# ---------------------------------------------------------------------------

class SingleCodebookProjector(nn.Module):
    """Projects between a single large-vocabulary codebook and multi-codebook space.

    For single-codebook codecs (e.g. WavTokenizer, X-Codec 2) the AR model
    operates over a single token stream with a larger vocabulary.  This module
    provides:
    - embed(): single-codebook token -> d_model embedding
    - project(): d_model hidden state -> single-codebook logits

    Args:
        vocab_size: Vocabulary size of the single codebook (e.g. 8192).
        d_model: Hidden dimension of the backbone.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = D_MODEL,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed single-codebook tokens.

        Args:
            tokens: [B, T] int64 token indices.

        Returns:
            embeddings: [B, T, d_model]
        """
        return self.embedding(tokens)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to single-codebook logits.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            logits: [B, T, vocab_size]
        """
        return self.output_proj(hidden_states)
