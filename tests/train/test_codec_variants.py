"""Tests for codec comparison condition B/C/D scaffolding.

Covers:
- NARRefinementHead output shapes (Condition B)
- DelayPatternScheduler delay/undelay round-trip (Condition C)
- SingleCodebookProjector embed and project shapes (Condition D)
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_core.constants import D_MODEL, N_CODEBOOKS, RVQ_VOCAB_SIZE
from tmrvc_train.models.codec_variants import (
    DelayPatternScheduler,
    NARRefinementHead,
    SingleCodebookProjector,
)

# ---------------------------------------------------------------------------
# Shared constants for fast tests
# ---------------------------------------------------------------------------

_D = 64  # small d_model
_B = 2
_T = 16
_K = N_CODEBOOKS  # 8


# ---------------------------------------------------------------------------
# Condition B: NARRefinementHead
# ---------------------------------------------------------------------------

class TestNARRefinementHead:
    def test_output_shape(self):
        head = NARRefinementHead(d_model=_D, n_nar_codebooks=_K - 1, codebook_size=RVQ_VOCAB_SIZE)
        hidden = torch.randn(_B, _T, _D)
        logits = head(hidden)
        assert logits.shape == (_B, _T, _K - 1, RVQ_VOCAB_SIZE)

    def test_default_constants(self):
        """Default args should use constants from tmrvc_core."""
        head = NARRefinementHead()
        assert head.d_model == D_MODEL
        assert head.n_nar_codebooks == N_CODEBOOKS - 1
        assert head.codebook_size == RVQ_VOCAB_SIZE

    def test_gradient_flows(self):
        head = NARRefinementHead(d_model=_D, n_nar_codebooks=_K - 1, codebook_size=32)
        hidden = torch.randn(_B, _T, _D, requires_grad=True)
        logits = head(hidden)
        logits.sum().backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape


# ---------------------------------------------------------------------------
# Condition C: DelayPatternScheduler
# ---------------------------------------------------------------------------

class TestDelayPatternScheduler:
    def test_delayed_shape(self):
        sched = DelayPatternScheduler(n_codebooks=_K)
        tokens = torch.randint(0, 1024, (_B, _K, _T))
        delayed = sched.prepare_delayed_input(tokens)
        assert delayed.shape == (_B, _K, _T + _K - 1)

    def test_round_trip(self):
        """delay -> undelay should recover the original tokens."""
        sched = DelayPatternScheduler(n_codebooks=_K, pad_token_id=-1)
        tokens = torch.randint(0, 1024, (_B, _K, _T))
        delayed = sched.prepare_delayed_input(tokens)
        recovered = sched.undelay_output(delayed)
        assert recovered.shape == tokens.shape
        assert torch.equal(recovered, tokens)

    def test_delay_offsets_correct(self):
        """CB_k should start at position k in the delayed output."""
        sched = DelayPatternScheduler(n_codebooks=4, pad_token_id=0)
        tokens = torch.arange(1, 4 * 5 + 1).reshape(1, 4, 5)  # [1, 4, 5]
        delayed = sched.prepare_delayed_input(tokens)

        # CB0 should have no leading padding
        assert delayed[0, 0, 0].item() == tokens[0, 0, 0].item()
        # CB1 should have 1 pad token at start
        assert delayed[0, 1, 0].item() == 0
        assert delayed[0, 1, 1].item() == tokens[0, 1, 0].item()
        # CB3 should have 3 pad tokens at start
        assert delayed[0, 3, 0].item() == 0
        assert delayed[0, 3, 1].item() == 0
        assert delayed[0, 3, 2].item() == 0
        assert delayed[0, 3, 3].item() == tokens[0, 3, 0].item()

    def test_undelay_too_short_raises(self):
        sched = DelayPatternScheduler(n_codebooks=_K)
        short = torch.zeros(_B, _K, _K - 1)  # T_delayed == K-1 => T_orig == 0
        with pytest.raises(ValueError, match="too short"):
            sched.undelay_output(short)


# ---------------------------------------------------------------------------
# Condition D: SingleCodebookProjector
# ---------------------------------------------------------------------------

class TestSingleCodebookProjector:
    def test_embed_shape(self):
        proj = SingleCodebookProjector(vocab_size=8192, d_model=_D)
        tokens = torch.randint(0, 8192, (_B, _T))
        emb = proj.embed(tokens)
        assert emb.shape == (_B, _T, _D)

    def test_project_shape(self):
        proj = SingleCodebookProjector(vocab_size=8192, d_model=_D)
        hidden = torch.randn(_B, _T, _D)
        logits = proj.project(hidden)
        assert logits.shape == (_B, _T, 8192)

    def test_default_constants(self):
        proj = SingleCodebookProjector()
        assert proj.d_model == D_MODEL
        assert proj.vocab_size == 8192

    def test_gradient_flows(self):
        proj = SingleCodebookProjector(vocab_size=256, d_model=_D)
        tokens = torch.randint(0, 256, (_B, _T))
        emb = proj.embed(tokens)
        logits = proj.project(emb)
        logits.sum().backward()
        # Embedding weight should have gradients
        assert proj.embedding.weight.grad is not None
