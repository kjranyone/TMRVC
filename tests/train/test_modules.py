"""Tests for CausalConvNeXtBlock, SemiCausalConvNeXtBlock, FiLMConditioner, etc."""

import torch
import pytest

from tmrvc_train.modules import (
    CausalConvNeXtBlock,
    FiLMConditioner,
    GlobalTimbreMemory,
    SemiCausalConvNeXtBlock,
    SinusoidalTimestepEmbedding,
    TimbreCrossAttention,
)


class TestCausalConvNeXtBlock:
    """Tests for CausalConvNeXtBlock."""

    def test_training_mode_shape(self):
        block = CausalConvNeXtBlock(channels=256, kernel_size=7, dilation=1)
        x = torch.randn(2, 256, 50)
        out, state = block(x)
        assert out.shape == (2, 256, 50)
        assert state is None

    def test_streaming_mode_shape(self):
        block = CausalConvNeXtBlock(channels=256, kernel_size=7, dilation=1)
        context_size = block.context_size  # (7-1)*1 = 6
        assert context_size == 6

        state_in = torch.zeros(1, 256, context_size)
        x = torch.randn(1, 256, 1)
        out, state_out = block(x, state_in)
        assert out.shape == (1, 256, 1)
        assert state_out.shape == (1, 256, context_size)

    def test_dilation_context_size(self):
        block = CausalConvNeXtBlock(channels=128, kernel_size=3, dilation=4)
        assert block.context_size == (3 - 1) * 4  # 8

    def test_causality(self):
        """Ensure output at time t only depends on input at time <= t."""
        block = CausalConvNeXtBlock(channels=64, kernel_size=7, dilation=1)
        block.eval()

        x1 = torch.randn(1, 64, 20)
        x2 = x1.clone()
        # Modify future frames (indices 15+)
        x2[:, :, 15:] = torch.randn(1, 64, 5)

        out1, _ = block(x1)
        out2, _ = block(x2)

        # Outputs at frame 10 should be identical (no look-ahead)
        torch.testing.assert_close(out1[:, :, :10], out2[:, :, :10])

    def test_streaming_matches_training(self):
        """Streaming frame-by-frame should match training full-sequence."""
        block = CausalConvNeXtBlock(channels=64, kernel_size=7, dilation=1)
        block.eval()

        T = 10
        x = torch.randn(1, 64, T)

        # Training mode
        with torch.no_grad():
            out_full, _ = block(x)

        # Streaming mode
        ctx = block.context_size
        state = torch.zeros(1, 64, ctx)
        out_frames = []
        with torch.no_grad():
            for t in range(T):
                frame = x[:, :, t:t+1]
                out_frame, state = block(frame, state)
                out_frames.append(out_frame)

        out_streaming = torch.cat(out_frames, dim=-1)
        torch.testing.assert_close(out_full, out_streaming, atol=1e-5, rtol=1e-4)

    def test_residual_connection(self):
        """Verify residual: output ≈ input when weights are zero-ish."""
        block = CausalConvNeXtBlock(channels=32, kernel_size=3, dilation=1)
        # Zero out conv weights for near-identity
        with torch.no_grad():
            block.pwconv2.weight.zero_()
            block.pwconv2.bias.zero_()
        x = torch.randn(1, 32, 10)
        out, _ = block(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-4)


class TestFiLMConditioner:
    """Tests for FiLMConditioner."""

    def test_shape(self):
        film = FiLMConditioner(d_cond=216, d_model=384)
        x = torch.randn(2, 384, 10)
        cond = torch.randn(2, 216)
        out = film(x, cond)
        assert out.shape == (2, 384, 10)

    def test_identity_when_gamma1_beta0(self):
        """FiLM with gamma=1, beta=0 should be identity."""
        film = FiLMConditioner(d_cond=4, d_model=8)
        # Set projection so gamma=1, beta=0
        with torch.no_grad():
            film.proj.weight.zero_()
            film.proj.bias.zero_()
            # gamma part (first 8) = 1, beta part (next 8) = 0
            film.proj.bias[:8] = 1.0
        x = torch.randn(1, 8, 5)
        cond = torch.zeros(1, 4)
        out = film(x, cond)
        torch.testing.assert_close(out, x)


class TestSinusoidalTimestepEmbedding:
    """Tests for SinusoidalTimestepEmbedding."""

    def test_shape(self):
        emb = SinusoidalTimestepEmbedding(d_model=512)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 512)

    def test_different_timesteps_produce_different_embeddings(self):
        emb = SinusoidalTimestepEmbedding(d_model=128)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        # All three should be different
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])

    def test_2d_input(self):
        emb = SinusoidalTimestepEmbedding(d_model=64)
        t = torch.tensor([[0.3], [0.7]])
        out = emb(t)
        assert out.shape == (2, 64)

    def test_odd_dimension(self):
        emb = SinusoidalTimestepEmbedding(d_model=65)
        t = torch.tensor([0.5])
        out = emb(t)
        assert out.shape == (1, 65)


class TestGlobalTimbreMemory:
    """Tests for GlobalTimbreMemory."""

    def test_global_timbre_memory_shape(self):
        """[B, 192] → [B, 8, 48]."""
        gtm = GlobalTimbreMemory(d_speaker=192, n_entries=8, d_entry=48)
        spk = torch.randn(2, 192)
        mem = gtm(spk)
        assert mem.shape == (2, 8, 48)


class TestTimbreCrossAttention:
    """Tests for TimbreCrossAttention."""

    def test_timbre_cross_attention_shape(self):
        """[B, 384, T] + [B, 8, 48] → [B, 384, T]."""
        attn = TimbreCrossAttention(d_model=384, d_entry=48, n_heads=4)
        content = torch.randn(2, 384, 10)
        memory = torch.randn(2, 8, 48)
        out = attn(content, memory)
        assert out.shape == (2, 384, 10)


class TestSemiCausalConvNeXtBlock:
    """Tests for SemiCausalConvNeXtBlock."""

    def test_semi_causal_block_training_shape(self):
        """Training: T=20 input → T=20 output (padding preserves length)."""
        block = SemiCausalConvNeXtBlock(channels=64, kernel_size=3, dilation=2, right_context=2)
        x = torch.randn(2, 64, 20)
        out, state = block(x)
        assert out.shape == (2, 64, 20)
        assert state is None

    def test_semi_causal_block_streaming_shape(self):
        """Streaming: T=7 + state → T=6 output (right_ctx=1 trims 1 frame)."""
        block = SemiCausalConvNeXtBlock(channels=64, kernel_size=3, dilation=1, right_context=1)
        # left_context = (3-1)*1 - 1 = 1
        assert block.left_context == 1
        state_in = torch.zeros(1, 64, block.left_context)
        x = torch.randn(1, 64, 7)
        out, state_out = block(x, state_in)
        assert out.shape == (1, 64, 6)
        assert state_out.shape == (1, 64, block.left_context)

    def test_semi_causal_block_state_advance_by_one(self):
        """State should advance by exactly 1 frame per call."""
        block = SemiCausalConvNeXtBlock(channels=32, kernel_size=3, dilation=1, right_context=1)
        block.eval()
        # left_context=1, right_context=1
        state = torch.zeros(1, 32, 1)

        x1 = torch.randn(1, 32, 3)
        _, state1 = block(x1, state)

        # State should contain [state[-0:] | x1[:,:,:1]][-left_ctx:]
        # = concat(zeros[1,32,1], x1[:,:,:1])[:,:,-1:] = x1[:,:,:1]
        torch.testing.assert_close(state1, x1[:, :, :1])

    def test_semi_causal_right_zero_equals_causal(self):
        """right_context=0 should produce identical output to CausalConvNeXtBlock."""
        torch.manual_seed(42)
        causal = CausalConvNeXtBlock(channels=64, kernel_size=3, dilation=2)
        semi = SemiCausalConvNeXtBlock(channels=64, kernel_size=3, dilation=2, right_context=0)
        # Copy weights
        semi.load_state_dict(causal.state_dict())
        causal.eval()
        semi.eval()

        x = torch.randn(1, 64, 15)
        with torch.no_grad():
            out_causal, _ = causal(x)
            out_semi, _ = semi(x)
        torch.testing.assert_close(out_causal, out_semi)
