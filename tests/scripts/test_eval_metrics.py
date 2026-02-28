"""Tests for tmrvc_train.eval_metrics module."""

from __future__ import annotations

import torch

from tmrvc_train.eval_metrics import (
    f0_correlation,
    speaker_embedding_cosine_similarity,
    utmos_proxy,
)


class TestSECS:
    def test_identical_embeddings(self):
        embed = torch.randn(192)
        score = speaker_embedding_cosine_similarity(embed, embed)
        assert abs(score - 1.0) < 1e-5

    def test_opposite_embeddings(self):
        embed = torch.randn(192)
        score = speaker_embedding_cosine_similarity(embed, -embed)
        assert abs(score - (-1.0)) < 1e-5

    def test_batch_mode(self):
        embed_a = torch.randn(4, 192)
        embed_b = torch.randn(4, 192)
        score = speaker_embedding_cosine_similarity(embed_a, embed_b)
        assert -1.0 <= score <= 1.0

    def test_1d_input(self):
        a = torch.randn(192)
        b = torch.randn(192)
        score = speaker_embedding_cosine_similarity(a, b)
        assert -1.0 <= score <= 1.0


class TestF0Correlation:
    def test_identical_f0(self):
        f0 = torch.tensor([200.0, 210.0, 220.0, 230.0, 240.0])
        corr = f0_correlation(f0, f0)
        assert abs(corr - 1.0) < 1e-5

    def test_opposite_f0(self):
        f0_a = torch.tensor([200.0, 210.0, 220.0, 230.0, 240.0])
        f0_b = torch.tensor([240.0, 230.0, 220.0, 210.0, 200.0])
        corr = f0_correlation(f0_a, f0_b)
        assert abs(corr - (-1.0)) < 1e-5

    def test_unvoiced_frames_excluded(self):
        f0_a = torch.tensor([200.0, 0.0, 220.0, 0.0, 240.0])
        f0_b = torch.tensor([200.0, 0.0, 220.0, 0.0, 240.0])
        corr = f0_correlation(f0_a, f0_b)
        assert abs(corr - 1.0) < 1e-5

    def test_all_unvoiced_returns_zero(self):
        f0_a = torch.zeros(10)
        f0_b = torch.zeros(10)
        corr = f0_correlation(f0_a, f0_b)
        assert corr == 0.0

    def test_2d_input(self):
        f0_a = torch.tensor([[200.0, 210.0, 220.0]])
        f0_b = torch.tensor([[200.0, 210.0, 220.0]])
        corr = f0_correlation(f0_a, f0_b)
        assert abs(corr - 1.0) < 1e-5

    def test_custom_voiced_threshold(self):
        f0_a = torch.tensor([5.0, 200.0, 210.0])
        f0_b = torch.tensor([5.0, 200.0, 210.0])
        # With default threshold=10, frame 0 is unvoiced
        corr = f0_correlation(f0_a, f0_b, voiced_threshold=10.0)
        assert abs(corr - 1.0) < 1e-5


class TestUTMOSProxy:
    def test_identical_mels(self):
        mel = torch.randn(80, 100)
        score = utmos_proxy(mel, mel)
        assert abs(score - 5.0) < 1e-4

    def test_different_mels(self):
        mel_a = torch.randn(80, 100)
        mel_b = torch.randn(80, 100)
        score = utmos_proxy(mel_a, mel_b)
        assert 0.0 <= score <= 5.0

    def test_3d_batch_input(self):
        mel_a = torch.randn(2, 80, 100)
        mel_b = torch.randn(2, 80, 100)
        score = utmos_proxy(mel_a, mel_b)
        assert 0.0 <= score <= 5.0

    def test_mismatched_lengths_aligned(self):
        mel_a = torch.randn(80, 100)
        mel_b = torch.randn(80, 120)
        score = utmos_proxy(mel_a, mel_b)
        assert 0.0 <= score <= 5.0
