"""Tests for ResidualVectorQuantizer mathematical correctness."""

import pytest
import torch
from tmrvc_train.models.emotion_codec import ResidualVectorQuantizer


class TestResidualVectorQuantizer:
    """RVQの数学的整合性テスト。"""

    def test_z_q_is_nonzero(self):
        """z_qが非ゼロであることを確認（バグ: in-place view代入が反映されない）。"""
        rvq = ResidualVectorQuantizer(
            n_codebooks=8, codebook_size=1024, codebook_dim=64
        )
        z = torch.randn(2, 100, 512)  # [B, T, D=8*64]
        z_q, indices, _ = rvq(z)

        # z_qは非ゼロでなければならない
        assert z_q.abs().sum() > 0, "z_q is all zeros - view assignment bug!"

    def test_z_q_shape(self):
        """出力形状が入力と一致することを確認。"""
        rvq = ResidualVectorQuantizer(
            n_codebooks=8, codebook_size=1024, codebook_dim=64
        )
        z = torch.randn(2, 100, 512)
        z_q, indices, _ = rvq(z)

        assert z_q.shape == z.shape, f"z_q shape {z_q.shape} != z shape {z.shape}"
        assert indices.shape == (2, 8, 100), (
            f"indices shape {indices.shape} != (B, n_codebooks, T)"
        )

    def test_indices_in_valid_range(self):
        """インデックスがcodebook_sizeの範囲内であることを確認。"""
        rvq = ResidualVectorQuantizer(
            n_codebooks=8, codebook_size=1024, codebook_dim=64
        )
        z = torch.randn(2, 100, 512)
        z_q, indices, _ = rvq(z)

        assert indices.min() >= 0, f"Negative index: {indices.min()}"
        assert indices.max() < 1024, f"Index exceeds codebook_size: {indices.max()}"

    def test_reconstruction_from_indices(self):
        """インデックスからz_qを再構成できることを確認。"""
        rvq = ResidualVectorQuantizer(
            n_codebooks=8, codebook_size=1024, codebook_dim=64
        )
        z = torch.randn(2, 100, 512)
        z_q, indices, _ = rvq(z)

        # インデックスから再構成
        z_q_reconstructed = torch.zeros_like(z_q)
        for i in range(8):
            z_q_reconstructed[:, :, i * 64 : (i + 1) * 64] = rvq.codebooks[i](
                indices[:, i, :]
            )

        # 一致することを確認（バグがあるとz_qはゼロで再構成と不一致）
        assert torch.allclose(z_q, z_q_reconstructed, atol=1e-5), (
            "z_q does not match reconstruction from indices!"
        )

    def test_gradients_flow(self):
        """勾配がz_qを通じて流れることを確認。"""
        rvq = ResidualVectorQuantizer(
            n_codebooks=8, codebook_size=1024, codebook_dim=64
        )
        z = torch.randn(2, 100, 512, requires_grad=True)
        z_q, indices, _ = rvq(z)

        loss = z_q.sum()
        loss.backward()

        # zに勾配が流れないのは正しい（quantizationは離散化）
        # しかしcodebookには勾配が流れるべき
        for i, cb in enumerate(rvq.codebooks):
            assert cb.weight.grad is not None, f"Codebook {i} has no gradient!"
