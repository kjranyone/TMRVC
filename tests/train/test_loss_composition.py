"""Tests for full v4 loss composition (Phase 3, Task 3-3).

Covers:
- All 9 loss terms produce non-zero gradients
- V4LossResult and V4LossConfig completeness
- compute_v4_total_loss produces correct weighted sum
- Tier weighting integration with loss composition
- Each loss term can be individually disabled via lambda=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class TestNineLossTermsNonZeroGradients:
    """Each of the 9 master-plan loss terms must produce non-zero gradients."""

    def test_loss1_codec_token_prediction(self):
        """Loss 1: codec token prediction produces gradients."""
        logits = torch.randn(2, 1024, 50, requires_grad=True)  # [B, vocab, T]
        targets = torch.randint(0, 1024, (2, 50))
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_loss2_control_token_prediction(self):
        """Loss 2: control token prediction produces gradients."""
        logits = torch.randn(2, 64, 50, requires_grad=True)  # [B, ctrl_vocab, T]
        targets = torch.randint(0, 64, (2, 50))
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_loss3_pointer_progression(self):
        """Loss 3: pointer advance loss produces gradients."""
        from tmrvc_train.models.uclm_loss import pointer_advance_loss

        logits = torch.randn(2, 50, 1, requires_grad=True)
        targets = torch.randint(0, 2, (2, 50)).float()
        loss = pointer_advance_loss(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_loss4_physical_supervision_12d(self):
        """Loss 4: explicit 12-D physical supervision produces gradients."""
        pred = torch.randn(2, 50, 12, requires_grad=True)
        target = torch.randn(2, 50, 12)
        mask = torch.ones(2, 50, 12, dtype=torch.bool)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = (loss * mask.float()).sum() / mask.float().sum()
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_loss5_acting_latent_kl(self):
        """Loss 5: acting latent KL regularization produces gradients."""
        from tmrvc_train.models.acting_losses import acting_latent_kl_loss

        mu = torch.randn(4, 24, requires_grad=True)
        logvar = torch.randn(4, 24, requires_grad=True)

        loss = acting_latent_kl_loss(mu, logvar, free_nats=0.0)  # free_nats=0 to ensure non-zero
        loss.backward()

        assert mu.grad is not None
        assert mu.grad.abs().sum() > 0
        assert logvar.grad is not None
        assert logvar.grad.abs().sum() > 0

    def test_loss6_disentanglement(self):
        """Loss 6: disentanglement loss produces gradients."""
        from tmrvc_train.models.acting_losses import disentanglement_loss

        physical = torch.randn(8, 50, 12, requires_grad=True)
        latent = torch.randn(8, 24, requires_grad=True)

        loss = disentanglement_loss(physical, latent)
        loss.backward()

        assert physical.grad is not None
        assert physical.grad.abs().sum() > 0

    def test_loss7_speaker_consistency(self):
        """Loss 7: speaker consistency loss produces gradients."""
        spk_embed = torch.randn(4, 192, requires_grad=True)

        # Normalize for cosine similarity
        spk_norm = F.normalize(spk_embed, dim=-1)
        sim = spk_norm @ spk_norm.T

        # Create same-speaker mask (first 2 are same, last 2 are same)
        speaker_ids = torch.tensor([0, 0, 1, 1])
        same_speaker = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        diag_mask = 1.0 - torch.eye(4)
        same_speaker = same_speaker * diag_mask

        positive_sim = (sim * same_speaker).sum() / same_speaker.sum().clamp(min=1)
        loss = 1.0 - positive_sim
        loss.backward()

        assert spk_embed.grad is not None
        assert spk_embed.grad.abs().sum() > 0

    def test_loss8_prosody_prediction(self):
        """Loss 8: prosody prediction loss produces gradients."""
        pred_prosody = torch.randn(2, 50, 128, requires_grad=True)
        target_prosody = torch.randn(2, 50, 128)

        loss = F.mse_loss(pred_prosody, target_prosody)
        loss.backward()

        assert pred_prosody.grad is not None
        assert pred_prosody.grad.abs().sum() > 0

    def test_loss9_semantic_alignment(self):
        """Loss 9: semantic alignment loss produces gradients."""
        from tmrvc_train.models.acting_losses import semantic_alignment_loss

        predicted = torch.randn(4, 24, requires_grad=True)
        target = torch.randn(4, 24)

        loss = semantic_alignment_loss(predicted, target)
        loss.backward()

        assert predicted.grad is not None
        assert predicted.grad.abs().sum() > 0


class TestV4LossConfigCompleteness:
    """V4LossConfig must have weights for all 9 loss terms."""

    def test_all_nine_loss_lambdas(self):
        from tmrvc_train.v4_loss import V4LossConfig

        config = V4LossConfig()
        required = [
            "lambda_codec",            # 1
            "lambda_control",          # 2
            "lambda_pointer",          # 3
            "lambda_physical",         # 4
            "lambda_acting_kl",        # 5
            "lambda_disentanglement",  # 6
            "lambda_speaker",          # 7
            "lambda_prosody",          # 8
            "lambda_semantic_align",   # 9
        ]
        for name in required:
            assert hasattr(config, name), f"Missing: {name}"
            val = getattr(config, name)
            assert val > 0, f"{name} should be > 0, got {val}"

    def test_bio_constraint_lambdas(self):
        from tmrvc_train.v4_loss import V4LossConfig

        config = V4LossConfig()
        assert config.lambda_bio_covariance > 0
        assert config.lambda_bio_transition > 0
        assert config.lambda_bio_implausibility > 0


class TestV4LossResultCompleteness:
    """V4LossResult must have fields for all loss components."""

    def test_all_fields_present(self):
        from tmrvc_train.v4_loss import V4LossResult

        result = V4LossResult()
        required = [
            "codec_loss", "control_loss", "pointer_loss",
            "physical_loss", "acting_kl_loss", "disentanglement_loss",
            "speaker_loss", "prosody_loss", "semantic_align_loss",
            "bio_covariance_loss", "bio_transition_loss", "bio_implausibility_loss",
        ]
        for field in required:
            assert hasattr(result, field), f"V4LossResult missing field: {field}"

    def test_to_dict_excludes_none(self):
        from tmrvc_train.v4_loss import V4LossResult

        result = V4LossResult(
            codec_loss=torch.tensor(1.0),
            physical_loss=torch.tensor(0.5),
        )
        d = result.to_dict()
        assert "codec_loss" in d
        assert "physical_loss" in d
        assert "control_loss" not in d  # None values excluded


class TestComputeV4TotalLoss:
    """Test compute_v4_total_loss weighted sum."""

    def test_basic_total(self):
        from tmrvc_train.v4_loss import V4LossConfig, V4LossResult, compute_v4_total_loss

        config = V4LossConfig(lambda_codec=1.0, lambda_physical=2.0)
        result = V4LossResult(
            codec_loss=torch.tensor(1.0),
            physical_loss=torch.tensor(0.5),
        )
        total = compute_v4_total_loss(result, config)
        # 1.0 * 1.0 + 2.0 * 0.5 = 2.0
        assert total.item() == pytest.approx(2.0)

    def test_zero_lambda_disables_term(self):
        from tmrvc_train.v4_loss import V4LossConfig, V4LossResult, compute_v4_total_loss

        config = V4LossConfig(lambda_codec=0.0, lambda_physical=1.0)
        result = V4LossResult(
            codec_loss=torch.tensor(10.0),
            physical_loss=torch.tensor(1.0),
        )
        total = compute_v4_total_loss(result, config)
        # codec disabled: 0.0 * 10.0 + 1.0 * 1.0 = 1.0
        assert total.item() == pytest.approx(1.0)

    def test_tier_weighting_reduces_loss(self):
        from tmrvc_train.v4_loss import V4LossConfig, V4LossResult, compute_v4_total_loss
        from tmrvc_data.v4_dataset import get_tier_loss_weights

        config = V4LossConfig()
        result = V4LossResult(
            codec_loss=torch.tensor(1.0),
            physical_loss=torch.tensor(1.0),
            acting_kl_loss=torch.tensor(1.0),
            speaker_loss=torch.tensor(1.0),
            prosody_loss=torch.tensor(1.0),
        )

        total_a = compute_v4_total_loss(
            V4LossResult(**{k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in vars(result).items()}),
            config,
            get_tier_loss_weights("tier_a"),
        )
        total_d = compute_v4_total_loss(
            V4LossResult(**{k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in vars(result).items()}),
            config,
            get_tier_loss_weights("tier_d"),
        )

        assert total_a.item() > total_d.item(), \
            f"Tier A total ({total_a.item()}) should exceed Tier D ({total_d.item()})"

    def test_no_losses_returns_zero(self):
        from tmrvc_train.v4_loss import V4LossConfig, V4LossResult, compute_v4_total_loss

        config = V4LossConfig()
        result = V4LossResult()  # all None
        total = compute_v4_total_loss(result, config)
        assert total.item() == 0.0

    def test_all_nine_losses_active(self):
        """When all 9 losses are set, total should be positive."""
        from tmrvc_train.v4_loss import V4LossConfig, V4LossResult, compute_v4_total_loss

        config = V4LossConfig()
        result = V4LossResult(
            codec_loss=torch.tensor(1.0),
            control_loss=torch.tensor(1.0),
            pointer_loss=torch.tensor(1.0),
            physical_loss=torch.tensor(1.0),
            acting_kl_loss=torch.tensor(1.0),
            disentanglement_loss=torch.tensor(1.0),
            speaker_loss=torch.tensor(1.0),
            prosody_loss=torch.tensor(1.0),
            semantic_align_loss=torch.tensor(1.0),
        )

        total = compute_v4_total_loss(result, config)
        assert total.item() > 0

        # Count how many individual lambdas contribute
        contributing = 0
        for name in ["lambda_codec", "lambda_control", "lambda_pointer",
                      "lambda_physical", "lambda_acting_kl",
                      "lambda_disentanglement", "lambda_speaker",
                      "lambda_prosody", "lambda_semantic_align"]:
            if getattr(config, name) > 0:
                contributing += 1
        assert contributing == 9, f"Expected 9 contributing losses, got {contributing}"


class TestBioConstraintIntegration:
    """Test biological constraint regularizer integration with loss."""

    def test_bio_losses_nonzero_gradient(self):
        from tmrvc_train.models.biological_constraints import BiologicalConstraintRegularizer

        reg = BiologicalConstraintRegularizer(d_physical=12, covariance_rank=8)
        physical = torch.randn(2, 50, 12, requires_grad=True)

        losses = reg(physical)
        total = losses["bio_total_loss"]
        total.backward()

        assert physical.grad is not None
        assert physical.grad.abs().sum() > 0

    def test_bio_covariance_loss_nonzero(self):
        from tmrvc_train.models.biological_constraints import BiologicalConstraintRegularizer

        reg = BiologicalConstraintRegularizer(d_physical=12)
        physical = torch.randn(4, 100, 12)
        loss = reg.compute_covariance_loss(physical)
        assert loss.item() > 0

    def test_bio_transition_loss_nonzero(self):
        from tmrvc_train.models.biological_constraints import BiologicalConstraintRegularizer

        reg = BiologicalConstraintRegularizer(d_physical=12)
        # Rapid changes should produce higher transition loss
        physical = torch.randn(2, 50, 12)
        loss = reg.compute_transition_loss(physical)
        assert loss.item() > 0

    def test_bio_implausibility_loss_nonzero(self):
        from tmrvc_train.models.biological_constraints import BiologicalConstraintRegularizer

        reg = BiologicalConstraintRegularizer(d_physical=12)
        physical = torch.randn(2, 50, 12)
        loss = reg.compute_implausibility_loss(physical)
        assert loss.item() > 0

    def test_transition_penalty_higher_for_jumpy_signals(self):
        """Signals with large frame-to-frame jumps should get higher penalty."""
        from tmrvc_train.models.biological_constraints import BiologicalConstraintRegularizer

        reg = BiologicalConstraintRegularizer(d_physical=12)

        # Smooth signal
        smooth = torch.linspace(0, 1, 50).unsqueeze(0).unsqueeze(-1).expand(2, 50, 12)
        loss_smooth = reg.compute_transition_loss(smooth)

        # Jumpy signal (alternating high/low)
        jumpy = torch.zeros(2, 50, 12)
        jumpy[:, ::2, :] = 1.0
        jumpy[:, 1::2, :] = -1.0
        loss_jumpy = reg.compute_transition_loss(jumpy)

        assert loss_jumpy.item() > loss_smooth.item(), \
            f"Jumpy ({loss_jumpy.item()}) should have higher penalty than smooth ({loss_smooth.item()})"


class TestActingLatentLossIntegration:
    """Test acting latent loss components in the full pipeline."""

    def test_kl_and_usage_combined(self):
        from tmrvc_train.models.acting_latent import ActingLatentEncoder
        from tmrvc_train.models.acting_losses import acting_latent_kl_loss, acting_latent_usage_loss

        encoder = ActingLatentEncoder(d_input=128, d_latent=24)
        encoder.train()

        ssl = torch.randn(4, 100, 128)
        latent, mu, logvar = encoder(ssl)

        kl = acting_latent_kl_loss(mu, logvar, free_nats=0.0)
        usage = acting_latent_usage_loss(latent)

        # Both should be finite
        assert torch.isfinite(kl)
        assert torch.isfinite(usage)

        # Combined loss should produce gradients on encoder
        total = kl + usage
        total.backward()

        for p in encoder.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0
                break
        else:
            pytest.fail("No gradients found on encoder parameters")

    def test_disentanglement_with_latent_encoder(self):
        from tmrvc_train.models.acting_latent import ActingLatentEncoder
        from tmrvc_train.models.acting_losses import disentanglement_loss

        encoder = ActingLatentEncoder(d_input=128, d_latent=24)
        encoder.train()

        ssl = torch.randn(8, 100, 128)
        latent, _, _ = encoder(ssl)

        physical = torch.randn(8, 100, 12)
        dis_loss = disentanglement_loss(physical, latent)

        assert torch.isfinite(dis_loss)
        assert dis_loss.item() >= 0
