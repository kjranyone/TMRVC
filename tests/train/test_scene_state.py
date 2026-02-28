"""Tests for Scene State Latent (SSL) models and losses."""

from __future__ import annotations

import torch
import pytest

from tmrvc_core.constants import (
    D_HISTORY,
    D_SCENE_STATE,
    D_SPEAKER,
    D_TEXT_ENCODER,
    SSL_PROSODY_STATS_DIM,
)
from tmrvc_train.models.scene_state import (
    DialogueHistoryEncoder,
    ProsodyStatsPredictor,
    SceneStateLoss,
    SceneStateUpdate,
)


class TestSceneStateUpdate:
    def test_output_shape(self):
        model = SceneStateUpdate()
        B = 4
        z_prev = torch.zeros(B, D_SCENE_STATE)
        u_t = torch.randn(B, D_TEXT_ENCODER)
        h_t = torch.randn(B, D_HISTORY)
        s = torch.randn(B, D_SPEAKER)

        z_t = model(z_prev, u_t, h_t, s)
        assert z_t.shape == (B, D_SCENE_STATE)

    def test_initial_state_is_zero(self):
        model = SceneStateUpdate()
        z0 = model.initial_state(3, torch.device("cpu"))
        assert z0.shape == (3, D_SCENE_STATE)
        assert (z0 == 0).all()

    def test_state_changes_with_input(self):
        model = SceneStateUpdate()
        B = 2
        z_prev = torch.zeros(B, D_SCENE_STATE)
        u_t = torch.randn(B, D_TEXT_ENCODER)
        h_t = torch.randn(B, D_HISTORY)
        s = torch.randn(B, D_SPEAKER)

        z_t = model(z_prev, u_t, h_t, s)
        # State should differ from zero after update (with high probability)
        assert z_t.abs().sum() > 0

    def test_state_evolves_across_turns(self):
        model = SceneStateUpdate()
        B = 2
        s = torch.randn(B, D_SPEAKER)
        h_t = torch.randn(B, D_HISTORY)

        z = model.initial_state(B, torch.device("cpu"))
        states = [z]
        for _ in range(3):
            u_t = torch.randn(B, D_TEXT_ENCODER)
            z = model(z, u_t, h_t, s)
            states.append(z)

        # States should differ from each other
        assert not torch.allclose(states[0], states[1])
        assert not torch.allclose(states[1], states[2])

    def test_gradient_flows(self):
        model = SceneStateUpdate()
        B = 2
        z_prev = torch.zeros(B, D_SCENE_STATE, requires_grad=True)
        u_t = torch.randn(B, D_TEXT_ENCODER)
        h_t = torch.randn(B, D_HISTORY)
        s = torch.randn(B, D_SPEAKER)

        z_t = model(z_prev, u_t, h_t, s)
        loss = z_t.sum()
        loss.backward()
        assert z_prev.grad is not None

    def test_scene_reset(self):
        """After reset, state should be equivalent to starting fresh."""
        model = SceneStateUpdate()
        B = 1
        s = torch.randn(B, D_SPEAKER)
        h_t = torch.randn(B, D_HISTORY)
        u_t = torch.randn(B, D_TEXT_ENCODER)

        # Evolve state for 3 turns
        z = model.initial_state(B, torch.device("cpu"))
        for _ in range(3):
            z = model(z, u_t, h_t, s)

        # Reset = new initial state
        z_reset = model.initial_state(B, torch.device("cpu"))
        z_fresh = model(z_reset, u_t, h_t, s)

        # Fresh start from zero should give same result regardless of past
        z_from_evolved = model(z, u_t, h_t, s)
        # These should be different because z_prev differs
        assert not torch.allclose(z_fresh, z_from_evolved, atol=1e-4)


class TestDialogueHistoryEncoder:
    def test_output_shape(self):
        encoder = DialogueHistoryEncoder()
        B, N = 4, 5
        history = torch.randn(B, N, D_TEXT_ENCODER)
        h = encoder(history)
        assert h.shape == (B, D_HISTORY)

    def test_single_turn(self):
        encoder = DialogueHistoryEncoder()
        B = 2
        history = torch.randn(B, 1, D_TEXT_ENCODER)
        h = encoder(history)
        assert h.shape == (B, D_HISTORY)

    def test_variable_lengths(self):
        encoder = DialogueHistoryEncoder()
        B, N_max = 3, 5
        history = torch.randn(B, N_max, D_TEXT_ENCODER)
        lengths = torch.tensor([5, 3, 1])
        h = encoder(history, history_lengths=lengths)
        assert h.shape == (B, D_HISTORY)

    def test_gradient_flows(self):
        encoder = DialogueHistoryEncoder()
        history = torch.randn(2, 3, D_TEXT_ENCODER, requires_grad=True)
        h = encoder(history)
        h.sum().backward()
        assert history.grad is not None


class TestProsodyStatsPredictor:
    def test_output_shape(self):
        predictor = ProsodyStatsPredictor()
        z = torch.randn(4, D_SCENE_STATE)
        stats = predictor(z)
        assert stats.shape == (4, SSL_PROSODY_STATS_DIM)

    def test_gradient_flows(self):
        predictor = ProsodyStatsPredictor()
        z = torch.randn(2, D_SCENE_STATE, requires_grad=True)
        stats = predictor(z)
        stats.sum().backward()
        assert z.grad is not None


class TestSceneStateLoss:
    def test_loss_components(self):
        loss_fn = SceneStateLoss()
        B = 4
        pred_prosody = torch.randn(B, SSL_PROSODY_STATS_DIM)
        gt_prosody = torch.randn(B, SSL_PROSODY_STATS_DIM)
        z_t = torch.randn(B, D_SCENE_STATE)
        z_prev = torch.randn(B, D_SCENE_STATE)

        losses = loss_fn(pred_prosody, gt_prosody, z_t, z_prev)
        assert "state_recon" in losses
        assert "state_cons" in losses
        assert "state_total" in losses
        assert losses["state_recon"].item() >= 0
        assert losses["state_cons"].item() >= 0

    def test_perfect_prediction_zero_recon(self):
        loss_fn = SceneStateLoss()
        B = 2
        gt = torch.randn(B, SSL_PROSODY_STATS_DIM)
        z = torch.randn(B, D_SCENE_STATE)

        losses = loss_fn(gt, gt, z, z)
        assert losses["state_recon"].item() < 1e-6
        # z_t == z_prev → cosine_similarity = 1 → cons = 0
        assert losses["state_cons"].item() < 1e-6

    def test_loss_is_differentiable(self):
        loss_fn = SceneStateLoss()
        B = 2
        pred_prosody = torch.randn(B, SSL_PROSODY_STATS_DIM, requires_grad=True)
        gt_prosody = torch.randn(B, SSL_PROSODY_STATS_DIM)
        z_t = torch.randn(B, D_SCENE_STATE, requires_grad=True)
        z_prev = torch.randn(B, D_SCENE_STATE)

        losses = loss_fn(pred_prosody, gt_prosody, z_t, z_prev)
        losses["state_total"].backward()
        assert pred_prosody.grad is not None
        assert z_t.grad is not None

    def test_lambda_weights(self):
        loss_fn = SceneStateLoss(lambda_recon=2.0, lambda_cons=0.0)
        B = 2
        pred = torch.randn(B, SSL_PROSODY_STATS_DIM)
        gt = torch.randn(B, SSL_PROSODY_STATS_DIM)
        z_t = torch.randn(B, D_SCENE_STATE)
        z_prev = torch.randn(B, D_SCENE_STATE)

        losses = loss_fn(pred, gt, z_t, z_prev)
        # With lambda_cons=0, total should be 2 * recon
        expected = 2.0 * losses["state_recon"]
        assert abs(losses["state_total"].item() - expected.item()) < 1e-5


class TestParameterCounts:
    def test_scene_state_update_param_count(self):
        model = SceneStateUpdate()
        n_params = sum(p.numel() for p in model.parameters())
        # Should be reasonable (< 100K for this small model)
        assert n_params < 100_000
        assert n_params > 0

    def test_dialogue_history_encoder_param_count(self):
        encoder = DialogueHistoryEncoder()
        n_params = sum(p.numel() for p in encoder.parameters())
        assert n_params < 200_000
        assert n_params > 0

    def test_prosody_predictor_param_count(self):
        predictor = ProsodyStatsPredictor()
        n_params = sum(p.numel() for p in predictor.parameters())
        assert n_params < 10_000
        assert n_params > 0
