"""Tests for EmotionClassifier model."""

from __future__ import annotations

import torch
import pytest

from tmrvc_core.constants import N_EMOTION_CATEGORIES, N_MELS
from tmrvc_train.models.emotion_classifier import EmotionClassifier


@pytest.fixture
def classifier():
    model = EmotionClassifier()
    model.eval()
    return model


class TestForward:
    def test_output_keys(self, classifier):
        mel = torch.randn(2, N_MELS, 100)
        out = classifier(mel)
        assert "emotion_logits" in out
        assert "vad" in out

    def test_output_shapes(self, classifier):
        B, T = 4, 80
        mel = torch.randn(B, N_MELS, T)
        out = classifier(mel)
        assert out["emotion_logits"].shape == (B, N_EMOTION_CATEGORIES)
        assert out["vad"].shape == (B, 3)

    def test_single_sample(self, classifier):
        mel = torch.randn(1, N_MELS, 50)
        out = classifier(mel)
        assert out["emotion_logits"].shape == (1, N_EMOTION_CATEGORIES)

    def test_variable_length(self, classifier):
        for T in [30, 100, 500]:
            mel = torch.randn(2, N_MELS, T)
            out = classifier(mel)
            assert out["emotion_logits"].shape == (2, N_EMOTION_CATEGORIES)


class TestPredict:
    def test_predict_keys(self, classifier):
        mel = torch.randn(3, N_MELS, 100)
        pred = classifier.predict(mel)
        assert "emotion_probs" in pred
        assert "emotion_ids" in pred
        assert "confidence" in pred
        assert "vad" in pred

    def test_predict_shapes(self, classifier):
        B = 5
        mel = torch.randn(B, N_MELS, 100)
        pred = classifier.predict(mel)
        assert pred["emotion_probs"].shape == (B, N_EMOTION_CATEGORIES)
        assert pred["emotion_ids"].shape == (B,)
        assert pred["confidence"].shape == (B,)
        assert pred["vad"].shape == (B, 3)

    def test_probs_sum_to_one(self, classifier):
        mel = torch.randn(4, N_MELS, 100)
        pred = classifier.predict(mel)
        sums = pred["emotion_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_confidence_range(self, classifier):
        mel = torch.randn(4, N_MELS, 100)
        pred = classifier.predict(mel)
        assert (pred["confidence"] >= 0.0).all()
        assert (pred["confidence"] <= 1.0).all()

    def test_emotion_ids_range(self, classifier):
        mel = torch.randn(4, N_MELS, 100)
        pred = classifier.predict(mel)
        assert (pred["emotion_ids"] >= 0).all()
        assert (pred["emotion_ids"] < N_EMOTION_CATEGORIES).all()

    def test_no_grad_in_predict(self, classifier):
        mel = torch.randn(2, N_MELS, 100)
        pred = classifier.predict(mel)
        assert not pred["emotion_probs"].requires_grad
        assert not pred["confidence"].requires_grad


class TestTraining:
    def test_backward(self):
        model = EmotionClassifier()
        model.train()
        mel = torch.randn(4, N_MELS, 100)
        target_cls = torch.randint(0, N_EMOTION_CATEGORIES, (4,))
        target_vad = torch.randn(4, 3)
        out = model(mel)
        cls_loss = torch.nn.functional.cross_entropy(out["emotion_logits"], target_cls)
        vad_loss = torch.nn.functional.mse_loss(out["vad"], target_vad)
        loss = cls_loss + vad_loss
        loss.backward()
        # Check gradients exist on all trainable params
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_custom_dims(self):
        model = EmotionClassifier(n_mels=40, hidden=64, n_classes=5)
        mel = torch.randn(2, 40, 50)
        out = model(mel)
        assert out["emotion_logits"].shape == (2, 5)
