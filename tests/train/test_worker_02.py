"""Tests for Worker 02: Training path, MAS, and Curriculum."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tmrvc_train.models.uclm_loss import monotonic_alignment_search
from tmrvc_train.trainer import UCLMTrainer, CurriculumScheduler
from tmrvc_train.models.uclm_model import DisentangledUCLM


def test_monotonic_alignment_search_smoke():
    B, L, T = 2, 10, 30
    # Create simple log_probs: diagonal-ish path should be preferred
    log_probs = torch.zeros(B, L, T)
    for b in range(B):
        for t in range(T):
            # Target phoneme index for frame t
            target_l = int(t * L / T)
            log_probs[b, target_l, t] = 1.0
            
    path = monotonic_alignment_search(log_probs)
    
    assert path.shape == (B, L, T)
    assert path.sum() == B * T
    # Check monotonicity: for each b, path[b, :, t] should have exactly one 1.0
    assert torch.all(path.sum(dim=1) == 1.0)
    # Phoneme index should be non-decreasing over time
    for b in range(B):
        phoneme_indices = path[b].argmax(dim=0)
        assert torch.all(phoneme_indices[1:] >= phoneme_indices[:-1])


def test_curriculum_scheduler_stages():
    sched = CurriculumScheduler(stage2_start=100, stage3_start=200)
    assert sched.get_stage(0) == 1
    assert sched.get_stage(100) == 2
    assert sched.get_stage(200) == 3
    
    # Check config overrides
    c1 = sched.get_config(0)
    assert c1["tts_prob"] == 0.2
    
    c3 = sched.get_config(250)
    assert "conditioning_dropout_prob" in c3


class TestUCLMTrainerV3:
    def _make_model(self):
        return DisentangledUCLM(
            d_model=64, n_heads=4, n_layers=1, num_speakers=2
        )

    def test_train_step_mas_mode(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = UCLMTrainer(
            model, optimizer, device="cpu", 
            pointer_target_source="mas", tts_prob=1.0
        )
        
        B, L, T = 2, 5, 10
        batch = {
            "target_a": torch.randint(0, 1024, (B, 8, T)),
            "target_b": torch.randint(0, 64, (B, 4, T)),
            "source_a_t": torch.randint(0, 1024, (B, 8, T)),
            "explicit_state": torch.randn(B, T, 12),
            "ssl_state": torch.randn(B, T, 128),
            "speaker_embed": torch.randn(B, 192),
            "speaker_id": torch.zeros(B, dtype=torch.long),
            "phoneme_ids": torch.randint(1, 200, (B, L)),
            "phoneme_lens": torch.full((B,), L, dtype=torch.long),
            "language_id": torch.zeros(B, dtype=torch.long),
        }
        
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["mode"] == 1 # TTS
        
    def test_cfg_dropout_logic(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters())
        # Force 100% dropout
        trainer = UCLMTrainer(
            model, optimizer, device="cpu", 
            conditioning_dropout_prob=1.0, tts_prob=1.0
        )
        
        B, L, T = 1, 5, 10
        batch = {
            "target_a": torch.randint(0, 1024, (B, 8, T)),
            "target_b": torch.randint(0, 64, (B, 4, T)),
            "source_a_t": torch.randint(0, 1024, (B, 8, T)),
            "explicit_state": torch.ones(B, T, 12), # all ones
            "ssl_state": torch.ones(B, T, 128),
            "speaker_embed": torch.ones(B, 192),
            "speaker_id": torch.zeros(B, dtype=torch.long),
            "phoneme_ids": torch.randint(1, 200, (B, L)),
            "phoneme_lens": torch.full((B,), L, dtype=torch.long),
            "language_id": torch.zeros(B, dtype=torch.long),
            "dialogue_context": torch.ones(B, 256),
        }
        
        # We need to spy on the model forward call to see if inputs are zeroed
        # but for now we just check it doesn't crash
        metrics = trainer.train_step(batch)
        assert "loss" in metrics

    def test_flow_matching_loss_integration(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = UCLMTrainer(
            model, optimizer, device="cpu", tts_prob=1.0
        )
        
        B, L, T = 2, 5, 10
        batch = {
            "target_a": torch.randint(0, 1024, (B, 8, T)),
            "target_b": torch.randint(0, 64, (B, 4, T)),
            "source_a_t": torch.randint(0, 1024, (B, 8, T)),
            "explicit_state": torch.randn(B, T, 12),
            "ssl_state": torch.randn(B, T, 128),
            "speaker_embed": torch.randn(B, 192),
            "speaker_id": torch.zeros(B, dtype=torch.long),
            "phoneme_ids": torch.randint(1, 200, (B, L)),
            "phoneme_lens": torch.full((B,), L, dtype=torch.long),
            "language_id": torch.zeros(B, dtype=torch.long),
            "prosody_targets": torch.randn(B, 128), # matches d_prosody from constants.yaml
        }
        
        metrics = trainer.train_step(batch)
        assert "loss_prosody" in metrics
        assert metrics["loss_prosody"] >= 0.0
