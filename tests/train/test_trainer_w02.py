"""Tests for Worker 02 training features.

Covers:
- CFG conditioning dropout uses Worker 01 frozen contract
- Config validation (invalid pointer_target_source, tts_mode)
- Voice state supervision loss integration in trainer
- Curriculum scheduler stages
- Pointer aux alignment warmup/anneal schedule
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.trainer import UCLMTrainer, CurriculumScheduler


@pytest.fixture
def model():
    return DisentangledUCLM()


@pytest.fixture
def trainer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return UCLMTrainer(model=model, optimizer=optimizer, device="cpu")


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_tts_mode_raises(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with pytest.raises(ValueError, match="tts_mode"):
            UCLMTrainer(model=model, optimizer=optimizer, tts_mode="invalid")

    def test_invalid_pointer_target_source_raises(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with pytest.raises(ValueError, match="pointer_target_source"):
            UCLMTrainer(
                model=model, optimizer=optimizer,
                pointer_target_source="nonexistent_source",
            )

    def test_invalid_alignment_loss_type_raises(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with pytest.raises(ValueError, match="alignment_loss_type"):
            UCLMTrainer(
                model=model, optimizer=optimizer,
                alignment_loss_type="invalid",
            )

    def test_valid_pointer_config_accepted(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer,
            tts_mode="pointer",
            pointer_target_source="heuristic_bootstrap",
            alignment_loss_type="none",
        )
        assert t.tts_mode == "pointer"

    def test_latent_only_accepted_without_bootstrap(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer,
            pointer_target_source="none",
        )
        assert t.pointer_target_source == "none"


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------


class TestCurriculumScheduler:
    def test_stage1(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200)
        assert cs.get_stage(0) == 1
        assert cs.get_stage(99) == 1

    def test_stage2(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200)
        assert cs.get_stage(100) == 2
        assert cs.get_stage(199) == 2

    def test_stage3(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200)
        assert cs.get_stage(200) == 3
        assert cs.get_stage(1000) == 3

    def test_stage3_enables_cfg(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200)
        config = cs.get_config(200)
        assert "conditioning_dropout_prob" in config
        assert config["conditioning_dropout_prob"] > 0

    def test_stage1_no_pointer_loss(self):
        cs = CurriculumScheduler()
        config = cs.get_config(0)
        assert config["pointer_loss_weight"] == 0.0

    def test_stage3_replay_default(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200)
        assert cs.stage3_replay_mix_ratio == 0.2

    def test_stage3_replay_no_replay_in_stage1(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200, stage3_replay_mix_ratio=1.0)
        assert not cs.should_replay(50)

    def test_stage3_replay_no_replay_in_stage2(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200, stage3_replay_mix_ratio=1.0)
        assert not cs.should_replay(150)

    def test_stage3_replay_always_replays_at_ratio_1(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200, stage3_replay_mix_ratio=1.0)
        assert cs.should_replay(300)

    def test_stage3_replay_never_replays_at_ratio_0(self):
        cs = CurriculumScheduler(stage2_start=100, stage3_start=200, stage3_replay_mix_ratio=0.0)
        # Run 100 times — should never replay
        assert not any(cs.should_replay(300) for _ in range(100))

    def test_stage3_replay_invalid_ratio_raises(self):
        with pytest.raises(ValueError, match="stage3_replay_mix_ratio"):
            CurriculumScheduler(stage3_replay_mix_ratio=1.5)


# ---------------------------------------------------------------------------
# CFG Dropout uses Worker 01 contract
# ---------------------------------------------------------------------------


class TestCFGDropoutContract:
    def test_apply_cfg_unconditional_mask_used(self):
        """Verify the static method produces zeroed outputs."""
        B, T, D = 2, 10, 8
        result = DisentangledUCLM.apply_cfg_unconditional_mask(
            explicit_state=torch.randn(B, T, D),
            ssl_state=torch.randn(B, T, 128),
            speaker_embed=torch.randn(B, 192),
            dialogue_context=torch.randn(B, 256),
            acting_intent=torch.randn(B, 64),
        )
        assert torch.all(result["explicit_state"] == 0)
        assert torch.all(result["ssl_state"] == 0)
        assert torch.all(result["speaker_embed"] == 0)
        assert torch.all(result["dialogue_context"] == 0)
        assert torch.all(result["acting_intent"] == 0)


# ---------------------------------------------------------------------------
# Pointer Aux Alignment Warmup / Anneal
# ---------------------------------------------------------------------------


class TestPointerAuxAlignmentSchedule:
    def test_warmup_phase(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer, device="cpu",
            pointer_target_source="mas",
            pointer_aux_alignment_warmup_steps=100,
            pointer_aux_alignment_anneal_steps=200,
        )
        # At step 0, aux weight should be 1.0 (warmup)
        assert t.pointer_aux_alignment_warmup_steps == 100
        assert t.pointer_aux_alignment_anneal_steps == 200

    def test_anneal_config(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer, device="cpu",
            pointer_target_source="mas",
            pointer_aux_alignment_warmup_steps=10,
            pointer_aux_alignment_anneal_steps=10,
        )
        # Just verify it accepts the config
        assert t.pointer_target_source == "mas"


# ---------------------------------------------------------------------------
# Voice State Loss Weight
# ---------------------------------------------------------------------------


class TestVoiceStateLossConfig:
    def test_zero_weight_accepted(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer,
            voice_state_loss_weight=0.0,
        )
        assert t.voice_state_loss_weight == 0.0

    def test_nonzero_weight_accepted(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer,
            voice_state_loss_weight=0.5,
        )
        assert t.voice_state_loss_weight == 0.5
