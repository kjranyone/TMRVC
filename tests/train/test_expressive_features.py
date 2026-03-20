"""Tests for v3 expressive training features.

Covers:
a) DialogueContextProjector shape and passthrough
b) Model forward_tts_pointer with dialogue/acting/prosody inputs
c) Dataset expressive_readiness_report
d) Trainer accepts expressive batch fields
e) Anti-collapse diversity loss
f) Schema new fields (phrase_pressure, breath_tendency)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tmrvc_train.models.uclm_model import (
    DialogueContextProjector,
    DisentangledUCLM,
)
from tmrvc_train.models.uclm_loss import context_diversity_loss, uclm_loss


# ---------------------------------------------------------------------------
# a) DialogueContextProjector
# ---------------------------------------------------------------------------


class TestDialogueContextProjector:
    def test_no_inputs_passthrough(self):
        proj = DialogueContextProjector(d_model=64, d_dialogue=32, d_acting=16, d_prosody=16)
        x = torch.randn(2, 10, 64)
        out = proj(x)
        assert torch.allclose(out, x)

    def test_dialogue_context_adds_bias(self):
        proj = DialogueContextProjector(d_model=64, d_dialogue=32, d_acting=16, d_prosody=16)
        x = torch.randn(2, 10, 64)
        ctx = torch.randn(2, 32)
        out = proj(x, dialogue_context=ctx)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_acting_intent_adds_bias(self):
        proj = DialogueContextProjector(d_model=64, d_dialogue=32, d_acting=16, d_prosody=16)
        x = torch.randn(2, 10, 64)
        act = torch.randn(2, 16)
        out = proj(x, acting_intent=act)
        assert out.shape == x.shape

    def test_prosody_latent_adds_local(self):
        proj = DialogueContextProjector(d_model=64, d_dialogue=32, d_acting=16, d_prosody=16)
        x = torch.randn(2, 10, 64)
        pro = torch.randn(2, 10, 16)
        out = proj(x, prosody_latent=pro)
        assert out.shape == x.shape

    def test_prosody_latent_different_length_interpolated(self):
        proj = DialogueContextProjector(d_model=64, d_dialogue=32, d_acting=16, d_prosody=16)
        x = torch.randn(2, 10, 64)
        pro = torch.randn(2, 5, 16)  # Different T
        out = proj(x, prosody_latent=pro)
        assert out.shape == x.shape

    def test_all_inputs_combined(self):
        proj = DialogueContextProjector(d_model=64, d_dialogue=32, d_acting=16, d_prosody=16)
        x = torch.randn(2, 10, 64)
        out = proj(
            x,
            dialogue_context=torch.randn(2, 32),
            acting_intent=torch.randn(2, 16),
            prosody_latent=torch.randn(2, 10, 16),
        )
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# b) Model forward_tts_pointer with expressive inputs
# ---------------------------------------------------------------------------


class TestModelExpressiveInputs:
    def test_forward_tts_pointer_with_dialogue_context(self):
        model = DisentangledUCLM(d_model=256, n_heads=4, n_layers=1, num_speakers=2)
        B, L, T = 2, 6, 10
        out = model.forward_tts_pointer(
            phoneme_ids=torch.randint(1, 200, (B, L)),
            language_ids=torch.zeros(B, dtype=torch.long),
            pointer_state=None,
            speaker_embed=torch.randn(B, 192),
            explicit_state=torch.randn(B, T, 12),
            ssl_state=torch.randn(B, T, 128),
            target_a=torch.zeros(B, 8, T, dtype=torch.long),
            target_b=torch.zeros(B, 4, T, dtype=torch.long),
            target_length=T,
            dialogue_context=torch.randn(B, 256),
        )
        assert out["logits_a"].shape[0] == B
        assert out["advance_logit"] is not None

    def test_forward_tts_pointer_with_all_expressive_inputs(self):
        model = DisentangledUCLM(d_model=256, n_heads=4, n_layers=1, num_speakers=2)
        B, L, T = 2, 6, 10
        out = model.forward_tts_pointer(
            phoneme_ids=torch.randint(1, 200, (B, L)),
            language_ids=torch.zeros(B, dtype=torch.long),
            pointer_state=None,
            speaker_embed=torch.randn(B, 192),
            explicit_state=torch.randn(B, T, 12),
            ssl_state=torch.randn(B, T, 128),
            target_a=torch.zeros(B, 8, T, dtype=torch.long),
            target_b=torch.zeros(B, 4, T, dtype=torch.long),
            target_length=T,
            dialogue_context=torch.randn(B, 256),
            acting_intent=torch.randn(B, 64),
            prosody_latent=torch.randn(B, T, 128),
        )
        assert out["logits_a"].shape[0] == B

    def test_forward_streaming_with_expressive_inputs(self):
        model = DisentangledUCLM(d_model=256, n_heads=4, n_layers=1, num_speakers=2)
        B = 1
        out = model.forward_streaming(
            queries=torch.randn(B, 1, 256),
            memory=torch.randn(B, 1, 256),
            a_ctx=torch.zeros(B, 8, 1, dtype=torch.long),
            b_ctx=torch.zeros(B, 4, 1, dtype=torch.long),
            speaker_embed=torch.randn(B, 192),
            explicit_state=torch.randn(B, 1, 12),
            ssl_state=torch.randn(B, 1, 128),
            dialogue_context=torch.randn(B, 256),
            acting_intent=torch.randn(B, 64),
        )
        assert "logits_a" in out


# ---------------------------------------------------------------------------
# c) Dataset expressive readiness report
# ---------------------------------------------------------------------------


class TestExpressiveReadinessReport:
    def test_report_with_no_expressive_data(self, tmp_path: Path):
        from tmrvc_train.dataset import DisentangledUCLMDataset

        _build_basic_cache(tmp_path)
        ds = DisentangledUCLMDataset(tmp_path / "cache")
        report = ds.expressive_readiness_report()
        assert report["total"] == 2
        assert report["with_dialogue_context"] == 0
        assert report["with_acting_intent"] == 0
        assert report["multi_take_texts"] == 0

    def test_report_with_expressive_data(self, tmp_path: Path):
        from tmrvc_train.dataset import DisentangledUCLMDataset

        _build_basic_cache(tmp_path, add_expressive=True)
        ds = DisentangledUCLMDataset(tmp_path / "cache")
        report = ds.expressive_readiness_report()
        assert report["total"] == 2
        assert report["with_dialogue_context"] == 2
        assert report["with_acting_intent"] == 2

    def test_multi_take_detection(self, tmp_path: Path):
        from tmrvc_train.dataset import DisentangledUCLMDataset

        _build_basic_cache(tmp_path, same_text=True)
        ds = DisentangledUCLMDataset(tmp_path / "cache")
        report = ds.expressive_readiness_report()
        assert report["multi_take_texts"] >= 1


# ---------------------------------------------------------------------------
# d) Unknown phone ratio
# ---------------------------------------------------------------------------


class TestUnknownPhoneRatio:
    def test_supervision_report_computes_unk_ratio(self, tmp_path: Path):
        from tmrvc_train.dataset import DisentangledUCLMDataset
        from tmrvc_data.g2p import UNK_ID

        _build_basic_cache(tmp_path, add_phonemes=True, unk_ratio=0.5)
        ds = DisentangledUCLMDataset(tmp_path / "cache")
        report = ds.supervision_report()
        assert report["unknown_phone_ratio"] > 0.0


# ---------------------------------------------------------------------------
# e) Anti-collapse diversity loss
# ---------------------------------------------------------------------------


class TestContextDiversityLoss:
    def test_no_groups_returns_zero(self):
        h = torch.randn(4, 10, 64)
        loss = context_diversity_loss(h, context_groups=None)
        assert loss.item() == 0.0

    def test_same_group_identical_hidden_produces_loss(self):
        h = torch.ones(2, 10, 64)  # Identical hidden states
        groups = torch.tensor([0, 0])  # Same group
        loss = context_diversity_loss(h, groups, margin=0.1)
        assert loss.item() > 0.0

    def test_different_groups_no_loss(self):
        h = torch.ones(2, 10, 64)
        groups = torch.tensor([0, 1])  # Different groups
        loss = context_diversity_loss(h, groups, margin=0.1)
        assert loss.item() == 0.0

    def test_diverse_hidden_low_loss(self):
        h = torch.randn(2, 10, 64)
        h[1] = -h[0]  # Very different
        groups = torch.tensor([0, 0])
        loss = context_diversity_loss(h, groups, margin=0.1)
        assert loss.item() < 0.5


# ---------------------------------------------------------------------------
# f) Schema new fields
# ---------------------------------------------------------------------------


class TestSchemaActingFields:
    def test_tts_request_phrase_pressure_default(self):
        from tmrvc_serve.schemas import PacingControlsSchema
        pacing = PacingControlsSchema()
        assert pacing.phrase_pressure == 0.0

    def test_tts_request_breath_tendency_default(self):
        from tmrvc_serve.schemas import PacingControlsSchema
        pacing = PacingControlsSchema()
        assert pacing.breath_tendency == 0.0

    def test_tts_request_phrase_pressure_range(self):
        from tmrvc_serve.schemas import PacingControlsSchema
        pacing = PacingControlsSchema(phrase_pressure=0.5)
        assert pacing.phrase_pressure == 0.5
        with pytest.raises(Exception):
            PacingControlsSchema(phrase_pressure=2.0)

    def test_tts_advanced_request_pacing_fields(self):
        from tmrvc_serve.schemas import TTSRequestAdvanced
        req = TTSRequestAdvanced(
            text="hello",
            pacing={"phrase_pressure": 0.3, "breath_tendency": -0.5},
        )
        assert req.pacing.phrase_pressure == 0.3
        assert req.pacing.breath_tendency == -0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_basic_cache(
    tmp_path: Path,
    *,
    add_expressive: bool = False,
    same_text: bool = False,
    add_phonemes: bool = False,
    unk_ratio: float = 0.0,
) -> None:
    cache = tmp_path / "cache"
    T = 20
    for i in range(2):
        utt_dir = cache / "ds1" / "train" / "spk0" / f"utt{i}"
        utt_dir.mkdir(parents=True, exist_ok=True)
        np.save(utt_dir / "codec_tokens.npy", np.random.randint(0, 1024, (8, T)))
        np.save(utt_dir / "explicit_state.npy", np.random.randn(T, 12).astype(np.float32))
        np.save(utt_dir / "ssl_state.npy", np.random.randn(T, 128).astype(np.float32))
        np.save(utt_dir / "spk_embed.npy", np.random.randn(192).astype(np.float32))
        text = "hello" if same_text else f"text{i}"
        meta = {"speaker_id": "spk0", "language_id": 0, "text": text}
        (utt_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        if add_expressive:
            np.save(utt_dir / "dialogue_context.npy", np.random.randn(256).astype(np.float32))
            np.save(utt_dir / "acting_intent.npy", np.random.randn(64).astype(np.float32))
            np.save(utt_dir / "prosody_targets.npy", np.random.randn(T, 128).astype(np.float32))

        if add_phonemes:
            from tmrvc_data.g2p import UNK_ID
            L = 8
            pids = np.random.randint(1, 200, (L,))
            n_unk = int(L * unk_ratio)
            pids[:n_unk] = UNK_ID
            np.save(utt_dir / "phoneme_ids.npy", pids)
