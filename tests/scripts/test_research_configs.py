"""Tests for research ablation configs (B0-B3) and CLI integration."""

from __future__ import annotations

from pathlib import Path

import yaml
import torch

from tmrvc_core.types import TTSBatch
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor
from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.tts_trainer import TTSTrainer, TTSTrainerConfig

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "research"

REQUIRED_KEYS = {"variant", "description", "model", "sampling", "test_split", "metrics", "output"}


def _load_yaml(name: str) -> dict:
    with open(CONFIGS_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_batch(B=2, L=10, T=50, with_events=False) -> TTSBatch:
    kw = {}
    if with_events:
        kw["breath_onsets"] = torch.zeros(B, T)
        kw["breath_onsets"][0, 5] = 1.0
        kw["breath_durations"] = torch.zeros(B, T)
        kw["breath_durations"][0, 5] = 300.0
        kw["breath_intensity"] = torch.zeros(B, T)
        kw["breath_intensity"][0, 5] = 0.8
        kw["pause_durations"] = torch.zeros(B, T)
    return TTSBatch(
        phoneme_ids=torch.randint(0, 100, (B, L)),
        durations=torch.full((B, L), T // L),
        language_ids=torch.zeros(B, dtype=torch.long),
        content=torch.randn(B, 256, T),
        f0=torch.abs(torch.randn(B, 1, T)) * 200 + 100,
        spk_embed=torch.randn(B, 192),
        mel_target=torch.randn(B, 80, T),
        frame_lengths=torch.full((B,), T, dtype=torch.long),
        phoneme_lengths=torch.full((B,), L, dtype=torch.long),
        content_dim=256,
        **kw,
    )


class TestConfigStructure:
    def test_b0_has_required_keys(self):
        cfg = _load_yaml("b0.yaml")
        assert REQUIRED_KEYS.issubset(cfg.keys())
        assert cfg["variant"] == "b0"

    def test_b1_has_required_keys(self):
        cfg = _load_yaml("b1.yaml")
        assert REQUIRED_KEYS.issubset(cfg.keys())
        assert cfg["variant"] == "b1"

    def test_b2_has_required_keys(self):
        cfg = _load_yaml("b2.yaml")
        assert REQUIRED_KEYS.issubset(cfg.keys())
        assert cfg["variant"] == "b2"

    def test_b3_has_required_keys(self):
        cfg = _load_yaml("b3.yaml")
        assert REQUIRED_KEYS.issubset(cfg.keys())
        assert cfg["variant"] == "b3"

    def test_all_share_test_split(self):
        b0 = _load_yaml("b0.yaml")
        for name in ["b1.yaml", "b2.yaml", "b3.yaml"]:
            cfg = _load_yaml(name)
            assert cfg["test_split"]["speakers"] == b0["test_split"]["speakers"], (
                f"{name} test_split.speakers differs from b0"
            )


class TestVariantFeatureFlags:
    def test_b0_no_extensions(self):
        cfg = _load_yaml("b0.yaml")
        model = cfg["model"]
        assert "ssl_enabled" not in model or not model.get("ssl_enabled")
        assert "bpeh_enabled" not in model or not model.get("bpeh_enabled")

    def test_b1_ssl_only(self):
        cfg = _load_yaml("b1.yaml")
        assert cfg["model"]["ssl_enabled"] is True
        assert cfg["model"]["bpeh_enabled"] is False
        assert "ssl" in cfg

    def test_b2_bpeh_only(self):
        cfg = _load_yaml("b2.yaml")
        assert cfg["model"]["ssl_enabled"] is False
        assert cfg["model"]["bpeh_enabled"] is True
        assert "bpeh" in cfg

    def test_b3_ssl_and_bpeh(self):
        cfg = _load_yaml("b3.yaml")
        assert cfg["model"]["ssl_enabled"] is True
        assert cfg["model"]["bpeh_enabled"] is True
        assert "ssl" in cfg
        assert "bpeh" in cfg


class TestVariantTraining:
    """Verify each variant can construct and run a training step."""

    def _build_trainer(self, tmp_path, enable_ssl=False, enable_bpeh=False) -> TTSTrainer:
        config = TTSTrainerConfig(
            max_steps=2,
            log_every=1,
            save_every=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            enable_ssl=enable_ssl,
            enable_bpeh=enable_bpeh,
        )
        text_encoder = TextEncoder()
        trainer = TTSTrainer(
            text_encoder=text_encoder,
            duration_predictor=DurationPredictor(),
            f0_predictor=F0Predictor(),
            content_synthesizer=ContentSynthesizer(),
            optimizer=torch.optim.Adam(text_encoder.parameters(), lr=1e-3),
            dataloader=[_make_batch(with_events=enable_bpeh)],
            config=config,
        )
        all_params = list(trainer._trainable_params())
        trainer.optimizer = torch.optim.Adam(all_params, lr=1e-3)
        return trainer

    def test_b0_trains(self, tmp_path):
        trainer = self._build_trainer(tmp_path)
        losses = trainer.train_step(_make_batch())
        assert "total" in losses
        assert "state_total" not in losses
        assert "event_total" not in losses

    def test_b1_trains(self, tmp_path):
        trainer = self._build_trainer(tmp_path, enable_ssl=True)
        losses = trainer.train_step(_make_batch())
        assert "state_total" in losses
        assert "event_total" not in losses

    def test_b2_trains(self, tmp_path):
        trainer = self._build_trainer(tmp_path, enable_bpeh=True)
        losses = trainer.train_step(_make_batch(with_events=True))
        assert "event_total" in losses
        assert "state_total" not in losses

    def test_b3_trains(self, tmp_path):
        trainer = self._build_trainer(tmp_path, enable_ssl=True, enable_bpeh=True)
        losses = trainer.train_step(_make_batch(with_events=True))
        assert "state_total" in losses
        assert "event_total" in losses
        assert not torch.tensor(losses["total"]).isnan()

    def test_b3_checkpoint_roundtrip(self, tmp_path):
        trainer = self._build_trainer(tmp_path, enable_ssl=True, enable_bpeh=True)
        trainer.train_step(_make_batch(with_events=True))
        ckpt_path = trainer.save_checkpoint()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ckpt["enable_ssl"] is True
        assert ckpt["enable_bpeh"] is True
        assert "ssl_state_update" in ckpt
        assert "bpeh_head" in ckpt


class TestCLIParser:
    def test_ssl_flag(self):
        from tmrvc_train.cli.train_tts import build_parser
        parser = build_parser()
        args = parser.parse_args(["--cache-dir", "data/cache", "--enable-ssl"])
        assert args.enable_ssl is True

    def test_bpeh_flag(self):
        from tmrvc_train.cli.train_tts import build_parser
        parser = build_parser()
        args = parser.parse_args(["--cache-dir", "data/cache", "--enable-bpeh"])
        assert args.enable_bpeh is True

    def test_both_flags(self):
        from tmrvc_train.cli.train_tts import build_parser
        parser = build_parser()
        args = parser.parse_args(["--cache-dir", "data/cache", "--enable-ssl", "--enable-bpeh"])
        assert args.enable_ssl is True
        assert args.enable_bpeh is True
