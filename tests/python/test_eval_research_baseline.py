"""Tests for scripts/eval_research_baseline.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySpeakerEncoder:
    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del sample_rate
        # Deterministic pseudo-embedding from waveform energy.
        scale = float(waveform.abs().mean().item()) if waveform.numel() > 0 else 0.0
        return torch.full((192,), scale, dtype=torch.float32)

def _make_cache_entry(tmp_path: Path, dataset: str, speaker: str, utt: str, T: int = 50) -> Path:
    """Create a minimal cache entry for testing."""
    utt_dir = tmp_path / dataset / "train" / speaker / utt
    utt_dir.mkdir(parents=True, exist_ok=True)
    np.save(utt_dir / "mel.npy", np.random.randn(80, T).astype(np.float32))
    np.save(utt_dir / "content.npy", np.random.randn(768, T).astype(np.float32))
    np.save(utt_dir / "f0.npy", np.abs(np.random.randn(1, T).astype(np.float32)) * 200 + 100)
    np.save(utt_dir / "spk_embed.npy", np.random.randn(192).astype(np.float32))
    meta = {"utterance_id": utt, "speaker_id": speaker, "n_frames": T, "content_dim": 768}
    with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return utt_dir


def _make_config(tmp_path: Path, speakers: list[str]) -> Path:
    """Create a minimal b0 config YAML."""
    config = {
        "variant": "b0_test",
        "sampling": {"steps": 2, "sway_coefficient": 0.0, "cfg_scale": 1.0},
        "test_split": {
            "datasets": ["testds"],
            "speakers": speakers,
            "max_utterances_per_speaker": 2,
        },
        "metrics": ["mel_mse", "secs", "f0_correlation", "utmos_proxy"],
        "output": {"save_audio": False},
    }
    config_path = tmp_path / "b0_test.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return config_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_deterministic_torch_randn(self):
        from scripts.eval_research_baseline import seed_everything

        seed_everything(123)
        a = torch.randn(10)
        seed_everything(123)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_deterministic_numpy(self):
        from scripts.eval_research_baseline import seed_everything

        seed_everything(456)
        a = np.random.randn(10)
        seed_everything(456)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)


class TestResolveTestSet:
    def test_resolves_matching_speakers(self, tmp_path):
        from scripts.eval_research_baseline import resolve_test_set
        from tmrvc_data.cache import FeatureCache

        _make_cache_entry(tmp_path, "testds", "spk001", "utt001")
        _make_cache_entry(tmp_path, "testds", "spk001", "utt002")
        _make_cache_entry(tmp_path, "testds", "spk001", "utt003")
        _make_cache_entry(tmp_path, "testds", "spk002", "utt001")

        cache = FeatureCache(tmp_path)
        config = {
            "test_split": {
                "datasets": ["testds"],
                "speakers": ["testds_spk001"],
                "max_utterances_per_speaker": 2,
            }
        }

        triples = resolve_test_set(cache, config)
        assert len(triples) == 2
        assert all(ds == "testds" for ds, _, _ in triples)
        assert all(sid == "spk001" for _, sid, _ in triples)

    def test_empty_when_no_match(self, tmp_path):
        from scripts.eval_research_baseline import resolve_test_set
        from tmrvc_data.cache import FeatureCache

        _make_cache_entry(tmp_path, "testds", "spk001", "utt001")
        cache = FeatureCache(tmp_path)
        config = {
            "test_split": {
                "datasets": ["testds"],
                "speakers": ["testds_nonexistent"],
                "max_utterances_per_speaker": 10,
            }
        }

        triples = resolve_test_set(cache, config)
        assert len(triples) == 0

    def test_all_speakers_when_empty_filter(self, tmp_path):
        from scripts.eval_research_baseline import resolve_test_set
        from tmrvc_data.cache import FeatureCache

        _make_cache_entry(tmp_path, "testds", "spk001", "utt001")
        _make_cache_entry(tmp_path, "testds", "spk002", "utt001")
        cache = FeatureCache(tmp_path)
        config = {
            "test_split": {
                "datasets": ["testds"],
                "speakers": [],
                "max_utterances_per_speaker": 10,
            }
        }

        triples = resolve_test_set(cache, config)
        assert len(triples) == 2


class TestEvaluate:
    def test_runs_and_produces_results(self, tmp_path):
        from scripts.eval_research_baseline import evaluate, seed_everything
        from tmrvc_data.cache import FeatureCache

        T = 30
        _make_cache_entry(tmp_path / "cache", "testds", "spk001", "utt001", T=T)
        _make_cache_entry(tmp_path / "cache", "testds", "spk001", "utt002", T=T)
        cache = FeatureCache(tmp_path / "cache")

        test_set = [
            ("testds", "spk001", "utt001"),
            ("testds", "spk001", "utt002"),
        ]

        config = {
            "variant": "b0_test",
            "sampling": {"steps": 2, "sway_coefficient": 0.0, "cfg_scale": 1.0},
            "evaluation": {"griffin_lim_iters": 2},
            "output": {"save_audio": False},
        }

        # Mock teacher: returns random mel
        mock_teacher = MagicMock()
        mock_teacher.return_value = torch.randn(1, 80, T)
        mock_teacher.eval = MagicMock()
        mock_teacher.parameters = MagicMock(return_value=iter([torch.randn(10)]))

        seed_everything(42)
        result = evaluate(
            teacher=mock_teacher,
            cache=cache,
            test_set=test_set,
            config=config,
            device=torch.device("cpu"),
            output_dir=tmp_path / "output",
            seed=42,
            speaker_encoder=_DummySpeakerEncoder(),
        )

        assert result.n_utterances == 2
        assert len(result.per_utterance) == 2
        assert "mel_mse" in result.aggregate
        assert "utmos" in result.aggregate
        assert result.aggregate["mel_mse"]["mean"] > 0

    def test_reproducibility(self, tmp_path):
        from scripts.eval_research_baseline import evaluate, seed_everything
        from tmrvc_data.cache import FeatureCache

        T = 20
        _make_cache_entry(tmp_path / "cache", "testds", "spk001", "utt001", T=T)
        cache = FeatureCache(tmp_path / "cache")
        test_set = [("testds", "spk001", "utt001")]

        config = {
            "variant": "b0_test",
            "sampling": {"steps": 2, "sway_coefficient": 0.0, "cfg_scale": 1.0},
            "evaluation": {"griffin_lim_iters": 2},
            "output": {"save_audio": False},
        }

        # Deterministic mock
        def mock_forward(x_t, t, **kwargs):
            return torch.zeros_like(x_t)  # deterministic velocity

        mock_teacher = MagicMock(side_effect=mock_forward)

        seed_everything(42)
        r1 = evaluate(
            mock_teacher,
            cache,
            test_set,
            config,
            torch.device("cpu"),
            None,
            seed=42,
            speaker_encoder=_DummySpeakerEncoder(),
        )
        seed_everything(42)
        r2 = evaluate(
            mock_teacher,
            cache,
            test_set,
            config,
            torch.device("cpu"),
            None,
            seed=42,
            speaker_encoder=_DummySpeakerEncoder(),
        )

        assert r1.per_utterance[0].mel_mse == r2.per_utterance[0].mel_mse
        assert r1.per_utterance[0].utmos == r2.per_utterance[0].utmos


class TestMainCLI:
    def test_missing_checkpoint_fails(self, tmp_path):
        """CLI should fail gracefully when checkpoint doesn't exist."""
        config_path = _make_config(tmp_path, ["testds_spk001"])

        with pytest.raises(Exception):
            from scripts.eval_research_baseline import main
            main([
                "--config", str(config_path),
                "--checkpoint", str(tmp_path / "nonexistent.pt"),
                "--cache-dir", str(tmp_path),
                "--seed", "42",
            ])


class TestLoadTeacherStrictness:
    def test_rejects_missing_model_state_dict(self, tmp_path):
        from scripts.eval_research_baseline import load_teacher

        ckpt_path = tmp_path / "invalid.pt"
        torch.save({"step": 1}, ckpt_path)

        with pytest.raises(RuntimeError, match="missing 'model_state_dict'"):
            load_teacher(ckpt_path, torch.device("cpu"))


class TestUtteranceResult:
    def test_dataclass_fields(self):
        from scripts.eval_research_baseline import UtteranceResult

        r = UtteranceResult(
            speaker_id="spk001",
            utterance_id="utt001",
            n_frames=100,
            mel_mse=0.1,
            secs=0.95,
            f0_corr=0.88,
            utmos=3.5,
        )
        assert r.speaker_id == "spk001"
        assert r.mel_mse == 0.1


class TestEvalResult:
    def test_serializable(self):
        from dataclasses import asdict
        from scripts.eval_research_baseline import EvalResult, UtteranceResult

        r = EvalResult(
            variant="b0",
            checkpoint="test.pt",
            seed=42,
            sampling_steps=32,
            sway_coefficient=1.0,
            cfg_scale=1.0,
            n_utterances=1,
            elapsed_sec=1.5,
            per_utterance=[
                UtteranceResult("spk", "utt", 100, 0.1, 0.95, 0.88, 3.5)
            ],
            aggregate={"mel_mse": {"mean": 0.1, "std": 0.0, "min": 0.1, "max": 0.1}},
        )
        d = asdict(r)
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["variant"] == "b0"
        assert len(parsed["per_utterance"]) == 1
