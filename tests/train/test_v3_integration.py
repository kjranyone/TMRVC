"""Integration tests for UCLM v3 pointer-based TTS training.

Tests cover:
a) v3 pointer training without durations.npy
b) v2 legacy training still works with durations.npy
c) CLI arg parsing for --tts-mode
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tmrvc_train.dataset import DisentangledUCLMDataset
from tmrvc_train.cli.train_uclm import main as train_uclm_main


# ---------------------------------------------------------------------------
# Helpers: create mock cache directories
# ---------------------------------------------------------------------------

_T = 20  # frames
_L = 8   # phoneme length


def _write_core_files(utt_dir: Path, *, speaker_id: str = "spk0") -> None:
    """Write the minimal required files for a valid utterance."""
    utt_dir.mkdir(parents=True, exist_ok=True)
    np.save(utt_dir / "codec_tokens.npy", np.random.randint(0, 1024, (8, _T)))
    np.save(utt_dir / "explicit_state.npy", np.random.randn(_T, 8).astype(np.float32))
    np.save(utt_dir / "ssl_state.npy", np.random.randn(_T, 128).astype(np.float32))
    np.save(utt_dir / "spk_embed.npy", np.random.randn(192).astype(np.float32))
    meta = {"speaker_id": speaker_id, "language_id": 0, "text": "hello"}
    (utt_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _add_phoneme_ids(utt_dir: Path) -> None:
    np.save(utt_dir / "phoneme_ids.npy", np.random.randint(1, 200, (_L,)))


def _add_durations(utt_dir: Path) -> None:
    # Make durations that roughly sum to _T
    durs = np.ones(_L, dtype=np.int64) * (_T // _L)
    durs[-1] += _T - durs.sum()
    np.save(utt_dir / "durations.npy", durs)


def _build_cache(tmp_path: Path, *, with_durations: bool, dataset: str = "ds1") -> Path:
    """Build a 2-utterance mock cache under tmp_path."""
    cache = tmp_path / "cache"
    for i in range(2):
        utt_dir = cache / dataset / "train" / "spk0" / f"utt{i}"
        _write_core_files(utt_dir, speaker_id="spk0")
        _add_phoneme_ids(utt_dir)
        if with_durations:
            _add_durations(utt_dir)
    return cache


# ---------------------------------------------------------------------------
# a) v3 pointer train without durations
# ---------------------------------------------------------------------------


class TestV3PointerTrainWithoutDurations:
    def test_dataset_loads_pointer_mode(self, tmp_path: Path):
        """Dataset in pointer mode should load utterances with phoneme_ids only."""
        cache = _build_cache(tmp_path, with_durations=False)
        ds = DisentangledUCLMDataset(cache, tts_mode="pointer")
        assert len(ds) == 2

        sample = ds[0]
        assert sample["phoneme_ids"] is not None
        assert sample["durations"] is None

    def test_trainer_step_pointer_no_durations(self, tmp_path: Path):
        """Full integration: dataset -> collate -> trainer.train_step in pointer mode."""
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        from tmrvc_train.trainer import UCLMTrainer
        from tmrvc_train.cli.train_uclm import collate_fn

        cache = _build_cache(tmp_path, with_durations=False)
        ds = DisentangledUCLMDataset(cache, tts_mode="pointer")
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)

        model = DisentangledUCLM(
            d_model=256, n_heads=4, n_layers=2, num_speakers=2
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = UCLMTrainer(model, optimizer, device="cpu", tts_prob=1.0, tts_mode="pointer")

        batch = next(iter(loader))
        # In pointer mode, durations should NOT be in the batch (all None -> omitted)
        assert "durations" not in batch or batch.get("durations") is None

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["mode"] == 1  # TTS


# ---------------------------------------------------------------------------
# b) v2 legacy train still works
# ---------------------------------------------------------------------------


class TestV2LegacyDatasetStillWorks:
    def test_dataset_loads_legacy_mode(self, tmp_path: Path):
        """Dataset in legacy_duration mode should load both phoneme_ids and durations."""
        cache = _build_cache(tmp_path, with_durations=True)
        ds = DisentangledUCLMDataset(cache, tts_mode="legacy_duration")
        assert len(ds) == 2

        sample = ds[0]
        assert sample["phoneme_ids"] is not None
        assert sample["durations"] is not None

    def test_trainer_uses_durations_as_pointer_targets(self, tmp_path: Path):
        """When durations are available (auto mode), trainer uses them as pointer targets."""
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        from tmrvc_train.trainer import UCLMTrainer
        from tmrvc_train.cli.train_uclm import collate_fn

        cache = _build_cache(tmp_path, with_durations=True)
        # Use auto mode to load both phoneme_ids and durations
        ds = DisentangledUCLMDataset(cache, tts_mode="auto")
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)

        model = DisentangledUCLM(
            d_model=256, n_heads=4, n_layers=2, num_speakers=2
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = UCLMTrainer(model, optimizer, device="cpu", tts_prob=1.0, tts_mode="pointer")

        batch = next(iter(loader))
        # In auto mode with durations files, durations should be available
        assert batch.get("durations") is not None

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["mode"] == 1  # TTS

    def test_legacy_mode_ignores_phoneme_only_utterances(self, tmp_path: Path):
        """In legacy_duration mode, utterances with phoneme_ids but no durations
        should have phoneme_ids=None (i.e. treated as VC-only)."""
        cache = _build_cache(tmp_path, with_durations=False)
        ds = DisentangledUCLMDataset(cache, tts_mode="legacy_duration")
        assert len(ds) == 2

        sample = ds[0]
        assert sample["phoneme_ids"] is None
        assert sample["durations"] is None


# ---------------------------------------------------------------------------
# c) CLI arg parsing for --tts-mode
# ---------------------------------------------------------------------------


class TestCliTtsModeArgParsing:
    def test_parse_pointer_mode(self, tmp_path: Path):
        """--tts-mode pointer should be accepted."""
        cache = _build_cache(tmp_path, with_durations=False)
        output = tmp_path / "out"

        # We only test that arg parsing succeeds and training starts;
        # max-steps=0 means it won't actually train but will parse and setup.
        # Actually max_steps=1 is the minimum to avoid division by zero etc.
        # We just ensure it doesn't raise on arg parsing.
        import argparse

        from tmrvc_train.cli.train_uclm import main

        argv = [
            "--cache-dir", str(cache),
            "--output-dir", str(output),
            "--batch-size", "2",
            "--max-steps", "1",
            "--device", "cpu",
            "--lr", "1e-3",
            "--tts-mode", "pointer",
            "--datasets", "ds1",
        ]
        # This will run 1 training step; we just want to verify no crash.
        main(argv)
        assert (output / "uclm_final.pt").exists()

    def test_parse_legacy_duration_mode(self, tmp_path: Path):
        """--tts-mode legacy_duration should be accepted."""
        cache = _build_cache(tmp_path, with_durations=True)
        output = tmp_path / "out"

        from tmrvc_train.cli.train_uclm import main

        argv = [
            "--cache-dir", str(cache),
            "--output-dir", str(output),
            "--batch-size", "2",
            "--max-steps", "1",
            "--device", "cpu",
            "--lr", "1e-3",
            "--tts-mode", "legacy_duration",
            "--datasets", "ds1",
        ]
        main(argv)
        assert (output / "uclm_final.pt").exists()

    def test_parse_invalid_mode_rejected(self, tmp_path: Path):
        """An invalid --tts-mode value should be rejected by argparse."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tmrvc_train.cli.train_uclm", "--tts-mode", "bogus"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()
