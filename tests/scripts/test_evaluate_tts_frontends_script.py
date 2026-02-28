"""Tests for scripts/evaluate_tts_frontends.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_eval_script():
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import evaluate_tts_frontends as mod
    return mod


class TestParser:
    def test_defaults(self):
        mod = _import_eval_script()
        parser = mod.build_parser()
        args = parser.parse_args(["scene.yaml"])
        assert args.script == Path("scene.yaml")
        assert args.frontends == ["tokenizer", "phoneme"]
        assert args.sentence_pause_ms == 120
        assert args.chunk_duration_ms == 100
        assert args.auto_style is False
        assert args.create_ab is False

    def test_all_args(self):
        mod = _import_eval_script()
        parser = mod.build_parser()
        args = parser.parse_args([
            "scene.yaml",
            "--output-dir", "out",
            "--tts-checkpoint", "ckpt/tts.pt",
            "--vc-checkpoint", "ckpt/vc.pt",
            "--device", "xpu",
            "--frontends", "tokenizer", "phoneme",
            "--sentence-pause-ms", "180",
            "--chunk-duration-ms", "80",
            "--speed", "0.9",
            "--auto-style",
            "--stage-blend-weight", "0.5",
            "--create-ab",
            "--seed", "99",
        ])
        assert args.output_dir == Path("out")
        assert args.tts_checkpoint == Path("ckpt/tts.pt")
        assert args.vc_checkpoint == Path("ckpt/vc.pt")
        assert args.device == "xpu"
        assert args.frontends == ["tokenizer", "phoneme"]
        assert args.sentence_pause_ms == 180
        assert args.chunk_duration_ms == 80
        assert args.speed == pytest.approx(0.9)
        assert args.auto_style is True
        assert args.stage_blend_weight == pytest.approx(0.5)
        assert args.create_ab is True
        assert args.seed == 99


class TestSummary:
    def test_summary_groups_by_frontend(self):
        mod = _import_eval_script()
        rows = [
            mod.EvalRow(
                entry_index=1,
                speaker="a",
                language="ja",
                frontend="tokenizer",
                source_text="x",
                spoken_text="x",
                stage_directions="",
                audio_path="a.wav",
                speed=1.0,
                sentence_pause_ms=120,
                duration_sec=1.0,
                samples=24000,
                wall_ms=800.0,
                first_chunk_ms=120.0,
                stream_total_ms=750.0,
                avg_sentence_ms=750.0,
                rtf=0.75,
            ),
            mod.EvalRow(
                entry_index=1,
                speaker="a",
                language="ja",
                frontend="phoneme",
                source_text="x",
                spoken_text="x",
                stage_directions="",
                audio_path="b.wav",
                speed=1.0,
                sentence_pause_ms=120,
                duration_sec=1.0,
                samples=24000,
                wall_ms=900.0,
                first_chunk_ms=140.0,
                stream_total_ms=850.0,
                avg_sentence_ms=850.0,
                rtf=0.85,
            ),
        ]
        s = mod._summary(rows)
        assert "tokenizer" in s
        assert "phoneme" in s
        assert s["tokenizer"]["entries"] == 1
        assert s["tokenizer"]["aggregate_rtf"] == pytest.approx(0.8)
        assert s["phoneme"]["mean_entry_rtf"] == pytest.approx(0.85)


class TestScriptLoading:
    def test_load_script_obj_accepts_entries_key(self, tmp_path):
        mod = _import_eval_script()
        p = tmp_path / "scene.yaml"
        p.write_text(
            (
                "title: demo\n"
                "characters:\n"
                "  a:\n"
                "    name: A\n"
                "entries:\n"
                "  - speaker: a\n"
                "    text: hello\n"
            ),
            encoding="utf-8",
        )
        script = mod._load_script_obj(p)
        assert len(script.entries) == 1
        assert script.entries[0].speaker == "a"
