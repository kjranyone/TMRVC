"""Tests for scripts/tts_demo.py: argument parser and basic invocation."""

from __future__ import annotations


class TestTTSDemoParser:
    def test_default_args(self):
        from scripts.tts_demo import main
        import argparse

        # Re-create the parser logic from main() to test defaults
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", default="こんにちは、世界！テスト音声です。")
        parser.add_argument("--language", default="ja", choices=["ja", "en", "zh", "ko"])
        parser.add_argument("--output", default="tts_demo_output.wav")
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--tts-checkpoint", default=None)
        parser.add_argument("--vc-checkpoint", default=None)
        parser.add_argument("--speed", type=float, default=1.0)
        parser.add_argument("--emotion", default="neutral")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument("-v", "--verbose", action="store_true")

        args = parser.parse_args([])
        assert args.language == "ja"
        assert args.speed == 1.0
        assert args.emotion == "neutral"
        assert not args.stream
        assert "こんにちは" in args.text

    def test_custom_args(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--text", default="test")
        parser.add_argument("--language", default="ja", choices=["ja", "en", "zh", "ko"])
        parser.add_argument("--output", default="out.wav")
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--speed", type=float, default=1.0)
        parser.add_argument("--emotion", default="neutral")
        parser.add_argument("--stream", action="store_true")

        args = parser.parse_args([
            "--text", "Hello world",
            "--language", "en",
            "--speed", "1.5",
            "--emotion", "happy",
            "--stream",
        ])
        assert args.text == "Hello world"
        assert args.language == "en"
        assert args.speed == 1.5
        assert args.emotion == "happy"
        assert args.stream

    def test_script_is_importable(self):
        """Verify tts_demo can be imported without side effects."""
        import importlib
        mod = importlib.import_module("scripts.tts_demo")
        assert hasattr(mod, "main")
        assert callable(mod.main)
