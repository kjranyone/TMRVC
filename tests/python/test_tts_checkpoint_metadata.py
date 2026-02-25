"""Tests for strict TTS checkpoint metadata handling."""

from __future__ import annotations

import pytest
import torch

from tmrvc_core.constants import PHONEME_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE
from tmrvc_serve.tts_engine import TTSEngine
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor
from tmrvc_train.models.text_encoder import TextEncoder


def _make_tts_ckpt(vocab_size: int, *, text_frontend: str | None) -> dict:
    ckpt: dict = {
        "text_encoder": TextEncoder(vocab_size=vocab_size).state_dict(),
        "duration_predictor": DurationPredictor().state_dict(),
        "f0_predictor": F0Predictor().state_dict(),
        "content_synthesizer": ContentSynthesizer().state_dict(),
    }
    if text_frontend is not None:
        ckpt["text_frontend"] = text_frontend
        ckpt["text_vocab_size"] = vocab_size
    return ckpt


class TestTTSCheckpointMetadata:
    def test_engine_default_frontend_is_tokenizer(self):
        engine = TTSEngine()
        assert engine._text_frontend == "tokenizer"

    def test_load_models_rejects_legacy_checkpoint_without_metadata(self, tmp_path):
        ckpt_path = tmp_path / "legacy_tts.pt"
        torch.save(_make_tts_ckpt(TOKENIZER_VOCAB_SIZE, text_frontend=None), ckpt_path)

        engine = TTSEngine(tts_checkpoint=ckpt_path, text_frontend="tokenizer")
        with pytest.raises(RuntimeError, match="Legacy TTS checkpoints are not supported"):
            engine.load_models()

    def test_load_models_rejects_frontend_mismatch(self, tmp_path):
        ckpt_path = tmp_path / "phoneme_tts.pt"
        torch.save(_make_tts_ckpt(PHONEME_VOCAB_SIZE, text_frontend="phoneme"), ckpt_path)

        engine = TTSEngine(tts_checkpoint=ckpt_path, text_frontend="tokenizer")
        with pytest.raises(RuntimeError, match="Checkpoint/frontend mismatch"):
            engine.load_models()

    def test_load_models_accepts_matching_metadata(self, tmp_path):
        ckpt_path = tmp_path / "tokenizer_tts.pt"
        torch.save(_make_tts_ckpt(TOKENIZER_VOCAB_SIZE, text_frontend="tokenizer"), ckpt_path)

        engine = TTSEngine(tts_checkpoint=ckpt_path, text_frontend="tokenizer")
        engine.load_models()
        assert engine._text_frontend == "tokenizer"
        assert engine._text_vocab_size == TOKENIZER_VOCAB_SIZE
