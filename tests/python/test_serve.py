"""Tests for the FastAPI TTS server schemas and utilities."""

import pytest


def _has_fastapi() -> bool:
    try:
        import fastapi
        return True
    except ImportError:
        return False


from tmrvc_serve.schemas import (
    CharacterCreateRequest,
    CharacterInfo,
    HealthResponse,
    Priority,
    TTSRequest,
    TTSResponse,
    WSAudioMessage,
    WSCancelRequest,
    WSConfigureRequest,
    WSError,
    WSQueueStatus,
    WSSkipped,
    WSSpeakRequest,
    WSStyleMessage,
)


class TestTTSRequest:
    def test_valid_request(self):
        req = TTSRequest(text="こんにちは", character_id="sakura")
        assert req.text == "こんにちは"
        assert req.speed == 1.0
        assert req.emotion is None
        assert req.style_preset == "default"
        assert req.context is None

    def test_with_options(self):
        req = TTSRequest(
            text="テスト",
            character_id="sakura",
            emotion="happy",
            style_preset="asmr_soft",
            speed=1.5,
            situation="学校で",
        )
        assert req.emotion == "happy"
        assert req.style_preset == "asmr_soft"
        assert req.speed == 1.5

    def test_speed_bounds(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            TTSRequest(text="test", character_id="x", speed=3.0)
        with pytest.raises(Exception):
            TTSRequest(text="test", character_id="x", speed=0.1)

    def test_invalid_style_preset(self):
        with pytest.raises(Exception):
            TTSRequest(text="test", character_id="x", style_preset="asmr")


class TestTTSResponse:
    def test_creation(self):
        resp = TTSResponse(
            audio_base64="AAAA",
            duration_sec=1.5,
        )
        assert resp.sample_rate == 24000


class TestCharacterSchemas:
    def test_character_info(self):
        info = CharacterInfo(id="sakura", name="桜")
        assert info.language == "ja"

    def test_character_create_request(self):
        req = CharacterCreateRequest(
            id="sakura",
            name="桜",
            personality="明るい",
        )
        assert req.id == "sakura"

    def test_character_create_request_rejects_unsupported_language(self):
        with pytest.raises(Exception):
            CharacterCreateRequest(
                id="sakura",
                name="Sakura",
                language="other",
            )


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.models_loaded is False
        assert h.characters_count == 0


class TestPriority:
    def test_values(self):
        assert Priority.URGENT == 0
        assert Priority.NORMAL == 1
        assert Priority.LOW == 2

    def test_ordering(self):
        assert Priority.URGENT < Priority.NORMAL < Priority.LOW

    def test_int_coercion(self):
        assert int(Priority.URGENT) == 0
        assert int(Priority.LOW) == 2


class TestWSSpeakRequest:
    def test_defaults(self):
        msg = WSSpeakRequest(text="こんにちは")
        assert msg.type == "speak"
        assert msg.priority == Priority.NORMAL
        assert msg.interrupt is False
        assert msg.character_id == ""
        assert msg.style_preset == "default"

    def test_with_priority_and_interrupt(self):
        msg = WSSpeakRequest(
            text="緊急です",
            character_id="sakura",
            emotion="surprised",
            priority=Priority.URGENT,
            interrupt=True,
        )
        assert msg.priority == Priority.URGENT
        assert msg.interrupt is True
        assert msg.emotion == "surprised"

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            WSSpeakRequest(text="")

    def test_speed_default_none(self):
        msg = WSSpeakRequest(text="test")
        assert msg.speed is None

    def test_speed_override(self):
        msg = WSSpeakRequest(text="test", speed=1.5)
        assert msg.speed == 1.5

    def test_speed_bounds(self):
        with pytest.raises(Exception):
            WSSpeakRequest(text="test", speed=0.1)
        with pytest.raises(Exception):
            WSSpeakRequest(text="test", speed=3.0)

    def test_style_preset_override(self):
        msg = WSSpeakRequest(text="test", style_preset="asmr_intimate")
        assert msg.style_preset == "asmr_intimate"

    def test_invalid_style_preset(self):
        with pytest.raises(Exception):
            WSSpeakRequest(text="test", style_preset="asmr")


class TestWSCancelRequest:
    def test_type(self):
        msg = WSCancelRequest()
        assert msg.type == "cancel"


class TestWSConfigureRequest:
    def test_defaults(self):
        msg = WSConfigureRequest()
        assert msg.type == "configure"
        assert msg.character_id is None
        assert msg.style_preset is None
        assert msg.speed is None

    def test_with_values(self):
        msg = WSConfigureRequest(character_id="yuki", style_preset="asmr_soft", speed=1.3)
        assert msg.character_id == "yuki"
        assert msg.style_preset == "asmr_soft"
        assert msg.speed == 1.3

    def test_speed_bounds(self):
        with pytest.raises(Exception):
            WSConfigureRequest(speed=0.1)
        with pytest.raises(Exception):
            WSConfigureRequest(speed=3.0)


class TestWSStyleMessage:
    def test_defaults(self):
        msg = WSStyleMessage()
        assert msg.type == "style"
        assert msg.emotion == "neutral"
        assert msg.seq == 0

    def test_with_values(self):
        msg = WSStyleMessage(emotion="happy", vad=[0.8, 0.6, 0.5], seq=3)
        assert msg.seq == 3
        assert msg.vad == [0.8, 0.6, 0.5]


class TestWSAudioMessage:
    def test_defaults(self):
        msg = WSAudioMessage()
        assert msg.type == "audio"
        assert msg.chunk_index == 0
        assert msg.is_last is False

    def test_with_data(self):
        msg = WSAudioMessage(data="AAAA", seq=1, chunk_index=5, is_last=True)
        assert msg.data == "AAAA"
        assert msg.seq == 1
        assert msg.is_last is True


class TestWSQueueStatus:
    def test_defaults(self):
        msg = WSQueueStatus()
        assert msg.type == "queue_status"
        assert msg.pending == 0
        assert msg.speaking is False

    def test_active(self):
        msg = WSQueueStatus(pending=3, speaking=True)
        assert msg.pending == 3


class TestWSSkipped:
    def test_creation(self):
        msg = WSSkipped(text="skipped text", reason="interrupted")
        assert msg.type == "skipped"
        assert msg.reason == "interrupted"


class TestWSError:
    def test_creation(self):
        msg = WSError(detail="something went wrong")
        assert msg.type == "error"
        assert msg.detail == "something went wrong"


class TestSpeakItemOrdering:
    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_priority_ordering(self):
        from tmrvc_serve.app import SpeakItem
        low = SpeakItem(priority=2, timestamp=1.0, text="low", character_id="a", emotion=None, style_preset="default", seq=1)
        normal = SpeakItem(priority=1, timestamp=2.0, text="normal", character_id="a", emotion=None, style_preset="default", seq=2)
        urgent = SpeakItem(priority=0, timestamp=3.0, text="urgent", character_id="a", emotion=None, style_preset="default", seq=3)

        items = sorted([low, normal, urgent])
        assert items[0].text == "urgent"
        assert items[1].text == "normal"
        assert items[2].text == "low"

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_fifo_within_same_priority(self):
        from tmrvc_serve.app import SpeakItem
        first = SpeakItem(priority=1, timestamp=1.0, text="first", character_id="a", emotion=None, style_preset="default", seq=1)
        second = SpeakItem(priority=1, timestamp=2.0, text="second", character_id="a", emotion=None, style_preset="default", seq=2)

        items = sorted([second, first])
        assert items[0].text == "first"
        assert items[1].text == "second"


class TestSynthesizeChunks:
    def test_chunk_sizes(self):
        """Verify synthesize_chunks yields correctly sized chunks."""
        import numpy as np
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True

        # Monkey-patch synthesize to return known-length audio
        total_samples = 7200  # 300ms @ 24kHz
        engine.synthesize = lambda *a, **kw: (np.zeros(total_samples, dtype=np.float32), 0.3)

        import torch
        chunks = list(engine.synthesize_chunks(
            text="test",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
        ))
        # 300ms / 100ms = 3 chunks
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) == 2400  # 100ms @ 24kHz

    def test_last_chunk_shorter(self):
        """Last chunk can be shorter than chunk_duration_ms."""
        import numpy as np
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True

        total_samples = 5000  # ~208ms
        engine.synthesize = lambda *a, **kw: (np.zeros(total_samples, dtype=np.float32), 0.208)

        import torch
        chunks = list(engine.synthesize_chunks(
            text="test",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
        ))
        # 5000 / 2400 = 2.08 → 3 chunks
        assert len(chunks) == 3
        assert len(chunks[0]) == 2400
        assert len(chunks[1]) == 2400
        assert len(chunks[2]) == 200  # remainder


class TestWavEncoding:
    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_audio_to_wav_base64(self):
        import base64
        import numpy as np
        from tmrvc_serve.app import _audio_to_wav_base64

        audio = np.zeros(2400, dtype=np.float32)  # 100ms of silence
        result = _audio_to_wav_base64(audio, sr=24000)

        # Should be valid base64
        decoded = base64.b64decode(result)
        # WAV header starts with RIFF
        assert decoded[:4] == b"RIFF"
        assert decoded[8:12] == b"WAVE"


class TestTTSEngineBackend:
    def test_vc_backend_rejects_legacy_converter_checkpoint(self, tmp_path):
        import torch
        from tmrvc_serve.tts_engine import TTSEngine
        from tmrvc_train.models.converter import ConverterStudent
        from tmrvc_train.models.vocoder import VocoderStudent

        ckpt_path = tmp_path / "vc_backend.pt"
        ckpt = {
            "converter": ConverterStudent().state_dict(),  # VC cond=224
            "vocoder": VocoderStudent().state_dict(),
        }
        torch.save(ckpt, ckpt_path)

        engine = TTSEngine(vc_checkpoint=ckpt_path)
        with pytest.raises(RuntimeError, match="Legacy VC converter checkpoints are not supported"):
            engine.load_models()

    def test_vc_backend_accepts_style_conditioned_converter(self, tmp_path):
        import torch
        from tmrvc_core.constants import D_SPEAKER, N_STYLE_PARAMS
        from tmrvc_serve.tts_engine import TTSEngine
        from tmrvc_train.models.converter import ConverterStudent
        from tmrvc_train.models.vocoder import VocoderStudent

        ckpt_path = tmp_path / "vc_backend_style.pt"
        ckpt = {
            "converter": ConverterStudent(n_acoustic_params=N_STYLE_PARAMS).state_dict(),
            "vocoder": VocoderStudent().state_dict(),
        }
        torch.save(ckpt, ckpt_path)

        engine = TTSEngine(vc_checkpoint=ckpt_path)
        engine.load_models()

        assert engine._converter is not None
        assert engine._converter.blocks[0].film.proj.in_features == D_SPEAKER + N_STYLE_PARAMS

    def test_language_id_mapping_for_korean(self, monkeypatch):
        import torch
        from tmrvc_core.constants import D_CONTENT, D_TEXT_ENCODER
        from tmrvc_data import g2p as g2p_module
        from tmrvc_serve.tts_engine import TTSEngine

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_language_ids = None

            def forward(self, phoneme_ids, language_ids):
                self.last_language_ids = language_ids.detach().clone()
                B, L = phoneme_ids.shape
                return torch.zeros(B, D_TEXT_ENCODER, L)

        class DummyDurationPredictor(torch.nn.Module):
            def forward(self, text_features, style):
                B, _, L = text_features.shape
                return torch.ones(B, L)

        class DummyF0Predictor(torch.nn.Module):
            def forward(self, expanded, style):
                B, _, T = expanded.shape
                return torch.zeros(B, 1, T), torch.zeros(B, 1, T)

        class DummyContentSynthesizer(torch.nn.Module):
            def forward(self, expanded):
                B, _, T = expanded.shape
                return torch.zeros(B, D_CONTENT, T)

        class _FakeG2PResult:
            def __init__(self):
                self.phoneme_ids = torch.tensor([2, 10, 3], dtype=torch.long)

        monkeypatch.setattr(g2p_module, "text_to_phonemes", lambda text, language: _FakeG2PResult())

        engine = TTSEngine()
        engine._models_loaded = True
        engine._text_encoder = DummyTextEncoder()
        engine._duration_predictor = DummyDurationPredictor()
        engine._f0_predictor = DummyF0Predictor()
        engine._content_synthesizer = DummyContentSynthesizer()
        engine._converter = None
        engine._vocoder = None

        _audio, _duration = engine.synthesize(
            text="안녕",
            language="ko",
            spk_embed=torch.zeros(192),
        )
        assert int(engine._text_encoder.last_language_ids.item()) == g2p_module.LANG_KO


class TestSynthesizeSentences:
    def _make_engine(self):
        """Create a TTSEngine with a mock synthesize method."""
        import numpy as np
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        from tmrvc_serve.tts_engine import TTSEngine, SynthesisMetrics

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._g2p_cache = OrderedDict()
        engine._lookahead_pool = ThreadPoolExecutor(max_workers=1)
        engine.last_metrics = None
        engine.last_stream_metrics = None

        # Each call returns 7200 samples (300ms) of ascending values
        call_count = [0]
        def _mock_synthesize(*args, **kwargs):
            call_count[0] += 1
            n = 7200
            audio = np.full(n, call_count[0], dtype=np.float32)
            engine.last_metrics = SynthesisMetrics(total_ms=10.0, output_duration_ms=300.0)
            return audio, n / 24000
        engine.synthesize = _mock_synthesize
        engine.prefetch_g2p = lambda *a, **kw: None  # no-op mock
        return engine

    def test_single_sentence_yields_all_samples(self):
        import torch
        engine = self._make_engine()

        chunks = list(engine.synthesize_sentences(
            text="テスト",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
            sentence_pause_ms=0,
        ))
        total = sum(len(c) for c in chunks)
        assert total == 7200  # single sentence, no crossfade tail held

    def test_multi_sentence_crossfade(self):
        import torch
        from tmrvc_serve.tts_engine import CROSSFADE_SAMPLES

        engine = self._make_engine()

        chunks = list(engine.synthesize_sentences(
            text="テスト。テスト。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
            sentence_pause_ms=0,
        ))
        total = sum(len(c) for c in chunks)
        # 2 sentences x 7200 samples, minus crossfade overlap
        expected = 7200 * 2 - CROSSFADE_SAMPLES
        assert total == expected

    def test_crossfade_blends_audio(self):
        import numpy as np
        import torch
        from tmrvc_serve.tts_engine import CROSSFADE_SAMPLES

        engine = self._make_engine()

        chunks = list(engine.synthesize_sentences(
            text="一文目。二文目。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=1000,
            sentence_pause_ms=0,
        ))
        audio = np.concatenate(chunks)

        # First sentence emits 7200 - 2400 = 4800 samples of value 1.0
        # Then crossfade region of 2400 samples (blend of 1.0 fade-out + 2.0 fade-in)
        crossfade_start = 7200 - CROSSFADE_SAMPLES
        crossfade_region = audio[crossfade_start:crossfade_start + CROSSFADE_SAMPLES]
        mid = CROSSFADE_SAMPLES // 2
        assert 1.2 < crossfade_region[mid] < 1.8

    def test_cancel_stops_generation(self):
        import threading
        import torch

        engine = self._make_engine()
        cancel = threading.Event()
        cancel.set()  # pre-cancelled

        chunks = list(engine.synthesize_sentences(
            text="一文目。二文目。三文目。",
            language="ja",
            spk_embed=torch.zeros(192),
            cancel=cancel,
            sentence_pause_ms=0,
        ))
        assert len(chunks) == 0

    def test_cancel_mid_stream(self):
        import threading
        import torch

        engine = self._make_engine()
        cancel = threading.Event()

        original = engine.synthesize
        call_count = [0]
        def _cancel_after_first(*args, **kwargs):
            call_count[0] += 1
            result = original(*args, **kwargs)
            if call_count[0] >= 1:
                cancel.set()
            return result
        engine.synthesize = _cancel_after_first

        chunks = list(engine.synthesize_sentences(
            text="一文目。二文目。三文目。",
            language="ja",
            spk_embed=torch.zeros(192),
            cancel=cancel,
            sentence_pause_ms=0,
        ))
        total = sum(len(c) for c in chunks)
        assert total < 7200 * 3

    def test_sentence_pause_inserts_silence(self):
        """Inter-sentence pause adds silence between sentences."""
        import numpy as np
        import torch

        engine = self._make_engine()
        pause_ms = 120
        pause_samples = int(24000 * pause_ms / 1000)  # 2880

        chunks_with_pause = list(engine.synthesize_sentences(
            text="一文目。二文目。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=1000,
            sentence_pause_ms=pause_ms,
        ))
        total_with = sum(len(c) for c in chunks_with_pause)

        # Reset engine for clean call counts
        engine2 = self._make_engine()
        chunks_without = list(engine2.synthesize_sentences(
            text="一文目。二文目。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=1000,
            sentence_pause_ms=0,
        ))
        total_without = sum(len(c) for c in chunks_without)

        # With pause should be longer by approximately pause_samples
        # (minus crossfade overlap adjustment)
        assert total_with > total_without
        diff = total_with - total_without
        # Pause adds pause_samples but crossfade absorbs CROSSFADE_SAMPLES
        # Actual diff depends on implementation but should be close to pause_samples
        assert diff > 0

    def test_sentence_pause_zero_disables(self):
        """sentence_pause_ms=0 should produce same output as before."""
        import numpy as np
        import torch
        from tmrvc_serve.tts_engine import CROSSFADE_SAMPLES

        engine = self._make_engine()

        chunks = list(engine.synthesize_sentences(
            text="テスト。テスト。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
            sentence_pause_ms=0,
        ))
        total = sum(len(c) for c in chunks)
        expected = 7200 * 2 - CROSSFADE_SAMPLES
        assert total == expected


class TestSynthesizeCancel:
    def test_synthesize_returns_none_on_cancel(self):
        import threading
        import torch
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._g2p_cache = {}
        engine.device = torch.device("cpu")

        cancel = threading.Event()
        cancel.set()

        # synthesize should check cancel after G2P and return None
        # We need to mock text_to_phonemes to not fail
        class _FakeG2PResult:
            phoneme_ids = torch.tensor([1, 2, 3], dtype=torch.long)

        import tmrvc_data.g2p as g2p_mod
        original = g2p_mod.text_to_phonemes
        g2p_mod.text_to_phonemes = lambda *a, **kw: _FakeG2PResult()
        try:
            result = engine.synthesize(
                text="test",
                language="ja",
                spk_embed=torch.zeros(192),
                cancel=cancel,
            )
            assert result is None
        finally:
            g2p_mod.text_to_phonemes = original


class TestG2PCache:
    def test_prefetch_populates_cache(self):
        import torch
        from collections import OrderedDict
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._g2p_cache = OrderedDict()

        class _FakeResult:
            phoneme_ids = torch.tensor([1, 2], dtype=torch.long)

        import tmrvc_data.g2p as g2p_mod
        original = g2p_mod.text_to_phonemes
        g2p_mod.text_to_phonemes = lambda *a, **kw: _FakeResult()
        try:
            engine.prefetch_g2p("hello", "en")
            assert ("hello", "en") in engine._g2p_cache
        finally:
            g2p_mod.text_to_phonemes = original

    def test_synthesize_uses_cached_g2p(self):
        import threading
        import torch
        from collections import OrderedDict
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._g2p_cache = OrderedDict()
        engine.device = torch.device("cpu")

        class _FakeResult:
            phoneme_ids = torch.tensor([1, 2], dtype=torch.long)

        # Pre-populate cache
        engine._g2p_cache[("test", "ja")] = _FakeResult()

        # Track whether text_to_phonemes is called
        called = [False]
        import tmrvc_data.g2p as g2p_mod
        original = g2p_mod.text_to_phonemes
        def _track(*a, **kw):
            called[0] = True
            return original(*a, **kw)
        g2p_mod.text_to_phonemes = _track

        cancel = threading.Event()
        cancel.set()  # will cancel after G2P, which is enough to test cache hit

        try:
            engine.synthesize("test", "ja", torch.zeros(192), cancel=cancel)
            # G2P should NOT have been called — cache was used
            assert not called[0]
            # Cache entry should be consumed (popped)
            assert ("test", "ja") not in engine._g2p_cache
        finally:
            g2p_mod.text_to_phonemes = original

    def test_lru_eviction(self):
        """Cache evicts oldest entries when exceeding max size."""
        import torch
        from collections import OrderedDict
        from tmrvc_serve.tts_engine import TTSEngine, G2P_CACHE_MAX_SIZE

        engine = TTSEngine.__new__(TTSEngine)
        engine._g2p_cache = OrderedDict()

        class _FakeResult:
            phoneme_ids = torch.tensor([1], dtype=torch.long)

        import tmrvc_data.g2p as g2p_mod
        original = g2p_mod.text_to_phonemes
        g2p_mod.text_to_phonemes = lambda *a, **kw: _FakeResult()
        try:
            # Fill cache beyond max
            for i in range(G2P_CACHE_MAX_SIZE + 10):
                engine.prefetch_g2p(f"text_{i}", "ja")

            assert len(engine._g2p_cache) == G2P_CACHE_MAX_SIZE
            # Oldest entries should be evicted
            assert (f"text_0", "ja") not in engine._g2p_cache
            # Newest entries should remain
            assert (f"text_{G2P_CACHE_MAX_SIZE + 9}", "ja") in engine._g2p_cache
        finally:
            g2p_mod.text_to_phonemes = original

    def test_prefetch_dedup(self):
        """Prefetching the same text twice doesn't call G2P again."""
        import torch
        from collections import OrderedDict
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._g2p_cache = OrderedDict()

        call_count = [0]
        class _FakeResult:
            phoneme_ids = torch.tensor([1], dtype=torch.long)

        import tmrvc_data.g2p as g2p_mod
        original = g2p_mod.text_to_phonemes
        def _counting(*a, **kw):
            call_count[0] += 1
            return _FakeResult()
        g2p_mod.text_to_phonemes = _counting
        try:
            engine.prefetch_g2p("hello", "en")
            engine.prefetch_g2p("hello", "en")
            assert call_count[0] == 1  # second call should be a no-op
        finally:
            g2p_mod.text_to_phonemes = original


class TestConstants:
    def test_crossfade_samples(self):
        from tmrvc_serve.tts_engine import (
            CROSSFADE_SAMPLES,
            FADEOUT_SAMPLES,
            SENTENCE_PAUSE_SAMPLES,
            G2P_CACHE_MAX_SIZE,
        )
        assert CROSSFADE_SAMPLES == 2400     # 100ms @ 24kHz
        assert FADEOUT_SAMPLES == 1200       # 50ms @ 24kHz
        assert SENTENCE_PAUSE_SAMPLES == 2880  # 120ms @ 24kHz
        assert G2P_CACHE_MAX_SIZE == 256


class TestStylePresetHelpers:
    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_default_preset_keeps_none_style(self):
        from tmrvc_serve.app import _resolve_style_preset

        style, cfg = _resolve_style_preset(None, "default")
        assert style is None
        assert cfg.speed_multiplier == 1.0

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_asmr_soft_preset_generates_whisper_style(self):
        from tmrvc_serve.app import _resolve_style_preset

        style, cfg = _resolve_style_preset(None, "asmr_soft")
        assert style is not None
        assert style.emotion == "whisper"
        assert style.energy < 0
        assert cfg.sentence_pause_ms > 120

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_speed_multiplier_is_clamped(self):
        from tmrvc_serve.app import _resolve_effective_speed, StylePresetConfig

        cfg = StylePresetConfig(speed_multiplier=0.1)
        assert _resolve_effective_speed(1.0, cfg) == 0.5


class TestTTSStreamRequestSchema:
    def test_defaults(self):
        from tmrvc_serve.schemas import TTSStreamRequest
        req = TTSStreamRequest(text="test", character_id="sakura")
        assert req.speed == 1.0
        assert req.chunk_duration_ms == 100
        assert req.emotion is None
        assert req.style_preset == "default"

    def test_with_options(self):
        from tmrvc_serve.schemas import TTSStreamRequest
        req = TTSStreamRequest(
            text="hello",
            character_id="yuki",
            emotion="happy",
            style_preset="asmr_intimate",
            speed=1.5,
            chunk_duration_ms=200,
        )
        assert req.emotion == "happy"
        assert req.style_preset == "asmr_intimate"
        assert req.chunk_duration_ms == 200

    def test_chunk_duration_bounds(self):
        from tmrvc_serve.schemas import TTSStreamRequest
        with pytest.raises(Exception):
            TTSStreamRequest(text="test", character_id="x", chunk_duration_ms=10)
        with pytest.raises(Exception):
            TTSStreamRequest(text="test", character_id="x", chunk_duration_ms=600)

    def test_invalid_style_preset(self):
        from tmrvc_serve.schemas import TTSStreamRequest
        with pytest.raises(Exception):
            TTSStreamRequest(text="test", character_id="x", style_preset="asmr")


# ---------------------------------------------------------------------------
# Round 3: Metrics, warmup, lookahead, auto-style tests
# ---------------------------------------------------------------------------


class TestSynthesisMetrics:
    def test_rtf_calculation(self):
        from tmrvc_serve.tts_engine import SynthesisMetrics
        m = SynthesisMetrics(total_ms=50.0, output_duration_ms=100.0)
        assert m.rtf == pytest.approx(0.5)

    def test_rtf_zero_output(self):
        from tmrvc_serve.tts_engine import SynthesisMetrics
        m = SynthesisMetrics(total_ms=50.0, output_duration_ms=0.0)
        assert m.rtf == 0.0

    def test_default_values(self):
        from tmrvc_serve.tts_engine import SynthesisMetrics
        m = SynthesisMetrics()
        assert m.g2p_ms == 0.0
        assert m.cancelled is False
        assert m.output_frames == 0


class TestStreamMetrics:
    def test_avg_sentence_ms(self):
        from tmrvc_serve.tts_engine import StreamMetrics, SynthesisMetrics
        m = StreamMetrics(
            sentence_count=2,
            per_sentence=[
                SynthesisMetrics(total_ms=30.0),
                SynthesisMetrics(total_ms=50.0),
            ],
        )
        assert m.avg_sentence_ms == pytest.approx(40.0)

    def test_avg_sentence_ms_empty(self):
        from tmrvc_serve.tts_engine import StreamMetrics
        m = StreamMetrics()
        assert m.avg_sentence_ms == 0.0


class TestWarmup:
    def test_warmup_sets_flag(self):
        import numpy as np
        from collections import OrderedDict
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._warmed_up = False
        engine._g2p_cache = OrderedDict()
        engine.device = __import__("torch").device("cpu")
        engine.last_metrics = None

        engine.synthesize = lambda *a, **kw: (np.zeros(2400, dtype=np.float32), 0.1)
        engine.warmup()
        assert engine._warmed_up is True

    def test_warmup_skips_if_already_warmed(self):
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._warmed_up = True

        call_count = [0]
        original_synth = lambda *a, **kw: None
        def counting_synth(*a, **kw):
            call_count[0] += 1
            return original_synth(*a, **kw)
        engine.synthesize = counting_synth
        engine.warmup()
        assert call_count[0] == 0

    def test_warmup_skips_if_not_loaded(self):
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = False
        engine._warmed_up = False

        engine.warmup()
        assert engine._warmed_up is False


class TestLookaheadSynthesis:
    def _make_engine(self):
        """Create engine with mock synthesize that tracks call order."""
        import numpy as np
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        from tmrvc_serve.tts_engine import TTSEngine

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._g2p_cache = OrderedDict()
        engine._lookahead_pool = ThreadPoolExecutor(max_workers=1)
        engine.last_metrics = None
        engine.last_stream_metrics = None

        call_log = []
        def _mock_synthesize(text, language, spk_embed, style=None,
                             speed=1.0, cancel=None):
            from tmrvc_serve.tts_engine import SynthesisMetrics
            call_log.append(text)
            n = 7200
            audio = np.ones(n, dtype=np.float32) * len(call_log)
            engine.last_metrics = SynthesisMetrics(total_ms=10.0, output_duration_ms=300.0)
            return audio, n / 24000
        engine.synthesize = _mock_synthesize
        engine.prefetch_g2p = lambda *a, **kw: None  # no-op mock
        return engine, call_log

    def test_all_sentences_synthesized(self):
        import torch
        engine, call_log = self._make_engine()

        chunks = list(engine.synthesize_sentences(
            text="一文目。二文目。三文目。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
            sentence_pause_ms=0,
        ))
        # All 3 sentences should be synthesized
        assert len(call_log) == 3
        assert sum(len(c) for c in chunks) > 0

    def test_stream_metrics_populated(self):
        import torch
        engine, _ = self._make_engine()

        list(engine.synthesize_sentences(
            text="テスト。テスト。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=100,
            sentence_pause_ms=0,
        ))
        sm = engine.last_stream_metrics
        assert sm is not None
        assert sm.sentence_count == 2
        assert len(sm.per_sentence) == 2
        assert sm.total_ms > 0
        assert sm.first_chunk_ms > 0


class TestAutoStyle:
    def _make_engine(self):
        import numpy as np
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        from tmrvc_serve.tts_engine import TTSEngine, SynthesisMetrics

        engine = TTSEngine.__new__(TTSEngine)
        engine._models_loaded = True
        engine._g2p_cache = OrderedDict()
        engine._lookahead_pool = ThreadPoolExecutor(max_workers=1)
        engine.last_metrics = None
        engine.last_stream_metrics = None

        styles_received = []
        def _mock_synthesize(text, language, spk_embed, style=None,
                             speed=1.0, cancel=None):
            styles_received.append(style)
            n = 7200
            engine.last_metrics = SynthesisMetrics(total_ms=10.0, output_duration_ms=300.0)
            return np.ones(n, dtype=np.float32), n / 24000
        engine.synthesize = _mock_synthesize
        engine.prefetch_g2p = lambda *a, **kw: None  # no-op mock
        return engine, styles_received

    def test_auto_style_infers_emotion(self):
        """auto_style=True should produce different styles for different sentences."""
        import torch
        engine, styles = self._make_engine()

        list(engine.synthesize_sentences(
            text="嬉しい！悲しい。",
            language="ja",
            spk_embed=torch.zeros(192),
            chunk_duration_ms=1000,
            sentence_pause_ms=0,
            auto_style=True,
        ))
        assert len(styles) == 2
        assert styles[0].emotion == "happy"
        assert styles[1].emotion == "sad"

    def test_auto_style_disabled(self):
        """auto_style=False should pass the same style to all sentences."""
        import torch
        from tmrvc_core.dialogue_types import StyleParams
        engine, styles = self._make_engine()

        base = StyleParams.neutral()
        list(engine.synthesize_sentences(
            text="嬉しい！悲しい。",
            language="ja",
            spk_embed=torch.zeros(192),
            style=base,
            chunk_duration_ms=1000,
            sentence_pause_ms=0,
            auto_style=False,
        ))
        assert len(styles) == 2
        assert styles[0] is base
        assert styles[1] is base

    def test_auto_style_with_explicit_base(self):
        """auto_style=True with a non-neutral base should derive from that base."""
        import torch
        from tmrvc_core.dialogue_types import StyleParams
        engine, styles = self._make_engine()

        base = StyleParams(emotion="neutral", energy=0.8)
        list(engine.synthesize_sentences(
            text="すごい！",
            language="ja",
            spk_embed=torch.zeros(192),
            style=base,
            chunk_duration_ms=1000,
            sentence_pause_ms=0,
            auto_style=True,
        ))
        assert len(styles) == 1
        # "すごい" → excited, energy should be boosted from the 0.8 base
        assert styles[0].emotion == "excited"

class TestHintSituationSchemas:
    def test_tts_request_accepts_hint(self):
        req = TTSRequest(text="test", character_id="x", hint="soft whisper")
        assert req.hint == "soft whisper"

    def test_tts_stream_accepts_context_and_situation(self):
        from tmrvc_serve.schemas import DialogueTurnSchema, TTSStreamRequest

        req = TTSStreamRequest(
            text="test",
            character_id="x",
            context=[DialogueTurnSchema(speaker="a", text="line")],
            situation="late night scene",
        )
        assert req.context is not None
        assert req.context[0].speaker == "a"
        assert req.situation == "late night scene"

    def test_ws_requests_accept_hint_and_situation(self):
        speak = WSSpeakRequest(text="hello", hint="breathy", situation="studio booth")
        assert speak.hint == "breathy"
        assert speak.situation == "studio booth"

        cfg = WSConfigureRequest(situation="rainy station platform")
        assert cfg.situation == "rainy station platform"


class TestDialogueStyleHelpers:
    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_blend_styles_keeps_non_neutral_base_emotion(self):
        from tmrvc_core.dialogue_types import StyleParams
        from tmrvc_serve.app import _blend_styles

        base = StyleParams(emotion="sad", valence=-0.4)
        hint = StyleParams(emotion="happy", valence=0.8, reasoning="hint")
        merged = _blend_styles(base, hint, overlay_weight=0.25, reason_tag="hint_soft")
        assert merged is not None
        assert merged.emotion == "sad"
        assert merged.valence > -0.4
        assert "hint_soft" in merged.reasoning

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_dialogue_dynamics_reply_to_question(self):
        from tmrvc_core.dialogue_types import DialogueTurn, StyleParams
        from tmrvc_serve.app import _apply_dialogue_dynamics

        base = StyleParams.neutral()
        history = [DialogueTurn(speaker="alice", text="Are you okay?", emotion="neutral")]
        adjusted = _apply_dialogue_dynamics(base, history, speaker="bob", text="yeah.")
        assert adjusted is not None
        assert adjusted.arousal > base.arousal
        assert "reply_to_question" in adjusted.reasoning

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_predict_style_uses_hint_as_soft_bias(self, monkeypatch):
        import asyncio
        from tmrvc_core.dialogue_types import CharacterProfile, StyleParams
        from tmrvc_serve import app as app_module

        class _DummyPredictor:
            async def predict(self, character, history, text, situation):
                return StyleParams.neutral()

            def predict_rule_based(self, text, character):
                if text == "base":
                    return StyleParams(emotion="neutral", valence=0.0, reasoning="base")
                return StyleParams(emotion="happy", valence=1.0, reasoning="hint")

        monkeypatch.setattr(app_module, "_context_predictor", _DummyPredictor())
        character = CharacterProfile(name="speaker")

        style = asyncio.run(app_module._predict_style_from_inputs(
            character=character,
            text="base",
            emotion=None,
            history=[],
            situation=None,
            hint="use warm acting",
            speaker="speaker",
        ))
        assert style is not None
        assert style.valence == pytest.approx(0.35, abs=1e-5)
        assert "hint_soft" in style.reasoning
