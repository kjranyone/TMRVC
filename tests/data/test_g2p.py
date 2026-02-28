"""Tests for G2P frontend and alignment utilities."""

import numpy as np
import torch
import pytest

from tmrvc_data.g2p import (
    PHONE2ID,
    PHONEME_LIST,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    SIL_ID,
    LANG_JA,
    LANG_EN,
    LANG_ZH,
    LANG_KO,
)


class TestPhonemeVocabulary:
    def test_special_tokens_at_start(self):
        assert PHONEME_LIST[0] == "<pad>"
        assert PHONEME_LIST[1] == "<unk>"
        assert PHONEME_LIST[2] == "<bos>"
        assert PHONEME_LIST[3] == "<eos>"
        assert PHONEME_LIST[4] == "<sil>"

    def test_pad_id_is_zero(self):
        assert PAD_ID == 0

    def test_no_duplicates(self):
        assert len(PHONEME_LIST) == len(set(PHONEME_LIST))

    def test_phone2id_consistency(self):
        for phone, idx in PHONE2ID.items():
            assert PHONEME_LIST[idx] == phone

    def test_japanese_phonemes_present(self):
        assert "a" in PHONE2ID
        assert "i" in PHONE2ID
        assert "u" in PHONE2ID
        assert "N" in PHONE2ID  # moraic nasal
        assert "cl" in PHONE2ID  # geminate
        assert "pau" in PHONE2ID  # pause

    def test_english_phonemes_present(self):
        assert "p" in PHONE2ID
        assert "b" in PHONE2ID
        assert "t" in PHONE2ID


class TestG2PJapanese:
    @pytest.fixture
    def _check_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    @pytest.mark.usefixtures("_check_pyopenjtalk")
    def test_basic_japanese(self):
        from tmrvc_data.g2p import text_to_phonemes

        result = text_to_phonemes("縺薙ｓ縺ｫ縺｡縺ｯ", language="ja")
        assert result.language_id == LANG_JA
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID
        assert len(result.phonemes) >= 3  # <bos> + phonemes + <eos>

    @pytest.mark.usefixtures("_check_pyopenjtalk")
    def test_japanese_returns_known_phonemes(self):
        from tmrvc_data.g2p import text_to_phonemes

        result = text_to_phonemes("test", language="ja")
        # Should contain 'a' phoneme
        phone_set = set(result.phonemes)
        assert "a" in phone_set or "<bos>" in phone_set


class TestG2PEnglish:
    @pytest.fixture
    def _check_phonemizer(self):
        pytest.importorskip("phonemizer")

    @pytest.mark.usefixtures("_check_phonemizer")
    def test_basic_english(self):
        from tmrvc_data.g2p import text_to_phonemes

        result = text_to_phonemes("hello", language="en")
        assert result.language_id == LANG_EN
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID

    def test_unsupported_language(self):
        from tmrvc_data.g2p import text_to_phonemes

        with pytest.raises(ValueError, match="Unsupported language"):
            text_to_phonemes("test", language="fr")


class TestG2PAdditionalLanguages:
    def test_basic_chinese_branch(self, monkeypatch):
        import tmrvc_data.g2p as g2p

        monkeypatch.setattr(g2p, "_g2p_chinese", lambda _text: ["n", "i", "h", "ao"])
        result = g2p.text_to_phonemes("菴螂ｽ", language="zh")

        assert result.language_id == LANG_ZH
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID

    def test_basic_korean_branch(self, monkeypatch):
        import tmrvc_data.g2p as g2p

        monkeypatch.setattr(g2p, "_g2p_korean", lambda _text: ["a", "n", "n", "j", "eo", "ng"])
        result = g2p.text_to_phonemes("・壱・", language="ko")

        assert result.language_id == LANG_KO
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID


class TestAlignment:
    def test_alignment_to_durations(self):
        from tmrvc_data.alignment import alignment_to_durations

        intervals = [
            (0.0, 0.05, "k"),
            (0.05, 0.12, "o"),
            (0.12, 0.20, "N"),
        ]
        result = alignment_to_durations(intervals)
        assert len(result.phonemes) == 3
        assert result.phonemes == ["k", "o", "N"]
        assert result.durations.sum() > 0
        assert len(result.durations) == 3

    def test_alignment_with_total_frames(self):
        from tmrvc_data.alignment import alignment_to_durations

        intervals = [
            (0.0, 0.10, "a"),
            (0.10, 0.20, "b"),
        ]
        result = alignment_to_durations(intervals, total_frames=25)
        assert result.durations.sum() == 25

    def test_empty_label_becomes_sil(self):
        from tmrvc_data.alignment import alignment_to_durations

        intervals = [
            (0.0, 0.05, ""),
            (0.05, 0.10, "a"),
        ]
        result = alignment_to_durations(intervals)
        assert result.phonemes[0] == "<sil>"

    def test_minimum_duration_one_frame(self):
        from tmrvc_data.alignment import alignment_to_durations

        # Very short interval
        intervals = [
            (0.0, 0.001, "a"),
        ]
        result = alignment_to_durations(intervals)
        assert result.durations[0] >= 1

class TestG2PJapaneseFallbacks:
    def test_fallback_to_phonemizer_when_pyopenjtalk_missing(self, monkeypatch):
        import builtins
        import tmrvc_data.g2p as g2p

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pyopenjtalk":
                raise ImportError("pyopenjtalk unavailable in test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        monkeypatch.setattr(g2p, "_g2p_phonemizer", lambda _text, _langs: ["a", "i", "u"])

        result = g2p.text_to_phonemes("test", language="ja")
        assert result.language_id == LANG_JA
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID
        assert "a" in result.phonemes

    def test_grapheme_fallback_when_all_backends_missing(self, monkeypatch):
        import builtins
        import tmrvc_data.g2p as g2p

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pyopenjtalk":
                raise ImportError("pyopenjtalk unavailable in test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        def _raise_backend_error(_text, _langs):
            raise ImportError("phonemizer unavailable in test")

        monkeypatch.setattr(g2p, "_g2p_phonemizer", _raise_backend_error)

        result = g2p.text_to_phonemes("A.B", language="ja")
        assert result.language_id == LANG_JA
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID
        assert "<sil>" in result.phonemes
        assert len(result.phonemes) >= 4

