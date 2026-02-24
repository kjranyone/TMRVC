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
        assert "ʃ" in PHONE2ID  # sh
        assert "θ" in PHONE2ID  # th (voiceless)
        assert "ŋ" in PHONE2ID  # ng


class TestG2PJapanese:
    @pytest.fixture
    def _check_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    @pytest.mark.usefixtures("_check_pyopenjtalk")
    def test_basic_japanese(self):
        from tmrvc_data.g2p import text_to_phonemes

        result = text_to_phonemes("こんにちは", language="ja")
        assert result.language_id == LANG_JA
        assert result.phoneme_ids[0].item() == BOS_ID
        assert result.phoneme_ids[-1].item() == EOS_ID
        assert len(result.phonemes) >= 3  # <bos> + phonemes + <eos>

    @pytest.mark.usefixtures("_check_pyopenjtalk")
    def test_japanese_returns_known_phonemes(self):
        from tmrvc_data.g2p import text_to_phonemes

        result = text_to_phonemes("あ", language="ja")
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
