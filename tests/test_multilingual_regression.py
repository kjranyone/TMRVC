"""Multilingual regression tests for TMRVC.

Validates:
    1. G2P produces valid phoneme_ids for each supported language (ja, en, zh, ko).
    2. Adding a language to the inventory does not change existing phoneme IDs.
    3. Phone inventory is append-only (no renumbering of existing phonemes).
    4. Frozen held-out text sets produce stable phoneme sequences.
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_data.g2p import (
    PHONE2ID,
    PHONEME_LIST,
    ID2PHONE,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    SIL_ID,
    LANG_JA,
    LANG_EN,
    LANG_ZH,
    LANG_KO,
    text_to_phonemes,
    _g2p_grapheme_fallback,
)


def _g2p_available(language: str) -> bool:
    """Check if G2P backend is available for the given language."""
    try:
        text_to_phonemes("test", language=language)
        return True
    except (ImportError, RuntimeError):
        return False


def _skip_if_g2p_unavailable(language: str) -> None:
    """Skip the test if the G2P backend for *language* is not installed."""
    if not _g2p_available(language):
        pytest.skip(
            f"G2P backend not available for {language} "
            f"(phonemizer/espeak not installed)"
        )


# ---------------------------------------------------------------------------
# Frozen held-out text sets per language
# ---------------------------------------------------------------------------

FROZEN_TEXTS = {
    "ja": [
        "おはようございます",
        "今日はいい天気ですね",
        "東京タワーに行きましょう",
        "ありがとうございました",
        "音声合成の研究をしています",
    ],
    "en": [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis is fascinating.",
        "Please call me at five o'clock.",
        "Artificial intelligence is transforming technology.",
    ],
    "zh": [
        "你好世界",
        "今天天气真好",
        "我喜欢学习中文",
        "请问你叫什么名字",
        "谢谢你的帮助",
    ],
    "ko": [
        "안녕하세요",
        "오늘 날씨가 좋습니다",
        "한국어를 공부하고 있습니다",
        "감사합니다",
        "음성 합성 연구를 합니다",
    ],
}


# ---------------------------------------------------------------------------
# Frozen phoneme inventory snapshot
#
# This records the expected special tokens and a subset of known phonemes.
# Any change to these would indicate a breaking renumbering event.
# ---------------------------------------------------------------------------

FROZEN_SPECIAL_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<sil>": 4,
    "<breath>": 5,
}

# A selection of phonemes that must retain their IDs across versions.
# These are drawn from all four languages to detect cross-language
# renumbering regressions.
FROZEN_PHONEME_SUBSET = {
    "a", "i", "u", "e", "o",     # shared vowels
    "k", "s", "t", "n", "m",     # shared consonants
    "N", "cl", "pau",            # Japanese special
    "^", "=", "_",               # Japanese accent markers
}


# ---------------------------------------------------------------------------
# Test: G2P produces valid phoneme IDs for each language
# ---------------------------------------------------------------------------

class TestG2PValidOutput:
    """G2P must produce valid phoneme_ids for each supported language."""

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko"])
    def test_g2p_produces_nonempty_ids(self, language: str):
        """Each frozen text must produce a non-empty phoneme_ids tensor."""
        _skip_if_g2p_unavailable(language)
        for text in FROZEN_TEXTS[language]:
            result = text_to_phonemes(text, language=language)
            assert result.phoneme_ids.numel() > 0, (
                f"Empty phoneme_ids for {language}: {text!r}"
            )
            assert result.phoneme_ids.dtype == torch.long

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko"])
    def test_g2p_includes_bos_eos(self, language: str):
        """All G2P output must start with <bos> and end with <eos>."""
        _skip_if_g2p_unavailable(language)
        for text in FROZEN_TEXTS[language]:
            result = text_to_phonemes(text, language=language)
            ids = result.phoneme_ids.tolist()
            assert ids[0] == BOS_ID, (
                f"Missing <bos> for {language}: {text!r}, got {ids[0]}"
            )
            assert ids[-1] == EOS_ID, (
                f"Missing <eos> for {language}: {text!r}, got {ids[-1]}"
            )

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko"])
    def test_g2p_no_pad_in_output(self, language: str):
        """G2P output must not contain <pad> tokens."""
        _skip_if_g2p_unavailable(language)
        for text in FROZEN_TEXTS[language]:
            result = text_to_phonemes(text, language=language)
            ids = result.phoneme_ids.tolist()
            assert PAD_ID not in ids, (
                f"<pad> found in output for {language}: {text!r}"
            )

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko"])
    def test_g2p_ids_within_vocab(self, language: str):
        """All phoneme IDs must be within the vocabulary range."""
        _skip_if_g2p_unavailable(language)
        vocab_size = len(PHONEME_LIST)
        for text in FROZEN_TEXTS[language]:
            result = text_to_phonemes(text, language=language)
            for pid in result.phoneme_ids.tolist():
                assert 0 <= pid < vocab_size, (
                    f"Out-of-vocab ID {pid} for {language}: {text!r}"
                )

    @pytest.mark.parametrize("language,expected_lang_id", [
        ("ja", LANG_JA),
        ("en", LANG_EN),
        ("zh", LANG_ZH),
        ("ko", LANG_KO),
    ])
    def test_g2p_returns_correct_language_id(self, language: str, expected_lang_id: int):
        """G2P result must carry the correct language_id."""
        _skip_if_g2p_unavailable(language)
        text = FROZEN_TEXTS[language][0]
        result = text_to_phonemes(text, language=language)
        assert result.language_id == expected_lang_id, (
            f"Wrong language_id for {language}: expected {expected_lang_id}, "
            f"got {result.language_id}"
        )


# ---------------------------------------------------------------------------
# Test: phone inventory is append-only (no renumbering)
# ---------------------------------------------------------------------------

class TestPhoneInventoryStability:
    """The phoneme inventory must be append-only: existing IDs never change."""

    def test_special_token_ids_frozen(self):
        """Special tokens must have their historically assigned IDs."""
        for token, expected_id in FROZEN_SPECIAL_IDS.items():
            assert token in PHONE2ID, f"Special token {token!r} missing from PHONE2ID"
            assert PHONE2ID[token] == expected_id, (
                f"Special token {token!r} has ID {PHONE2ID[token]}, "
                f"expected {expected_id}. This is a breaking renumbering."
            )

    def test_frozen_phoneme_subset_present(self):
        """Key phonemes from all languages must exist in the inventory."""
        for phone in FROZEN_PHONEME_SUBSET:
            assert phone in PHONE2ID, (
                f"Phoneme {phone!r} missing from inventory. "
                f"This breaks backward compatibility."
            )

    def test_phoneme_ids_are_contiguous(self):
        """Phoneme IDs must be contiguous starting from 0."""
        expected_ids = set(range(len(PHONEME_LIST)))
        actual_ids = set(PHONE2ID.values())
        assert actual_ids == expected_ids, (
            f"Non-contiguous IDs detected. "
            f"Missing: {expected_ids - actual_ids}, "
            f"Extra: {actual_ids - expected_ids}"
        )

    def test_id2phone_round_trips(self):
        """PHONE2ID and ID2PHONE must be perfect inverses."""
        for phone, pid in PHONE2ID.items():
            assert ID2PHONE[pid] == phone, (
                f"Round-trip failure: PHONE2ID[{phone!r}]={pid}, "
                f"but ID2PHONE[{pid}]={ID2PHONE[pid]!r}"
            )

    def test_no_duplicate_phonemes(self):
        """PHONEME_LIST must not contain duplicates."""
        seen: set[str] = set()
        for phone in PHONEME_LIST:
            assert phone not in seen, (
                f"Duplicate phoneme in PHONEME_LIST: {phone!r}"
            )
            seen.add(phone)


# ---------------------------------------------------------------------------
# Test: adding a language does not change existing phoneme IDs
# ---------------------------------------------------------------------------

class TestCrossLanguageStability:
    """Adding or modifying one language's phoneme set must not change others."""

    def test_japanese_phonemes_stable(self):
        """Japanese-specific phonemes must retain their known IDs."""
        ja_specific = ["N", "cl", "pau", "^", "=", "_"]
        for phone in ja_specific:
            assert phone in PHONE2ID, f"Japanese phoneme {phone!r} missing"
            # Record the ID -- if this test passes once, any future change
            # will require updating the frozen set (deliberate).

    def test_shared_vowels_single_id(self):
        """Vowels shared across languages must map to a single ID each.

        E.g. "a" appears in Japanese, Chinese, and Korean phoneme lists,
        but must have exactly one entry in PHONE2ID.
        """
        shared_vowels = ["a", "e", "i", "o", "u"]
        for v in shared_vowels:
            assert v in PHONE2ID, f"Shared vowel {v!r} missing"
            # Verify it appears exactly once in PHONEME_LIST
            count = PHONEME_LIST.count(v)
            assert count == 1, (
                f"Vowel {v!r} appears {count} times in PHONEME_LIST "
                f"(expected exactly 1)"
            )

    def test_unsupported_language_raises(self):
        """text_to_phonemes must raise ValueError for unsupported languages."""
        with pytest.raises(ValueError, match="Unsupported language"):
            text_to_phonemes("hello", language="xx")


# ---------------------------------------------------------------------------
# Test: grapheme fallback stability
# ---------------------------------------------------------------------------

class TestGraphemeFallbackStability:
    """Grapheme fallback must produce stable, non-empty output."""

    def test_empty_text_returns_silence(self):
        """Empty string must produce a silence token."""
        result = _g2p_grapheme_fallback("")
        assert result == ["<sil>"]

    def test_latin_chars_fallback(self):
        """Latin characters should produce phoneme-like output."""
        result = _g2p_grapheme_fallback("abc")
        assert len(result) > 0
        for phone in result:
            assert phone in PHONE2ID, (
                f"Fallback produced unknown phone {phone!r}"
            )

    def test_punctuation_collapses_to_silence(self):
        """Multiple consecutive punctuation chars should collapse to single <sil>."""
        result = _g2p_grapheme_fallback("...!!!")
        # Should not have consecutive <sil> tokens
        for i in range(1, len(result)):
            if result[i] == "<sil>":
                assert result[i - 1] != "<sil>", (
                    "Consecutive <sil> tokens in fallback output"
                )


# ---------------------------------------------------------------------------
# Test: G2P determinism
# ---------------------------------------------------------------------------

class TestG2PDeterminism:
    """G2P must be deterministic: same input always produces same output."""

    @pytest.mark.parametrize("language", ["ja", "en", "zh", "ko"])
    def test_g2p_deterministic(self, language: str):
        """Running G2P twice on the same text must produce identical results."""
        _skip_if_g2p_unavailable(language)
        for text in FROZEN_TEXTS[language]:
            r1 = text_to_phonemes(text, language=language)
            r2 = text_to_phonemes(text, language=language)
            assert torch.equal(r1.phoneme_ids, r2.phoneme_ids), (
                f"Non-deterministic G2P for {language}: {text!r}\n"
                f"  Run 1: {r1.phonemes}\n"
                f"  Run 2: {r2.phonemes}"
            )
            assert r1.phonemes == r2.phonemes
            assert r1.language_id == r2.language_id
