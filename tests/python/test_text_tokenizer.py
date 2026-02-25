"""Tests for tokenizer-first text frontend."""

from __future__ import annotations

import pytest

from tmrvc_data.text_tokenizer import (
    BOS_ID,
    EOS_ID,
    TOKENIZER_VOCAB_SIZE,
    text_to_tokens,
)


class TestTextTokenizer:
    def test_vocab_size(self):
        assert TOKENIZER_VOCAB_SIZE == 262

    def test_basic_tokenization(self):
        result = text_to_tokens("hello", language="en")
        assert result.language_id == 1
        assert int(result.token_ids[0].item()) == BOS_ID
        assert int(result.token_ids[-1].item()) == EOS_ID
        assert len(result.token_ids) >= 3

    def test_multilingual_language_ids(self):
        assert text_to_tokens("a", language="ja").language_id == 0
        assert text_to_tokens("a", language="en").language_id == 1
        assert text_to_tokens("a", language="zh").language_id == 2
        assert text_to_tokens("a", language="ko").language_id == 3

    def test_empty_text_inserts_sil(self):
        result = text_to_tokens("", language="ja")
        # <bos>, <sil>, <eos>
        assert len(result.token_ids) == 3

    def test_unsupported_language(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            text_to_tokens("hello", language="fr")

