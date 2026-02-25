"""Tokenizer-first text frontend for multilingual TTS.

This module provides a lightweight byte-level tokenizer that does not
depend on external G2P backends. It is intended as a robust baseline for
E2E tokenizer-based TTS training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tmrvc_data.g2p import LANG_EN, LANG_JA, LANG_KO, LANG_ZH

# Keep special IDs aligned with g2p.py for seamless model reuse.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SIL_ID = 4
BREATH_ID = 5

_BYTE_OFFSET = 6
TOKENIZER_VOCAB_SIZE = _BYTE_OFFSET + 256  # 262

_LANG_TO_ID = {
    "ja": LANG_JA,
    "en": LANG_EN,
    "zh": LANG_ZH,
    "ko": LANG_KO,
}


@dataclass
class TokenizerResult:
    """Result of tokenizer-based text frontend."""

    token_ids: torch.Tensor  # [L] int64
    tokens: list[str]
    language_id: int


def text_to_tokens(
    text: str,
    language: str = "ja",
) -> TokenizerResult:
    """Convert text to byte-level token IDs.

    Args:
        text: Raw input text.
        language: One of ``ja/en/zh/ko``.
    """
    if language not in _LANG_TO_ID:
        raise ValueError(f"Unsupported language: {language}")

    payload = text.encode("utf-8")
    token_ids = [BOS_ID]
    tokens = ["<bos>"]

    if not payload:
        token_ids.append(SIL_ID)
        tokens.append("<sil>")
    else:
        for b in payload:
            token_ids.append(_BYTE_OFFSET + int(b))
            tokens.append(f"<0x{b:02x}>")

    token_ids.append(EOS_ID)
    tokens.append("<eos>")

    return TokenizerResult(
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        tokens=tokens,
        language_id=_LANG_TO_ID[language],
    )

