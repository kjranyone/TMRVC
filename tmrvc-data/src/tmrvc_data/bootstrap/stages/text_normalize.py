"""Stage 8: Text normalization and G2P phoneme ID generation.

Normalises the transcript text and converts it to phoneme IDs using
the existing G2P module (tmrvc_data.g2p).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)

# Common text normalisation patterns
_MULTI_SPACE = re.compile(r"\s+")
_URL_PATTERN = re.compile(r"https?://\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")


class TextNormalizeStage:
    """Text normalization + G2P phoneme_ids generation.

    Normalises the transcript text (clean whitespace, expand common
    abbreviations, remove URLs) then converts to phoneme IDs using
    ``tmrvc_data.g2p.text_to_phonemes``.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Normalise text and generate phoneme IDs for each utterance."""
        for utt in utterances:
            if utt.is_rejected:
                utt.stage_completed = BootstrapStage.TEXT_NORMALIZATION
                continue

            if not utt.text_transcript:
                utt.phoneme_ids = np.array([], dtype=np.int64)
                utt.stage_completed = BootstrapStage.TEXT_NORMALIZATION
                continue

            try:
                # 1. Normalise the text
                normalised = self._normalize_text(
                    utt.text_transcript, language=utt.language,
                )

                # 2. Convert to phoneme IDs
                phoneme_ids = self._text_to_phoneme_ids(
                    normalised, language=utt.language,
                )
                utt.phoneme_ids = phoneme_ids

                # Update transcript with normalised form
                utt.text_transcript = normalised

            except Exception as exc:
                logger.warning(
                    "Text normalize failed for %s: %s",
                    utt.utterance_id, exc,
                )
                utt.phoneme_ids = np.array([], dtype=np.int64)
                utt.warnings.append(f"text_normalize_error:{exc}")

            utt.stage_completed = BootstrapStage.TEXT_NORMALIZATION

        logger.info("TextNormalize: processed %d utterances", len(utterances))
        return utterances

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str, language: str = "") -> str:
        """Normalise transcript text.

        - Strip leading/trailing whitespace
        - Collapse multiple spaces
        - Remove URLs and email addresses
        - Language-specific normalisation
        """
        text = text.strip()

        # Remove URLs and emails
        text = _URL_PATTERN.sub("", text)
        text = _EMAIL_PATTERN.sub("", text)

        # Collapse whitespace
        text = _MULTI_SPACE.sub(" ", text).strip()

        # Language-specific normalization
        if language == "ja":
            text = TextNormalizeStage._normalize_japanese(text)
        elif language == "en":
            text = TextNormalizeStage._normalize_english(text)
        elif language == "zh":
            text = TextNormalizeStage._normalize_chinese(text)

        return text

    @staticmethod
    def _normalize_japanese(text: str) -> str:
        """Japanese-specific text normalization."""
        # Convert full-width ASCII to half-width
        result = []
        for ch in text:
            cp = ord(ch)
            if 0xFF01 <= cp <= 0xFF5E:
                result.append(chr(cp - 0xFEE0))
            elif ch == '\u3000':  # Full-width space
                result.append(' ')
            else:
                result.append(ch)
        text = "".join(result)

        # Normalize common Japanese punctuation
        text = text.replace("？", "?").replace("！", "!")
        text = text.replace("。", "。").replace("、", "、")

        return text

    @staticmethod
    def _normalize_english(text: str) -> str:
        """English-specific text normalization."""
        # Expand common abbreviations
        replacements = {
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Dr.": "Doctor",
            "St.": "Street",
            "etc.": "etcetera",
            "vs.": "versus",
        }
        for abbr, expansion in replacements.items():
            text = text.replace(abbr, expansion)

        return text

    @staticmethod
    def _normalize_chinese(text: str) -> str:
        """Chinese-specific text normalization."""
        # Convert traditional Chinese punctuation to simplified equivalents
        text = text.replace("\u300C", "\u201C").replace("\u300D", "\u201D")
        text = text.replace("\u300E", "\u201C").replace("\u300F", "\u201D")
        return text

    # ------------------------------------------------------------------
    # G2P conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _text_to_phoneme_ids(
        text: str, language: str = "",
    ) -> np.ndarray:
        """Convert normalised text to phoneme IDs using the G2P module.

        Falls back to a simple character-level tokenisation if G2P
        backends are unavailable.
        """
        # Map language codes to G2P expected format
        lang_map = {
            "ja": "ja",
            "en": "en",
            "zh": "zh",
            "ko": "ko",
            "japanese": "ja",
            "english": "en",
            "chinese": "zh",
            "korean": "ko",
        }
        g2p_lang = lang_map.get(language, "en")  # Default to English

        try:
            from tmrvc_data.g2p import text_to_phonemes

            result = text_to_phonemes(text, language=g2p_lang)
            return result.phoneme_ids.numpy().astype(np.int64)
        except Exception as exc:
            logger.debug(
                "G2P conversion failed (%s), using character fallback: %s",
                g2p_lang, exc,
            )

        # Fallback: simple character-to-ID mapping
        try:
            from tmrvc_data.g2p import PHONE2ID, UNK_ID, BOS_ID, EOS_ID

            ids = [BOS_ID]
            for ch in text:
                ch_lower = ch.lower()
                if ch_lower in PHONE2ID:
                    ids.append(PHONE2ID[ch_lower])
                elif ch == " ":
                    from tmrvc_data.g2p import SIL_ID
                    ids.append(SIL_ID)
                else:
                    ids.append(UNK_ID)
            ids.append(EOS_ID)
            return np.array(ids, dtype=np.int64)
        except ImportError:
            # Ultimate fallback: ordinal values
            ids = [ord(ch) % 200 for ch in text]
            return np.array(ids, dtype=np.int64)
