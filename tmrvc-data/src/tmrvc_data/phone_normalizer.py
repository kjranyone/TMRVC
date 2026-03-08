"""Phone normalization pipeline with alias mapping (Worker 03).

Implements a deterministic 5-stage normalization:
1. Unicode normalization
2. Backend-specific cleanup
3. Alias-table lookup
4. Canonical-symbol validation
5. Fallback to UNK_ID

The alias table is loaded from configs/phoneme_aliases.yaml.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from tmrvc_data.g2p import PHONE2ID, UNK_ID

logger = logging.getLogger(__name__)

_DEFAULT_ALIASES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "phoneme_aliases.yaml"
)


@dataclass
class AliasEntry:
    """A single alias mapping entry."""

    source_symbol: str
    normalized_symbol: str
    canonical_symbol: str
    language: str
    source_backend: str
    action: Literal["map", "drop", "unk"]
    note: str = ""


@dataclass
class NormalizationStats:
    """Accumulated statistics from phone normalization."""

    total_phones: int = 0
    direct_hits: int = 0
    alias_hits: int = 0
    drops: int = 0
    unk_fallbacks: int = 0
    unmapped_counts: dict[str, int] = field(default_factory=dict)

    @property
    def direct_hit_ratio(self) -> float:
        return self.direct_hits / max(self.total_phones, 1)

    @property
    def alias_hit_ratio(self) -> float:
        return self.alias_hits / max(self.total_phones, 1)

    @property
    def unk_ratio(self) -> float:
        return self.unk_fallbacks / max(self.total_phones, 1)

    def top_unmapped(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(self.unmapped_counts.items(), key=lambda x: -x[1])[:n]


class PhoneNormalizer:
    """Deterministic phone normalization pipeline.

    Loads alias entries from YAML and applies the 5-stage normalization
    to map external phone symbols into the canonical PHONE2ID inventory.
    """

    def __init__(self, aliases_path: str | Path | None = None) -> None:
        self.aliases_path = Path(aliases_path) if aliases_path else _DEFAULT_ALIASES_PATH
        self.entries: list[AliasEntry] = []
        self._lookup: dict[tuple[str, str, str], AliasEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.aliases_path.exists():
            logger.warning("Alias file not found: %s", self.aliases_path)
            return
        with open(self.aliases_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for entry_dict in data.get("aliases", []):
            entry = AliasEntry(**entry_dict)
            self.entries.append(entry)
            # Key: (normalized_symbol, language, source_backend)
            # Also register with "all" wildcards
            self._lookup[(entry.normalized_symbol, entry.language, entry.source_backend)] = entry

    def _find_alias(
        self, normalized: str, language: str, backend: str
    ) -> AliasEntry | None:
        """Look up alias with fallback to wildcard language/backend."""
        for lang in (language, "all"):
            for be in (backend, "all"):
                key = (normalized, lang, be)
                if key in self._lookup:
                    return self._lookup[key]
        return None

    def normalize_phone(
        self,
        raw_symbol: str,
        language: str = "all",
        backend: str = "all",
    ) -> tuple[int | None, str]:
        """Normalize a single phone symbol to canonical ID.

        Returns:
            (phone_id, action) where action is "direct", "alias", "drop", or "unk".
            phone_id is None when action is "drop".
        """
        # Stage 1: Unicode normalization
        normalized = unicodedata.normalize("NFC", raw_symbol)

        # Stage 2: Backend-specific cleanup (strip whitespace, lowercase for non-IPA)
        normalized = normalized.strip()

        # Stage 3: Direct lookup in canonical inventory
        if normalized in PHONE2ID:
            return PHONE2ID[normalized], "direct"

        # Stage 4: Alias-table lookup
        alias = self._find_alias(normalized, language, backend)
        if alias is not None:
            if alias.action == "drop":
                return None, "drop"
            if alias.action == "unk":
                return UNK_ID, "unk"
            # action == "map"
            canonical = alias.canonical_symbol
            if canonical in PHONE2ID:
                return PHONE2ID[canonical], "alias"
            logger.warning(
                "Alias maps %r -> %r but %r not in PHONE2ID",
                raw_symbol, canonical, canonical,
            )
            return UNK_ID, "unk"

        # Stage 5: Fallback to UNK
        return UNK_ID, "unk"

    def normalize_sequence(
        self,
        phones: list[str],
        language: str = "all",
        backend: str = "all",
        stats: NormalizationStats | None = None,
    ) -> list[int]:
        """Normalize a sequence of phone symbols to canonical IDs.

        Dropped phones are omitted from the output.
        """
        if stats is None:
            stats = NormalizationStats()
        ids: list[int] = []
        for phone in phones:
            stats.total_phones += 1
            pid, action = self.normalize_phone(phone, language, backend)
            if action == "direct":
                stats.direct_hits += 1
                ids.append(pid)  # type: ignore[arg-type]
            elif action == "alias":
                stats.alias_hits += 1
                ids.append(pid)  # type: ignore[arg-type]
            elif action == "drop":
                stats.drops += 1
            else:  # unk
                stats.unk_fallbacks += 1
                stats.unmapped_counts[phone] = stats.unmapped_counts.get(phone, 0) + 1
                ids.append(pid)  # type: ignore[arg-type]
        return ids
