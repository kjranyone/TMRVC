"""Tests for phone normalization pipeline (Worker 03)."""

from __future__ import annotations

import pytest

from tmrvc_data.phone_normalizer import (
    PhoneNormalizer,
    NormalizationStats,
)
from tmrvc_data.g2p import PHONE2ID, UNK_ID


class TestPhoneNormalizerDirect:
    """Test direct canonical hits."""

    @pytest.fixture
    def normalizer(self):
        return PhoneNormalizer()

    def test_canonical_phone_direct_hit(self, normalizer):
        pid, action = normalizer.normalize_phone("a")
        assert action == "direct"
        assert pid == PHONE2ID["a"]

    def test_canonical_special_direct_hit(self, normalizer):
        pid, action = normalizer.normalize_phone("<sil>")
        assert action == "direct"
        assert pid == PHONE2ID["<sil>"]

    def test_unknown_phone_returns_unk(self, normalizer):
        pid, action = normalizer.normalize_phone("ZZZZZ_nonexistent")
        assert action == "unk"
        assert pid == UNK_ID


class TestPhoneNormalizerAlias:
    """Test alias mapping from phoneme_aliases.yaml."""

    @pytest.fixture
    def normalizer(self):
        return PhoneNormalizer()

    def test_sil_maps_to_pau(self, normalizer):
        pid, action = normalizer.normalize_phone("sil", language="all")
        assert action == "alias"
        assert pid == PHONE2ID["pau"]

    def test_sp_maps_to_pau(self, normalizer):
        pid, action = normalizer.normalize_phone("sp", language="all")
        assert action == "alias"
        assert pid == PHONE2ID["pau"]

    def test_spn_dropped(self, normalizer):
        pid, action = normalizer.normalize_phone("spn", backend="mfa")
        assert action == "drop"
        assert pid is None


class TestNormalizationStats:
    """Test statistics accumulation."""

    @pytest.fixture
    def normalizer(self):
        return PhoneNormalizer()

    def test_sequence_stats(self, normalizer):
        phones = ["a", "sil", "ZZZZZ", "k", "spn"]
        stats = NormalizationStats()
        ids = normalizer.normalize_sequence(
            phones, language="all", backend="mfa", stats=stats
        )
        assert stats.total_phones == 5
        assert stats.direct_hits == 2  # a, k
        assert stats.alias_hits == 1  # sil -> pau
        assert stats.drops == 1  # spn
        assert stats.unk_fallbacks == 1  # ZZZZZ
        assert len(ids) == 4  # dropped spn

    def test_ratios(self):
        stats = NormalizationStats(
            total_phones=100, direct_hits=80, alias_hits=10, unk_fallbacks=10
        )
        assert abs(stats.direct_hit_ratio - 0.8) < 1e-6
        assert abs(stats.alias_hit_ratio - 0.1) < 1e-6
        assert abs(stats.unk_ratio - 0.1) < 1e-6

    def test_top_unmapped(self):
        stats = NormalizationStats(
            unmapped_counts={"X": 10, "Y": 5, "Z": 20}
        )
        top = stats.top_unmapped(2)
        assert top[0] == ("Z", 20)
        assert top[1] == ("X", 10)


class TestAliasMappingDeterminism:
    """Alias mapping must be deterministic."""

    def test_same_input_same_output(self):
        n1 = PhoneNormalizer()
        n2 = PhoneNormalizer()
        phones = ["a", "sil", "sp", "k", "ZZZZZ"]
        ids1 = n1.normalize_sequence(phones)
        ids2 = n2.normalize_sequence(phones)
        assert ids1 == ids2
