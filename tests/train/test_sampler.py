"""Tests for tmrvc_data.sampler module."""

from __future__ import annotations

from tmrvc_data.sampler import BalancedSpeakerSampler, SpeakerGroupConfig


class TestBalancedSpeakerSampler:
    def test_all_indices_emitted(self):
        speaker_ids = ["A", "A", "A", "B", "B", "C"]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42)

        indices = list(sampler)
        assert sorted(indices) == [0, 1, 2, 3, 4, 5]

    def test_length(self):
        speaker_ids = ["A", "A", "B", "B"]
        sampler = BalancedSpeakerSampler(speaker_ids)
        assert len(sampler) == 4

    def test_deterministic_with_seed(self):
        speaker_ids = ["A", "A", "B", "B", "C"]
        s1 = list(BalancedSpeakerSampler(speaker_ids, seed=0))
        s2 = list(BalancedSpeakerSampler(speaker_ids, seed=0))
        assert s1 == s2

    def test_different_seeds_differ(self):
        speaker_ids = ["A"] * 10 + ["B"] * 10
        s1 = list(BalancedSpeakerSampler(speaker_ids, seed=1))
        s2 = list(BalancedSpeakerSampler(speaker_ids, seed=2))
        assert s1 != s2

    def test_speaker_groups_weight(self):
        # A has 2 utterances, B has 2 utterances
        speaker_ids = ["A", "A", "B", "B"]
        groups = [SpeakerGroupConfig(speakers=["A"], weight=2)]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42, speaker_groups=groups)

        indices = list(sampler)
        assert sorted(indices) == [0, 1, 2, 3]

    def test_fnmatch_patterns(self):
        speaker_ids = ["moe_001", "moe_002", "normal_001"]
        groups = [SpeakerGroupConfig(speakers=["moe_*"], weight=3)]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42, speaker_groups=groups)

        indices = list(sampler)
        assert sorted(indices) == [0, 1, 2]

    def test_single_speaker(self):
        speaker_ids = ["A", "A", "A"]
        sampler = BalancedSpeakerSampler(speaker_ids, seed=42)
        indices = list(sampler)
        assert sorted(indices) == [0, 1, 2]

    def test_empty(self):
        sampler = BalancedSpeakerSampler([], seed=42)
        assert list(sampler) == []
        assert len(sampler) == 0
