"""Tests for bootstrap alignment projection (Worker 03)."""

from __future__ import annotations

import math

import pytest

from tmrvc_data.bootstrap_alignment import (
    AlignmentSpan,
    BootstrapAlignment,
    samples_to_frames,
    seconds_to_frame,
    project_word_timestamps,
    SAMPLE_RATE,
    HOP_LENGTH,
)


class TestFrameConvention:
    """Verify frozen frame convention matches tmrvc-core."""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 24000

    def test_hop_length(self):
        assert HOP_LENGTH == 240

    def test_samples_to_frames_exact(self):
        assert samples_to_frames(24000) == 100

    def test_samples_to_frames_ceil(self):
        assert samples_to_frames(24001) == 101
        assert samples_to_frames(23999) == 100

    def test_seconds_to_frame(self):
        # 0.5s = 12000 samples => frame 50
        assert seconds_to_frame(0.5) == 50
        assert seconds_to_frame(0.0) == 0
        assert seconds_to_frame(1.0) == 100


class TestBootstrapAlignmentValidation:
    def test_valid_alignment(self):
        spans = [
            AlignmentSpan(text_unit_index=0, start_frame=0, end_frame=50),
            AlignmentSpan(text_unit_index=1, start_frame=50, end_frame=100),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=2, num_frames=100)
        errors = ba.validate()
        assert not errors

    def test_non_monotonic_detected(self):
        spans = [
            AlignmentSpan(text_unit_index=1, start_frame=0, end_frame=50),
            AlignmentSpan(text_unit_index=0, start_frame=50, end_frame=100),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=2, num_frames=100)
        errors = ba.validate()
        assert any("non-monotonic" in e for e in errors)

    def test_overlapping_frames_detected(self):
        spans = [
            AlignmentSpan(text_unit_index=0, start_frame=0, end_frame=60),
            AlignmentSpan(text_unit_index=1, start_frame=50, end_frame=100),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=2, num_frames=100)
        errors = ba.validate()
        assert any("overlapping" in e for e in errors)

    def test_missing_indices_detected(self):
        spans = [
            AlignmentSpan(text_unit_index=0, start_frame=0, end_frame=50),
            # index 1 missing
            AlignmentSpan(text_unit_index=2, start_frame=50, end_frame=100),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=3, num_frames=100)
        errors = ba.validate()
        assert any("missing" in e for e in errors)

    def test_exceeding_num_frames_detected(self):
        spans = [
            AlignmentSpan(text_unit_index=0, start_frame=0, end_frame=150),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=1, num_frames=100)
        errors = ba.validate()
        assert any("exceeds" in e for e in errors)

    def test_zero_length_span_detected(self):
        spans = [
            AlignmentSpan(text_unit_index=0, start_frame=50, end_frame=50),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=1, num_frames=100)
        errors = ba.validate()
        assert any("zero-length" in e for e in errors)

    def test_empty_alignment_fails(self):
        ba = BootstrapAlignment(spans=[], num_text_units=5, num_frames=100)
        errors = ba.validate()
        assert any("no spans" in e for e in errors)


class TestBootstrapAlignmentSerialization:
    def test_roundtrip(self):
        spans = [
            AlignmentSpan(text_unit_index=0, start_frame=0, end_frame=50, confidence=0.9),
            AlignmentSpan(text_unit_index=1, start_frame=50, end_frame=100, confidence=0.8),
        ]
        ba = BootstrapAlignment(spans=spans, num_text_units=2, num_frames=100)
        d = ba.to_dict()
        ba2 = BootstrapAlignment.from_dict(d)
        assert len(ba2.spans) == 2
        assert ba2.spans[0].confidence == 0.9
        assert ba2.num_frames == 100


class TestProjectWordTimestamps:
    def test_basic_projection(self):
        word_timestamps = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        phoneme_ids = [10, 11, 12, 20, 21, 22]  # 3 phones per word
        word_to_phoneme_map = [(0, 3), (3, 6)]
        num_samples = 24000

        ba = project_word_timestamps(
            word_timestamps, phoneme_ids, word_to_phoneme_map, num_samples
        )
        assert len(ba.spans) == 6
        assert ba.num_text_units == 6
        assert ba.num_frames == 100

        # Check monotonicity
        errors = ba.validate()
        assert not errors

    def test_projection_is_deterministic(self):
        word_timestamps = [
            {"word": "a", "start": 0.0, "end": 0.3},
            {"word": "b", "start": 0.3, "end": 0.7},
        ]
        phoneme_ids = [1, 2, 3, 4]
        word_to_phoneme_map = [(0, 2), (2, 4)]
        num_samples = 16800

        ba1 = project_word_timestamps(
            word_timestamps, phoneme_ids, word_to_phoneme_map, num_samples
        )
        ba2 = project_word_timestamps(
            word_timestamps, phoneme_ids, word_to_phoneme_map, num_samples
        )
        assert ba1.to_dict() == ba2.to_dict()

    def test_projection_covers_all_phones(self):
        word_timestamps = [
            {"word": "test", "start": 0.0, "end": 1.0},
        ]
        phoneme_ids = [1, 2, 3, 4, 5]
        word_to_phoneme_map = [(0, 5)]
        num_samples = 24000

        ba = project_word_timestamps(
            word_timestamps, phoneme_ids, word_to_phoneme_map, num_samples
        )
        covered = {s.text_unit_index for s in ba.spans}
        assert covered == {0, 1, 2, 3, 4}
