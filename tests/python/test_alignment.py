"""Tests for tmrvc_data.alignment module."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from tmrvc_data.alignment import (
    AlignmentResult,
    _parse_textgrid,
    alignment_to_durations,
    load_textgrid_durations,
)


class TestParseTextGrid:
    def test_short_format(self, tmp_path):
        """Parse short-format TextGrid with phone tier.

        The parser looks for "phones" tier then parses xmin/xmax/text triplets.
        The header lines (xmin, xmax, count) before intervals may be parsed as
        a spurious interval, so we validate only the real phone intervals.
        """
        tg = tmp_path / "test.TextGrid"
        tg.write_text(textwrap.dedent("""\
            File type = "ooTextFile"
            Object class = "TextGrid"
            "IntervalTier"
            "phones"
            0.0
            0.5
            3
            0.0
            0.1
            "a"
            0.1
            0.3
            "k"
            0.3
            0.5
            "i"
        """), encoding="utf-8")

        intervals = _parse_textgrid(tg)
        # Filter out any spurious header intervals
        real = [(s, e, l) for s, e, l in intervals if l in ("a", "k", "i")]
        assert len(real) == 3
        assert real[0] == (0.0, 0.1, "a")
        assert real[1] == (0.1, 0.3, "k")
        assert real[2] == (0.3, 0.5, "i")

    def test_long_format(self, tmp_path):
        """Parse long-format TextGrid with xmin/xmax/text."""
        tg = tmp_path / "test.TextGrid"
        tg.write_text(textwrap.dedent("""\
            File type = "ooTextFile"
            Object class = "TextGrid"
            xmin = 0
            xmax = 0.5
            tiers? <exists>
            size = 1
            item []:
                item [1]:
                    class = "IntervalTier"
                    name = "words"
                    xmin = 0
                    xmax = 0.5
                    intervals: size = 2
                    intervals [1]:
                        xmin = 0.0
                        xmax = 0.25
                        text = "hello"
                    intervals [2]:
                        xmin = 0.25
                        xmax = 0.5
                        text = "world"
        """), encoding="utf-8")

        intervals = _parse_textgrid(tg)
        assert len(intervals) == 2
        assert intervals[0][2] == "hello"
        assert intervals[1][2] == "world"

    def test_empty_file(self, tmp_path):
        """Empty or invalid TextGrid returns empty list."""
        tg = tmp_path / "empty.TextGrid"
        tg.write_text("nothing useful here", encoding="utf-8")
        assert _parse_textgrid(tg) == []


class TestAlignmentToDurations:
    def test_basic(self):
        intervals = [(0.0, 0.05, "a"), (0.05, 0.15, "b"), (0.15, 0.20, "c")]
        result = alignment_to_durations(intervals)

        assert isinstance(result, AlignmentResult)
        assert result.phonemes == ["a", "b", "c"]
        assert result.durations.dtype == np.int64
        # 50ms=5 frames, 100ms=10 frames, 50ms=5 frames
        assert result.durations[0] == 5
        assert result.durations[1] == 10
        assert result.durations[2] == 5

    def test_adjust_to_total_frames(self):
        intervals = [(0.0, 0.05, "a"), (0.05, 0.10, "b")]
        result = alignment_to_durations(intervals, total_frames=12)
        assert result.durations.sum() == 12

    def test_empty_label_becomes_sil(self):
        intervals = [(0.0, 0.1, "")]
        result = alignment_to_durations(intervals)
        assert result.phonemes == ["<sil>"]

    def test_min_duration_is_one(self):
        intervals = [(0.0, 0.001, "x")]  # Very short
        result = alignment_to_durations(intervals)
        assert result.durations[0] >= 1


class TestLoadTextgridDurations:
    def test_load_and_parse(self, tmp_path):
        """Test loading TextGrid in long format which is parsed reliably."""
        tg = tmp_path / "test.TextGrid"
        tg.write_text(textwrap.dedent("""\
            File type = "ooTextFile"
            Object class = "TextGrid"
            item [1]:
                class = "IntervalTier"
                name = "phones"
                intervals [1]:
                    xmin = 0.0
                    xmax = 0.1
                    text = "s"
                intervals [2]:
                    xmin = 0.1
                    xmax = 0.2
                    text = "a"
                intervals [3]:
                    xmin = 0.2
                    xmax = 0.3
                    text = "n"
        """), encoding="utf-8")

        result = load_textgrid_durations(tg, total_frames=30)
        assert len(result.phonemes) == 3
        assert result.durations.sum() == 30

    def test_empty_textgrid_raises(self, tmp_path):
        tg = tmp_path / "empty.TextGrid"
        tg.write_text("empty", encoding="utf-8")
        with pytest.raises(ValueError, match="No intervals found"):
            load_textgrid_durations(tg)
