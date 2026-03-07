"""Tests for Phoneme mapping, MFA alignment parity, and data model contracts.

.. deprecated:: v3
    Legacy alignment tests below are skipped. The alignment module is legacy v2 tooling.
    Non-legacy tests (CurationRecord, pseudo_annotation) run normally.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Non-legacy tests (always run)
# ---------------------------------------------------------------------------


class TestCurationRecordFields:
    """Verify CurationRecord has expected speaker clustering fields."""

    def test_speaker_cluster_field_exists(self):
        from tmrvc_data.curation.models import CurationRecord
        rec = CurationRecord(
            record_id="test_001",
            source_path="/tmp/test.wav",
            audio_hash="abc123",
        )
        assert hasattr(rec, "speaker_cluster")
        assert rec.speaker_cluster is None

    def test_diarization_confidence_field_exists(self):
        from tmrvc_data.curation.models import CurationRecord
        rec = CurationRecord(
            record_id="test_002",
            source_path="/tmp/test.wav",
            audio_hash="def456",
            speaker_cluster="spk_0",
            diarization_confidence=0.95,
        )
        assert rec.diarization_confidence == 0.95

    def test_speaker_fields_in_to_dict(self):
        from tmrvc_data.curation.models import CurationRecord
        rec = CurationRecord(
            record_id="test_003",
            source_path="/tmp/test.wav",
            audio_hash="ghi789",
            speaker_cluster="spk_1",
            diarization_confidence=0.88,
        )
        d = rec.to_dict()
        assert "speaker_cluster" in d
        assert "diarization_confidence" in d
        assert d["speaker_cluster"] == "spk_1"
        assert d["diarization_confidence"] == 0.88


class TestPseudoAnnotationQualitySummary:
    """Verify quality_summary returns expected keys."""

    def test_quality_summary_empty_results(self):
        from tmrvc_data.pseudo_annotation import quality_summary
        summary = quality_summary([])
        expected_keys = {
            "total_segments",
            "total_duration_sec",
            "mean_asr_confidence",
            "mean_quality_score",
            "mean_snr_db",
            "speaker_counts",
            "language_counts",
            "event_stats",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["total_segments"] == 0

    def test_quality_summary_with_results(self):
        from tmrvc_data.pseudo_annotation import (
            PseudoAnnotationResult,
            quality_summary,
        )
        results = [
            PseudoAnnotationResult(
                start_sec=0.0,
                end_sec=1.5,
                text="hello",
                asr_confidence=0.9,
                speaker_cluster=0,
                quality_score=0.85,
                snr_db=25.0,
                detected_language="en",
            ),
            PseudoAnnotationResult(
                start_sec=2.0,
                end_sec=3.0,
                text="world",
                asr_confidence=0.8,
                speaker_cluster=1,
                quality_score=0.7,
                snr_db=20.0,
                detected_language="en",
            ),
        ]
        summary = quality_summary(results)
        assert summary["total_segments"] == 2
        assert summary["total_duration_sec"] > 0
        assert "speaker_counts" in summary
        assert "language_counts" in summary
        assert "event_stats" in summary
        assert summary["mean_asr_confidence"] > 0


# ---------------------------------------------------------------------------
# Legacy tests (skipped in v3)
# ---------------------------------------------------------------------------

try:
    from tmrvc_data.alignment import load_textgrid_durations, alignment_to_durations
except ImportError:
    load_textgrid_durations = None
    alignment_to_durations = None

try:
    from tmrvc_data.g2p import text_to_phonemes, PHONE2ID, UNK_ID, BOS_ID, EOS_ID
except ImportError:
    text_to_phonemes = None
    PHONE2ID = {}
    UNK_ID = BOS_ID = EOS_ID = 0


@pytest.mark.skip(reason="Legacy v2 alignment tests — alignment module removed in v3")
class TestPhonemeAlignmentParity:
    def test_g2p_to_id_mapping(self):
        """Ensure G2P output can be mapped to IDs correctly."""
        text = "こんにちは"
        result = text_to_phonemes(text, language="ja")
        
        assert result.phonemes[0] == "<bos>"
        assert result.phonemes[-1] == "<eos>"
        
        # Verify IDs match
        for p, pid in zip(result.phonemes, result.phoneme_ids):
            assert PHONE2ID.get(p, UNK_ID) == pid.item()

    def test_textgrid_alignment_frame_parity(self, tmp_path):
        """Ensure MFA durations sum up exactly to total_frames."""
        tg_path = tmp_path / "test.TextGrid"
        # 10ms hop -> 100 frames/sec
        # 0.1s = 10 frames, 0.2s = 20 frames, 0.15s = 15 frames
        # Total = 45 frames
        tg_content = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 0.45
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.45
        intervals: size = 3
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = "k"
        intervals [2]:
            xmin = 0.1
            xmax = 0.3
            text = "o"
        intervals [3]:
            xmin = 0.3
            xmax = 0.45
            text = "n"
"""
        tg_path.write_text(tg_content)
        
        # 1. Test without total_frames constraint
        res = load_textgrid_durations(tg_path)
        assert res.durations.sum() == 45
        assert list(res.durations) == [10, 20, 15]
        
        # 2. Test with total_frames constraint (forcing parity)
        # If actual audio is slightly different, e.g., 47 frames
        res_constrained = load_textgrid_durations(tg_path, total_frames=47)
        assert res_constrained.durations.sum() == 47
        # Last phoneme should absorb the difference
        assert res_constrained.durations[-1] == 17 

    def test_alignment_with_bos_eos_logic(self):
        """Verify the logic used in scripts/annotate/run_forced_alignment.py for BOS/EOS."""
        intervals = [(0.0, 0.1, "a"), (0.1, 0.2, "b")]
        alignment = alignment_to_durations(intervals, total_frames=20)
        
        # This mirrors the logic in the script
        ids = [BOS_ID]
        durs = [0]
        for phone, dur in zip(alignment.phonemes, alignment.durations):
            ids.append(PHONE2ID.get(phone, UNK_ID))
            durs.append(int(dur))
        ids.append(EOS_ID)
        durs.append(0)
        
        assert ids == [BOS_ID, PHONE2ID["a"], PHONE2ID["b"], EOS_ID]
        assert sum(durs) == 20
        assert len(ids) == len(durs)
