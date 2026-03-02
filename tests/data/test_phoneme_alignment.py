"""Tests for Phoneme mapping and MFA alignment parity."""

import numpy as np
import pytest
import torch

from tmrvc_data.alignment import load_textgrid_durations, alignment_to_durations
from tmrvc_data.g2p import text_to_phonemes, PHONE2ID, UNK_ID, BOS_ID, EOS_ID


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
