"""Tests for Worker 03 dataset contract, text supervision, and metrics.

Covers:
- Dataset returns valid TTS sample without durations.npy (pointer mode)
- Quality-score based filtering
- Few-shot prompt eligibility and curation meta fields
- Supervision report distinguishes text_supervision from canonical_text_unit coverage
- voice_state artifact shape and reporting
- text_suprasegmentals artifact shape and alignment
- bootstrap_alignment canonical format
- DatasetReport field completeness with new Worker 03 fields
- Supervision scanner utility
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures: synthetic cache directory
# ---------------------------------------------------------------------------


def _make_utt_dir(
    base: Path,
    dataset: str = "test_ds",
    speaker: str = "spk01",
    utt_id: str = "utt001",
    n_frames: int = 100,
    n_codebooks: int = 8,
    d_voice_state: int = 8,
    n_phones: int = 10,
    include_durations: bool = False,
    include_suprasegmentals: bool = False,
    include_voice_state_targets: bool = False,
    include_bootstrap_alignment: bool = False,
    meta_overrides: dict | None = None,
) -> Path:
    """Create a synthetic utterance cache directory."""
    utt_dir = base / dataset / "train" / speaker / utt_id
    utt_dir.mkdir(parents=True, exist_ok=True)

    # Codec tokens [n_codebooks, T]
    np.save(utt_dir / "codec_tokens.npy", np.zeros((n_codebooks, n_frames), dtype=np.int64))

    # Voice state [T, 8]
    np.save(utt_dir / "voice_state.npy", np.random.randn(n_frames, d_voice_state).astype(np.float32))

    # Speaker embedding [192]
    np.save(utt_dir / "spk_embed.npy", np.random.randn(192).astype(np.float32))

    # Phoneme IDs [L]
    pids = np.arange(n_phones, dtype=np.int64)
    np.save(utt_dir / "phoneme_ids.npy", pids)

    # Optional: durations
    if include_durations:
        durations = np.full(n_phones, n_frames // n_phones, dtype=np.int64)
        durations[-1] += n_frames - durations.sum()
        np.save(utt_dir / "durations.npy", durations)

    # Optional: suprasegmentals [L, 4]
    if include_suprasegmentals:
        np.save(
            utt_dir / "text_suprasegmentals.npy",
            np.random.randn(n_phones, 4).astype(np.float32),
        )

    # Optional: voice state supervision
    if include_voice_state_targets:
        np.save(utt_dir / "voice_state_targets.npy", np.random.randn(n_frames, 8).astype(np.float32))
        np.save(utt_dir / "voice_state_observed_mask.npy", np.ones((n_frames, 8), dtype=bool))
        np.save(utt_dir / "voice_state_confidence.npy", np.ones((n_frames, 8), dtype=np.float32) * 0.9)

    # Optional: bootstrap alignment
    if include_bootstrap_alignment:
        ba_data = {
            "phoneme_indices": list(range(n_frames)),
            "frame_count": n_frames,
            "phoneme_count": n_phones,
        }
        with open(utt_dir / "bootstrap_alignment.json", "w", encoding="utf-8") as f:
            json.dump(ba_data, f)

    # meta.json
    meta = {
        "utterance_id": utt_id,
        "speaker_id": speaker,
        "n_frames": n_frames,
        "text": f"Test utterance {utt_id}",
        "language": "ja",
    }
    if meta_overrides:
        meta.update(meta_overrides)
    with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return utt_dir


@pytest.fixture
def cache_dir_pointer_mode(tmp_path):
    """Cache dir with utterance that has phoneme_ids but NO durations."""
    _make_utt_dir(tmp_path, include_durations=False)
    return tmp_path


@pytest.fixture
def cache_dir_with_durations(tmp_path):
    """Cache dir with utterance that has both phoneme_ids AND durations."""
    _make_utt_dir(tmp_path, include_durations=True)
    return tmp_path


@pytest.fixture
def cache_dir_quality_filtered(tmp_path):
    """Cache dir with utterances at different quality scores."""
    _make_utt_dir(tmp_path, utt_id="utt_high", meta_overrides={"quality_score": 0.9})
    _make_utt_dir(tmp_path, utt_id="utt_low", meta_overrides={"quality_score": 0.2})
    _make_utt_dir(tmp_path, utt_id="utt_mid", meta_overrides={"quality_score": 0.5})
    return tmp_path


@pytest.fixture
def cache_dir_full(tmp_path):
    """Cache dir with all optional artifacts."""
    _make_utt_dir(
        tmp_path,
        utt_id="full_utt",
        include_durations=True,
        include_suprasegmentals=True,
        include_voice_state_targets=True,
        include_bootstrap_alignment=True,
        meta_overrides={
            "quality_score": 0.95,
            "prompt_eligible": True,
            "prompt_pair_id": "pair_001",
            "curation_record_id": "rec_001",
            "promotion_bucket": "tier_a",
            "curation_pass": "pass_2",
        },
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Task 1: durations.npy optional in pointer mode
# ---------------------------------------------------------------------------


class TestDurationsOptional:
    """Dataset returns valid TTS sample without durations.npy in pointer mode."""

    def test_pointer_mode_loads_without_durations(self, cache_dir_pointer_mode):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_pointer_mode, tts_mode="pointer")
        assert len(ds) == 1
        sample = ds[0]
        assert sample["phoneme_ids"] is not None
        assert sample["durations"] is None

    def test_auto_mode_loads_without_durations(self, cache_dir_pointer_mode):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_pointer_mode, tts_mode="auto")
        assert len(ds) == 1
        sample = ds[0]
        assert sample["phoneme_ids"] is not None
        assert sample["durations"] is None

    def test_legacy_mode_skips_without_durations(self, cache_dir_pointer_mode):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_pointer_mode, tts_mode="legacy_duration")
        assert len(ds) == 1
        sample = ds[0]
        # legacy_duration mode: phoneme_ids is None when durations missing
        assert sample["phoneme_ids"] is None
        assert sample["durations"] is None

    def test_with_durations_loaded_in_auto(self, cache_dir_with_durations):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_with_durations, tts_mode="auto")
        sample = ds[0]
        assert sample["phoneme_ids"] is not None
        assert sample["durations"] is not None

    def test_pointer_mode_ignores_existing_durations(self, cache_dir_with_durations):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_with_durations, tts_mode="pointer")
        sample = ds[0]
        assert sample["phoneme_ids"] is not None
        assert sample["durations"] is None


# ---------------------------------------------------------------------------
# Task 5: text_suprasegmentals
# ---------------------------------------------------------------------------


class TestTextSuprasegmentals:
    """text_suprasegmentals.npy shape and alignment with phoneme_ids."""

    def test_shape_L_by_4(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["text_suprasegmentals"] is not None
        L = sample["phoneme_ids"].shape[0]
        assert sample["text_suprasegmentals"].shape == (L, 4)

    def test_index_aligned_with_phoneme_ids(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["text_suprasegmentals"].shape[0] == sample["phoneme_ids"].shape[0]


# ---------------------------------------------------------------------------
# Task 3: bootstrap_alignment.json canonical format
# ---------------------------------------------------------------------------


class TestBootstrapAlignmentFormat:
    """bootstrap_alignment.json schema: phoneme_indices, frame_count, phoneme_count."""

    def test_loads_from_cache(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        ba = sample["bootstrap_alignment"]
        assert ba is not None
        assert "phoneme_indices" in ba
        assert "frame_count" in ba
        assert "phoneme_count" in ba
        assert isinstance(ba["phoneme_indices"], torch.Tensor)
        assert ba["phoneme_indices"].dtype == torch.long

    def test_phoneme_indices_length(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        ba = sample["bootstrap_alignment"]
        assert ba["frame_count"] == sample["n_frames"]


# ---------------------------------------------------------------------------
# Task 4: voice_state supervision artifacts
# ---------------------------------------------------------------------------


class TestVoiceStateSupervision:
    """voice_state supervision artifact shapes."""

    def test_targets_loaded(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["voice_state_targets"] is not None
        assert sample["voice_state_targets"].shape == (100, 8)

    def test_observed_mask_loaded(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["voice_state_observed_mask"] is not None
        assert sample["voice_state_observed_mask"].dtype == torch.bool
        assert sample["voice_state_observed_mask"].shape == (100, 8)

    def test_confidence_loaded(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["voice_state_confidence"] is not None
        assert sample["voice_state_confidence"].shape == (100, 8)


# ---------------------------------------------------------------------------
# Task 6: Few-shot prompt pairing metadata
# ---------------------------------------------------------------------------


class TestPromptPairingMetadata:
    """meta.json prompt_eligible and prompt_pair_id fields."""

    def test_prompt_eligible_exposed(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["prompt_eligible"] is True

    def test_prompt_pair_id_exposed(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["prompt_pair_id"] == "pair_001"

    def test_absent_prompt_fields_are_none(self, cache_dir_pointer_mode):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_pointer_mode, tts_mode="pointer")
        sample = ds[0]
        assert sample["prompt_eligible"] is None
        assert sample["prompt_pair_id"] is None


# ---------------------------------------------------------------------------
# Task 7: Curated asset consumption contract
# ---------------------------------------------------------------------------


class TestCurationMetaFields:
    """meta.json curation fields exposed by dataset loader."""

    def test_curation_fields_exposed(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        assert sample["curation_record_id"] == "rec_001"
        assert sample["promotion_bucket"] == "tier_a"
        assert sample["curation_pass"] == "pass_2"
        assert sample["quality_score"] == 0.95

    def test_absent_curation_fields_are_none(self, cache_dir_pointer_mode):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_pointer_mode, tts_mode="pointer")
        sample = ds[0]
        assert sample["curation_record_id"] is None
        assert sample["promotion_bucket"] is None


# ---------------------------------------------------------------------------
# Task 8: Quality filtering
# ---------------------------------------------------------------------------


class TestQualityFiltering:
    """min_quality_score filter skips low-quality samples."""

    def test_no_filter_returns_all(self, cache_dir_quality_filtered):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_quality_filtered, min_quality_score=0.0)
        assert len(ds) == 3

    def test_high_threshold_filters(self, cache_dir_quality_filtered):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_quality_filtered, min_quality_score=0.8)
        assert len(ds) == 1
        assert ds[0]["utterance_id"] == "utt_high"

    def test_medium_threshold_filters(self, cache_dir_quality_filtered):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_quality_filtered, min_quality_score=0.4)
        assert len(ds) == 2

    def test_backward_compat_quality_score_threshold(self, cache_dir_quality_filtered):
        """Legacy quality_score_threshold param still works."""
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_quality_filtered, quality_score_threshold=0.8)
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# Task 2: Supervision report
# ---------------------------------------------------------------------------


class TestSupervisionReport:
    """Supervision report from UCLMDataset."""

    def test_report_distinguishes_coverage(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        report = ds.supervision_report()
        assert "text_supervision_coverage" in report
        assert "canonical_text_unit_coverage" in report
        assert "legacy_duration_coverage" in report
        # All should be 1.0 since the full cache has everything
        assert report["text_supervision_coverage"] == 1.0
        assert report["canonical_text_unit_coverage"] == 1.0
        assert report["legacy_duration_coverage"] == 1.0

    def test_report_voice_state_coverage(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        report = ds.supervision_report()
        assert report["voice_state_supervision_coverage"] == 1.0
        assert report["voice_state_observed_ratio"] > 0.0

    def test_report_prompt_coverage(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        report = ds.supervision_report()
        assert report["prompt_pairing_coverage"] == 1.0

    def test_report_curation_coverage(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        report = ds.supervision_report()
        assert report["curation_record_coverage"] == 1.0

    def test_report_suprasegmental_coverage(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        report = ds.supervision_report()
        assert report["suprasegmental_coverage"] == 1.0

    def test_report_zero_on_empty(self, tmp_path):
        from tmrvc_data.uclm_dataset import UCLMDataset

        ds = UCLMDataset(cache_dir=tmp_path, tts_mode="pointer")
        report = ds.supervision_report()
        assert report["num_utterances"] == 0
        assert report["text_supervision_coverage"] == 0.0


# ---------------------------------------------------------------------------
# Supervision scanner utility
# ---------------------------------------------------------------------------


class TestSupervisionScanner:
    """Supervision scanner produces valid DatasetReport."""

    def test_scan_produces_report(self, cache_dir_full):
        from tmrvc_data.supervision_scanner import scan_cache_dir

        report = scan_cache_dir(cache_dir_full, dataset_name="test_scan")
        assert report.dataset_name == "test_scan"
        assert report.num_utterances == 1
        assert report.canonical_text_unit_coverage == 1.0
        assert report.voice_state_supervision_coverage == 1.0
        assert report.suprasegmental_coverage == 1.0
        assert report.prompt_pairing_coverage == 1.0
        assert report.curation_record_coverage == 1.0

    def test_scan_validates(self, cache_dir_full):
        from tmrvc_data.supervision_scanner import scan_cache_dir

        report = scan_cache_dir(cache_dir_full, dataset_name="test_scan")
        errors = report.validate()
        assert not errors

    def test_scan_empty_dir(self, tmp_path):
        from tmrvc_data.supervision_scanner import scan_cache_dir

        report = scan_cache_dir(tmp_path, dataset_name="empty")
        assert report.num_utterances == 0


# ---------------------------------------------------------------------------
# DatasetReport field completeness (updated for Worker 03 additions)
# ---------------------------------------------------------------------------


class TestDatasetReportFieldsWorker03:
    """Verify DatasetReport has all Worker 03 required fields."""

    def test_all_required_fields_present(self):
        from tmrvc_data.dataset_report import DatasetReport, REQUIRED_REPORT_FIELDS

        report = DatasetReport()
        report_dict = report.to_dict()
        missing = REQUIRED_REPORT_FIELDS - set(report_dict.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_new_worker03_fields_in_required(self):
        from tmrvc_data.dataset_report import REQUIRED_REPORT_FIELDS

        assert "prompt_pairing_coverage" in REQUIRED_REPORT_FIELDS
        assert "prompt_leakage_risk_count" in REQUIRED_REPORT_FIELDS
        assert "suprasegmental_coverage" in REQUIRED_REPORT_FIELDS
        assert "bootstrap_alignment_coverage" in REQUIRED_REPORT_FIELDS
        assert "curation_record_coverage" in REQUIRED_REPORT_FIELDS

    def test_validation_covers_new_fields(self):
        from tmrvc_data.dataset_report import DatasetReport

        report = DatasetReport(
            dataset_name="test",
            prompt_pairing_coverage=1.5,
        )
        errors = report.validate()
        assert any("prompt_pairing_coverage" in e for e in errors)


# ---------------------------------------------------------------------------
# Collation test (pointer mode, no durations)
# ---------------------------------------------------------------------------


class TestCollationPointerMode:
    """Collate batch works with pointer mode (no durations)."""

    def test_collate_no_durations(self, cache_dir_pointer_mode):
        from tmrvc_data.uclm_dataset import UCLMDataset, collate_uclm_batch

        ds = UCLMDataset(cache_dir=cache_dir_pointer_mode, tts_mode="pointer")
        sample = ds[0]
        batch = collate_uclm_batch([sample])
        assert batch.durations is None
        assert batch.phoneme_ids is not None
        assert batch.phoneme_ids.shape[0] == 1

    def test_collate_with_suprasegmentals(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset, collate_uclm_batch

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        batch = collate_uclm_batch([sample])
        assert batch.text_suprasegmentals is not None
        assert batch.text_suprasegmentals.shape[-1] == 4

    def test_collate_with_voice_state_targets(self, cache_dir_full):
        from tmrvc_data.uclm_dataset import UCLMDataset, collate_uclm_batch

        ds = UCLMDataset(cache_dir=cache_dir_full, tts_mode="pointer")
        sample = ds[0]
        batch = collate_uclm_batch([sample])
        assert batch.voice_state_targets is not None
        assert batch.voice_state_observed_mask is not None
        assert batch.voice_state_confidence is not None


# ---------------------------------------------------------------------------
# VOICE_STATE_DIMS and SUPRASEGMENTAL_DIMS constants
# ---------------------------------------------------------------------------


class TestCanonicalDimConstants:
    """Verify canonical dimension name constants."""

    def test_voice_state_dims_count(self):
        from tmrvc_data.uclm_dataset import VOICE_STATE_DIMS

        assert len(VOICE_STATE_DIMS) == 8

    def test_voice_state_dims_names(self):
        from tmrvc_data.uclm_dataset import VOICE_STATE_DIMS

        assert VOICE_STATE_DIMS[0] == "pitch_level"
        assert VOICE_STATE_DIMS[5] == "breathiness"

    def test_suprasegmental_dims_count(self):
        from tmrvc_data.uclm_dataset import SUPRASEGMENTAL_DIMS

        assert len(SUPRASEGMENTAL_DIMS) == 4

    def test_suprasegmental_dims_names(self):
        from tmrvc_data.uclm_dataset import SUPRASEGMENTAL_DIMS

        assert SUPRASEGMENTAL_DIMS[0] == "accent_upstep"
        assert SUPRASEGMENTAL_DIMS[3] == "lexical_tone_id"
