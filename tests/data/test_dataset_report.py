"""Tests for dataset report (Worker 03)."""

from __future__ import annotations

import pytest

from tmrvc_data.dataset_report import DatasetReport, REQUIRED_REPORT_FIELDS


class TestDatasetReportFields:
    """Verify report has all required fields from Worker 03 spec."""

    def test_all_required_fields_present(self):
        report = DatasetReport()
        report_dict = report.to_dict()
        missing = REQUIRED_REPORT_FIELDS - set(report_dict.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_required_fields_frozenset_matches_dataclass(self):
        import dataclasses
        dc_fields = {f.name for f in dataclasses.fields(DatasetReport)}
        assert REQUIRED_REPORT_FIELDS.issubset(dc_fields)


class TestDatasetReportValidation:
    def test_empty_name_fails(self):
        report = DatasetReport()
        errors = report.validate()
        assert any("dataset_name" in e for e in errors)

    def test_valid_report_passes(self):
        report = DatasetReport(
            dataset_name="test_corpus",
            num_utterances=100,
            text_supervision_coverage=0.9,
            canonical_text_unit_coverage=0.85,
        )
        errors = report.validate()
        assert not errors

    def test_out_of_range_coverage_fails(self):
        report = DatasetReport(
            dataset_name="test",
            text_supervision_coverage=1.5,
        )
        errors = report.validate()
        assert any("text_supervision_coverage" in e for e in errors)

    def test_negative_utterances_fails(self):
        report = DatasetReport(
            dataset_name="test",
            num_utterances=-1,
        )
        errors = report.validate()
        assert any("num_utterances" in e for e in errors)


class TestDatasetReportSerialization:
    def test_to_dict_roundtrip(self):
        report = DatasetReport(
            dataset_name="corpus_a",
            split="train",
            num_utterances=500,
            text_supervision_coverage=0.95,
            active_phone_inventory=["a", "k", "s"],
            unmapped_phone_counts={"X": 3},
        )
        d = report.to_dict()
        assert d["dataset_name"] == "corpus_a"
        assert d["active_phone_inventory"] == ["a", "k", "s"]
        assert d["unmapped_phone_counts"] == {"X": 3}


class TestSupervisionCoverageDistinction:
    """Report must distinguish text_supervision from canonical_text_unit coverage."""

    def test_text_vs_canonical_independent(self):
        report = DatasetReport(
            dataset_name="test",
            text_supervision_coverage=0.9,
            canonical_text_unit_coverage=0.7,
        )
        assert report.text_supervision_coverage != report.canonical_text_unit_coverage
