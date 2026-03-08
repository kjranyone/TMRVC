"""Tests for Worker 07: Curation Orchestration and Manifest Contract.

Covers:
- CurationDataService (SQLite CRUD, optimistic locking, audit trail)
- StaleVersionError / InvalidTransitionError
- StageResult model
- CurationStage base class + StageRegistry
- Orchestrator retry logic and stage-addressable execution
- Record lifecycle / state transitions
- Pass management
- CLI parser
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from tmrvc_data.curation.errors import (
    CurationError,
    InvalidTransitionError,
    StageExecutionError,
    StaleVersionError,
)
from tmrvc_data.curation.models import (
    CurationRecord,
    PromotionBucket,
    Provenance,
    RecordStatus,
    StageResult,
    VALID_TRANSITIONS,
)
from tmrvc_data.curation.service import CurationDataService
from tmrvc_data.curation.stage_framework import (
    CurationStage,
    StageRegistry,
    STAGE_NAMES,
    create_default_stage_registry,
)
from tmrvc_data.curation.orchestrator import CurationOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    record_id: str = "r1",
    status: RecordStatus = RecordStatus.INGESTED,
    **kwargs: Any,
) -> CurationRecord:
    defaults = dict(
        record_id=record_id,
        source_path="/tmp/audio.wav",
        audio_hash="abc123",
        duration_sec=5.0,
        status=status,
    )
    defaults.update(kwargs)
    return CurationRecord(**defaults)


class DummyStage(CurationStage):
    """A trivial stage for testing."""

    stage_num = 1
    stage_name = "cleanup"

    def __init__(self, *, fail: bool = False, retryable: bool = False):
        self._fail = fail
        self._retryable = retryable
        self.call_count = 0

    def process(self, record: CurationRecord, **kwargs: Any) -> StageResult:
        self.call_count += 1
        if self._fail:
            return StageResult(
                success=False,
                error="test failure",
                retryable=self._retryable,
            )
        return StageResult(
            success=True,
            outputs={"cleaned": True},
            confidence=0.95,
        )


class ExplodingStage(CurationStage):
    """A stage that raises on first N attempts then succeeds."""

    stage_num = 2
    stage_name = "separation"

    def __init__(self, fail_count: int = 2):
        self._fail_count = fail_count
        self.call_count = 0

    def process(self, record: CurationRecord, **kwargs: Any) -> StageResult:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise RuntimeError("transient failure")
        return StageResult(success=True, outputs={"separated": True})


# ===========================================================================
# StageResult model
# ===========================================================================


class TestStageResult:
    def test_success_result(self):
        r = StageResult(success=True, outputs={"key": "val"}, confidence=0.9)
        assert r.success is True
        assert r.confidence == 0.9
        assert r.outputs["key"] == "val"

    def test_failure_result(self):
        r = StageResult(success=False, error="boom", retryable=True)
        assert r.success is False
        assert r.retryable is True
        assert r.error == "boom"


# ===========================================================================
# VALID_TRANSITIONS
# ===========================================================================


class TestValidTransitions:
    def test_ingested_to_annotating(self):
        assert RecordStatus.ANNOTATING in VALID_TRANSITIONS[RecordStatus.INGESTED]

    def test_ingested_to_promoted_invalid(self):
        assert RecordStatus.PROMOTED not in VALID_TRANSITIONS[RecordStatus.INGESTED]

    def test_promoted_to_exported(self):
        assert RecordStatus.EXPORTED in VALID_TRANSITIONS[RecordStatus.PROMOTED]

    def test_exported_status_value(self):
        assert RecordStatus.EXPORTED.value == "exported"

    def test_every_status_has_transitions(self):
        for status in RecordStatus:
            assert status in VALID_TRANSITIONS


# ===========================================================================
# StaleVersionError
# ===========================================================================


class TestStaleVersionError:
    def test_attributes(self):
        err = StaleVersionError("r1", 3, 5)
        assert err.record_id == "r1"
        assert err.expected_version == 3
        assert err.actual_version == 5
        assert "r1" in str(err)
        assert "3" in str(err)

    def test_is_curation_error(self):
        err = StaleVersionError("r1", 1)
        assert isinstance(err, CurationError)


class TestInvalidTransitionError:
    def test_attributes(self):
        err = InvalidTransitionError("r1", "ingested", "exported")
        assert err.record_id == "r1"
        assert err.from_status == "ingested"
        assert err.to_status == "exported"


class TestStageExecutionError:
    def test_retryable_flag(self):
        err = StageExecutionError(2, "r1", "timeout", retryable=True)
        assert err.retryable is True
        assert err.stage == 2


# ===========================================================================
# CurationDataService (SQLite)
# ===========================================================================


class TestCurationDataService:
    def test_create_and_get(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record()
        svc.create_record(record)

        fetched = svc.get_record("r1")
        assert fetched is not None
        assert fetched.record_id == "r1"
        assert fetched.source_path == "/tmp/audio.wav"
        assert fetched.metadata_version == 1

    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        assert svc.get_record("nope") is None

    def test_create_duplicate_raises(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        svc.create_record(_make_record())
        with pytest.raises(sqlite3.IntegrityError):
            svc.create_record(_make_record())

    def test_update_increments_version(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record()
        svc.create_record(record)

        record.status = RecordStatus.ANNOTATING
        svc.update_record(record, actor_id="test", action_type="annotate")

        fetched = svc.get_record("r1")
        assert fetched is not None
        assert fetched.metadata_version == 2
        assert fetched.status == RecordStatus.ANNOTATING

    def test_optimistic_lock_stale_version(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record()
        svc.create_record(record)

        # Simulate stale read: version is 1, but we pretend it's 99
        record.metadata_version = 99
        record.status = RecordStatus.ANNOTATING
        with pytest.raises(StaleVersionError) as exc_info:
            svc.update_record(
                record, actor_id="test", action_type="stale_test"
            )
        assert exc_info.value.expected_version == 99
        assert exc_info.value.actual_version == 1

    def test_invalid_transition_raises(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record(status=RecordStatus.INGESTED)
        svc.create_record(record)

        record.status = RecordStatus.EXPORTED  # invalid from INGESTED
        with pytest.raises(InvalidTransitionError):
            svc.update_record(record, actor_id="test", action_type="bad")

    def test_valid_transition_succeeds(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record(status=RecordStatus.INGESTED)
        svc.create_record(record)

        record.status = RecordStatus.ANNOTATING
        svc.update_record(record, actor_id="test", action_type="annotate")
        assert svc.get_record("r1").status == RecordStatus.ANNOTATING

    def test_skip_transition_validation(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record(status=RecordStatus.INGESTED)
        svc.create_record(record)

        record.status = RecordStatus.EXPORTED
        svc.update_record(
            record,
            actor_id="test",
            action_type="override",
            validate_transition=False,
        )
        assert svc.get_record("r1").status == RecordStatus.EXPORTED

    def test_list_records_no_filter(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        for i in range(5):
            svc.create_record(_make_record(record_id=f"r{i}"))

        records = svc.list_records(limit=10)
        assert len(records) == 5

    def test_list_records_filter_status(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        svc.create_record(_make_record("a1", status=RecordStatus.INGESTED))
        svc.create_record(_make_record("a2", status=RecordStatus.ANNOTATING))

        results = svc.list_records(status="ingested")
        assert len(results) == 1
        assert results[0].record_id == "a1"

    def test_list_records_filter_quality(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        r1 = _make_record("q1", quality_score=0.3)
        r2 = _make_record("q2", quality_score=0.9)
        svc.create_record(r1)
        svc.create_record(r2)

        high = svc.list_records(min_quality=0.8)
        assert len(high) == 1
        assert high[0].record_id == "q2"

    def test_count_records(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        for i in range(3):
            svc.create_record(_make_record(record_id=f"c{i}"))
        assert svc.count_records() == 3
        assert svc.count_records(status="ingested") == 3

    def test_batch_create(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        records = [_make_record(record_id=f"b{i}") for i in range(10)]
        inserted = svc.batch_create(records)
        assert inserted == 10

        # Duplicates are skipped
        inserted2 = svc.batch_create(records)
        assert inserted2 == 0

    def test_batch_update(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        records = [_make_record(record_id=f"u{i}") for i in range(3)]
        svc.batch_create(records)

        for r in records:
            r.status = RecordStatus.ANNOTATING

        updated, stale = svc.batch_update(
            records, actor_id="batch", action_type="annotate"
        )
        assert updated == 3
        assert len(stale) == 0

        # Verify versions incremented
        for r in records:
            fetched = svc.get_record(r.record_id)
            assert fetched.metadata_version == 2

    def test_batch_update_stale_detection(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        r = _make_record("stale1")
        svc.create_record(r)

        # Mess up the version
        r.metadata_version = 99
        r.status = RecordStatus.ANNOTATING
        updated, stale = svc.batch_update(
            [r], actor_id="test", action_type="bad"
        )
        assert updated == 0
        assert "stale1" in stale

    def test_audit_log(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record()
        svc.create_record(record)

        record.status = RecordStatus.ANNOTATING
        svc.update_record(
            record,
            actor_id="alice",
            action_type="annotate",
            rationale="starting annotation pass",
        )

        log = svc.get_audit_log("r1")
        assert len(log) == 1
        entry = log[0]
        assert entry["actor_id"] == "alice"
        assert entry["action_type"] == "annotate"
        assert entry["rationale"] == "starting annotation pass"
        assert entry["before_state"] is not None
        assert entry["after_state"] is not None

        # Verify before/after contain valid JSON
        before = json.loads(entry["before_state"])
        after = json.loads(entry["after_state"])
        assert before["status"] == "ingested"
        assert after["status"] == "annotating"

    def test_audit_log_multiple_updates(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record()
        svc.create_record(record)

        record.status = RecordStatus.ANNOTATING
        svc.update_record(record, actor_id="a", action_type="step1")

        record.status = RecordStatus.SCORED
        svc.update_record(record, actor_id="b", action_type="step2")

        log = svc.get_audit_log("r1")
        assert len(log) == 2
        # Most recent first
        assert log[0]["action_type"] == "step2"
        assert log[1]["action_type"] == "step1"

    def test_status_summary(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        svc.create_record(_make_record("s1", status=RecordStatus.INGESTED))
        svc.create_record(_make_record("s2", status=RecordStatus.INGESTED))
        svc.create_record(_make_record("s3", status=RecordStatus.ANNOTATING))

        summary = svc.status_summary()
        assert summary["total_records"] == 3
        assert summary["status"]["ingested"] == 2
        assert summary["status"]["annotating"] == 1

    def test_wal_mode_enabled(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        conn = svc._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_close(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        svc.create_record(_make_record())
        svc.close()
        # After close, a new connection is created on next access
        fetched = svc.get_record("r1")
        assert fetched is not None


# ===========================================================================
# StageRegistry
# ===========================================================================


class TestStageRegistry:
    def test_register_and_get(self):
        reg = StageRegistry()
        stage = DummyStage()
        reg.register(stage)
        assert reg.get(1) is stage

    def test_get_nonexistent(self):
        reg = StageRegistry()
        assert reg.get(99) is None

    def test_registered_stages(self):
        reg = StageRegistry()
        reg.register(DummyStage())
        reg.register(ExplodingStage())
        assert reg.registered_stages() == [1, 2]

    def test_contains(self):
        reg = StageRegistry()
        reg.register(DummyStage())
        assert 1 in reg
        assert 99 not in reg

    def test_get_all(self):
        reg = StageRegistry()
        s1 = DummyStage()
        s2 = DummyStage()
        reg.register(s1)
        reg.register(s2)
        all_stages = reg.get_all(1)
        assert len(all_stages) == 2

    def test_default_registry_empty(self):
        reg = create_default_stage_registry()
        assert len(reg.registered_stages()) == 0

    def test_stage_names_cover_0_to_9(self):
        for i in range(10):
            assert i in STAGE_NAMES


# ===========================================================================
# Orchestrator: retry logic
# ===========================================================================


class TestOrchestratorRetry:
    def test_retry_on_exception(self, tmp_path: Path):
        reg = StageRegistry()
        stage = ExplodingStage(fail_count=2)
        reg.register(stage)

        orch = CurationOrchestrator(
            tmp_path,
            stage_registry=reg,
            max_retries=3,
            retry_backoff=0.01,
        )
        record = _make_record()
        orch.records[record.record_id] = record

        results = orch.run_stage_num(2)
        assert results["r1"].success is True
        assert stage.call_count == 3  # 2 failures + 1 success

    def test_max_retries_exhausted(self, tmp_path: Path):
        reg = StageRegistry()
        stage = ExplodingStage(fail_count=5)
        reg.register(stage)

        orch = CurationOrchestrator(
            tmp_path,
            stage_registry=reg,
            max_retries=3,
            retry_backoff=0.01,
        )
        record = _make_record()
        orch.records[record.record_id] = record

        results = orch.run_stage_num(2)
        assert results["r1"].success is False
        assert stage.call_count == 3

    def test_no_retry_on_non_retryable(self, tmp_path: Path):
        reg = StageRegistry()
        stage = DummyStage(fail=True, retryable=False)
        reg.register(stage)

        orch = CurationOrchestrator(
            tmp_path,
            stage_registry=reg,
            max_retries=3,
            retry_backoff=0.01,
        )
        record = _make_record()
        orch.records[record.record_id] = record

        results = orch.run_stage_num(1)
        assert results["r1"].success is False
        assert stage.call_count == 1  # no retries

    def test_retry_on_retryable_failure(self, tmp_path: Path):
        reg = StageRegistry()
        stage = DummyStage(fail=True, retryable=True)
        reg.register(stage)

        orch = CurationOrchestrator(
            tmp_path,
            stage_registry=reg,
            max_retries=3,
            retry_backoff=0.01,
        )
        record = _make_record()
        orch.records[record.record_id] = record

        results = orch.run_stage_num(1)
        assert results["r1"].success is False
        assert stage.call_count == 3  # retried until exhausted


# ===========================================================================
# Orchestrator: stage-addressable execution
# ===========================================================================


class TestOrchestratorStageExec:
    def test_run_stage_num_success(self, tmp_path: Path):
        reg = StageRegistry()
        reg.register(DummyStage())

        orch = CurationOrchestrator(tmp_path, stage_registry=reg)
        orch.records["r1"] = _make_record()

        results = orch.run_stage_num(1)
        assert "r1" in results
        assert results["r1"].success is True
        # Outputs merged into record
        assert orch.records["r1"].attributes.get("cleaned") is True

    def test_run_stage_num_no_registry(self, tmp_path: Path):
        orch = CurationOrchestrator(tmp_path)
        orch.records["r1"] = _make_record()
        with pytest.raises(RuntimeError, match="No StageRegistry"):
            orch.run_stage_num(1)

    def test_run_stage_num_missing_impl(self, tmp_path: Path):
        reg = StageRegistry()
        orch = CurationOrchestrator(tmp_path, stage_registry=reg)
        orch.records["r1"] = _make_record()
        with pytest.raises(ValueError, match="No implementation"):
            orch.run_stage_num(99)

    def test_run_stage_num_skips_processed(self, tmp_path: Path):
        reg = StageRegistry()
        stage = DummyStage()
        reg.register(stage)

        orch = CurationOrchestrator(tmp_path, stage_registry=reg)
        # Record already has cleanup provenance
        r = _make_record()
        r.providers["cleanup"] = Provenance(
            stage="cleanup", provider="p", version="1", timestamp=0,
        )
        orch.records["r1"] = r

        results = orch.run_stage_num(1, force=False)
        assert len(results) == 0
        assert stage.call_count == 0

    def test_run_stage_num_force_reprocess(self, tmp_path: Path):
        reg = StageRegistry()
        stage = DummyStage()
        reg.register(stage)

        orch = CurationOrchestrator(tmp_path, stage_registry=reg)
        r = _make_record()
        r.providers["cleanup"] = Provenance(
            stage="cleanup", provider="p", version="1", timestamp=0,
        )
        orch.records["r1"] = r

        results = orch.run_stage_num(1, force=True)
        assert len(results) == 1
        assert stage.call_count == 1

    def test_run_stage_num_target_records(self, tmp_path: Path):
        reg = StageRegistry()
        stage = DummyStage()
        reg.register(stage)

        orch = CurationOrchestrator(tmp_path, stage_registry=reg)
        orch.records["r1"] = _make_record("r1")
        orch.records["r2"] = _make_record("r2")
        orch.records["r3"] = _make_record("r3")

        results = orch.run_stage_num(1, record_ids=["r2"])
        assert list(results.keys()) == ["r2"]

    def test_run_all_stages(self, tmp_path: Path):
        reg = StageRegistry()
        reg.register(DummyStage())
        reg.register(ExplodingStage(fail_count=0))

        orch = CurationOrchestrator(
            tmp_path, stage_registry=reg, retry_backoff=0.01,
        )
        orch.records["r1"] = _make_record()

        all_results = orch.run_all_stages(start_stage=1, end_stage=2)
        assert 1 in all_results
        assert 2 in all_results

    def test_resume(self, tmp_path: Path):
        reg = StageRegistry()
        stage = DummyStage()
        reg.register(stage)

        orch = CurationOrchestrator(tmp_path, stage_registry=reg)
        orch.records["r1"] = _make_record()

        all_results = orch.resume()
        # Should have run stage 1 (cleanup)
        assert 1 in all_results


# ===========================================================================
# Record lifecycle
# ===========================================================================


class TestRecordLifecycle:
    def test_full_lifecycle_in_service(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record(status=RecordStatus.INGESTED)
        svc.create_record(record)

        # ingested -> annotating
        record.status = RecordStatus.ANNOTATING
        svc.update_record(record, actor_id="w07", action_type="cleanup")

        # annotating -> scored
        record.status = RecordStatus.SCORED
        svc.update_record(record, actor_id="w09", action_type="score")

        # scored -> promoted
        record.status = RecordStatus.PROMOTED
        record.promotion_bucket = PromotionBucket.TTS_MAINLINE
        svc.update_record(record, actor_id="w09", action_type="promote")

        # promoted -> exported
        record.status = RecordStatus.EXPORTED
        svc.update_record(record, actor_id="w10", action_type="export")

        final = svc.get_record("r1")
        assert final.status == RecordStatus.EXPORTED
        assert final.metadata_version == 5  # 1 + 4 updates
        assert final.promotion_bucket == PromotionBucket.TTS_MAINLINE

    def test_review_loop(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record(status=RecordStatus.INGESTED)
        svc.create_record(record)

        # ingested -> annotating -> scored -> review -> annotating -> scored -> promoted
        for new_status in [
            RecordStatus.ANNOTATING,
            RecordStatus.SCORED,
            RecordStatus.REVIEW,
            RecordStatus.ANNOTATING,
            RecordStatus.SCORED,
            RecordStatus.PROMOTED,
        ]:
            record.status = new_status
            svc.update_record(record, actor_id="system", action_type="cycle")

        final = svc.get_record("r1")
        assert final.status == RecordStatus.PROMOTED
        assert final.metadata_version == 7

    def test_rejected_rescue(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record(status=RecordStatus.INGESTED)
        svc.create_record(record)

        # rejected -> annotating (rescue)
        record.status = RecordStatus.REJECTED
        svc.update_record(
            record, actor_id="system", action_type="reject",
            validate_transition=False,
        )

        record.status = RecordStatus.ANNOTATING
        svc.update_record(
            record, actor_id="admin", action_type="rescue",
            rationale="re-process with better provider",
        )

        final = svc.get_record("r1")
        assert final.status == RecordStatus.ANNOTATING


# ===========================================================================
# Pass management
# ===========================================================================


class TestPassManagement:
    def test_pass_0_initial(self):
        record = _make_record()
        assert record.pass_index == 0

    def test_pass_increment(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        record = _make_record()
        svc.create_record(record)

        # Simulate pass 1 refinement
        record.pass_index = 1
        record.status = RecordStatus.ANNOTATING
        svc.update_record(record, actor_id="system", action_type="pass1")

        fetched = svc.get_record("r1")
        assert fetched.pass_index == 1

    def test_list_by_pass(self, tmp_path: Path):
        svc = CurationDataService(tmp_path / "curation.db")
        r0 = _make_record("p0", pass_index=0)
        r1 = _make_record("p1", pass_index=1)
        svc.batch_create([r0, r1])

        pass0 = svc.list_records(curation_pass=0)
        assert len(pass0) == 1
        assert pass0[0].record_id == "p0"


# ===========================================================================
# CLI parser tests
# ===========================================================================


class TestCurationCLI:
    def test_parser_ingest(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args(["ingest", "/tmp/audio"])
        assert args.command == "ingest"
        assert args.path == Path("/tmp/audio")

    def test_parser_run_stage(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args(["run-stage", "3", "--force"])
        assert args.command == "run-stage"
        assert args.stage_num == 3
        assert args.force is True

    def test_parser_resume(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args(["resume"])
        assert args.command == "resume"

    def test_parser_status(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_parser_promote(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args(["promote"])
        assert args.command == "promote"

    def test_parser_export(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "export", "--export-dir", "/tmp/out",
        ])
        assert args.command == "export"
        assert args.export_dir == Path("/tmp/out")

    def test_parser_with_db(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--db", "/tmp/curation.db", "status",
        ])
        assert args.db == Path("/tmp/curation.db")

    def test_parser_with_output_dir(self):
        from tmrvc_data.cli.curation import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--output-dir", "/tmp/work", "status",
        ])
        assert args.output_dir == Path("/tmp/work")


# ===========================================================================
# Orchestrator backward compatibility
# ===========================================================================


class TestOrchestratorBackcompat:
    """Ensure the existing callback-based API still works."""

    def test_run_stage_callback(self, tmp_path: Path):
        orch = CurationOrchestrator(tmp_path)
        record = _make_record()
        orch.update_record(record)

        call_count = [0]

        def processor(r: CurationRecord) -> Optional[CurationRecord]:
            call_count[0] += 1
            return r

        orch.run_stage("test_stage", processor)
        assert call_count[0] == 1

    def test_manifest_round_trip(self, tmp_path: Path):
        orch = CurationOrchestrator(tmp_path)
        record = _make_record()
        orch.update_record(record)
        orch.save_manifest()

        orch2 = CurationOrchestrator(tmp_path)
        assert "r1" in orch2.records

    def test_optimistic_lock_in_orchestrator(self, tmp_path: Path):
        orch = CurationOrchestrator(tmp_path)
        record = _make_record()
        orch.update_record(record)

        # Record now at version 2 (incremented on update)
        # Trying with stale version should fail
        r2 = _make_record()
        with pytest.raises(ValueError, match="Conflict"):
            orch.update_record(r2, expected_version=1)

    def test_available_stages(self):
        stages = CurationOrchestrator.available_stages()
        assert "cleanup" in stages
        assert "ingest" in stages
