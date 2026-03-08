"""Tests for /ui/* routes, SSE event types, idempotency middleware,
conflict response types, and artifact contract (Worker 04, tasks 13/20/21/22).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

# ---------------------------------------------------------------------------
# SSE event schema tests (task 20)
# ---------------------------------------------------------------------------


class TestSSEEventType:
    def test_all_required_types_exist(self):
        from tmrvc_serve.events import SSEEventType

        required = [
            "job_progress",
            "job_blocked_human",
            "job_failed",
            "job_completed",
            "take_ready",
            "telemetry_update",
        ]
        actual = [e.value for e in SSEEventType]
        for rt in required:
            assert rt in actual, f"Missing SSE event type: {rt}"

    def test_event_type_is_str_enum(self):
        from tmrvc_serve.events import SSEEventType

        assert isinstance(SSEEventType.JOB_PROGRESS.value, str)


class TestSSEEvent:
    def test_defaults(self):
        from tmrvc_serve.events import SSEEvent, SSEEventType

        evt = SSEEvent(event_type=SSEEventType.JOB_PROGRESS, job_id="j1")
        assert evt.event_type == SSEEventType.JOB_PROGRESS
        assert evt.job_id == "j1"
        assert evt.payload_version == 1
        assert isinstance(evt.timestamp, datetime)
        assert evt.event_id  # non-empty

    def test_required_envelope_fields(self):
        from tmrvc_serve.events import SSEEvent, SSEEventType

        evt = SSEEvent(
            event_type=SSEEventType.TAKE_READY,
            job_id="j2",
            object_type="take",
            object_id="t1",
            data={"audio_url": "/audio/t1.wav"},
        )
        d = evt.model_dump(mode="json")
        for field in ("event_type", "job_id", "object_type", "object_id", "timestamp", "payload_version"):
            assert field in d, f"Missing envelope field: {field}"

    def test_to_sse_wire_format(self):
        from tmrvc_serve.events import SSEEvent, SSEEventType

        evt = SSEEvent(
            event_type=SSEEventType.JOB_COMPLETED,
            job_id="j3",
            object_type="dataset",
            object_id="d1",
        )
        wire = evt.to_sse()
        assert wire.startswith("id: ")
        assert "event: job_completed\n" in wire
        assert "data: " in wire
        assert wire.endswith("\n\n")

        # The data line should be valid JSON
        for line in wire.strip().split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                assert payload["event_type"] == "job_completed"
                assert payload["job_id"] == "j3"

    def test_serialization_roundtrip(self):
        from tmrvc_serve.events import SSEEvent, SSEEventType

        evt = SSEEvent(
            event_type=SSEEventType.JOB_FAILED,
            job_id="j4",
            data={"error": "OOM"},
        )
        d = evt.model_dump(mode="json")
        restored = SSEEvent(**d)
        assert restored.event_type == SSEEventType.JOB_FAILED
        assert restored.data["error"] == "OOM"


# ---------------------------------------------------------------------------
# Conflict type tests (task 21)
# ---------------------------------------------------------------------------


class TestConflictTypes:
    def test_all_conflict_types_exist(self):
        from tmrvc_serve.middleware import ConflictType

        required = ["stale_version", "locked_by_other", "already_submitted", "policy_forbidden"]
        actual = [ct.value for ct in ConflictType]
        for rt in required:
            assert rt in actual, f"Missing conflict type: {rt}"

    def test_conflict_detail_model(self):
        from tmrvc_serve.middleware import ConflictDetail, ConflictType

        detail = ConflictDetail(
            conflict_type=ConflictType.STALE_VERSION,
            message="Version mismatch",
            current_version=5,
        )
        d = detail.model_dump(mode="json")
        assert d["conflict_type"] == "stale_version"
        assert d["current_version"] == 5

    def test_raise_conflict(self):
        from fastapi import HTTPException
        from tmrvc_serve.middleware import ConflictType, raise_conflict

        with pytest.raises(HTTPException) as exc_info:
            raise_conflict(
                ConflictType.LOCKED_BY_OTHER,
                "Resource locked",
                locked_by="user-42",
            )
        assert exc_info.value.status_code == 409
        detail = exc_info.value.detail
        assert detail["conflict_type"] == "locked_by_other"
        assert detail["locked_by"] == "user-42"


# ---------------------------------------------------------------------------
# Idempotency middleware tests (task 21)
# ---------------------------------------------------------------------------


class TestIdempotencyMiddleware:
    def test_cache_entry_expiry(self):
        from tmrvc_serve.middleware import _CacheEntry

        entry = _CacheEntry(status_code=200, body=b"ok", headers={}, ttl=0)
        # TTL=0 means it should be expired immediately (or very nearly)
        time.sleep(0.01)
        assert entry.expired is True

    def test_cache_entry_not_expired(self):
        from tmrvc_serve.middleware import _CacheEntry

        entry = _CacheEntry(status_code=200, body=b"ok", headers={}, ttl=300)
        assert entry.expired is False

    def test_middleware_instantiation(self):
        """Verify the middleware can be instantiated with a dummy ASGI app."""
        from tmrvc_serve.middleware import IdempotencyMiddleware

        async def dummy_app(scope, receive, send):
            pass

        mw = IdempotencyMiddleware(dummy_app, ttl=60, max_cache_size=100)
        assert mw._ttl == 60
        assert mw._max_cache_size == 100
        assert len(mw._cache) == 0


# ---------------------------------------------------------------------------
# New /ui/* route schema tests (task 13)
# ---------------------------------------------------------------------------


class TestUIRouteSchemas:
    def test_job_status_response(self):
        from tmrvc_serve.routes.ui import JobStatusResponse

        resp = JobStatusResponse(
            job_id="j1",
            job_type="dataset_upload",
            status="completed",
            progress=1.0,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
        )
        assert resp.job_id == "j1"
        assert resp.status == "completed"

    def test_dataset_register_request(self):
        from tmrvc_serve.routes.ui import DatasetRegisterRequest

        req = DatasetRegisterRequest(
            name="test-ds",
            path="/data/test",
            language="en",
        )
        assert req.name == "test-ds"
        assert req.language == "en"

    def test_curation_run_request(self):
        from tmrvc_serve.routes.ui import CurationRunRequest

        req = CurationRunRequest(dataset_id="ds-001")
        assert req.policy == "default"

    def test_workshop_generate_request(self):
        from tmrvc_serve.routes.ui import WorkshopGenerateRequest

        req = WorkshopGenerateRequest(
            character_id="char-a",
            text="Hello world",
        )
        assert req.style_preset == "default"

    def test_eval_submit_request_validation(self):
        from tmrvc_serve.routes.ui import EvalSubmitRequest

        req = EvalSubmitRequest(rating=3.5, notes="Good quality")
        assert 1.0 <= req.rating <= 5.0

        with pytest.raises(Exception):
            EvalSubmitRequest(rating=0.0)  # below minimum

        with pytest.raises(Exception):
            EvalSubmitRequest(rating=6.0)  # above maximum


# ---------------------------------------------------------------------------
# Artifact response contract test (task 22)
# ---------------------------------------------------------------------------


class TestArtifactResponse:
    def test_artifact_response_from_ui_routes(self):
        from tmrvc_serve.routes.ui import ArtifactResponse

        art = ArtifactResponse(
            artifact_id="art-001",
            artifact_type="take_bundle",
            download_url="/artifacts/art-001/download",
            provenance_summary={"take_id": "t1", "character_id": "c1"},
        )
        d = art.model_dump(mode="json")
        assert d["artifact_id"] == "art-001"
        assert d["artifact_type"] == "take_bundle"
        assert d["download_url"] == "/artifacts/art-001/download"
        assert d["provenance_summary"]["take_id"] == "t1"

    def test_artifact_response_from_schemas(self):
        from tmrvc_serve.schemas import ArtifactResponse

        art = ArtifactResponse(
            artifact_id="art-002",
            artifact_type="training_bundle",
            download_url="/artifacts/art-002/download",
        )
        assert art.artifact_type == "training_bundle"
        assert art.expires_at is None

    def test_artifact_types_coverage(self):
        """Verify all three artifact types can be represented."""
        from tmrvc_serve.routes.ui import ArtifactResponse

        for art_type in ("training_bundle", "eval_bundle", "take_bundle"):
            art = ArtifactResponse(
                artifact_id="x",
                artifact_type=art_type,
                download_url="/x",
            )
            assert art.artifact_type == art_type


# ---------------------------------------------------------------------------
# Router registration tests
# ---------------------------------------------------------------------------


class TestUIRouterRegistration:
    def test_prefix_is_ui(self):
        from tmrvc_serve.routes.ui import router

        assert router.prefix == "/ui"

    def test_required_routes_exist(self):
        from tmrvc_serve.routes.ui import router

        route_paths = [r.path for r in router.routes]
        expected_paths = [
            "/ui/datasets/upload",
            "/ui/datasets/register",
            "/ui/jobs/{job_id}",
            "/ui/jobs/{job_id}/events",
            "/ui/curation/runs",
            "/ui/curation/runs/{run_id}/resume",
            "/ui/curation/runs/{run_id}/stop",
            "/ui/curation/records",
            "/ui/curation/records/{record_id}",
            "/ui/workshop/generate",
            "/ui/workshop/takes/{take_id}/pin",
            "/ui/workshop/takes/{take_id}/export",
            "/ui/workshop/sessions",
            "/ui/eval/sessions",
            "/ui/eval/assignments/{assignment_id}",
            "/ui/eval/assignments/{assignment_id}/submit",
        ]
        for ep in expected_paths:
            assert ep in route_paths, f"Missing route: {ep} (available: {sorted(route_paths)})"
