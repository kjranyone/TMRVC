"""Tests for admin management routes (v3).

Covers:
- AdminHealthResponse schema validation
- TelemetryResponse schema validation
- RuntimeContractResponse schema validation
- LoadModelRequest schema validation
- Router prefix is /admin
"""

from __future__ import annotations

import pytest

from tmrvc_serve.routes.admin import (
    AdminHealthResponse,
    LoadModelRequest,
    RuntimeContractResponse,
    TelemetryResponse,
    router,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestAdminHealthResponse:
    def test_defaults(self):
        resp = AdminHealthResponse()
        assert resp.status == "ok"
        assert resp.models_loaded is False
        assert resp.device == "cpu"
        assert resp.cuda_available is False
        assert resp.cuda_memory_allocated_mb == 0.0
        assert resp.cuda_memory_reserved_mb == 0.0
        assert resp.uptime_seconds == 0.0

    def test_with_custom_values(self):
        resp = AdminHealthResponse(
            status="ok",
            models_loaded=True,
            device="cuda:0",
            cuda_available=True,
            cuda_memory_allocated_mb=1024.0,
            cuda_memory_reserved_mb=2048.0,
            uptime_seconds=123.45,
        )
        assert resp.models_loaded is True
        assert resp.device == "cuda:0"
        assert resp.cuda_memory_allocated_mb == 1024.0

    def test_serialization_roundtrip(self):
        resp = AdminHealthResponse(models_loaded=True, device="cuda:1")
        data = resp.model_dump()
        restored = AdminHealthResponse(**data)
        assert restored.models_loaded is True
        assert restored.device == "cuda:1"


class TestTelemetryResponse:
    def test_defaults(self):
        resp = TelemetryResponse()
        assert resp.vram_allocated_mb == 0.0
        assert resp.vram_reserved_mb == 0.0
        assert resp.avg_tts_latency_ms == 0.0
        assert resp.avg_vc_latency_ms == 0.0
        assert resp.tts_mode == "pointer"
        assert resp.model_checkpoint == ""

    def test_with_custom_values(self):
        resp = TelemetryResponse(
            vram_allocated_mb=512.0,
            avg_tts_latency_ms=35.0,
            tts_mode="pointer",
            model_checkpoint="/path/to/model.pt",
        )
        assert resp.vram_allocated_mb == 512.0
        assert resp.avg_tts_latency_ms == 35.0
        assert resp.model_checkpoint == "/path/to/model.pt"


class TestRuntimeContractResponse:
    def test_defaults(self):
        resp = RuntimeContractResponse()
        assert resp.tts_mode == "pointer"
        assert "text_index" in resp.pointer_fields
        assert "progress" in resp.pointer_fields
        assert "finished" in resp.pointer_fields
        assert "stall_frames" in resp.pointer_fields
        assert resp.voice_state_dims == 8
        assert resp.supports_few_shot is True
        assert resp.supports_dialogue_context is True
        assert resp.supports_acting_intent is True

    def test_pacing_controls(self):
        resp = RuntimeContractResponse()
        for ctrl in ("pace", "hold_bias", "boundary_bias", "phrase_pressure", "breath_tendency"):
            assert ctrl in resp.pacing_controls, f"Missing pacing control: {ctrl}"

    def test_serialization_roundtrip(self):
        resp = RuntimeContractResponse(tts_mode="legacy_duration")
        data = resp.model_dump()
        restored = RuntimeContractResponse(**data)
        assert restored.tts_mode == "legacy_duration"


class TestLoadModelRequest:
    def test_required_fields(self):
        req = LoadModelRequest(
            uclm_checkpoint="/path/uclm.pt",
            codec_checkpoint="/path/codec.pt",
        )
        assert req.uclm_checkpoint == "/path/uclm.pt"
        assert req.codec_checkpoint == "/path/codec.pt"

    def test_missing_fields_raises(self):
        with pytest.raises(Exception):
            LoadModelRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Router prefix test
# ---------------------------------------------------------------------------


class TestRouterPrefix:
    def test_prefix_is_admin(self):
        assert router.prefix == "/admin"

    def test_router_has_routes(self):
        route_paths = [r.path for r in router.routes]
        # Routes include the router prefix in their path
        assert "/admin/health" in route_paths
        assert "/admin/telemetry" in route_paths
        assert "/admin/runtime_contract" in route_paths
        assert "/admin/load_model" in route_paths
        assert "/admin/models" in route_paths
