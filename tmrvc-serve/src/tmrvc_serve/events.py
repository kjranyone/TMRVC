"""SSE event types for WebUI job and telemetry streams (Worker 04, task 20).

Every SSE event follows a uniform envelope so that the UI can dispatch on
``event_type`` without inspecting the payload shape.  The ``payload_version``
field allows forward-compatible schema evolution.

Event stream is resumable via ``Last-Event-ID`` or equivalent cursor by
supplying the ``last_event_id`` query parameter on reconnect.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SSEEventType(str, Enum):
    """Canonical SSE event types emitted by /ui/jobs/{job_id}/events."""

    JOB_PROGRESS = "job_progress"
    JOB_BLOCKED_HUMAN = "job_blocked_human"
    JOB_FAILED = "job_failed"
    JOB_COMPLETED = "job_completed"
    TAKE_READY = "take_ready"
    TELEMETRY_UPDATE = "telemetry_update"


class SSEEvent(BaseModel):
    """Uniform SSE event envelope.

    Fields
    ------
    event_type : SSEEventType
        Discriminator used by the UI event dispatcher.
    job_id : str
        Job that originated this event (may be empty for global telemetry).
    object_type : str
        Semantic type of the affected object (e.g. ``"dataset"``, ``"take"``).
    object_id : str
        Unique ID of the affected object.
    timestamp : datetime
        Server-side UTC timestamp when the event was created.
    payload_version : int
        Schema version for forward compatibility (starts at 1).
    data : dict
        Free-form payload whose schema is keyed by ``event_type`` and
        ``payload_version``.
    event_id : str
        Unique event ID for ``Last-Event-ID`` based resumption.
    """

    event_type: SSEEventType
    job_id: str = ""
    object_type: str = ""
    object_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload_version: int = 1
    data: dict[str, Any] = Field(default_factory=dict)
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_sse(self) -> str:
        """Serialize to the SSE wire format (``event: ... \\ndata: ...\\n\\n``)."""
        payload = self.model_dump(mode="json")
        return (
            f"id: {self.event_id}\n"
            f"event: {self.event_type.value}\n"
            f"data: {json.dumps(payload)}\n\n"
        )
