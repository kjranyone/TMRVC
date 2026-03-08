"""Idempotency and optimistic-locking middleware (Worker 04, tasks 21).

Provides:
- ``IdempotencyMiddleware`` -- ASGI middleware that caches responses keyed by
  the ``Idempotency-Key`` request header.
- ``ConflictType`` -- typed conflict reasons for UI-originated write failures.
- ``raise_conflict`` -- convenience helper to raise a structured 409 response.
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, Callable

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conflict types (task 21)
# ---------------------------------------------------------------------------


class ConflictType(str, Enum):
    """Typed conflict reasons returned in 409 responses."""

    STALE_VERSION = "stale_version"
    LOCKED_BY_OTHER = "locked_by_other"
    ALREADY_SUBMITTED = "already_submitted"
    POLICY_FORBIDDEN = "policy_forbidden"


class ConflictDetail(BaseModel):
    """Structured body for 409 Conflict responses."""

    conflict_type: ConflictType
    message: str = ""
    current_version: int | None = None
    locked_by: str | None = None


def raise_conflict(
    conflict_type: ConflictType,
    message: str = "",
    *,
    current_version: int | None = None,
    locked_by: str | None = None,
) -> None:
    """Raise an ``HTTPException`` with a structured 409 body."""
    detail = ConflictDetail(
        conflict_type=conflict_type,
        message=message,
        current_version=current_version,
        locked_by=locked_by,
    )
    raise HTTPException(status_code=409, detail=detail.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Idempotency cache entry
# ---------------------------------------------------------------------------

_DEFAULT_TTL_SECONDS: int = 300  # 5 minutes


class _CacheEntry:
    __slots__ = ("status_code", "body", "headers", "created_at", "ttl")

    def __init__(
        self,
        status_code: int,
        body: bytes,
        headers: dict[str, str],
        ttl: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers
        self.created_at = time.monotonic()
        self.ttl = ttl

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl


# ---------------------------------------------------------------------------
# Idempotency middleware
# ---------------------------------------------------------------------------


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware implementing idempotency-key based response caching.

    When a request includes the ``Idempotency-Key`` header:
    - If the key is already cached and not expired, the cached response is
      returned immediately.
    - Otherwise the request is processed normally and the response is cached
      under that key.

    The cache is in-memory (a plain ``dict``).  Expired entries are lazily
    evicted on each lookup.

    Parameters
    ----------
    app : ASGI application
    ttl : int
        Time-to-live in seconds for cached responses (default 300).
    max_cache_size : int
        Maximum number of entries before oldest entries are evicted (default 4096).
    """

    def __init__(self, app: Any, ttl: int = _DEFAULT_TTL_SECONDS, max_cache_size: int = 4096) -> None:
        super().__init__(app)
        self._cache: dict[str, _CacheEntry] = {}
        self._ttl = ttl
        self._max_cache_size = max_cache_size

    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove expired entries from the cache."""
        expired_keys = [k for k, v in self._cache.items() if v.expired]
        for k in expired_keys:
            del self._cache[k]

    def _evict_oldest(self) -> None:
        """Evict the oldest entry when cache exceeds max size."""
        if len(self._cache) <= self._max_cache_size:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]

    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        idem_key = request.headers.get("idempotency-key") or request.headers.get("Idempotency-Key")

        if idem_key is None:
            return await call_next(request)

        # Check cache
        self._evict_expired()
        cached = self._cache.get(idem_key)
        if cached is not None and not cached.expired:
            logger.debug("Idempotency cache hit for key=%s", idem_key)
            return Response(
                content=cached.body,
                status_code=cached.status_code,
                headers=cached.headers,
                media_type="application/json",
            )

        # Process request
        response = await call_next(request)

        # Read and cache the response body
        body_parts: list[bytes] = []
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            if isinstance(chunk, str):
                body_parts.append(chunk.encode("utf-8"))
            else:
                body_parts.append(chunk)
        body = b"".join(body_parts)

        # Only cache successful responses (2xx)
        if 200 <= response.status_code < 300:
            resp_headers = dict(response.headers)
            entry = _CacheEntry(
                status_code=response.status_code,
                body=body,
                headers=resp_headers,
                ttl=self._ttl,
            )
            self._cache[idem_key] = entry
            self._evict_oldest()
            logger.debug("Idempotency cache store for key=%s", idem_key)

        # Return a fresh response with the consumed body
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type", "application/json"),
        )
