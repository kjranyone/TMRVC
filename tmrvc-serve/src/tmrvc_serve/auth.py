"""Authentication and Authorization Layer for TMRVC API.

Supports:
- API Key authentication (service-to-service)
- JWT authentication (user sessions)
- Rate limiting per key/user
- Usage tracking and quotas
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

API_KEY_HEADER = "X-API-Key"
JWT_HEADER = HTTPBearer()


class AuthConfig(BaseModel):
    """Authentication configuration."""

    api_keys_secret: str = ""  # HMAC secret for API key validation
    jwt_secret: str = ""  # JWT signing secret
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24

    # Rate limits
    default_rate_limit_rpm: int = 60  # requests per minute
    default_rate_limit_concurrent: int = 5  # concurrent sessions

    # Quotas
    default_quota_seconds: int = 3600  # audio seconds per day


def get_config() -> AuthConfig:
    """Load config from environment."""
    return AuthConfig(
        api_keys_secret=os.getenv("TMRVC_API_KEYS_SECRET", ""),
        jwt_secret=os.getenv("TMRVC_JWT_SECRET", "dev-secret-change-in-prod"),
        jwt_algorithm=os.getenv("TMRVC_JWT_ALGORITHM", "HS256"),
        jwt_expire_hours=int(os.getenv("TMRVC_JWT_EXPIRE_HOURS", "24")),
        default_rate_limit_rpm=int(os.getenv("TMRVC_RATE_LIMIT_RPM", "60")),
        default_rate_limit_concurrent=int(
            os.getenv("TMRVC_RATE_LIMIT_CONCURRENT", "5")
        ),
        default_quota_seconds=int(os.getenv("TMRVC_QUOTA_SECONDS", "3600")),
    )


# ============================================================
# Models
# ============================================================


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""

    ADMIN = "admin"  # Full access, no limits
    ENTERPRISE = "enterprise"  # High limits, priority queue
    PRO = "pro"  # Standard limits
    FREE = "free"  # Basic limits, lower priority


@dataclass
class RateLimits:
    """Rate limit configuration per tier."""

    requests_per_minute: int
    concurrent_sessions: int
    audio_seconds_per_day: int
    priority: int  # Lower = higher priority (0 = admin)

    @classmethod
    def for_role(cls, role: UserRole) -> "RateLimits":
        limits = {
            UserRole.ADMIN: cls(
                requests_per_minute=1000,
                concurrent_sessions=50,
                audio_seconds_per_day=999999,
                priority=0,
            ),
            UserRole.ENTERPRISE: cls(
                requests_per_minute=300,
                concurrent_sessions=20,
                audio_seconds_per_day=36000,  # 10 hours
                priority=1,
            ),
            UserRole.PRO: cls(
                requests_per_minute=120,
                concurrent_sessions=10,
                audio_seconds_per_day=7200,  # 2 hours
                priority=2,
            ),
            UserRole.FREE: cls(
                requests_per_minute=30,
                concurrent_sessions=3,
                audio_seconds_per_day=600,  # 10 minutes
                priority=3,
            ),
        }
        return limits.get(role, limits[UserRole.FREE])


@dataclass
class APIKey:
    """API key metadata."""

    key_hash: str  # SHA256 hash of the key
    key_prefix: str  # First 8 chars for identification
    tenant_id: str
    user_id: str
    role: UserRole
    rate_limits: RateLimits
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # None = never expires

    # Usage tracking
    total_requests: int = 0
    total_audio_seconds: float = 0.0
    last_request: float | None = None


@dataclass
class UserSession:
    """JWT-authenticated user session."""

    user_id: str
    tenant_id: str
    role: UserRole
    email: str | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 86400)


@dataclass
class RequestContext:
    """Request context with auth info."""

    tenant_id: str
    user_id: str
    role: UserRole
    rate_limits: RateLimits
    api_key: str | None = None  # Key prefix for logging
    session: UserSession | None = None


# ============================================================
# API Key Store (In-memory, replace with DB in production)
# ============================================================


class APIKeyStore:
    """Manages API keys and their metadata.

    Production: Replace with Redis/PostgreSQL.
    """

    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # key_hash -> APIKey
        self._tenant_keys: dict[str, set[str]] = {}  # tenant_id -> set of key_hashes

    def _hash_key(self, api_key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _get_prefix(self, api_key: str) -> str:
        """Get display prefix for logging."""
        return api_key[:8] + "..." if len(api_key) > 8 else api_key

    def create_key(
        self,
        tenant_id: str,
        user_id: str,
        role: UserRole,
        expires_days: int | None = None,
    ) -> str:
        """Generate new API key."""
        # Generate random key: tmrvc_<32 random chars>
        random_part = os.urandom(24).hex()
        api_key = f"tmrvc_{random_part}"

        key_hash = self._hash_key(api_key)
        key_prefix = self._get_prefix(api_key)

        expires_at = None
        if expires_days:
            expires_at = time.time() + expires_days * 86400

        key_meta = APIKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            rate_limits=RateLimits.for_role(role),
            expires_at=expires_at,
        )

        self._keys[key_hash] = key_meta

        if tenant_id not in self._tenant_keys:
            self._tenant_keys[tenant_id] = set()
        self._tenant_keys[tenant_id].add(key_hash)

        logger.info(
            "Created API key %s for tenant=%s role=%s", key_prefix, tenant_id, role
        )
        return api_key

    def validate_key(self, api_key: str) -> APIKey | None:
        """Validate API key and return metadata."""
        if not api_key.startswith("tmrvc_"):
            return None

        key_hash = self._hash_key(api_key)
        key_meta = self._keys.get(key_hash)

        if not key_meta:
            return None

        if not key_meta.enabled:
            return None

        if key_meta.expires_at and time.time() > key_meta.expires_at:
            return None

        # Update usage
        key_meta.total_requests += 1
        key_meta.last_request = time.time()

        return key_meta

    def revoke_key(self, key_prefix: str) -> bool:
        """Revoke an API key by prefix."""
        for key_hash, key_meta in self._keys.items():
            if key_meta.key_prefix == key_prefix:
                key_meta.enabled = False
                logger.info("Revoked API key %s", key_prefix)
                return True
        return False

    def get_tenant_keys(self, tenant_id: str) -> list[APIKey]:
        """Get all keys for a tenant."""
        key_hashes = self._tenant_keys.get(tenant_id, set())
        return [self._keys[h] for h in key_hashes if h in self._keys]

    def record_usage(self, api_key: str, audio_seconds: float) -> None:
        """Record audio usage for quota tracking."""
        key_hash = self._hash_key(api_key)
        if key_hash in self._keys:
            self._keys[key_hash].total_audio_seconds += audio_seconds


# Global store (singleton)
_key_store = APIKeyStore()


def get_key_store() -> APIKeyStore:
    return _key_store


# ============================================================
# Rate Limiter
# ============================================================


class RateLimiter:
    """Token bucket rate limiter with sliding window.

    Production: Replace with Redis with Lua scripts.
    """

    def __init__(self):
        # tenant_id -> {timestamp: count}
        self._requests: dict[str, list[float]] = {}
        # tenant_id -> concurrent count
        self._concurrent: dict[str, int] = {}
        # tenant_id -> daily audio seconds
        self._daily_usage: dict[str, dict[str, float]] = {}  # {date: seconds}

    def check_rate_limit(
        self,
        tenant_id: str,
        limits: RateLimits,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is within rate limits.

        Returns: (allowed, metadata)
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        if tenant_id in self._requests:
            self._requests[tenant_id] = [
                ts for ts in self._requests[tenant_id] if ts > minute_ago
            ]
        else:
            self._requests[tenant_id] = []

        # Check RPM
        current_rpm = len(self._requests[tenant_id])
        if current_rpm >= limits.requests_per_minute:
            return False, {
                "reason": "rate_limit_rpm",
                "limit": limits.requests_per_minute,
                "current": current_rpm,
                "retry_after": 60 - (now - min(self._requests[tenant_id])),
            }

        # Check concurrent
        current_concurrent = self._concurrent.get(tenant_id, 0)
        if current_concurrent >= limits.concurrent_sessions:
            return False, {
                "reason": "rate_limit_concurrent",
                "limit": limits.concurrent_sessions,
                "current": current_concurrent,
            }

        # Check daily quota
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily = self._daily_usage.get(tenant_id, {})
        today_usage = daily.get(today, 0)
        if today_usage >= limits.audio_seconds_per_day:
            return False, {
                "reason": "quota_exceeded",
                "limit": limits.audio_seconds_per_day,
                "current": today_usage,
            }

        # Allow request
        self._requests[tenant_id].append(now)
        return True, {"remaining_rpm": limits.requests_per_minute - current_rpm - 1}

    def start_session(self, tenant_id: str) -> None:
        """Increment concurrent session count."""
        self._concurrent[tenant_id] = self._concurrent.get(tenant_id, 0) + 1

    def end_session(self, tenant_id: str, audio_seconds: float = 0) -> None:
        """Decrement concurrent session and record usage."""
        current = self._concurrent.get(tenant_id, 0)
        if current > 0:
            self._concurrent[tenant_id] = current - 1

        # Record daily usage
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if tenant_id not in self._daily_usage:
            self._daily_usage[tenant_id] = {}
        self._daily_usage[tenant_id][today] = (
            self._daily_usage[tenant_id].get(today, 0) + audio_seconds
        )


# Global limiter
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


# ============================================================
# JWT Handling
# ============================================================


def create_jwt_token(
    user_id: str,
    tenant_id: str,
    role: UserRole,
    email: str | None = None,
) -> str:
    """Create JWT token for user session."""
    import jwt

    config = get_config()
    expires = datetime.utcnow() + timedelta(hours=config.jwt_expire_hours)

    payload = {
        "sub": user_id,
        "tenant": tenant_id,
        "role": role.value,
        "email": email,
        "exp": expires,
        "iat": datetime.utcnow(),
    }

    return jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)


def decode_jwt_token(token: str) -> UserSession | None:
    """Decode and validate JWT token."""
    import jwt

    config = get_config()

    try:
        payload = jwt.decode(
            token, config.jwt_secret, algorithms=[config.jwt_algorithm]
        )
        return UserSession(
            user_id=payload["sub"],
            tenant_id=payload["tenant"],
            role=UserRole(payload["role"]),
            email=payload.get("email"),
            expires_at=payload["exp"],
        )
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token: %s", e)
        return None


# ============================================================
# FastAPI Dependencies
# ============================================================

api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
jwt_bearer = HTTPBearer(auto_error=False)


async def get_api_key_auth(
    request: Request,
    api_key: Annotated[str | None, Depends(api_key_header)] = None,
) -> RequestContext | None:
    """Authenticate via API key."""
    if not api_key:
        return None

    store = get_key_store()
    key_meta = store.validate_key(api_key)

    if not key_meta:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    limiter = get_rate_limiter()
    allowed, info = limiter.check_rate_limit(
        key_meta.tenant_id,
        key_meta.rate_limits,
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=info,
        )

    return RequestContext(
        tenant_id=key_meta.tenant_id,
        user_id=key_meta.user_id,
        role=key_meta.role,
        rate_limits=key_meta.rate_limits,
        api_key=key_meta.key_prefix,
    )


async def get_jwt_auth(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(jwt_bearer)
    ] = None,
) -> RequestContext | None:
    """Authenticate via JWT."""
    if not credentials:
        return None

    session = decode_jwt_token(credentials.credentials)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    limits = RateLimits.for_role(session.role)

    limiter = get_rate_limiter()
    allowed, info = limiter.check_rate_limit(session.tenant_id, limits)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=info,
        )

    return RequestContext(
        tenant_id=session.tenant_id,
        user_id=session.user_id,
        role=session.role,
        rate_limits=limits,
        session=session,
    )


async def get_auth(
    request: Request,
    api_key_ctx: Annotated[RequestContext | None, Depends(get_api_key_auth)] = None,
    jwt_ctx: Annotated[RequestContext | None, Depends(get_jwt_auth)] = None,
) -> RequestContext:
    """Get authentication context (API key or JWT).

    Raises 401 if neither is valid.
    """
    ctx = api_key_ctx or jwt_ctx

    if not ctx:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key or JWT token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Store context in request state for later use
    request.state.auth = ctx
    return ctx


async def get_optional_auth(
    api_key_ctx: Annotated[RequestContext | None, Depends(get_api_key_auth)] = None,
    jwt_ctx: Annotated[RequestContext | None, Depends(get_jwt_auth)] = None,
) -> RequestContext | None:
    """Optional authentication (for endpoints that work with/without auth)."""
    return api_key_ctx or jwt_ctx


def require_role(*roles: UserRole) -> Callable:
    """Dependency that requires specific role."""

    async def check_role(
        ctx: Annotated[RequestContext, Depends(get_auth)],
    ) -> RequestContext:
        if ctx.role not in roles and ctx.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {ctx.role} not authorized. Required: {roles}",
            )
        return ctx

    return check_role


# Type alias for dependency injection
AuthContext = Annotated[RequestContext, Depends(get_auth)]
OptionalAuth = Annotated[RequestContext | None, Depends(get_optional_auth)]
AdminOnly = Annotated[RequestContext, Depends(require_role(UserRole.ADMIN))]
EnterpriseOnly = Annotated[
    RequestContext, Depends(require_role(UserRole.ENTERPRISE, UserRole.ADMIN))
]
