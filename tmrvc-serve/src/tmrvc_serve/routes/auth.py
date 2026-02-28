"""Authentication endpoints for API key and JWT management."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

from tmrvc_serve.auth import (
    AuthConfig,
    APIKeyStore,
    RateLimits,
    RequestContext,
    UserRole,
    create_jwt_token,
    decode_jwt_token,
    get_auth,
    get_config,
    get_key_store,
    get_rate_limiter,
    require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================
# Request/Response Models
# ============================================================


class TokenRequest(BaseModel):
    """Login request for JWT token."""

    email: EmailStr
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: str | None = None


class RefreshRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class APIKeyCreateRequest(BaseModel):
    """Request to create new API key."""

    user_id: str = Field(..., min_length=1, max_length=64)
    role: UserRole = UserRole.PRO
    expires_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key creation response."""

    api_key: str
    key_prefix: str
    role: str
    expires_at: str | None


class APIKeyInfo(BaseModel):
    """API key information."""

    prefix: str
    role: str
    enabled: bool
    created_at: str
    expires_at: str | None
    total_requests: int
    total_audio_seconds: float
    last_request: str | None


class APIKeyListResponse(BaseModel):
    """List of API keys."""

    keys: list[APIKeyInfo]


class LogoutResponse(BaseModel):
    """Logout response."""

    success: bool
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str
    detail: dict | None = None


# ============================================================
# Mock User Store (Replace with database in production)
# ============================================================

# Demo users for development
_DEMO_USERS = {
    "admin@tmrvc.example.com": {
        "password": "admin123",  # In production: use bcrypt hash
        "tenant_id": "admin",
        "role": UserRole.ADMIN,
    },
    "enterprise@tmrvc.example.com": {
        "password": "enterprise123",
        "tenant_id": "enterprise",
        "role": UserRole.ENTERPRISE,
    },
    "pro@tmrvc.example.com": {
        "password": "pro12345",  # 8 chars minimum
        "tenant_id": "pro",
        "role": UserRole.PRO,
    },
    "free@tmrvc.example.com": {
        "password": "free12345",  # 8 chars minimum
        "tenant_id": "free",
        "role": UserRole.FREE,
    },
}


def _validate_user(email: str, password: str) -> dict | None:
    """Validate user credentials (mock implementation)."""
    user = _DEMO_USERS.get(email)
    if user and user["password"] == password:
        return {
            "user_id": email,
            "tenant_id": user["tenant_id"],
            "role": user["role"],
        }
    return None


# ============================================================
# Endpoints
# ============================================================


@router.post("/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    """Login with email/password to get JWT token.

    Demo credentials:
    - admin@tmrvc.local / admin123
    - enterprise@tmrvc.local / enterprise123
    - pro@tmrvc.local / pro123
    - free@tmrvc.local / free123
    """
    user = _validate_user(request.email, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_credentials",
                "message": "Invalid email or password",
            },
        )

    config = get_config()

    access_token = create_jwt_token(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role=user["role"],
        email=request.email,
    )

    # Generate refresh token (simplified - use proper refresh token in production)
    refresh_token = create_jwt_token(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role=user["role"],
        email=request.email,
    )

    logger.info(
        "User logged in: %s (tenant=%s role=%s)",
        request.email,
        user["tenant_id"],
        user["role"],
    )

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=config.jwt_expire_hours * 3600,
        refresh_token=refresh_token,
    )


@router.post("/token/form", response_model=TokenResponse)
async def login_form(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible login (for Swagger UI)."""
    request = TokenRequest(email=form_data.username, password=form_data.password)
    return await login(request)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using refresh token."""
    session = decode_jwt_token(request.refresh_token)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_refresh_token",
                "message": "Invalid or expired refresh token",
            },
        )

    config = get_config()

    access_token = create_jwt_token(
        user_id=session.user_id,
        tenant_id=session.tenant_id,
        role=session.role,
        email=session.email,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=config.jwt_expire_hours * 3600,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(ctx: RequestContext = Depends(get_auth)):
    """Logout and invalidate token.

    Note: In production, add token to revocation list in Redis.
    """
    logger.info("User logged out: %s (tenant=%s)", ctx.user_id, ctx.tenant_id)
    return LogoutResponse(success=True, message="Logged out successfully")


@router.get("/me")
async def get_current_user(ctx: RequestContext = Depends(get_auth)):
    """Get current authenticated user info."""
    return {
        "user_id": ctx.user_id,
        "tenant_id": ctx.tenant_id,
        "role": ctx.role,
        "rate_limits": {
            "requests_per_minute": ctx.rate_limits.requests_per_minute,
            "concurrent_sessions": ctx.rate_limits.concurrent_sessions,
            "audio_seconds_per_day": ctx.rate_limits.audio_seconds_per_day,
        },
    }


# ============================================================
# API Key Management (Admin only)
# ============================================================


@router.get("/keys", response_model=APIKeyListResponse)
async def list_api_keys(ctx: RequestContext = Depends(require_role(UserRole.ADMIN))):
    """List all API keys for tenant (admin only)."""
    store = get_key_store()
    keys = store.get_tenant_keys(ctx.tenant_id)

    return APIKeyListResponse(
        keys=[
            APIKeyInfo(
                prefix=k.key_prefix,
                role=k.role.value,
                enabled=k.enabled,
                created_at=datetime.fromtimestamp(k.created_at).isoformat(),
                expires_at=datetime.fromtimestamp(k.expires_at).isoformat()
                if k.expires_at
                else None,
                total_requests=k.total_requests,
                total_audio_seconds=k.total_audio_seconds,
                last_request=datetime.fromtimestamp(k.last_request).isoformat()
                if k.last_request
                else None,
            )
            for k in keys
        ]
    )


@router.post("/keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    ctx: RequestContext = Depends(require_role(UserRole.ADMIN)),
):
    """Create new API key (admin only).

    The API key is only returned once - store it securely.
    """
    store = get_key_store()

    # Limit role to max of admin's role
    role = request.role
    if ctx.role != UserRole.ADMIN and role.value > ctx.role.value:
        role = ctx.role

    api_key = store.create_key(
        tenant_id=ctx.tenant_id,
        user_id=request.user_id,
        role=role,
        expires_days=request.expires_days,
    )

    key_hash = store._hash_key(api_key)
    key_meta = store._keys.get(key_hash)

    logger.info(
        "API key created: %s tenant=%s user=%s role=%s",
        key_meta.key_prefix,
        ctx.tenant_id,
        request.user_id,
        role,
    )

    return APIKeyResponse(
        api_key=api_key,
        key_prefix=key_meta.key_prefix,
        role=role.value,
        expires_at=datetime.fromtimestamp(key_meta.expires_at).isoformat()
        if key_meta.expires_at
        else None,
    )


@router.delete("/keys/{key_prefix}")
async def revoke_api_key(
    key_prefix: str,
    ctx: RequestContext = Depends(require_role(UserRole.ADMIN)),
):
    """Revoke an API key (admin only)."""
    store = get_key_store()
    success = store.revoke_key(key_prefix)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"API key {key_prefix} not found"},
        )

    logger.info("API key revoked: %s admin=%s", key_prefix, ctx.user_id)

    return {"revoked": key_prefix}


@router.get("/usage")
async def get_usage(ctx: RequestContext = Depends(get_auth)):
    """Get current usage and quota for tenant."""
    limiter = get_rate_limiter()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    daily_usage = limiter._daily_usage.get(ctx.tenant_id, {}).get(today, 0)
    concurrent = limiter._concurrent.get(ctx.tenant_id, 0)

    return {
        "tenant_id": ctx.tenant_id,
        "role": ctx.role,
        "usage": {
            "daily_audio_seconds": daily_usage,
            "daily_quota_seconds": ctx.rate_limits.audio_seconds_per_day,
            "concurrent_sessions": concurrent,
            "concurrent_limit": ctx.rate_limits.concurrent_sessions,
            "remaining_quota_seconds": max(
                0, ctx.rate_limits.audio_seconds_per_day - daily_usage
            ),
        },
        "limits": {
            "requests_per_minute": ctx.rate_limits.requests_per_minute,
            "concurrent_sessions": ctx.rate_limits.concurrent_sessions,
            "audio_seconds_per_day": ctx.rate_limits.audio_seconds_per_day,
        },
    }
