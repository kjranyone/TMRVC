# TMRVC API Authentication & Authorization Specification

## 1. Overview

TMRVC API はマルチテナント対応の Voice Conversion / TTS サービスとして、以下の認証・認可機能を提供する:

- **API Key認証**: サービス間通信向け
- **JWT認証**: ユーザーセッション向け
- **ロールベースアクセス制御**: 4段階のロール
- **レート制限**: RPM / 同時接続 / 日次クォータ
- **使用量追跡**: テナントごとの利用統計

## 2. Authentication Methods

### 2.1 API Key Authentication

Service-to-service 通信向けの静的認証。

**Header:**
```
X-API-Key: tmrvc_<48 hex chars>
```

**Key Format:**
```
tmrvc_<random 48 hex chars>
例: tmrvc_a1b2c3d4e5f6789012345678901234567890abcd
```

**Validation:**
1. Key が `tmrvc_` プレフィックスを持つか
2. SHA256 hash を計算しストアと照合
3. 有効期限チェック
4. enabled フラグチェック

### 2.2 JWT Authentication

User session 向けのトークン認証。

**Header:**
```
Authorization: Bearer <jwt_token>
```

**JWT Payload:**
```json
{
  "sub": "user_id",
  "tenant": "tenant_id",
  "role": "pro",
  "email": "user@example.com",
  "exp": 1704067200,
  "iat": 1703980800
}
```

**Token Generation:**
```python
POST /auth/token
{
  "email": "user@example.com",
  "password": "..."
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 86400
}
```

## 3. Authorization (RBAC)

### 3.1 Roles

| Role | Priority | Description |
|------|----------|-------------|
| `admin` | 0 | Full access, no limits |
| `enterprise` | 1 | High limits, priority queue |
| `pro` | 2 | Standard limits |
| `free` | 3 | Basic limits, lower priority |

### 3.2 Rate Limits per Role

| Role | Requests/min | Concurrent Sessions | Daily Audio Quota |
|------|-------------|---------------------|-------------------|
| `admin` | 1000 | 50 | Unlimited |
| `enterprise` | 300 | 20 | 10 hours |
| `pro` | 120 | 10 | 2 hours |
| `free` | 30 | 3 | 10 minutes |

### 3.3 Endpoint Permissions

| Endpoint | free | pro | enterprise | admin |
|----------|------|-----|------------|-------|
| `WS /vc/stream` | ✓ | ✓ | ✓ | ✓ |
| `POST /tts` | ✓ | ✓ | ✓ | ✓ |
| `POST /tts/stream` | ✓ | ✓ | ✓ | ✓ |
| `GET /vc/stats` | ✗ | ✗ | ✓ | ✓ |
| `GET /auth/keys` | ✗ | ✗ | ✗ | ✓ |
| `POST /auth/keys` | ✗ | ✗ | ✗ | ✓ |
| `DELETE /auth/keys/{prefix}` | ✗ | ✗ | ✗ | ✓ |

## 4. Rate Limiting

### 4.1 Sliding Window RPM

- 直近60秒のリクエスト数をカウント
- 制限超過時は `429 Too Many Requests` を返却
- `Retry-After` ヘッダーで再試行可能時刻を通知

### 4.2 Concurrent Session Limit

- WebSocket接続ごとにカウント
- 接続開始時にインクリメント、終了時にデクリメント
- 制限超過時は `1013 Try Again Later` で接続拒否

### 4.3 Daily Quota

- 1日の音声処理秒数を追跡
- UTC 00:00 にリセット
- 制限超過時はエラーレスポンス

## 5. API Endpoints

### 5.1 Authentication

```
POST /auth/token
  Request:  { "email": string, "password": string }
  Response: { "access_token": string, "token_type": "Bearer", "expires_in": int }
  Auth:     None (public)

POST /auth/refresh
  Request:  { "refresh_token": string }
  Response: { "access_token": string, "token_type": "Bearer", "expires_in": int }
  Auth:     None (refresh token required)

POST /auth/logout
  Request:  {}
  Response: { "success": bool }
  Auth:     JWT
```

### 5.2 API Key Management (Admin only)

```
GET /auth/keys
  Response: { "keys": [{ "prefix": string, "role": string, "enabled": bool, ... }] }
  Auth:     Admin

POST /auth/keys
  Request:  { "user_id": string, "role": string, "expires_days": int? }
  Response: { "api_key": string, "role": string }
  Auth:     Admin

DELETE /auth/keys/{prefix}
  Response: { "revoked": string }
  Auth:     Admin
```

### 5.3 Voice Conversion

```
WebSocket /vc/stream?api_key={key}
  Auth:     API Key (query param for WebSocket compatibility)

POST /vc/batch
  Request:  { "speaker_embedding": [192 floats], "audio": [N floats] }
  Response: { "rtf": float, "elapsed_ms": float, "audio": [N floats] }
  Auth:     API Key or JWT

GET /vc/stats
  Response: { "active_sessions": int, "max_sessions": int, "is_ready": bool }
  Auth:     Enterprise or Admin
```

## 6. Error Responses

### 6.1 Authentication Errors

```json
// 401 Unauthorized
{
  "error": "unauthorized",
  "message": "API key or JWT token required",
  "detail": null
}

// 401 Unauthorized (invalid key)
{
  "error": "invalid_api_key",
  "message": "Invalid or expired API key",
  "detail": null
}

// 401 Unauthorized (expired token)
{
  "error": "token_expired",
  "message": "JWT token has expired",
  "detail": { "expired_at": "2024-01-01T00:00:00Z" }
}
```

### 6.2 Authorization Errors

```json
// 403 Forbidden
{
  "error": "forbidden",
  "message": "Role 'free' not authorized. Required: ['enterprise', 'admin']",
  "detail": { "current_role": "free", "required_roles": ["enterprise", "admin"] }
}
```

### 6.3 Rate Limit Errors

```json
// 429 Too Many Requests (RPM)
{
  "error": "rate_limit_rpm",
  "message": "Rate limit exceeded",
  "detail": {
    "limit": 30,
    "current": 30,
    "retry_after": 45.2
  }
}

// 429 Too Many Requests (Concurrent)
{
  "error": "rate_limit_concurrent",
  "message": "Concurrent session limit exceeded",
  "detail": {
    "limit": 3,
    "current": 3
  }
}

// 429 Too Many Requests (Quota)
{
  "error": "quota_exceeded",
  "message": "Daily audio quota exceeded",
  "detail": {
    "limit_seconds": 600,
    "used_seconds": 605,
    "resets_at": "2024-01-02T00:00:00Z"
  }
}
```

## 7. Data Models

### 7.1 API Key

```typescript
interface APIKey {
  key_hash: string;        // SHA256(api_key)
  key_prefix: string;      // "tmrvc_a1b2..."
  tenant_id: string;
  user_id: string;
  role: "admin" | "enterprise" | "pro" | "free";
  enabled: boolean;
  created_at: number;      // Unix timestamp
  expires_at: number | null;
  
  // Usage tracking
  total_requests: number;
  total_audio_seconds: number;
  last_request: number | null;
}
```

### 7.2 RequestContext

```typescript
interface RequestContext {
  tenant_id: string;
  user_id: string;
  role: string;
  rate_limits: RateLimits;
  api_key?: string;        // Key prefix (for logging)
  session?: UserSession;   // If JWT auth
}
```

### 7.3 RateLimits

```typescript
interface RateLimits {
  requests_per_minute: number;
  concurrent_sessions: number;
  audio_seconds_per_day: number;
  priority: number;        // 0 = highest
}
```

## 8. WebSocket Authentication

WebSocket はカスタムヘッダーをサポートしないため、以下の方法で認証:

### 8.1 Query Parameter

```
ws://localhost:8000/vc/stream?api_key=tmrvc_a1b2c3...
```

### 8.2 First Message Auth (Alternative)

```
1. Connect (unauthenticated)
2. Client sends: { "type": "auth", "api_key": "..." }
3. Server responds: { "type": "auth_ok", "session_id": "..." }
4. Continue with normal streaming
```

## 9. Production Configuration

### 9.1 Environment Variables

```bash
# Secrets
TMRVC_API_KEYS_SECRET=your-hmac-secret-min-32-chars
TMRVC_JWT_SECRET=your-jwt-secret-min-32-chars

# JWT
TMRVC_JWT_ALGORITHM=HS256
TMRVC_JWT_EXPIRE_HOURS=24

# Rate Limits (defaults)
TMRVC_RATE_LIMIT_RPM=60
TMRVC_RATE_LIMIT_CONCURRENT=5
TMRVC_QUOTA_SECONDS=3600
```

### 9.2 Redis Storage (Production)

Replace in-memory stores with Redis:

```
# Key format
api_key:{key_hash} -> APIKey (JSON)
tenant:{tenant_id}:keys -> SET of key_hashes
tenant:{tenant_id}:rpm:{minute} -> COUNT
tenant:{tenant_id}:concurrent -> COUNT
tenant:{tenant_id}:quota:{date} -> FLOAT (seconds)
```

### 9.3 Database Schema (PostgreSQL)

```sql
CREATE TABLE api_keys (
  key_hash VARCHAR(64) PRIMARY KEY,
  key_prefix VARCHAR(16) NOT NULL,
  tenant_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  role VARCHAR(16) NOT NULL,
  enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT NOW(),
  expires_at TIMESTAMP,
  total_requests BIGINT DEFAULT 0,
  total_audio_seconds FLOAT DEFAULT 0,
  last_request TIMESTAMP
);

CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
```

## 10. Security Considerations

### 10.1 API Key Storage

- 平文キーは保存しない (SHA256 hash のみ)
- キー生成時にのみ1回返却 (再表示不可)
- 失効したキーはログに残さない

### 10.2 JWT Security

- `HS256` アルゴリズム使用
- 有効期限は最大24時間
- Refresh token は別途管理
- トークン失効リスト対応可能

### 10.3 Rate Limiting

- テナント単位で制限 (IPベースではない)
- 分散環境では Redis で一元管理
- ブルートフォース攻撃防止

### 10.4 Audit Logging

```
[auth] API key created: tmrvc_a1b2... tenant=T1 role=pro
[auth] Session started: sid=abc123 tenant=T1 user=U1
[auth] Session closed: sid=abc123 audio=45.2s
[auth] Rate limit exceeded: tenant=T1 type=rpm current=31
[auth] API key revoked: tmrvc_a1b2... admin=admin@company.com
```

## 11. Implementation Checklist

- [x] API Key generation and validation
- [x] JWT token generation and validation
- [x] Role-based access control
- [x] RPM rate limiting (sliding window)
- [x] Concurrent session limiting
- [x] Daily quota tracking
- [x] FastAPI dependencies
- [x] WebSocket query param auth
- [ ] Refresh token flow
- [ ] Redis storage backend
- [ ] PostgreSQL key store
- [ ] Audit logging
- [ ] Key rotation API
- [ ] Tenant management API
