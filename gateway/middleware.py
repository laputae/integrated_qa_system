import json
import time
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from base import RequestContext, logger
from gateway.audit import AuditEventType, get_audit_logger
from gateway.rate_limiter import RateLimiter
from gateway.security import SecurityFilter

AUTH_WHITELIST = {
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh",
    "/health",
    "/ready",
    "/status",
    "/",
    "/docs",
    "/openapi.json",
    "/api/sources",
}

# Paths that should skip SQL injection scanning (parameterized queries only)
_SKIP_SQL_SCAN_PATHS = {"/api/query", "/api/stream"}


def _get_client_ip(request: Request) -> str:
    """Resolve client IP, respecting X-Forwarded-For for reverse proxy setups."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_whitelisted(path: str) -> bool:
    if path in AUTH_WHITELIST:
        return True
    if path.startswith("/static"):
        return True
    return False


class GatewayMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._rate_limiter: Optional[RateLimiter] = None
        self._metrics_auth_user: Optional[str] = None
        self._metrics_auth_password: Optional[str] = None

    @staticmethod
    def _get_config():
        from base.config import Config
        return Config()

    @property
    def rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter()
        return self._rate_limiter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        client_ip = _get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        audit = get_audit_logger()

        request_id = str(uuid.uuid4())
        RequestContext.set(request_id=request_id)
        start_time = time.time()

        # ---- Layer 1: SecurityFilter (all requests) ----
        if path.startswith("/api/"):
            body_bytes = None
            if request.method in ("POST", "PUT", "PATCH"):
                try:
                    body_bytes = await request.body()
                    # Skip SQL injection scan on query endpoints (they use
                    # parameterized queries — false positives on edu content)
                    if path in _SKIP_SQL_SCAN_PATHS:
                        scan_ok, scan_error = SecurityFilter.detect_xss(
                            body_bytes.decode("utf-8")
                        ), None
                        if scan_ok:
                            scan_ok, scan_error = True, None
                        else:
                            scan_ok, scan_error = False, scan_ok
                    else:
                        scan_ok, scan_error = SecurityFilter.scan(
                            body_bytes.decode("utf-8")
                        )
                    if not scan_ok:
                        detail = {"path": path, "reason": scan_error, "ip": client_ip}
                        audit.log(AuditEventType.SQL_INJECTION_ATTEMPT if "SQL" in str(scan_error)
                                  else AuditEventType.XSS_ATTEMPT,
                                  ip_address=client_ip, user_agent=user_agent, detail=detail)
                        return JSONResponse(
                            status_code=400,
                            content={"detail": f"请求被安全过滤器拦截: {scan_error}"},
                        )
                    # Reconstruct request body for downstream
                    from starlette.requests import Request as StarletteRequest
                    from starlette.datastructures import Headers

                    async def receive():
                        return {"type": "http.request", "body": body_bytes}

                    request = StarletteRequest(
                        scope={**request.scope, "headers": request.scope.get("headers", [])},
                        receive=receive,
                    )
                except Exception as e:
                    logger.warning(f"请求体重构失败 ({path}): {e}")

        # ---- Layer 1.5: /metrics Basic Auth (if configured) ----
        if path == "/metrics":
            metrics_auth_user = getattr(self, "_metrics_auth_user", None)
            if metrics_auth_user is None:
                config = self._get_config()
                metrics_auth_user = config.METRICS_AUTH_USER
                self._metrics_auth_user = metrics_auth_user
                self._metrics_auth_password = config.METRICS_AUTH_PASSWORD
            if self._metrics_auth_user and self._metrics_auth_password:
                import base64
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Basic "):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "未提供认证信息"},
                        headers={"WWW-Authenticate": "Basic realm=\"metrics\""},
                    )
                try:
                    credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
                    username, password = credentials.split(":", 1)
                    if username != self._metrics_auth_user or password != self._metrics_auth_password:
                        raise ValueError
                except Exception:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "认证信息无效"},
                        headers={"WWW-Authenticate": "Basic realm=\"metrics\""},
                    )

        # ---- Layer 2: RateLimiter (all requests) ----
        if path.startswith("/api/"):
            rate_allowed = True
            if path == "/api/auth/login":
                rate_allowed = self.rate_limiter.check_login_limit(client_ip)
            elif path == "/api/auth/register":
                rate_allowed = self.rate_limiter.check_register_limit(client_ip)
            # For business endpoints, rate limit is checked after JWT verification
            if not rate_allowed:
                audit.log(AuditEventType.RATE_LIMIT_EXCEEDED,
                          ip_address=client_ip, user_agent=user_agent,
                          detail={"path": path})
                return JSONResponse(
                    status_code=429,
                    content={"detail": "请求过于频繁，请稍后再试"},
                )

        # ---- Layer 3: AuthMiddleware — JWT check for non-whitelisted paths ----
        if path.startswith("/api/") and not _is_whitelisted(path):
            from gateway.auth import decode_access_token, get_token_jti
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                audit.log(AuditEventType.UNAUTHORIZED_ACCESS,
                          ip_address=client_ip, user_agent=user_agent,
                          detail={"path": path, "reason": "missing_token"})
                return JSONResponse(
                    status_code=401,
                    content={"detail": "未提供认证令牌"},
                )
            token = auth_header[7:]
            # Check blacklist
            from mysql_qa import RedisClient
            redis_client = RedisClient()
            jti = get_token_jti(token)
            if jti and redis_client.is_token_blacklisted(jti):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Token已失效"},
                )
            try:
                payload = decode_access_token(token)
                # Inject user info into request scope for downstream
                tenant_id = payload.get("tenant_id", 0)
                request.scope["current_user"] = {
                    "user_id": payload["user_id"],
                    "username": payload["username"],
                    "tenant_id": tenant_id,
                }
                RequestContext.set(
                    user_id=payload["user_id"], tenant_id=tenant_id
                )
                # Rate limit for business endpoints
                user_id = payload["user_id"]
                if path == "/api/query" or path == "/api/stream":
                    check_func = self.rate_limiter.check_query_limit if path == "/api/query" \
                        else self.rate_limiter.check_stream_limit
                    if not check_func(user_id, tenant_id):
                        audit.log(AuditEventType.RATE_LIMIT_EXCEEDED,
                                  user_id=user_id, ip_address=client_ip,
                                  user_agent=user_agent, detail={"path": path})
                        return JSONResponse(
                            status_code=429,
                            content={"detail": "请求过于频繁，请稍后再试"},
                        )
            except Exception:
                audit.log(AuditEventType.UNAUTHORIZED_ACCESS,
                          ip_address=client_ip, user_agent=user_agent,
                          detail={"path": path, "reason": "invalid_token"})
                return JSONResponse(
                    status_code=401,
                    content={"detail": "令牌无效或已过期"},
                )

        # ---- Layer 4: Proceed to business layer ----
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"{request.method} {path} -> {status_code} "
                f"[{duration_ms:.1f}ms]"
            )
