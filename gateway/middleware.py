import json
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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

    @property
    def rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter()
        return self._rate_limiter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        audit = get_audit_logger()

        # ---- Layer 1: SecurityFilter (all requests) ----
        if path.startswith("/api/"):
            body_bytes = None
            if request.method in ("POST", "PUT", "PATCH"):
                try:
                    body_bytes = await request.body()
                    body_text = body_bytes.decode("utf-8")
                    scan_ok, scan_error = SecurityFilter.scan(body_text)
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
                except Exception:
                    pass

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
        response = await call_next(request)
        return response
