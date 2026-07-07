"""Shared auth whitelist — single source of truth for paths that skip JWT verification."""

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


def is_whitelisted(path: str) -> bool:
    if path in AUTH_WHITELIST:
        return True
    if path.startswith("/static"):
        return True
    return False
