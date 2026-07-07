from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gateway.auth import decode_access_token
from gateway.auth_whitelist import is_whitelisted
from mysql_qa import get_redis_client

security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> dict:
    if is_whitelisted(request.url.path):
        return {"user_id": 0, "username": "anonymous", "tenant_id": 0}

    # 中间件已解析过，直接复用
    cached = request.scope.get("current_user")
    if cached is not None:
        return cached

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证令牌",
        )

    token = credentials.credentials

    redis_client = get_redis_client()

    from gateway.auth import get_token_jti
    jti = get_token_jti(token)
    if jti and redis_client.is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token已失效",
        )

    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期",
        )

    return {
        "user_id": payload["user_id"],
        "username": payload["username"],
        "tenant_id": payload.get("tenant_id", 0),
        "jti": payload.get("jti"),
    }


def require_auth(user: dict = Depends(get_current_user)) -> dict:
    if user["user_id"] == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="此操作需要登录",
        )
    return user
