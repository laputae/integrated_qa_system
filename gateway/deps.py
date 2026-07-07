from fastapi import Depends, HTTPException, Request, status

from gateway.auth_whitelist import is_whitelisted


async def get_current_user(request: Request) -> dict:
    if is_whitelisted(request.url.path):
        return {"user_id": 0, "username": "anonymous", "tenant_id": 0}

    # 中间件已解析过，直接复用
    cached = request.scope.get("current_user")
    if cached is not None:
        return cached

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未提供认证令牌",
    )


def require_auth(user: dict = Depends(get_current_user)) -> dict:
    if user["user_id"] == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="此操作需要登录",
        )
    return user
