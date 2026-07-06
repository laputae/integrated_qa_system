"""Auth endpoints: register, login, refresh, logout + greeting patterns."""
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from db_models.base import SessionLocal
from repositories.user_repo import UserRepository
from repositories.tenant_repo import TenantRepository
from gateway.auth import (
    create_access_token, create_refresh_token, decode_refresh_token,
    hash_password, verify_password, get_token_ttl,
)
from gateway.security import SecurityFilter
from gateway.deps import require_auth
from gateway.audit import AuditLogger, AuditEventType, get_audit_logger
from mysql_qa import get_redis_client
from routers.v1.schemas import RegisterRequest, LoginRequest, RefreshRequest

router = APIRouter()

# ========== Greeting Patterns ==========

GREETING_PATTERNS = [
    {"pattern": r"^(你好|您好|hi|hello)",
     "response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"},
    {"pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
     "response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"},
    {"pattern": r"^(在吗|在不在|有人吗)",
     "response": "我在！我是黑马程序员，随时为你解答问题！"},
    {"pattern": r"^(干嘛呢|你在干嘛|做什么)",
     "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"},
]


def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


# ========== Auth Endpoints ==========

@router.post("/api/auth/register")
async def register(request: RegisterRequest):
    audit = get_audit_logger()
    valid, err = SecurityFilter.validate_username(request.username)
    if not valid:
        return JSONResponse(status_code=400, content={"detail": err})
    valid, err = SecurityFilter.validate_password(request.password)
    if not valid:
        return JSONResponse(status_code=400, content={"detail": err})

    tenant_repo = TenantRepository(SessionLocal)
    tenant = tenant_repo.get_or_create(request.tenant_name)
    if not tenant.is_active:
        return JSONResponse(status_code=403, content={"detail": "该租户已被禁用"})

    repo = UserRepository(SessionLocal)
    if repo.username_exists(request.username, tenant.id):
        return JSONResponse(status_code=400, content={"detail": "用户名已存在"})

    password_hash = hash_password(request.password)
    user = repo.create(request.username, password_hash, tenant.id)

    access_token = create_access_token(user.id, user.username, tenant.id)
    refresh_token, jti, expires_at = create_refresh_token(user.id, user.username, tenant.id)

    from db_models.refresh_token import RefreshToken
    with SessionLocal() as session:
        rt = RefreshToken(
            user_id=user.id, tenant_id=tenant.id,
            token_jti=jti, expires_at=expires_at, device_info=None
        )
        session.add(rt)
        session.commit()

    audit.log(AuditEventType.REGISTER_SUCCESS, user_id=user.id, tenant_id=tenant.id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "username": user.username,
        "user_id": user.id,
        "tenant_name": tenant.name,
    }


@router.post("/api/auth/login")
async def login(request: LoginRequest):
    audit = get_audit_logger()

    tenant_repo = TenantRepository(SessionLocal)
    tenant = tenant_repo.get_by_name(request.tenant_name)
    if not tenant or not tenant.is_active:
        audit.log(AuditEventType.LOGIN_FAILED,
                  detail={"username": request.username, "tenant": request.tenant_name})
        return JSONResponse(status_code=401, content={"detail": "用户名或密码错误"})

    repo = UserRepository(SessionLocal)
    user = repo.get_by_username(request.username, tenant.id)
    if not user or not verify_password(request.password, user.password_hash):
        audit.log(AuditEventType.LOGIN_FAILED,
                  detail={"username": request.username, "tenant": request.tenant_name})
        return JSONResponse(status_code=401, content={"detail": "用户名或密码错误"})

    access_token = create_access_token(user.id, user.username, tenant.id)
    refresh_token, jti, expires_at = create_refresh_token(user.id, user.username, tenant.id)

    from db_models.refresh_token import RefreshToken
    with SessionLocal() as session:
        rt = RefreshToken(
            user_id=user.id, tenant_id=tenant.id,
            token_jti=jti, expires_at=expires_at, device_info=None
        )
        session.add(rt)
        session.commit()

    audit.log(AuditEventType.LOGIN_SUCCESS, user_id=user.id, tenant_id=tenant.id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "username": user.username,
        "user_id": user.id,
        "tenant_name": tenant.name,
    }


@router.post("/api/auth/refresh")
async def refresh_token(request: RefreshRequest):
    audit = get_audit_logger()

    try:
        payload = decode_refresh_token(request.refresh_token)
    except Exception:
        return JSONResponse(status_code=401, content={"detail": "Refresh Token 无效或已过期"})

    redis_client = get_redis_client()
    jti = payload.get("jti")
    if jti and redis_client.is_token_blacklisted(jti):
        return JSONResponse(status_code=401, content={"detail": "Refresh Token 已失效"})

    redis_client.blacklist_token(jti, get_token_ttl(request.refresh_token))
    from db_models.refresh_token import RefreshToken
    with SessionLocal() as session:
        rt = session.query(RefreshToken).filter(RefreshToken.token_jti == jti).first()
        if rt:
            rt.revoked = True
            session.commit()

    user_id = payload["user_id"]
    username = payload["username"]
    tenant_id = payload.get("tenant_id", 0)

    new_access_token = create_access_token(user_id, username, tenant_id)
    new_refresh_token, new_jti, new_expires_at = create_refresh_token(user_id, username, tenant_id)

    with SessionLocal() as session:
        rt = RefreshToken(
            user_id=user_id, tenant_id=tenant_id,
            token_jti=new_jti, expires_at=new_expires_at, device_info=None
        )
        session.add(rt)
        session.commit()

    audit.log(AuditEventType.TOKEN_REFRESH, user_id=user_id, tenant_id=tenant_id)

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "username": username,
        "user_id": user_id,
    }


@router.post("/api/auth/logout")
async def logout(user: dict = Depends(require_auth)):
    audit = get_audit_logger()
    redis_client = get_redis_client()
    jti = user.get("jti")
    if jti:
        redis_client.blacklist_token(jti, 3600)

    audit.log(AuditEventType.LOGOUT, user_id=user["user_id"],
              tenant_id=user.get("tenant_id", 0))
    return {"message": "已登出"}
