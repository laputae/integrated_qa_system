from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
import os
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional, List, Dict, Any
import time
import re

from new_main import IntegratedQASystem
from gateway.middleware import GatewayMiddleware
from gateway.auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    verify_password,
    get_token_jti,
    get_token_ttl,
)
from gateway.security import SecurityFilter
from gateway.deps import get_current_user, require_auth
from gateway.audit import AuditLogger, AuditEventType, get_audit_logger
from db_models.base import init_db, SessionLocal, Base, engine
from repositories.user_repo import UserRepository
from repositories.conversation_repo import ConversationRepository
from repositories.tenant_repo import TenantRepository
from mysql_qa import RedisClient

app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")

app.add_middleware(GatewayMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)

qa_system = IntegratedQASystem()

# ========== Pydantic Models ==========

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float

class RegisterRequest(BaseModel):
    username: str
    password: str
    tenant_name: str = "default"

class LoginRequest(BaseModel):
    username: str
    password: str
    tenant_name: str = "default"

class RefreshRequest(BaseModel):
    refresh_token: str

# ========== Greeting Patterns ==========

GREETING_PATTERNS = [
    {"pattern": r"^(你好|您好|hi|hello)", "response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"},
    {"pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)", "response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"},
    {"pattern": r"^(在吗|在不在|有人吗)", "response": "我在！我是黑马程序员，随时为你解答问题！"},
    {"pattern": r"^(干嘛呢|你在干嘛|做什么)", "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"},
]


def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


# ========== Static & Root ==========

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ========== Auth Endpoints ==========

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    audit = get_audit_logger()
    # Security checks
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

    # Store refresh token
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


@app.post("/api/auth/login")
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


@app.post("/api/auth/refresh")
async def refresh_token(request: RefreshRequest):
    audit = get_audit_logger()

    try:
        payload = decode_refresh_token(request.refresh_token)
    except Exception:
        return JSONResponse(status_code=401, content={"detail": "Refresh Token 无效或已过期"})

    # Check blacklist
    redis_client = RedisClient()
    jti = payload.get("jti")
    if jti and redis_client.is_token_blacklisted(jti):
        return JSONResponse(status_code=401, content={"detail": "Refresh Token 已失效"})

    # Revoke old refresh token
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


@app.post("/api/auth/logout")
async def logout(user: dict = Depends(require_auth)):
    audit = get_audit_logger()
    redis_client = RedisClient()
    jti = user.get("jti")
    if jti:
        redis_client.blacklist_token(jti, 3600)

    audit.log(AuditEventType.LOGOUT, user_id=user["user_id"],
              tenant_id=user.get("tenant_id", 0))
    return {"message": "已登出"}


# ========== Session Endpoints ==========

@app.post("/api/create_session")
async def create_session(user: dict = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "user_id": user["user_id"]}


@app.get("/api/sessions")
async def get_user_sessions(user: dict = Depends(require_auth)):
    repo = ConversationRepository(SessionLocal)
    sessions = repo.get_user_sessions(user["user_id"], tenant_id=user["tenant_id"])
    return {"sessions": sessions, "username": user["username"]}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str, user: dict = Depends(require_auth)):
    repo = ConversationRepository(SessionLocal)
    history = repo.get_session_history(session_id, user["user_id"],
                                       tenant_id=user["tenant_id"])
    return {"session_id": session_id, "history": history}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str, user: dict = Depends(require_auth)):
    audit = get_audit_logger()
    repo = ConversationRepository(SessionLocal)
    success = repo.delete_session(session_id, user["user_id"],
                                  tenant_id=user["tenant_id"])
    if success:
        audit.log(AuditEventType.HISTORY_CLEARED, user_id=user["user_id"],
                  tenant_id=user["tenant_id"],
                  detail={"session_id": session_id})
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


# ========== Query Endpoint ==========

@app.post("/api/query")
async def query(request: QueryRequest, user: dict = Depends(get_current_user)):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# ========== WebSocket Endpoint ==========

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    user_id = 0
    username = "anonymous"
    tenant_id = 0

    if token:
        try:
            payload = decode_access_token(token)
            redis_client = RedisClient()
            jti = payload.get("jti")
            if not jti or not redis_client.is_token_blacklisted(jti):
                user_id = payload["user_id"]
                username = payload["username"]
                tenant_id = payload.get("tenant_id", 0)
        except Exception:
            await websocket.close(code=4001, reason="令牌无效或已过期")
            return
    else:
        await websocket.close(code=4001, reason="未提供认证令牌")
        return

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query_text = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()

            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({"type": "start", "session_id": session_id})

            greeting_response = check_greeting(query_text)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token", "token": greeting_response,
                        "session_id": session_id
                    })
                    await websocket.send_json({
                        "type": "end", "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break

            collected_answer = ""
            for token_val, is_complete in qa_system.query(
                query_text, user_id=user_id, tenant_id=tenant_id,
                source_filter=source_filter, session_id=session_id
            ):
                collected_answer += token_val
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end", "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token_val and websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token", "token": token_val,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end", "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
