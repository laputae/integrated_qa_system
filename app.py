from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
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

from main import IntegratedQASystem
from base import logger
from base.settings import validate_config
from base.health import DegradationLevel
from prometheus_fastapi_instrumentator import Instrumentator
from gateway.middleware import GatewayMiddleware
from gateway.security_headers import SecurityHeadersMiddleware
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

qa_system = IntegratedQASystem()

# GAP-14: Fail-fast config validation at startup
validate_config()
logger.info("Config validation passed — all required settings present.")

# asyncio.Semaphore for concurrency control — limits simultaneous LLM calls
_llm_semaphore = asyncio.Semaphore(qa_system.config.MAX_CONCURRENT_LLM_CALLS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await qa_system.health.start_background_recovery()
    if qa_system.eval_service:
        await qa_system.eval_service.start_periodic_eval()
    yield
    if qa_system.eval_service:
        await qa_system.eval_service.stop_periodic_eval()
    await qa_system.health.close()


app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统", lifespan=lifespan)

app.add_middleware(GatewayMiddleware)

if qa_system.config.SECURE_HEADERS_ENABLED:
    app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=qa_system.config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

os.makedirs("static", exist_ok=True)

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

class EvalRunRequest(BaseModel):
    dataset: Optional[list] = None
    triggered_by: str = "manual"

class ChunkConfigResponse(BaseModel):
    default_strategy: str
    doc_type_strategies: Dict[str, str]
    semantic_model_path: str
    semantic_device: str
    semantic_fallback_strategy: str
    parent_chunk_size: int
    child_chunk_size: int
    chunk_overlap: int

class ChunkConfigUpdate(BaseModel):
    default_strategy: Optional[str] = None
    doc_type_strategies: Optional[Dict[str, str]] = None
    semantic_model_path: Optional[str] = None
    semantic_device: Optional[str] = None
    semantic_fallback_strategy: Optional[str] = None
    parent_chunk_size: Optional[int] = None
    child_chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

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
    """Liveness probe: Is the process alive?"""
    return {"status": "healthy", "service": "integrated_qa_system"}


@app.get("/ready")
async def readiness_check():
    """Readiness probe: Can the app serve traffic?"""
    is_ready = qa_system.health.is_ready()
    return {
        "status": "ready" if is_ready else "not_ready",
        "degradation_level": qa_system.health.get_degradation_level().value,
    }


@app.get("/status")
async def status_detail():
    """Detailed status: Per-component health breakdown."""
    return qa_system.health.get_status_response()


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


class DeleteHistoryRequest(BaseModel):
    session_ids: list[str]


@app.post("/api/history/delete")
async def delete_history(request: DeleteHistoryRequest, user: dict = Depends(require_auth)):
    if not request.session_ids:
        raise HTTPException(status_code=400, detail="请选择要删除的会话")
    audit = get_audit_logger()
    repo = ConversationRepository(SessionLocal)
    count = repo.soft_delete_sessions(request.session_ids, user["user_id"],
                                      tenant_id=user["tenant_id"])
    if count > 0:
        audit.log(AuditEventType.HISTORY_DELETED, user_id=user["user_id"],
                  tenant_id=user["tenant_id"],
                  detail={"session_ids": request.session_ids, "count": count})
        return {"status": "success", "message": f"已删除 {len(request.session_ids)} 个会话的对话记录"}
    else:
        raise HTTPException(status_code=404, detail="未找到可删除的对话记录")


@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


# ========== Eval Endpoints ==========

@app.post("/api/eval/run")
async def eval_run(request: EvalRunRequest, user: dict = Depends(require_auth)):
    if qa_system.eval_service is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "评估服务未初始化，无法执行评估"},
        )

    # Create the run record synchronously before launching background work
    chunk_snapshot = None
    try:
        from base.chunk_config import ChunkConfigManager
        chunk_snapshot = ChunkConfigManager().get_config()
    except Exception:
        pass
    run = qa_system.eval_service.repo.create_run(
        triggered_by=request.triggered_by,
        chunk_config_snapshot=chunk_snapshot,
    )

    async def run_background():
        await qa_system.eval_service.run_evaluation_async(
            dataset=request.dataset,
            triggered_by=request.triggered_by,
            run_id=run.id,
        )

    asyncio.create_task(run_background())

    return JSONResponse(
        status_code=202,
        content={
            "run_id": run.id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "message": f"评估已启动，请使用 GET /api/eval/runs/{run.id} 查询结果",
        },
    )


@app.get("/api/eval/runs")
async def eval_list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_auth),
):
    if qa_system.eval_service is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "评估服务未初始化"},
        )

    runs = qa_system.eval_service.repo.get_runs(limit=limit, offset=offset)
    total = qa_system.eval_service.repo.count_runs()

    return {
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "total_questions": r.total_questions,
                "avg_faithfulness": r.avg_faithfulness,
                "avg_answer_relevancy": r.avg_answer_relevancy,
                "avg_context_precision": r.avg_context_precision,
                "avg_context_recall": r.avg_context_recall,
                "triggered_by": r.triggered_by,
            }
            for r in runs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/eval/runs/{run_id}")
async def eval_get_run(
    run_id: int,
    include_contexts: bool = Query(False),
    user: dict = Depends(require_auth),
):
    if qa_system.eval_service is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "评估服务未初始化"},
        )

    run = qa_system.eval_service.repo.get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "评估记录不存在"})

    results = qa_system.eval_service.repo.get_results_for_run(run_id)
    results_data = []
    for r in results:
        item = {
            "id": r.id,
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer": r.answer,
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
            "source_filter": r.source_filter,
        }
        if include_contexts and r.contexts:
            try:
                item["contexts"] = json.loads(r.contexts)
            except (json.JSONDecodeError, TypeError):
                item["contexts"] = [r.contexts]
        results_data.append(item)

    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "total_questions": run.total_questions,
            "avg_faithfulness": run.avg_faithfulness,
            "avg_answer_relevancy": run.avg_answer_relevancy,
            "avg_context_precision": run.avg_context_precision,
            "avg_context_recall": run.avg_context_recall,
            "error_message": run.error_message,
            "triggered_by": run.triggered_by,
        },
        "results": results_data,
    }


@app.get("/api/eval/trends")
async def eval_trends(
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(require_auth),
):
    if qa_system.eval_service is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "评估服务未初始化"},
        )

    return qa_system.eval_service.get_trends(limit=limit)


@app.get("/api/eval/status")
async def eval_status(user: dict = Depends(require_auth)):
    if qa_system.eval_service is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "评估服务未初始化"},
        )

    return qa_system.eval_service.get_quality_status()


# ========== Chunk Config Endpoints ==========

@app.get("/api/chunk-config", response_model=ChunkConfigResponse)
async def get_chunk_config(user: dict = Depends(require_auth)):
    from base.chunk_config import ChunkConfigManager
    mgr = ChunkConfigManager()
    cfg = mgr.get_config()
    return ChunkConfigResponse(
        default_strategy=cfg["default_strategy"],
        doc_type_strategies=cfg["doc_type_strategies"],
        semantic_model_path=cfg["semantic_model_path"],
        semantic_device=cfg["semantic_device"],
        semantic_fallback_strategy=cfg["semantic_fallback_strategy"],
        parent_chunk_size=cfg["parent_chunk_size"],
        child_chunk_size=cfg["child_chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )


@app.put("/api/chunk-config", response_model=ChunkConfigResponse)
async def update_chunk_config(
    update: ChunkConfigUpdate,
    user: dict = Depends(require_auth),
):
    from repositories.user_repo import UserRepository
    from db_models.base import SessionLocal
    repo = UserRepository(SessionLocal)
    if not repo.is_admin_user(user["user_id"]):
        raise HTTPException(status_code=403, detail="需要管理员权限")

    from base.chunk_config import ChunkConfigManager
    mgr = ChunkConfigManager()
    updates = {k: v for k, v in update.model_dump(exclude_none=True).items()}
    mgr.update_config(updates)
    logger.info("Chunk config updated by user %s: %s", user["username"], list(updates.keys()))
    cfg = mgr.get_config()
    return ChunkConfigResponse(
        default_strategy=cfg["default_strategy"],
        doc_type_strategies=cfg["doc_type_strategies"],
        semantic_model_path=cfg["semantic_model_path"],
        semantic_device=cfg["semantic_device"],
        semantic_fallback_strategy=cfg["semantic_fallback_strategy"],
        parent_chunk_size=cfg["parent_chunk_size"],
        child_chunk_size=cfg["child_chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )


@app.post("/api/chunk-config/reload", response_model=ChunkConfigResponse)
async def reload_chunk_config(user: dict = Depends(require_auth)):
    from repositories.user_repo import UserRepository
    from db_models.base import SessionLocal
    repo = UserRepository(SessionLocal)
    if not repo.is_admin_user(user["user_id"]):
        raise HTTPException(status_code=403, detail="需要管理员权限")

    from base.chunk_config import ChunkConfigManager
    mgr = ChunkConfigManager()
    mgr.reload()
    logger.info("Chunk config reloaded from config.ini by user %s", user["username"])
    cfg = mgr.get_config()
    return ChunkConfigResponse(
        default_strategy=cfg["default_strategy"],
        doc_type_strategies=cfg["doc_type_strategies"],
        semantic_model_path=cfg["semantic_model_path"],
        semantic_device=cfg["semantic_device"],
        semantic_fallback_strategy=cfg["semantic_fallback_strategy"],
        parent_chunk_size=cfg["parent_chunk_size"],
        child_chunk_size=cfg["child_chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )


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

    # Degradation check: Level 4 (no MySQL) → 503
    if qa_system.health.get_degradation_level() == DegradationLevel.LEVEL4_NO_MYSQL:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "系统维护中，暂无法处理查询，请联系管理员。",
                "session_id": session_id,
            },
        )

    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        # If RAG is degraded (Level 2+), tell the user directly
        level = qa_system.health.get_degradation_level()
        if level >= DegradationLevel.LEVEL2_NO_MILVUS:
            return {
                "answer": "未找到答案",
                "is_streaming": False,
                "session_id": session_id,
                "processing_time": time.time() - start_time,
            }
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
            external_context = request_data.get("external_context")
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
            async for token_val, is_complete in qa_system.aquery(
                query_text, _llm_semaphore,
                user_id=user_id, tenant_id=tenant_id,
                source_filter=source_filter, session_id=session_id,
                external_context=external_context
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
                    # HallucinationGuard 结果通知
                    guard_result = getattr(qa_system, '_last_guard_result', None)
                    if guard_result is not None and guard_result.is_hallucinated:
                        if websocket.client_state == websocket.client_state.CONNECTED:
                            await websocket.send_json({
                                "type": "hallucination_warning",
                                "message": "部分回答内容可能缺乏文档依据，建议核实后使用。",
                                "details": guard_result.details,
                                "score": guard_result.score,
                                "session_id": session_id,
                            })
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end", "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
