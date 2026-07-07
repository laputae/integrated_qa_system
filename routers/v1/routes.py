"""REST API routes: root/health, sessions, query, sources."""
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from base.health import DegradationLevel
from db_models.base import SessionLocal
from gateway.audit import AuditEventType, get_audit_logger
from gateway.deps import get_current_user, require_auth
from repositories.conversation_repo import ConversationRepository
from routers.v1.auth import check_greeting
from routers.v1.schemas import DeleteHistoryRequest, QueryRequest

router = APIRouter()


def _get_qa_system():
    import app as app_module
    return app_module.qa_system


# ========== Root & Health ==========

@router.get("/")
async def read_root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "integrated_qa_system"}


@router.get("/ready")
async def readiness_check():
    qa_system = _get_qa_system()
    is_ready = qa_system.health.is_ready()
    return {
        "status": "ready" if is_ready else "not_ready",
        "degradation_level": qa_system.health.get_degradation_level().value,
    }


@router.get("/status")
async def status_detail():
    qa_system = _get_qa_system()
    return qa_system.health.get_status_response()


# ========== Session Endpoints ==========

@router.post("/api/create_session")
async def create_session(user: dict = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "user_id": user["user_id"]}


@router.get("/api/sessions")
async def get_user_sessions(user: dict = Depends(require_auth)):
    repo = ConversationRepository(SessionLocal)
    sessions = repo.get_user_sessions(user["user_id"], tenant_id=user["tenant_id"])
    return {"sessions": sessions, "username": user["username"]}


@router.get("/api/history/{session_id}")
async def get_history(session_id: str, user: dict = Depends(require_auth)):
    repo = ConversationRepository(SessionLocal)
    history = repo.get_session_history(session_id, user["user_id"],
                                       tenant_id=user["tenant_id"])
    return {"session_id": session_id, "history": history}


@router.post("/api/history/delete")
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
        return {"status": "success", "message": f"已删除 {count} 个会话的对话记录"}
    else:
        raise HTTPException(status_code=404, detail="未找到可删除的对话记录")


@router.get("/api/sources")
async def get_sources():
    qa_system = _get_qa_system()
    return {"sources": qa_system.config.VALID_SOURCES}


# ========== Query Endpoint ==========

@router.post("/api/query")
async def query(request: QueryRequest, user: dict = Depends(get_current_user)):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }

    qa_system = _get_qa_system()

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
            "processing_time": time.time() - start_time,
        }

    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time,
    }
