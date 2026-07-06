"""Chunk config admin endpoints."""
from fastapi import APIRouter, Depends, HTTPException

from db_models.base import SessionLocal
from repositories.user_repo import UserRepository
from gateway.deps import require_auth
from routers.v1.schemas import ChunkConfigResponse, ChunkConfigUpdate

router = APIRouter()


@router.get("/api/chunk-config", response_model=ChunkConfigResponse)
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


@router.put("/api/chunk-config", response_model=ChunkConfigResponse)
async def update_chunk_config(update: ChunkConfigUpdate, user: dict = Depends(require_auth)):
    repo = UserRepository(SessionLocal)
    if not repo.is_admin_user(user["user_id"]):
        raise HTTPException(status_code=403, detail="需要管理员权限")

    from base.chunk_config import ChunkConfigManager
    from base import logger
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


@router.post("/api/chunk-config/reload", response_model=ChunkConfigResponse)
async def reload_chunk_config(user: dict = Depends(require_auth)):
    repo = UserRepository(SessionLocal)
    if not repo.is_admin_user(user["user_id"]):
        raise HTTPException(status_code=403, detail="需要管理员权限")

    from base.chunk_config import ChunkConfigManager
    from base import logger
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
