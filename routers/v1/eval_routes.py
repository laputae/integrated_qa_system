"""Eval API endpoints."""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from db_models.base import SessionLocal
from gateway.deps import require_auth
from repositories.user_repo import UserRepository
from routers.v1.schemas import EvalRunRequest

router = APIRouter()


def _get_qa_system():
    import app as app_module

    return app_module.qa_system


def _require_eval_admin(user: dict):
    repo = UserRepository(SessionLocal)
    if not repo.is_admin_user(user["user_id"]):
        raise HTTPException(status_code=403, detail="需要管理员权限")


@router.post("/api/eval/run")
async def eval_run(request: EvalRunRequest, user: dict = Depends(require_auth)):
    _require_eval_admin(user)
    qa_system = _get_qa_system()
    if qa_system.eval_service is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "评估服务未初始化，无法执行评估"},
        )

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


@router.get("/api/eval/runs")
async def eval_list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_auth),
):
    _require_eval_admin(user)
    qa_system = _get_qa_system()
    if qa_system.eval_service is None:
        return JSONResponse(status_code=503, content={"detail": "评估服务未初始化"})

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


@router.get("/api/eval/runs/{run_id}")
async def eval_get_run(
    run_id: int,
    include_contexts: bool = Query(False),
    user: dict = Depends(require_auth),
):
    _require_eval_admin(user)
    qa_system = _get_qa_system()
    if qa_system.eval_service is None:
        return JSONResponse(status_code=503, content={"detail": "评估服务未初始化"})

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


@router.get("/api/eval/trends")
async def eval_trends(
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(require_auth),
):
    _require_eval_admin(user)
    qa_system = _get_qa_system()
    if qa_system.eval_service is None:
        return JSONResponse(status_code=503, content={"detail": "评估服务未初始化"})
    return qa_system.eval_service.get_trends(limit=limit)


@router.get("/api/eval/status")
async def eval_status(user: dict = Depends(require_auth)):
    _require_eval_admin(user)
    qa_system = _get_qa_system()
    if qa_system.eval_service is None:
        return JSONResponse(status_code=503, content={"detail": "评估服务未初始化"})
    return qa_system.eval_service.get_quality_status()
