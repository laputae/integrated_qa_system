"""
Quality reporting helpers for eval — trend analysis, regression detection, status.

Extracted from eval_service.py to keep each module under 300 lines.
"""
from base import Config


def check_regression(repo, config: Config) -> dict:
    """Check if avg_faithfulness has been below threshold for N consecutive runs."""
    threshold = config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD
    n = config.EVAL_REGRESSION_CONSECUTIVE_RUNS

    values = repo.get_recent_metrics("faithfulness", limit=n)
    if len(values) < n:
        return {
            "detected": False,
            "details": f"仅有 {len(values)} 次历史评估（需要 {n} 次）",
            "current_value": values[0] if values else None,
            "threshold": threshold,
        }

    current_value = values[0]
    if current_value is None:
        return {"detected": False, "details": "最新评估无 faithfulness 分数",
                "current_value": None, "threshold": threshold}

    if all(v is not None and v < threshold for v in values):
        return {
            "detected": True,
            "details": f" faithfulness 已连续 {n} 次低于阈值 {threshold}（当前值: {current_value:.3f}）",
            "current_value": current_value,
            "threshold": threshold,
        }

    return {"detected": False, "details": None, "current_value": current_value,
            "threshold": threshold}


def get_quality_status(repo, config: Config) -> dict:
    """Current quality snapshot for health check / status API."""
    latest = repo.get_latest_completed()
    regression = check_regression(repo, config)

    if latest is None:
        return {
            "latest_run": None,
            "regression": regression,
            "quality_status": "unknown",
            "trend_direction": "stable",
            "total_runs": repo.count_runs(),
        }

    faithfulness = latest.avg_faithfulness or 0.0

    if faithfulness < config.EVAL_QUALITY_CRITICAL_THRESHOLD:
        quality_status = "critical"
    elif faithfulness < config.EVAL_QUALITY_WARNING_THRESHOLD:
        quality_status = "warning"
    else:
        quality_status = "good"

    trend = _compute_trend_direction(repo)

    return {
        "latest_run": {
            "id": latest.id,
            "status": latest.status,
            "started_at": latest.started_at.isoformat() if latest.started_at else None,
            "completed_at": latest.completed_at.isoformat() if latest.completed_at else None,
            "total_questions": latest.total_questions,
            "avg_faithfulness": latest.avg_faithfulness,
            "avg_answer_relevancy": latest.avg_answer_relevancy,
            "avg_context_precision": latest.avg_context_precision,
            "avg_context_recall": latest.avg_context_recall,
            "triggered_by": latest.triggered_by,
        },
        "regression": regression,
        "quality_status": quality_status,
        "trend_direction": trend,
        "total_runs": repo.count_runs(),
    }


def get_trends(repo, limit: int = 20) -> dict:
    """Return metric trends over time for dashboard."""
    runs = repo.get_runs(limit=limit, offset=0)
    completed = [r for r in runs if r.status == "completed"]

    return {
        "runs": [
            {
                "id": r.id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "avg_faithfulness": r.avg_faithfulness,
                "avg_answer_relevancy": r.avg_answer_relevancy,
                "avg_context_precision": r.avg_context_precision,
                "avg_context_recall": r.avg_context_recall,
                "triggered_by": r.triggered_by,
            }
            for r in completed
        ],
        "faithfulness": [r.avg_faithfulness for r in completed if r.avg_faithfulness is not None],
        "answer_relevancy": [r.avg_answer_relevancy for r in completed if r.avg_answer_relevancy is not None],
        "context_precision": [r.avg_context_precision for r in completed if r.avg_context_precision is not None],
        "context_recall": [r.avg_context_recall for r in completed if r.avg_context_recall is not None],
    }


def get_latest_metrics(repo) -> dict | None:
    latest = repo.get_latest_completed()
    if latest is None:
        return None
    return {
        "run_id": latest.id,
        "avg_faithfulness": latest.avg_faithfulness,
        "avg_answer_relevancy": latest.avg_answer_relevancy,
        "avg_context_precision": latest.avg_context_precision,
        "avg_context_recall": latest.avg_context_recall,
        "completed_at": latest.completed_at.isoformat() if latest.completed_at else None,
    }


def _compute_trend_direction(repo) -> str:
    """Compute trend direction from the last 5 faithfulness values."""
    values = repo.get_recent_metrics("faithfulness", limit=5)
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return "stable"

    recent_slice = values[:3]
    older_slice = values[-3:]
    recent_avg = sum(recent_slice) / len(recent_slice)
    older_avg = sum(older_slice) / len(older_slice)
    diff = recent_avg - older_avg

    if diff > 0.05:
        return "improving"
    elif diff < -0.05:
        return "declining"
    return "stable"
