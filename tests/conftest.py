"""Shared fixtures for eval pipeline tests."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_dataset():
    return [
        {"question": "人工智能就业课的课程版本是什么？", "ground_truth": "V6.0", "source_filter": None},
        {
            "question": "课程的一句话概括是什么？",
            "ground_truth": "解锁大模型新技能成就高薪AI人才",
            "source_filter": None,
        },
        {"question": "课程优势有哪些？", "ground_truth": "热门岗位覆盖、与大厂深入合作", "source_filter": None},
    ]


@pytest.fixture
def sample_run():
    from db_models.eval_run import EvalRun

    return EvalRun(
        id=1,
        status="running",
        started_at=datetime.now(),
        total_questions=0,
        triggered_by="manual",
    )


@pytest.fixture
def sample_result():
    from db_models.eval_result import EvalResult

    return EvalResult(
        id=1,
        run_id=1,
        question="测试问题？",
        ground_truth="测试答案",
        answer="测试回答",
        contexts=json.dumps(["ctx1", "ctx2"], ensure_ascii=False),
        faithfulness=0.85,
        answer_relevancy=0.90,
        context_precision=0.78,
        context_recall=0.82,
    )


@pytest.fixture
def mock_session_factory():
    mock_session = MagicMock()
    mock_factory = MagicMock()
    mock_factory.return_value.__enter__.return_value = mock_session
    mock_factory.return_value.__exit__.return_value = None
    return mock_factory, mock_session


@pytest.fixture
def eval_repo(mock_session_factory):
    from repositories.eval_repo import EvalRepository

    factory, _ = mock_session_factory
    return EvalRepository(factory)


@pytest.fixture
def test_app():
    """Create a FastAPI TestClient with a mocked qa_system."""
    import app as app_module
    from app import app as fastapi_app

    original_qa = app_module.qa_system

    mock_qa = MagicMock()
    mock_qa.eval_service = MagicMock()
    mock_qa.eval_service.repo = MagicMock()
    app_module.qa_system = mock_qa

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)

    yield client, mock_qa

    app_module.qa_system = original_qa
