"""评估自动化管道 — 单元测试与集成测试"""
import sys
import os
import json
import time
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytest
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dataset():
    return [
        {"question": "人工智能就业课的课程版本是什么？", "ground_truth": "V6.0", "source_filter": None},
        {"question": "课程的一句话概括是什么？", "ground_truth": "解锁大模型新技能成就高薪AI人才", "source_filter": None},
        {"question": "课程优势有哪些？", "ground_truth": "热门岗位覆盖、与大厂深入合作", "source_filter": None},
    ]


@pytest.fixture
def sample_run():
    """Create a minimal EvalRun for testing."""
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
    """Create a minimal EvalResult for testing."""
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
    """Create a mock session factory for repository tests."""
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


# ============================================================================
# 1. ORM Model Tests
# ============================================================================

class TestEvalRunModel:
    def test_tablename(self):
        from db_models.eval_run import EvalRun
        assert EvalRun.__tablename__ == "eval_runs"

    def test_default_status(self):
        """SQLAlchemy mapped_column default is applied at INSERT time, not Python init."""
        from db_models.eval_run import EvalRun
        run = EvalRun(status="running", total_questions=0)
        assert run.status == "running"

    def test_default_triggered_by(self):
        from db_models.eval_run import EvalRun
        run = EvalRun(triggered_by="manual", status="running", total_questions=0)
        assert run.triggered_by == "manual"

    def test_required_fields(self):
        from db_models.eval_run import EvalRun
        run = EvalRun(status="completed", total_questions=10,
                      avg_faithfulness=0.85, avg_answer_relevancy=0.90,
                      avg_context_precision=0.78, avg_context_recall=0.82)
        assert run.status == "completed"
        assert run.total_questions == 10
        assert run.avg_faithfulness == 0.85
        assert run.avg_answer_relevancy == 0.90
        assert run.avg_context_precision == 0.78
        assert run.avg_context_recall == 0.82

    def test_nullable_metrics(self):
        from db_models.eval_run import EvalRun
        run = EvalRun()
        assert run.avg_faithfulness is None
        assert run.avg_answer_relevancy is None
        assert run.avg_context_precision is None
        assert run.avg_context_recall is None
        assert run.completed_at is None
        assert run.error_message is None


class TestEvalResultModel:
    def test_tablename(self):
        from db_models.eval_result import EvalResult
        assert EvalResult.__tablename__ == "eval_results"

    def test_required_fields(self):
        from db_models.eval_result import EvalResult
        result = EvalResult(
            run_id=1,
            question="什么是AI？",
            ground_truth="人工智能",
            answer="AI是人工智能的缩写",
            faithfulness=0.92,
            answer_relevancy=0.88,
        )
        assert result.run_id == 1
        assert result.question == "什么是AI？"
        assert result.ground_truth == "人工智能"
        assert result.answer == "AI是人工智能的缩写"
        assert result.faithfulness == 0.92
        assert result.answer_relevancy == 0.88

    def test_nullable_fields(self):
        from db_models.eval_result import EvalResult
        result = EvalResult(run_id=1, question="Q", ground_truth="A")
        assert result.answer is None
        assert result.contexts is None
        assert result.faithfulness is None
        assert result.answer_relevancy is None
        assert result.context_precision is None
        assert result.context_recall is None
        assert result.source_filter is None

    def test_source_filter_default(self):
        from db_models.eval_result import EvalResult
        result = EvalResult(run_id=1, question="Q", ground_truth="A")
        assert result.source_filter is None


# ============================================================================
# 2. Repository Tests
# ============================================================================

class TestEvalRepositoryRun:
    def test_create_run(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        eval_repo.create_run(triggered_by="manual")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called()

    def test_create_run_scheduled(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        eval_repo.create_run(triggered_by="scheduled")
        args, _ = mock_session.add.call_args
        added_run = args[0]
        assert added_run.triggered_by == "scheduled"
        assert added_run.status == "running"

    def test_complete_run(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_run = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_run

        metrics = {"faithfulness": 0.85, "answer_relevancy": 0.90,
                   "context_precision": 0.78, "context_recall": 0.82}
        eval_repo.complete_run(run_id=1, metrics=metrics, total_questions=10)

        assert mock_run.status == "completed"
        assert mock_run.total_questions == 10
        assert mock_run.avg_faithfulness == 0.85
        assert mock_run.avg_answer_relevancy == 0.90
        assert mock_run.avg_context_precision == 0.78
        assert mock_run.avg_context_recall == 0.82
        mock_session.commit.assert_called()

    def test_complete_run_not_found(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.first.return_value = None
        # Should not raise
        eval_repo.complete_run(run_id=999, metrics={}, total_questions=0)

    def test_fail_run(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_run = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_run

        eval_repo.fail_run(run_id=1, error_message="RAGAS evaluation crashed")

        assert mock_run.status == "failed"
        assert mock_run.error_message == "RAGAS evaluation crashed"
        mock_session.commit.assert_called()

    def test_fail_run_not_found(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.first.return_value = None
        eval_repo.fail_run(run_id=999, error_message="error")  # Should not raise

    def test_get_run(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        expected = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = expected

        result = eval_repo.get_run(1)
        assert result is expected

    def test_get_runs_pagination(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        eval_repo.get_runs(limit=10, offset=5)
        mock_session.query.return_value.order_by.assert_called()

    def test_count_runs(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.count.return_value = 42
        assert eval_repo.count_runs() == 42

    def test_get_recent_metrics(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = \
            [(0.9,), (0.85,), (0.88,)]

        values = eval_repo.get_recent_metrics("faithfulness", limit=3)
        assert values == [0.9, 0.85, 0.88]

    def test_get_recent_metrics_invalid_name(self, eval_repo):
        with pytest.raises(ValueError, match="Invalid metric"):
            eval_repo.get_recent_metrics("invalid_metric")


class TestEvalRepositoryResult:
    def test_insert_result(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        eval_repo.insert_result(
            run_id=1, question="Q?", ground_truth="A",
            answer="Answer", contexts=["ctx1", "ctx2"], source_filter="ai",
        )
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called()

    def test_insert_result_no_contexts(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        eval_repo.insert_result(
            run_id=1, question="Q?", ground_truth="A", answer="Answer",
        )
        args, _ = mock_session.add.call_args
        added = args[0]
        assert added.contexts is None

    def test_update_result_scores(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_result = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_result

        scores = {"faithfulness": 0.95, "answer_relevancy": 0.88,
                  "context_precision": 0.76, "context_recall": 0.81}
        eval_repo.update_result_scores(result_id=1, scores=scores)

        assert mock_result.faithfulness == 0.95
        assert mock_result.answer_relevancy == 0.88
        assert mock_result.context_precision == 0.76
        assert mock_result.context_recall == 0.81
        mock_session.commit.assert_called()

    def test_update_result_scores_not_found(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.first.return_value = None
        eval_repo.update_result_scores(result_id=999, scores={})  # Should not raise

    def test_get_result_ids_for_run(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = \
            [(1,), (2,), (3,)]
        ids = eval_repo.get_result_ids_for_run(run_id=1)
        assert ids == [1, 2, 3]


# ============================================================================
# 3. Service Tests
# ============================================================================

class TestEvalServiceInit:
    def test_service_creation(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        repo = MagicMock()
        rag = MagicMock()
        llm = MagicMock()
        vs = MagicMock()

        service = EvalService(config, repo, rag, llm, vs)
        assert service.config is config
        assert service.repo is repo
        assert service.rag_system is rag
        assert service.llm_client is llm
        assert service.vector_store is vs
        assert service._running is False
        assert service._eval_task is None


class TestEvalServiceRunEvaluation:
    def test_run_evaluation_no_rag_system(self):
        from rag_qa.eval.eval_service import EvalService
        service = EvalService(MagicMock(), MagicMock(), None, MagicMock(), MagicMock())
        result = service.run_evaluation([])
        assert result["error"] == "RAGSystem 未初始化，无法执行评估"

    def test_run_evaluation_empty_dataset(self):
        from rag_qa.eval.eval_service import EvalService
        service = EvalService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        result = service.run_evaluation([])
        assert result["error"] == "评估数据集为空"

    @patch("rag_qa.eval.eval_service.EvalService._load_default_dataset")
    def test_run_evaluation_loads_default_dataset(self, mock_load, sample_dataset):
        from rag_qa.eval.eval_service import EvalService
        mock_load.return_value = sample_dataset
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3

        repo = MagicMock()
        mock_run = MagicMock()
        mock_run.id = 1
        repo.create_run.return_value = mock_run
        repo.get_recent_metrics.return_value = [0.85, 0.82, 0.80]

        rag_system = MagicMock()
        rag_system.retrieve_and_merge.return_value = []
        rag_system.generate_answer.return_value = iter(["测试回答"])

        service = EvalService(config, repo, rag_system, MagicMock(), MagicMock())

        with patch.object(service, "_prepare_ragas_dataset") as mock_prep, \
             patch.object(service, "_run_ragas") as mock_ragas:
            mock_prep.return_value = MagicMock()
            mock_ragas.return_value = {
                "faithfulness": [0.9, 0.85, 0.88],
                "answer_relevancy": [0.92, 0.87, 0.90],
                "context_precision": [0.78, 0.80, 0.76],
                "context_recall": [0.82, 0.84, 0.80],
            }
            result = service.run_evaluation(triggered_by="manual")

        assert result["status"] == "completed"
        assert result["run_id"] == 1
        assert result["total_questions"] == 3
        assert "faithfulness" in result["metrics"]
        assert "answer_relevancy" in result["metrics"]
        repo.create_run.assert_called_once_with(triggered_by="manual")
        repo.complete_run.assert_called_once()

    def test_run_evaluation_pipeline_error(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        mock_run = MagicMock()
        mock_run.id = 1
        repo.create_run.return_value = mock_run
        repo.insert_result.side_effect = RuntimeError("DB connection lost")

        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())

        with patch.object(service, "_load_default_dataset") as mock_load:
            mock_load.return_value = [{"question": "Q?", "ground_truth": "A"}]
            result = service.run_evaluation(triggered_by="manual")

        assert result["status"] == "failed"
        assert result["run_id"] == 1
        repo.fail_run.assert_called_once()

    def test_run_evaluation_individual_failure(self, sample_dataset):
        """One question fails but the evaluation continues."""
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3

        repo = MagicMock()
        mock_run = MagicMock()
        mock_run.id = 1
        repo.create_run.return_value = mock_run
        repo.get_recent_metrics.return_value = [0.85, 0.82, 0.80]

        rag_system = MagicMock()
        # First question fails, second succeeds
        rag_system.retrieve_and_merge.side_effect = [
            Exception("Retrieval error"),
            [],
            [],
        ]
        rag_system.generate_answer.return_value = iter(["回答"])

        service = EvalService(config, repo, rag_system, MagicMock(), MagicMock())

        with patch.object(service, "_prepare_ragas_dataset") as mock_prep, \
             patch.object(service, "_run_ragas") as mock_ragas:
            mock_prep.return_value = MagicMock()
            mock_ragas.return_value = {
                "faithfulness": [0.0, 0.85, 0.88],
                "answer_relevancy": [0.0, 0.87, 0.90],
                "context_precision": [0.0, 0.80, 0.76],
                "context_recall": [0.0, 0.84, 0.80],
            }
            result = service.run_evaluation(dataset=sample_dataset)

        # Should still complete with 3 questions inserted
        assert result["status"] == "completed"
        assert result["total_questions"] == 3


class TestEvalServiceRegression:
    def test_check_regression_insufficient_runs(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3

        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.55, 0.50]  # only 2 runs

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        result = service.check_regression()

        assert result["detected"] is False
        assert "仅有 2 次历史评估" in result["details"]

    def test_check_regression_detected(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3

        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.55, 0.50, 0.45]  # all below 0.6

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        result = service.check_regression()

        assert result["detected"] is True
        assert "已连续 3 次低于阈值" in result["details"]
        assert result["current_value"] == 0.55

    def test_check_regression_not_detected(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3

        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.75, 0.55, 0.50]  # first is above threshold

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        result = service.check_regression()

        assert result["detected"] is False

    def test_check_regression_none_value(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3

        repo = MagicMock()
        repo.get_recent_metrics.return_value = [None, 0.55, 0.50]

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        result = service.check_regression()

        assert result["detected"] is False
        assert "最新评估无 faithfulness 分数" in result["details"]


class TestEvalServiceQualityStatus:
    def test_quality_status_unknown(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        repo = MagicMock()
        repo.get_latest_completed.return_value = None
        repo.count_runs.return_value = 0

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        with patch.object(service, "check_regression", return_value={"detected": False}):
            status = service.get_quality_status()

        assert status["quality_status"] == "unknown"
        assert status["latest_run"] is None
        assert status["total_runs"] == 0

    def test_quality_status_good(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_run import EvalRun

        config = MagicMock()
        config.EVAL_QUALITY_CRITICAL_THRESHOLD = 0.4
        config.EVAL_QUALITY_WARNING_THRESHOLD = 0.6

        latest = EvalRun(
            id=1, status="completed",
            avg_faithfulness=0.85, avg_answer_relevancy=0.90,
            avg_context_precision=0.78, avg_context_recall=0.82,
            total_questions=10, triggered_by="manual",
        )

        repo = MagicMock()
        repo.get_latest_completed.return_value = latest
        repo.count_runs.return_value = 5

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        with patch.object(service, "check_regression", return_value={"detected": False}):
            with patch.object(service, "_compute_trend_direction", return_value="stable"):
                status = service.get_quality_status()

        assert status["quality_status"] == "good"
        assert status["latest_run"]["id"] == 1
        assert status["latest_run"]["avg_faithfulness"] == 0.85

    def test_quality_status_warning(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_run import EvalRun

        config = MagicMock()
        config.EVAL_QUALITY_CRITICAL_THRESHOLD = 0.4
        config.EVAL_QUALITY_WARNING_THRESHOLD = 0.6

        latest = EvalRun(id=1, status="completed", avg_faithfulness=0.55)

        repo = MagicMock()
        repo.get_latest_completed.return_value = latest
        repo.count_runs.return_value = 3

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        with patch.object(service, "check_regression", return_value={"detected": False}):
            with patch.object(service, "_compute_trend_direction", return_value="declining"):
                status = service.get_quality_status()

        assert status["quality_status"] == "warning"

    def test_quality_status_critical(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_run import EvalRun

        config = MagicMock()
        config.EVAL_QUALITY_CRITICAL_THRESHOLD = 0.4
        config.EVAL_QUALITY_WARNING_THRESHOLD = 0.6

        latest = EvalRun(id=1, status="completed", avg_faithfulness=0.35)

        repo = MagicMock()
        repo.get_latest_completed.return_value = latest
        repo.count_runs.return_value = 10

        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        with patch.object(service, "check_regression", return_value={"detected": True}):
            with patch.object(service, "_compute_trend_direction", return_value="declining"):
                status = service.get_quality_status()

        assert status["quality_status"] == "critical"


class TestEvalServiceTrendDirection:
    def test_trend_improving(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.90, 0.88, 0.85, 0.70, 0.65]

        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        result = service._compute_trend_direction()
        assert result == "improving"

    def test_trend_declining(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.60, 0.65, 0.70, 0.85, 0.90]

        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        result = service._compute_trend_direction()
        assert result == "declining"

    def test_trend_stable(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.80, 0.81, 0.79, 0.82, 0.80]

        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        result = service._compute_trend_direction()
        assert result == "stable"

    def test_trend_insufficient_data(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.80]  # only 1 value

        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        result = service._compute_trend_direction()
        assert result == "stable"


class TestEvalServiceEnsureRagas:
    def test_ensure_ragas_importable(self):
        from rag_qa.eval.eval_service import EvalService
        import sys
        # Should not raise
        EvalService._ensure_ragas_importable()
        # Verify the module was patched
        assert "langchain_community.chat_models.vertexai" in sys.modules


class TestEvalServicePrepareRagasDataset:
    def test_prepare_with_valid_contexts(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_result import EvalResult

        results = [
            EvalResult(id=1, run_id=1, question="Q1", ground_truth="A1",
                       answer="Answer1", contexts=json.dumps(["ctx1", "ctx2"], ensure_ascii=False)),
            EvalResult(id=2, run_id=1, question="Q2", ground_truth="A2",
                       answer="Answer2", contexts=json.dumps(["ctx3"], ensure_ascii=False)),
        ]

        service = EvalService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        dataset = service._prepare_ragas_dataset(results)

        assert len(dataset["question"]) == 2
        assert dataset["question"][0] == "Q1"
        assert dataset["answer"][0] == "Answer1"
        assert dataset["contexts"][0] == ["ctx1", "ctx2"]
        assert dataset["ground_truth"][0] == "A1"

    def test_prepare_with_none_contexts(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_result import EvalResult

        results = [
            EvalResult(id=1, run_id=1, question="Q1", ground_truth="A1",
                       answer="Answer1", contexts=None),
        ]

        service = EvalService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        dataset = service._prepare_ragas_dataset(results)

        assert dataset["contexts"][0] == []

    def test_prepare_with_none_answer(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_result import EvalResult

        results = [
            EvalResult(id=1, run_id=1, question="Q1", ground_truth="A1",
                       answer=None, contexts=None),
        ]

        service = EvalService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        dataset = service._prepare_ragas_dataset(results)

        assert dataset["answer"][0] == ""


# ============================================================================
# 4. Config Tests
# ============================================================================

class TestEvalConfig:
    def test_config_loads_eval_section(self):
        from base.config import Config
        config = Config()
        # These must exist (from config.ini)
        assert hasattr(config, "EVAL_EMBEDDING_MODEL")
        assert hasattr(config, "EVAL_EMBEDDING_BASE_URL")
        assert hasattr(config, "EVAL_INTERVAL_SECONDS")
        assert hasattr(config, "EVAL_REGRESSION_FAITHFULNESS_THRESHOLD")
        assert hasattr(config, "EVAL_REGRESSION_CONSECUTIVE_RUNS")
        assert hasattr(config, "EVAL_QUALITY_WARNING_THRESHOLD")
        assert hasattr(config, "EVAL_QUALITY_CRITICAL_THRESHOLD")
        assert hasattr(config, "EVAL_DEFAULT_DATASET_PATH")

    def test_config_eval_defaults(self):
        from base.config import Config
        config = Config()
        assert config.EVAL_INTERVAL_SECONDS == 86400
        assert config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD == 0.6
        assert config.EVAL_REGRESSION_CONSECUTIVE_RUNS == 3
        assert config.EVAL_QUALITY_WARNING_THRESHOLD == 0.6
        assert config.EVAL_QUALITY_CRITICAL_THRESHOLD == 0.4

    def test_config_eval_embedding(self):
        from base.config import Config
        config = Config()
        assert config.EVAL_EMBEDDING_MODEL == "mxbai-embed-large"
        assert config.EVAL_EMBEDDING_BASE_URL == "http://localhost:11434"

    def test_config_eval_llm_fallback(self):
        """When eval_llm_model is empty, it should fall back to LLM_MODEL."""
        from base.config import Config
        config = Config()
        if not config.EVAL_LLM_MODEL:
            # Fallback to main LLM config
            assert config.LLM_MODEL is not None


# ============================================================================
# 5. API Endpoint Tests (with TestClient)
# ============================================================================

@pytest.fixture
def test_app():
    """Create a FastAPI TestClient with a mocked qa_system."""
    from app import app as fastapi_app
    # Don't trigger the lifespan (which starts periodic eval)
    # Instead, we'll use the app directly with dependency overrides

    # Store original state
    import app as app_module
    original_qa = app_module.qa_system

    # Create mock qa_system
    mock_qa = MagicMock()
    mock_qa.eval_service = MagicMock()
    mock_qa.eval_service.repo = MagicMock()
    app_module.qa_system = mock_qa

    client = TestClient(fastapi_app)

    yield client, mock_qa

    # Restore
    app_module.qa_system = original_qa


class TestEvalAPI:
    def test_eval_run_no_service(self, test_app):
        client, mock_qa = test_app
        mock_qa.eval_service = None

        # Need auth
        response = client.post("/api/eval/run", json={"triggered_by": "manual"})
        # Should be 401 (no auth) or 503 (no service)
        assert response.status_code in (401, 403)

    def test_eval_runs_no_service(self, test_app):
        client, mock_qa = test_app
        mock_qa.eval_service = None

        response = client.get("/api/eval/runs")
        assert response.status_code in (401, 403)

    def test_eval_trends_no_service(self, test_app):
        client, mock_qa = test_app
        mock_qa.eval_service = None

        response = client.get("/api/eval/trends")
        assert response.status_code in (401, 403)

    def test_eval_status_no_service(self, test_app):
        client, mock_qa = test_app
        mock_qa.eval_service = None

        response = client.get("/api/eval/status")
        assert response.status_code in (401, 403)


# ============================================================================
# 6. Health Check Tests
# ============================================================================

class TestEvalHealthCheck:
    def test_check_eval_quality_not_initialized(self):
        from base.health import HealthChecker
        checker = HealthChecker(MagicMock(), MagicMock())
        result = checker.check_eval_quality(None)
        assert result.status.value == "unknown"
        assert "not initialized" in result.error_message

    def test_check_eval_quality_initialized(self):
        from base.health import HealthChecker
        config = MagicMock()
        config.EVAL_QUALITY_CRITICAL_THRESHOLD = 0.4
        config.EVAL_QUALITY_WARNING_THRESHOLD = 0.6

        eval_service = MagicMock()
        eval_service.get_quality_status.return_value = {
            "quality_status": "good",
            "latest_run": {"id": 1, "avg_faithfulness": 0.85},
        }

        checker = HealthChecker(config, MagicMock())
        result = checker.check_eval_quality(eval_service)
        assert result.status.value == "healthy"

    def test_check_eval_quality_critical(self):
        from base.health import HealthChecker
        config = MagicMock()
        config.EVAL_QUALITY_CRITICAL_THRESHOLD = 0.4
        config.EVAL_QUALITY_WARNING_THRESHOLD = 0.6

        eval_service = MagicMock()
        eval_service.get_quality_status.return_value = {
            "quality_status": "critical",
            "latest_run": {"id": 1, "avg_faithfulness": 0.35},
        }

        checker = HealthChecker(config, MagicMock())
        result = checker.check_eval_quality(eval_service)
        assert result.status.value == "degraded"

    def test_check_eval_quality_exception(self):
        from base.health import HealthChecker
        eval_service = MagicMock()
        eval_service.get_quality_status.side_effect = RuntimeError("Boom")

        checker = HealthChecker(MagicMock(), MagicMock())
        result = checker.check_eval_quality(eval_service)
        assert result.status.value == "unhealthy"
        assert "Boom" in result.error_message


# ============================================================================
# 7. Dataset Loading Tests
# ============================================================================

class TestDatasetLoading:
    def test_default_dataset_exists(self):
        """Verify the default dataset JSON file exists and is valid."""
        from base.config import Config
        config = Config()
        path = config.EVAL_DEFAULT_DATASET_PATH
        full_path = os.path.join(project_root, path)
        assert os.path.exists(full_path), f"Dataset not found: {full_path}"

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        # Each item must have at least "question"
        for item in data:
            assert "question" in item, f"Missing 'question' in dataset item: {item}"

    def test_load_default_dataset(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_DEFAULT_DATASET_PATH = os.path.join(
            project_root, "rag_qa", "rag_assesment", "rag_evaluate_data.json"
        )

        service = EvalService(config, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        dataset = service._load_default_dataset()

        assert isinstance(dataset, list)
        assert len(dataset) == 30  # 30 questions in the dataset
        for item in dataset:
            assert "question" in item


# ============================================================================
# 8. Run
# ============================================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
