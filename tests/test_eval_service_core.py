"""评估自动化管道 — Service Tests (core)"""
import pytest
from unittest.mock import MagicMock, patch


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

        with patch("rag_qa.eval.eval_service.prepare_ragas_dataset") as mock_prep, \
             patch("rag_qa.eval.eval_service.run_ragas") as mock_ragas:
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
        rag_system.retrieve_and_merge.side_effect = [
            Exception("Retrieval error"),
            [],
            [],
        ]
        rag_system.generate_answer.return_value = iter(["回答"])

        service = EvalService(config, repo, rag_system, MagicMock(), MagicMock())

        with patch("rag_qa.eval.eval_service.prepare_ragas_dataset") as mock_prep, \
             patch("rag_qa.eval.eval_service.run_ragas") as mock_ragas:
            mock_prep.return_value = MagicMock()
            mock_ragas.return_value = {
                "faithfulness": [0.0, 0.85, 0.88],
                "answer_relevancy": [0.0, 0.87, 0.90],
                "context_precision": [0.0, 0.80, 0.76],
                "context_recall": [0.0, 0.84, 0.80],
            }
            result = service.run_evaluation(dataset=sample_dataset)

        assert result["status"] == "completed"
        assert result["total_questions"] == 3


class TestEvalServiceRegression:
    def test_check_regression_insufficient_runs(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.55, 0.50]
        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        result = service.check_regression()
        assert result["detected"] is False

    def test_check_regression_detected(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.55, 0.50, 0.45]
        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        result = service.check_regression()
        assert result["detected"] is True

    def test_check_regression_not_detected(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = 0.6
        config.EVAL_REGRESSION_CONSECUTIVE_RUNS = 3
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.75, 0.55, 0.50]
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


class TestEvalServiceQualityStatus:
    def test_quality_status_unknown(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        repo = MagicMock()
        repo.get_latest_completed.return_value = None
        repo.count_runs.return_value = 0
        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        with patch("rag_qa.eval.quality_reporter.check_regression", return_value={"detected": False}):
            status = service.get_quality_status()
        assert status["quality_status"] == "unknown"

    def test_quality_status_good(self):
        from rag_qa.eval.eval_service import EvalService
        from db_models.eval_run import EvalRun

        config = MagicMock()
        config.EVAL_QUALITY_CRITICAL_THRESHOLD = 0.4
        config.EVAL_QUALITY_WARNING_THRESHOLD = 0.6

        latest = EvalRun(id=1, status="completed", avg_faithfulness=0.85,
                         avg_answer_relevancy=0.90, avg_context_precision=0.78,
                         avg_context_recall=0.82, total_questions=10, triggered_by="manual")
        repo = MagicMock()
        repo.get_latest_completed.return_value = latest
        repo.count_runs.return_value = 5
        service = EvalService(config, repo, MagicMock(), MagicMock(), MagicMock())
        with patch("rag_qa.eval.quality_reporter.check_regression", return_value={"detected": False}):
            status = service.get_quality_status()
        assert status["quality_status"] == "good"

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
        with patch("rag_qa.eval.quality_reporter.check_regression", return_value={"detected": False}):
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
        with patch("rag_qa.eval.quality_reporter.check_regression", return_value={"detected": True}), \
             patch("rag_qa.eval.quality_reporter._compute_trend_direction", return_value="declining"):
            status = service.get_quality_status()
        assert status["quality_status"] == "critical"


class TestEvalServiceTrendDirection:
    def test_trend_improving(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.90, 0.88, 0.85, 0.70, 0.65]
        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        assert service._compute_trend_direction() == "improving"

    def test_trend_declining(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.60, 0.65, 0.70, 0.85, 0.90]
        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        assert service._compute_trend_direction() == "declining"

    def test_trend_stable(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.80, 0.81, 0.79, 0.82, 0.80]
        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        assert service._compute_trend_direction() == "stable"

    def test_trend_insufficient_data(self):
        from rag_qa.eval.eval_service import EvalService
        repo = MagicMock()
        repo.get_recent_metrics.return_value = [0.80]
        service = EvalService(MagicMock(), repo, MagicMock(), MagicMock(), MagicMock())
        assert service._compute_trend_direction() == "stable"
