"""评估自动化管道 — Config, API, Health, and Dataset tests"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestEvalConfig:
    def test_config_loads_eval_section(self):
        from base.config import Config
        config = Config()
        assert hasattr(config, "EVAL_EMBEDDING_MODEL")
        assert hasattr(config, "EVAL_INTERVAL_SECONDS")
        assert hasattr(config, "EVAL_REGRESSION_FAITHFULNESS_THRESHOLD")
        assert hasattr(config, "EVAL_QUALITY_WARNING_THRESHOLD")
        assert hasattr(config, "EVAL_QUALITY_CRITICAL_THRESHOLD")
        assert hasattr(config, "EVAL_DEFAULT_DATASET_PATH")

    def test_config_eval_defaults(self):
        from base.config import Config
        config = Config()
        assert config.EVAL_INTERVAL_SECONDS == 21600
        assert config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD == 0.7
        assert config.EVAL_REGRESSION_CONSECUTIVE_RUNS == 3
        assert config.EVAL_QUALITY_WARNING_THRESHOLD == 0.7
        assert config.EVAL_QUALITY_CRITICAL_THRESHOLD == 0.4

    def test_config_eval_embedding(self):
        from base.config import Config
        config = Config()
        assert config.EVAL_EMBEDDING_MODEL == "mxbai-embed-large"
        assert config.EVAL_EMBEDDING_BASE_URL == "http://localhost:11434"

    def test_config_eval_llm_fallback(self):
        from base.config import Config
        config = Config()
        if not config.EVAL_LLM_MODEL:
            assert config.LLM_MODEL is not None


class TestEvalAPI:
    def test_eval_run_no_service(self, test_app):
        client, mock_qa = test_app
        mock_qa.eval_service = None
        response = client.post("/api/eval/run", json={"triggered_by": "manual"})
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


class TestEvalHealthCheck:
    def test_check_eval_quality_not_initialized(self):
        from base.health import HealthChecker
        checker = HealthChecker(MagicMock(), MagicMock())
        result = checker.check_eval_quality(None)
        assert result.status.value == "unknown"

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


class TestDatasetLoading:
    def test_default_dataset_exists(self):
        from base.config import Config
        config = Config()
        path = config.EVAL_DEFAULT_DATASET_PATH
        full_path = os.path.join(_project_root, path)
        assert os.path.exists(full_path), f"Dataset not found: {full_path}"

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        for item in data:
            assert "question" in item

    def test_load_default_dataset(self):
        from rag_qa.eval.eval_service import EvalService
        config = MagicMock()
        config.EVAL_DEFAULT_DATASET_PATH = os.path.join(
            _project_root, "rag_qa", "rag_assesment", "rag_evaluate_data.json"
        )
        service = EvalService(config, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        dataset = service._load_default_dataset()
        assert isinstance(dataset, list)
        assert len(dataset) == 30
        for item in dataset:
            assert "question" in item
