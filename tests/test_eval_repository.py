"""评估自动化管道 — Repository Tests"""
from unittest.mock import MagicMock

import pytest


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
        mock_session.commit.assert_called()

    def test_complete_run_not_found(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.first.return_value = None
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
        eval_repo.fail_run(run_id=999, error_message="error")

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
        eval_repo.insert_result(run_id=1, question="Q?", ground_truth="A", answer="Answer")
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
        mock_session.commit.assert_called()

    def test_update_result_scores_not_found(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.first.return_value = None
        eval_repo.update_result_scores(result_id=999, scores={})

    def test_get_result_ids_for_run(self, eval_repo, mock_session_factory):
        _, mock_session = mock_session_factory
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = \
            [(1,), (2,), (3,)]
        ids = eval_repo.get_result_ids_for_run(run_id=1)
        assert ids == [1, 2, 3]
