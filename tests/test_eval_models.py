"""评估自动化管道 — ORM Model Tests"""


class TestEvalRunModel:
    def test_tablename(self):
        from db_models.eval_run import EvalRun

        assert EvalRun.__tablename__ == "eval_runs"

    def test_default_status(self):
        from db_models.eval_run import EvalRun

        run = EvalRun(status="running", total_questions=0)
        assert run.status == "running"

    def test_default_triggered_by(self):
        from db_models.eval_run import EvalRun

        run = EvalRun(triggered_by="manual", status="running", total_questions=0)
        assert run.triggered_by == "manual"

    def test_required_fields(self):
        from db_models.eval_run import EvalRun

        run = EvalRun(
            status="completed",
            total_questions=10,
            avg_faithfulness=0.85,
            avg_answer_relevancy=0.90,
            avg_context_precision=0.78,
            avg_context_recall=0.82,
        )
        assert run.status == "completed"
        assert run.total_questions == 10
        assert run.avg_faithfulness == 0.85

    def test_nullable_metrics(self):
        from db_models.eval_run import EvalRun

        run = EvalRun()
        assert run.avg_faithfulness is None
        assert run.avg_answer_relevancy is None
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
        assert result.faithfulness == 0.92

    def test_nullable_fields(self):
        from db_models.eval_result import EvalResult

        result = EvalResult(run_id=1, question="Q", ground_truth="A")
        assert result.answer is None
        assert result.contexts is None
        assert result.faithfulness is None

    def test_source_filter_default(self):
        from db_models.eval_result import EvalResult

        result = EvalResult(run_id=1, question="Q", ground_truth="A")
        assert result.source_filter is None
