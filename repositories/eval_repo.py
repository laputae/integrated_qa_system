import json
from datetime import datetime
from typing import Optional

from sqlalchemy import desc

from db_models.eval_run import EvalRun
from db_models.eval_result import EvalResult


class EvalRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    # ---- Run CRUD ----

    def create_run(self, triggered_by: str = "manual") -> EvalRun:
        with self.session_factory() as session:
            run = EvalRun(status="running", triggered_by=triggered_by)
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def complete_run(self, run_id: int, metrics: dict, total_questions: int):
        with self.session_factory() as session:
            run = session.query(EvalRun).filter(EvalRun.id == run_id).first()
            if run is None:
                return
            run.status = "completed"
            run.completed_at = datetime.now()
            run.total_questions = total_questions
            run.avg_faithfulness = metrics.get("faithfulness")
            run.avg_answer_relevancy = metrics.get("answer_relevancy")
            run.avg_context_precision = metrics.get("context_precision")
            run.avg_context_recall = metrics.get("context_recall")
            session.commit()

    def fail_run(self, run_id: int, error_message: str):
        with self.session_factory() as session:
            run = session.query(EvalRun).filter(EvalRun.id == run_id).first()
            if run is None:
                return
            run.status = "failed"
            run.completed_at = datetime.now()
            run.error_message = error_message
            session.commit()

    def get_run(self, run_id: int) -> Optional[EvalRun]:
        with self.session_factory() as session:
            return session.query(EvalRun).filter(EvalRun.id == run_id).first()

    def get_runs(self, limit: int = 20, offset: int = 0) -> list[EvalRun]:
        with self.session_factory() as session:
            return (
                session.query(EvalRun)
                .order_by(desc(EvalRun.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )

    def count_runs(self) -> int:
        with self.session_factory() as session:
            return session.query(EvalRun).count()

    def get_latest_completed(self) -> Optional[EvalRun]:
        with self.session_factory() as session:
            return (
                session.query(EvalRun)
                .filter(EvalRun.status == "completed")
                .order_by(desc(EvalRun.completed_at))
                .first()
            )

    def get_recent_metrics(self, metric_name: str, limit: int = 10) -> list[float]:
        valid_metrics = {
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        }
        if metric_name not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric_name}")

        column_name = f"avg_{metric_name}"
        with self.session_factory() as session:
            column = getattr(EvalRun, column_name)
            rows = (
                session.query(column)
                .filter(EvalRun.status == "completed")
                .filter(column.isnot(None))
                .order_by(desc(EvalRun.completed_at))
                .limit(limit)
                .all()
            )
            return [row[0] for row in rows]

    # ---- Result CRUD ----

    def insert_result(self, run_id: int, question: str, ground_truth: str,
                      answer: str | None = None,
                      contexts: list[str] | None = None,
                      source_filter: str | None = None) -> EvalResult:
        with self.session_factory() as session:
            result = EvalResult(
                run_id=run_id,
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=json.dumps(contexts, ensure_ascii=False) if contexts else None,
                source_filter=source_filter,
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            return result

    def update_result_scores(self, result_id: int, scores: dict):
        with self.session_factory() as session:
            result = session.query(EvalResult).filter(EvalResult.id == result_id).first()
            if result is None:
                return
            result.faithfulness = scores.get("faithfulness")
            result.answer_relevancy = scores.get("answer_relevancy")
            result.context_precision = scores.get("context_precision")
            result.context_recall = scores.get("context_recall")
            session.commit()

    def get_results_for_run(self, run_id: int) -> list[EvalResult]:
        with self.session_factory() as session:
            return (
                session.query(EvalResult)
                .filter(EvalResult.run_id == run_id)
                .order_by(EvalResult.id)
                .all()
            )

    def get_result_ids_for_run(self, run_id: int) -> list[int]:
        with self.session_factory() as session:
            rows = (
                session.query(EvalResult.id)
                .filter(EvalResult.run_id == run_id)
                .order_by(EvalResult.id)
                .all()
            )
            return [row[0] for row in rows]
