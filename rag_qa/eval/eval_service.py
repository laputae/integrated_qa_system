import asyncio
import json
import os
import time

from base import Config, logger
from rag_qa.eval.quality_reporter import (
    check_regression,
    get_latest_metrics,
    get_quality_status,
    get_trends,
)
from rag_qa.eval.ragas_runner import (
    prepare_ragas_dataset,
    run_ragas,
)


class EvalService:
    """Evaluation automation pipeline + continuous quality monitoring."""

    def __init__(self, config: Config, repo, rag_system, llm_client, vector_store, executor=None):
        self.config = config
        self.repo = repo
        self.rag_system = rag_system
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.executor = executor
        self.logger = logger
        self._eval_task = None
        self._running = False

    # ================================================================
    # Public API
    # ================================================================

    def run_evaluation(
        self,
        dataset: list | None = None,
        triggered_by: str = "manual",
        chunk_config_snapshot: dict | None = None,
        run_id: int | None = None,
    ) -> dict:
        """Run a full RAGAS evaluation synchronously (call via asyncio.to_thread)."""
        start_time = time.time()

        if self.rag_system is None:
            return {"error": "RAGSystem 未初始化，无法执行评估"}

        # 1. Load dataset
        if dataset is None:
            dataset = self._load_default_dataset()
        if not dataset:
            return {"error": "评估数据集为空"}

        # 2. Create or reuse run record
        run = None
        if run_id is not None:
            run = self.repo.get_run(run_id)
        if run is None:
            run = self.repo.create_run(
                triggered_by=triggered_by,
                chunk_config_snapshot=chunk_config_snapshot,
            )
        run_id = run.id
        self.logger.info(f"[Eval] 开始评估 run_id={run_id}, 问题数={len(dataset)}, 触发方式={triggered_by}")

        try:
            # 3. Run each question through the production pipeline
            pipeline_results = []
            for item in dataset:
                question = item["question"]
                ground_truth = item.get("ground_truth", "")
                source_filter = item.get("source_filter")

                try:
                    answer, contexts = self._run_through_pipeline(question, source_filter)
                except Exception as e:
                    self.logger.warning(f"[Eval] 管线执行失败 (问题: '{question[:30]}...'): {e}")
                    answer = None
                    contexts = []

                result = self.repo.insert_result(
                    run_id=run_id,
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer,
                    contexts=contexts,
                    source_filter=source_filter,
                )
                pipeline_results.append(result)

            # 4. Run RAGAS metrics
            result_ids = [r.id for r in pipeline_results]
            ragas_dataset = prepare_ragas_dataset(pipeline_results)
            ragas_scores = run_ragas(ragas_dataset, self.config, self.logger)

            # 5. Write per-question scores back
            for i, result_id in enumerate(result_ids):
                scores = {}
                for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                    if metric_name in ragas_scores and i < len(ragas_scores[metric_name]):
                        scores[metric_name] = float(ragas_scores[metric_name][i])
                self.repo.update_result_scores(result_id, scores)

            # 6. Compute aggregate metrics
            total = len(dataset)
            aggregates = {}
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric_name in ragas_scores and ragas_scores[metric_name]:
                    values = [v for v in ragas_scores[metric_name] if v is not None]
                    aggregates[metric_name] = float(sum(values) / len(values)) if values else None
                else:
                    aggregates[metric_name] = None

            self.repo.complete_run(run_id, aggregates, total)

            elapsed = time.time() - start_time
            self.logger.info(
                f"[Eval] 评估完成 run_id={run_id}, "
                f"faithfulness={aggregates.get('faithfulness')}, "
                f"answer_relevancy={aggregates.get('answer_relevancy')}, "
                f"耗时={elapsed:.1f}s"
            )

        except Exception as e:
            self.logger.error(f"[Eval] 评估失败 run_id={run_id}: {e}")
            self.repo.fail_run(run_id, str(e))
            return {"run_id": run_id, "status": "failed", "error": str(e)}

        # 7. Check for regression
        regression = None
        try:
            regression = self.check_regression()
        except Exception as e:
            self.logger.warning(f"[Eval] 回归检测失败 run_id={run_id}: {e}")

        return {
            "run_id": run_id,
            "status": "completed",
            "total_questions": total,
            "metrics": aggregates,
            "regression": regression,
            "elapsed_seconds": round(elapsed, 1),
        }

    async def run_evaluation_async(
        self,
        dataset: list | None = None,
        triggered_by: str = "manual",
        chunk_config_snapshot: dict | None = None,
        run_id: int | None = None,
    ) -> dict:
        """Async wrapper for run_evaluation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.run_evaluation,
            dataset,
            triggered_by,
            chunk_config_snapshot,
            run_id,
        )

    # ================================================================
    # Quality reporting (delegates to quality_reporter)
    # ================================================================

    def get_quality_status(self) -> dict:
        return get_quality_status(self.repo, self.config)

    def get_trends(self, limit: int = 20) -> dict:
        return get_trends(self.repo, limit)

    def get_latest_metrics(self) -> dict | None:
        return get_latest_metrics(self.repo)

    def check_regression(self) -> dict:
        return check_regression(self.repo, self.config)

    # ================================================================
    # Background periodic evaluation
    # ================================================================

    async def start_periodic_eval(self):
        interval = self.config.EVAL_INTERVAL_SECONDS
        if interval <= 0:
            self.logger.info("[Eval] 周期评估已禁用 (eval_interval_seconds=0)")
            return
        self._running = True
        loop = asyncio.get_running_loop()
        self._eval_task = loop.create_task(self._eval_loop(interval))
        self.logger.info(f"[Eval] 周期评估已启动 (间隔 {interval}s)")

    async def _eval_loop(self, interval: int):
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                self.logger.info("[Eval] 开始周期评估...")
                result = await self.run_evaluation_async(triggered_by="scheduled")
                self.logger.info(f"[Eval] 周期评估完成: {result.get('status')}")
            except asyncio.CancelledError:
                self.logger.info("[Eval] 周期评估任务已取消")
                break
            except Exception as e:
                self.logger.error(f"[Eval] 周期评估异常: {e}")

    async def stop_periodic_eval(self):
        self._running = False
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
            self._eval_task = None

    # ================================================================
    # Internal helpers
    # ================================================================

    def _load_default_dataset(self) -> list[dict]:
        path = self.config.EVAL_DEFAULT_DATASET_PATH
        if not os.path.isabs(path):
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                path,
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _run_through_pipeline(self, question: str, source_filter: str | None = None) -> tuple[str | None, list[str]]:
        """Run a question through the production RAG pipeline."""
        # Step 1: Retrieve contexts
        try:
            context_docs = self.rag_system.retrieve_and_merge(question, source_filter=source_filter)
            contexts = [doc.page_content for doc in context_docs] if context_docs else []
        except Exception as e:
            self.logger.warning(f"[Eval] 检索失败 (问题: '{question[:30]}...'): {e}")
            contexts = []

        # Step 2: Generate answer via the RAG pipeline
        try:
            answer_tokens = []
            for token in self.rag_system.generate_answer(
                question, source_filter=source_filter, history=None, external_context=None
            ):
                if token:
                    answer_tokens.append(token)
            answer = "".join(answer_tokens) if answer_tokens else None
        except Exception as e:
            self.logger.warning(f"[Eval] LLM生成失败 (问题: '{question[:30]}...'): {e}")
            answer = None

        return answer, contexts

    def _compute_trend_direction(self) -> str:
        """Compute trend direction from the last 5 faithfulness values."""
        from rag_qa.eval.quality_reporter import _compute_trend_direction

        return _compute_trend_direction(self.repo)
