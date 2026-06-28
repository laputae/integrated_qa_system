import asyncio
import json
import os
import time

from base import logger, Config


class EvalService:
    """Evaluation automation pipeline + continuous quality monitoring."""

    def __init__(self, config: Config, repo, rag_system, llm_client, vector_store,
                 executor=None):
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

    def run_evaluation(self, dataset: list | None = None,
                       triggered_by: str = "manual",
                       chunk_config_snapshot: dict | None = None,
                       run_id: int | None = None) -> dict:
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
            ragas_dataset = self._prepare_ragas_dataset(pipeline_results)
            ragas_scores = self._run_ragas(ragas_dataset)

            # 5. Write per-question scores back
            for i, result_id in enumerate(result_ids):
                scores = {}
                for metric_name in ["faithfulness", "answer_relevancy",
                                     "context_precision", "context_recall"]:
                    if metric_name in ragas_scores and i < len(ragas_scores[metric_name]):
                        scores[metric_name] = float(ragas_scores[metric_name][i])
                self.repo.update_result_scores(result_id, scores)

            # 6. Compute aggregate metrics
            total = len(dataset)
            aggregates = {}
            for metric_name in ["faithfulness", "answer_relevancy",
                                "context_precision", "context_recall"]:
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

        # 7. Check for regression (outside try block — won't override run status on failure)
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

    async def run_evaluation_async(self, dataset: list | None = None,
                                   triggered_by: str = "manual",
                                   chunk_config_snapshot: dict | None = None,
                                   run_id: int | None = None) -> dict:
        """Async wrapper for run_evaluation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self.run_evaluation, dataset, triggered_by, chunk_config_snapshot, run_id,
        )

    def get_quality_status(self) -> dict:
        """Current quality snapshot for health check / status API."""
        latest = self.repo.get_latest_completed()
        regression = self.check_regression()

        if latest is None:
            return {
                "latest_run": None,
                "regression": regression,
                "quality_status": "unknown",
                "trend_direction": "stable",
                "total_runs": self.repo.count_runs(),
            }

        faithfulness = latest.avg_faithfulness or 0.0

        if faithfulness < self.config.EVAL_QUALITY_CRITICAL_THRESHOLD:
            quality_status = "critical"
        elif faithfulness < self.config.EVAL_QUALITY_WARNING_THRESHOLD:
            quality_status = "warning"
        else:
            quality_status = "good"

        trend = self._compute_trend_direction()

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
            "total_runs": self.repo.count_runs(),
        }

    def get_trends(self, limit: int = 20) -> dict:
        """Return metric trends over time for dashboard."""
        runs = self.repo.get_runs(limit=limit, offset=0)
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

    def get_latest_metrics(self) -> dict | None:
        latest = self.repo.get_latest_completed()
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

    def check_regression(self) -> dict:
        """Check if avg_faithfulness has been below threshold for N consecutive runs."""
        threshold = self.config.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD
        n = self.config.EVAL_REGRESSION_CONSECUTIVE_RUNS

        values = self.repo.get_recent_metrics("faithfulness", limit=n)
        if len(values) < n:
            return {
                "detected": False,
                "details": f"仅有 {len(values)} 次历史评估（需要 {n} 次）",
                "current_value": values[0] if values else None,
                "threshold": threshold,
            }

        current_value = values[0]
        if current_value is None:
            return {"detected": False, "details": "最新评估无 faithfulness 分数", "current_value": None, "threshold": threshold}

        if all(v is not None and v < threshold for v in values):
            return {
                "detected": True,
                "details": f" faithfulness 已连续 {n} 次低于阈值 {threshold}（当前值: {current_value:.3f}）",
                "current_value": current_value,
                "threshold": threshold,
            }

        return {"detected": False, "details": None, "current_value": current_value, "threshold": threshold}

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

    @staticmethod
    def _ensure_ragas_importable():
        """Monkey-patch missing langchain_community modules that ragas requires."""
        import sys
        import langchain_community.chat_models as chat_models_module

        if "langchain_community.chat_models.vertexai" not in sys.modules:
            import types
            vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")

            class ChatVertexAI:
                def __init__(self, *args, **kwargs):
                    raise ImportError(
                        "ChatVertexAI is not available. Install langchain-google-vertexai "
                        "or use a different LLM for RAGAS evaluation."
                    )

            vertexai_module.ChatVertexAI = ChatVertexAI
            sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module
            setattr(chat_models_module, "vertexai", vertexai_module)

    def _load_default_dataset(self) -> list[dict]:
        path = self.config.EVAL_DEFAULT_DATASET_PATH
        if not os.path.isabs(path):
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                path,
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run_through_pipeline(self, question: str,
                               source_filter: str | None = None) -> tuple[str | None, list[str]]:
        """Run a question through the production RAG pipeline.

        Returns (answer, contexts) where contexts is a list of context strings.
        """
        # Step 1: Retrieve contexts
        try:
            context_docs = self.rag_system.retrieve_and_merge(
                question, source_filter=source_filter
            )
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

    def _prepare_ragas_dataset(self, results: list) -> "Dataset":
        from datasets import Dataset

        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        for r in results:
            data["question"].append(r.question)
            data["answer"].append(r.answer or "")
            data["ground_truth"].append(r.ground_truth)
            if r.contexts:
                try:
                    ctx = json.loads(r.contexts)
                except (json.JSONDecodeError, TypeError):
                    ctx = [r.contexts]
            else:
                ctx = []
            data["contexts"].append(ctx)

        return Dataset.from_dict(data)

    def _create_langchain_llm(self):
        from openai import OpenAI
        from ragas.llms import llm_factory

        model = self.config.EVAL_LLM_MODEL or self.config.LLM_MODEL
        base_url = self.config.EVAL_LLM_BASE_URL or self.config.DASHSCOPE_BASE_URL
        api_key = self.config.DASHSCOPE_API_KEY

        client = OpenAI(api_key=api_key, base_url=base_url)
        return llm_factory(model, client=client)

    def _create_langchain_embeddings(self):
        from openai import OpenAI
        from ragas.embeddings.base import embedding_factory

        base_url = self.config.EVAL_EMBEDDING_BASE_URL
        model = self.config.EVAL_EMBEDDING_MODEL
        api_key = self.config.DASHSCOPE_API_KEY

        if "11434" in base_url or "ollama" in base_url.lower():
            client = OpenAI(api_key="ollama", base_url=base_url.rstrip("/") + "/v1")
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)

        return embedding_factory("openai", model=model, client=client)

    def _run_ragas(self, dataset: "Dataset") -> dict:
        self._ensure_ragas_importable()

        from ragas import evaluate
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )

        llm = self._create_langchain_llm()
        embeddings = self._create_langchain_embeddings()

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                ContextPrecision(llm=llm),
                ContextRecall(llm=llm),
            ],
            llm=llm,
            embeddings=embeddings,
        )

        return {
            "faithfulness": result.get("faithfulness", []),
            "answer_relevancy": result.get("answer_relevancy", []),
            "context_precision": result.get("context_precision", []),
            "context_recall": result.get("context_recall", []),
        }

    def _compute_trend_direction(self) -> str:
        """Compute trend direction from the last 5 faithfulness values."""
        values = self.repo.get_recent_metrics("faithfulness", limit=5)
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
