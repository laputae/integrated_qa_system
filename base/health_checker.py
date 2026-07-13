"""
HealthChecker — performs actual dependency probes for each component.

Extracted from health.py to keep each module under 300 lines.
"""

import time

from base import Config, logger
from base.health_types import ComponentHealth, HealthStatus


class HealthChecker:
    """Performs individual health checks for each dependency.

    All check methods accept None for the dependency and return UNHEALTHY
    instead of crashing, so the caller never receives an unhandled exception.
    """

    def __init__(self, config: Config, logger_instance=None):
        self.config = config
        self.logger = logger_instance or logger

    # ------ MySQL ------

    def check_mysql(self, engine) -> ComponentHealth:
        result = ComponentHealth(name="mysql")
        if engine is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "MySQL engine is None (init failed)"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            result.status = HealthStatus.HEALTHY
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"MySQL 健康检查失败: {e}")
        result.last_checked = time.time()
        return result

    # ------ Redis ------

    def check_redis(self, redis_client) -> ComponentHealth:
        result = ComponentHealth(name="redis")
        if redis_client is None or getattr(redis_client, "client", None) is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Redis client is None (init failed)"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            redis_client.client.ping()
            result.status = HealthStatus.HEALTHY
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"Redis 健康检查失败: {e}")
        result.last_checked = time.time()
        return result

    # ------ Milvus ------

    def check_milvus(self, vector_store) -> ComponentHealth:
        result = ComponentHealth(name="milvus")
        if vector_store is None or getattr(vector_store, "client", None) is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Milvus client is None (init failed)"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            collections = vector_store.client.list_collections()
            result.status = HealthStatus.HEALTHY
            result.latency_ms = (time.time() - start) * 1000
            result.to_dict = lambda: {
                **result.__dict__,
                "collections": collections,
            }
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"Milvus 健康检查失败: {e}")
        result.last_checked = time.time()
        return result

    # ------ LLM ------

    def check_llm(self, openai_client, config: Config) -> ComponentHealth:
        result = ComponentHealth(name="llm")
        if openai_client is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "OpenAI client is None (init failed or no API key)"
            result.last_checked = time.time()
            return result
        # Verify the client object is usable without making a paid API call
        try:
            api_key = getattr(openai_client, "api_key", None)
            if not api_key:
                result.status = HealthStatus.UNHEALTHY
                result.error_message = "API key is empty"
            else:
                result.status = HealthStatus.HEALTHY
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
        result.last_checked = time.time()
        return result

    # ------ Embedding model ------

    def check_embedding(self, vector_store) -> ComponentHealth:
        result = ComponentHealth(name="embedding")
        if vector_store is None or getattr(vector_store, "embedding_function", None) is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Embedding function is None (init failed)"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            ef = vector_store.embedding_function
            # Call embedding function directly (returns {"dense": [...], "sparse": [...]})
            _ = ef(["健康检查测试"])
            result.status = HealthStatus.HEALTHY
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"Embedding 健康检查失败: {e}")
        result.last_checked = time.time()
        return result

    # ------ Reranker ------

    def check_reranker(self, vector_store) -> ComponentHealth:
        result = ComponentHealth(name="reranker")
        if vector_store is None or getattr(vector_store, "reranker", None) is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Reranker model is None (init failed)"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            _ = vector_store.reranker.predict([("健康检查", "健康检查")])
            result.status = HealthStatus.HEALTHY
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"Reranker 健康检查失败: {e}")
        result.last_checked = time.time()
        return result

    # ------ Query Classifier ------

    def check_classifier(self, rag_system) -> ComponentHealth:
        result = ComponentHealth(name="classifier")
        if rag_system is None or getattr(rag_system, "query_classifier", None) is None:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Query classifier is None (init failed)"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            qc = rag_system.query_classifier
            if qc.model is None:
                result.status = HealthStatus.UNHEALTHY
                result.error_message = "BERT model not loaded"
            else:
                _ = qc.predict_with_confidence("测试问题")
                result.status = HealthStatus.HEALTHY
                result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"Classifier 健康检查失败: {e}")
        result.last_checked = time.time()
        return result

    # ------ LLM Reranker ------

    def check_llm_reranker(self, config: Config) -> ComponentHealth:
        result = ComponentHealth(name="llm_reranker")
        try:
            if not config.LLM_RERANKER_ENABLED:
                result.status = HealthStatus.HEALTHY
                result.error_message = "LLM reranker is disabled in config"
            elif config.LLM_RERANKER_CRITICAL_MIN_LENGTH < 1:
                result.status = HealthStatus.UNHEALTHY
                result.error_message = "llm_reranker.critical_min_length must be >= 1"
            elif config.LLM_RERANKER_LISTWISE_K < 1:
                result.status = HealthStatus.UNHEALTHY
                result.error_message = "llm_reranker.listwise_k must be >= 1"
            elif not config.LLM_RERANKER_CRITICAL_STRATEGIES:
                result.status = HealthStatus.DEGRADED
                result.error_message = "No critical strategies configured, LLM reranker will never trigger"
            else:
                result.status = HealthStatus.HEALTHY
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
        result.last_checked = time.time()
        return result

    # ------ HallucinationGuard ------

    def check_hallucination_guard(self, rag_system) -> ComponentHealth:
        result = ComponentHealth(name="hallucination_guard")
        if rag_system is None:
            result.status = HealthStatus.UNKNOWN
            result.error_message = "RAGSystem 未初始化"
            result.last_checked = time.time()
            return result

        guard = getattr(rag_system, "hallucination_guard", None)
        if guard is None:
            result.status = HealthStatus.DEGRADED
            result.error_message = "HallucinationGuard 未启用 (hallucination_guard.enabled=false)"
            result.last_checked = time.time()
            return result

        # Verify the model is loaded and callable
        try:
            if guard.model is None:
                result.status = HealthStatus.UNHEALTHY
                result.error_message = "NLI model not loaded"
            else:
                result.status = HealthStatus.HEALTHY
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
        result.last_checked = time.time()
        return result

    # ------ Eval Quality ------

    def check_eval_quality(self, eval_service) -> ComponentHealth:
        result = ComponentHealth(name="eval_quality")
        if eval_service is None:
            result.status = HealthStatus.UNKNOWN
            result.error_message = "Eval service not initialized"
            result.last_checked = time.time()
            return result

        start = time.time()
        try:
            quality = eval_service.get_quality_status()
            qs = quality.get("quality_status", "unknown")
            regression = quality.get("regression", {})
            if qs == "critical":
                result.status = HealthStatus.DEGRADED
                result.error_message = "评估质量严重下降 (faithfulness < critical threshold)"
            elif regression.get("detected"):
                result.status = HealthStatus.DEGRADED
                result.error_message = regression.get("details", "检测到质量回归")
            elif qs == "warning":
                result.status = HealthStatus.DEGRADED
                result.error_message = "评估质量低于警告阈值"
            else:
                result.status = HealthStatus.HEALTHY
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            self.logger.warning(f"Eval quality 健康检查失败: {e}")
        result.last_checked = time.time()
        return result
