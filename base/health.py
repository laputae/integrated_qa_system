# -*- coding:utf-8 -*-
"""
Health check + multi-level degradation system.

Provides:
  - Real dependency health checks (MySQL, Redis, Milvus, LLM, embedding, reranker, classifier)
  - Multi-level degradation (Level 0 full → Level 4 no MySQL)
  - Circuit breaker to avoid hammering downed services
  - Auto-recovery background task
"""
import time
import threading
import asyncio
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict

from base import logger, Config
from base.metrics import qa_component_health, qa_degradation_level


# ============================================================
# Enums
# ============================================================

class HealthStatus(Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DegradationLevel(IntEnum):
    LEVEL0_FULL = 0       # all healthy
    LEVEL1_NO_REDIS = 1   # Redis down, no caching
    LEVEL2_NO_MILVUS = 2  # Milvus/embedding/reranker down, BM25 only
    LEVEL3_NO_LLM = 3     # LLM down, BM25 or raw context
    LEVEL4_NO_MYSQL = 4   # MySQL down, 503


# Maps component name to the degradation level it triggers when unhealthy
_COMPONENT_DEGRADATION_MAP = {
    "mysql": DegradationLevel.LEVEL4_NO_MYSQL,
    "llm": DegradationLevel.LEVEL3_NO_LLM,
    "milvus": DegradationLevel.LEVEL2_NO_MILVUS,
    "embedding": DegradationLevel.LEVEL2_NO_MILVUS,
    "reranker": DegradationLevel.LEVEL2_NO_MILVUS,
    "classifier": DegradationLevel.LEVEL2_NO_MILVUS,
    "redis": DegradationLevel.LEVEL1_NO_REDIS,
}

# Odered for display
_COMPONENT_ORDER = ["mysql", "redis", "milvus", "llm", "embedding", "reranker", "classifier", "eval_quality"]


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0.0
    last_checked: float = 0.0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 1),
            "last_checked": self.last_checked,
            "error_message": self.error_message,
        }


# ============================================================
# Circuit Breaker
# ============================================================

class CircuitBreaker:
    """Three-state circuit breaker to avoid hammering downed services."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> str:
        return self._state

    def record_success(self):
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
        self._consecutive_failures = 0

    def record_failure(self):
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= self.failure_threshold:
            self._state = self.OPEN

    def can_probe(self) -> bool:
        """Should we attempt a health check now?"""
        if self._state == self.CLOSED:
            return True
        if self._state == self.HALF_OPEN:
            return True
        # OPEN state: only probe if cooldown has expired
        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.cooldown_seconds:
            self._state = self.HALF_OPEN
            return True
        return False

    def reset(self):
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0


# ============================================================
# Health Checker — Performs actual dependency probes
# ============================================================

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
        if redis_client is None or getattr(redis_client, 'client', None) is None:
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
        if vector_store is None or getattr(vector_store, 'client', None) is None:
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
            api_key = getattr(openai_client, 'api_key', None)
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
        if vector_store is None or getattr(vector_store, 'embedding_function', None) is None:
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
        if vector_store is None or getattr(vector_store, 'reranker', None) is None:
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
        if rag_system is None or getattr(rag_system, 'query_classifier', None) is None:
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
                result.error_message = f"评估质量严重下降 (faithfulness < critical threshold)"
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


# ============================================================
# System Health — Central orchestrator
# ============================================================

class SystemHealth:
    """Orchestrates health checks, degradation level, circuit breakers, and auto-recovery."""

    def __init__(self, config: Config, logger_instance=None):
        self.config = config
        self.logger = logger_instance or logger
        self._startup_time = time.time()
        self._lock = threading.Lock()
        self._last_full_check = 0.0
        self._cache_ttl = config.HEALTH_CACHE_TTL
        self._recovery_interval = config.HEALTH_RECOVERY_INTERVAL

        self._checker = HealthChecker(config, self.logger)

        # Registered components: name -> (checker_callable, ComponentHealth, CircuitBreaker)
        self._components: Dict[str, tuple] = {}
        self._health_cache: Dict[str, ComponentHealth] = {}

        self._background_task: Optional[asyncio.Task] = None

    # ------- Registration -------

    def register_component(
        self,
        name: str,
        checker: Callable[[], ComponentHealth],
    ):
        cb = CircuitBreaker(
            failure_threshold=self.config.HEALTH_CIRCUIT_BREAKER_THRESHOLD,
            cooldown_seconds=self.config.HEALTH_CIRCUIT_BREAKER_COOLDOWN,
        )
        component = ComponentHealth(name=name)
        self._components[name] = (checker, component, cb)

    # ------- Core check logic -------

    def _run_check(self, name: str) -> ComponentHealth:
        """Run a single component check, respecting circuit breaker state."""
        checker, component, cb = self._components[name]

        if not cb.can_probe():
            # Circuit is OPEN and cooldown hasn't expired — return cached
            return component

        try:
            result = checker()
            if result.status == HealthStatus.HEALTHY:
                cb.record_success()
                component.consecutive_failures = 0
                component.status = HealthStatus.HEALTHY
                component.error_message = ""
            else:
                cb.record_failure()
                component.consecutive_failures += 1
                component.last_failure_time = time.time()
                component.status = HealthStatus.UNHEALTHY
                component.error_message = result.error_message
            component.latency_ms = result.latency_ms
            component.last_checked = result.last_checked
        except Exception as e:
            cb.record_failure()
            component.consecutive_failures += 1
            component.last_failure_time = time.time()
            component.status = HealthStatus.UNHEALTHY
            component.error_message = str(e)
            component.last_checked = time.time()
            self.logger.warning(f"健康检查异常 ({name}): {e}")

        return component

    def check_all(self) -> Dict[str, ComponentHealth]:
        """Run a full health check across all registered components."""
        with self._lock:
            for name in self._components:
                self._health_cache[name] = self._run_check(name)
            self._last_full_check = time.time()
        self._export_metrics()
        return dict(self._health_cache)

    def _export_metrics(self):
        """Update Prometheus gauges from current health state."""
        for name, component in self._health_cache.items():
            qa_component_health.labels(component=name).set(
                1 if component.status == HealthStatus.HEALTHY else 0
            )
        qa_degradation_level.set(int(self._compute_degradation_level()))

    def _compute_degradation_level(self) -> DegradationLevel:
        """Compute degradation level from cache without TTL refresh (for metrics)."""
        max_level = DegradationLevel.LEVEL0_FULL
        for name, component in self._health_cache.items():
            if component.status == HealthStatus.UNHEALTHY:
                level = _COMPONENT_DEGRADATION_MAP.get(name)
                if level is not None and level > max_level:
                    max_level = level
        return max_level

    def _get_cached_or_refresh(self) -> Dict[str, ComponentHealth]:
        """Return cached results if still fresh, otherwise re-check."""
        if time.time() - self._last_full_check < self._cache_ttl and self._health_cache:
            return dict(self._health_cache)
        return self.check_all()

    # ------- Degradation level -------

    def get_degradation_level(self) -> DegradationLevel:
        """Compute current degradation level from cached health status."""
        self._get_cached_or_refresh()
        return self._compute_degradation_level()

    def is_ready(self) -> bool:
        """Can the app serve traffic? Level 4 (no MySQL) means not ready."""
        return self.get_degradation_level() < DegradationLevel.LEVEL4_NO_MYSQL

    # ------- Status response -------

    def get_status_response(self) -> dict:
        health = self._get_cached_or_refresh()
        level = self.get_degradation_level()
        label_map = {
            0: "full",
            1: "no_redis",
            2: "no_milvus",
            3: "no_llm",
            4: "no_mysql",
        }
        components = {}
        for name in _COMPONENT_ORDER:
            if name in health:
                components[name] = health[name].to_dict()

        overall = "healthy" if level == 0 else ("degraded" if level < 4 else "unhealthy")
        return {
            "status": overall,
            "degradation_level": level.value,
            "degradation_label": label_map.get(level.value, "unknown"),
            "components": components,
            "uptime_seconds": round(time.time() - self._startup_time, 0),
            "cache_ttl": self._cache_ttl,
        }

    # ------- Background auto-recovery -------

    async def start_background_recovery(self):
        """Start the background recovery loop as an asyncio task."""
        loop = asyncio.get_running_loop()
        self._background_task = loop.create_task(self._recovery_loop())
        self.logger.info(
            f"后台恢复任务已启动 (间隔 {self._recovery_interval}s)"
        )

    async def _recovery_loop(self):
        while True:
            try:
                await asyncio.sleep(self._recovery_interval)
                self._recover_unhealthy()
            except asyncio.CancelledError:
                self.logger.info("后台恢复任务已取消")
                break
            except Exception as e:
                self.logger.error(f"后台恢复循环异常: {e}")

    def _recover_unhealthy(self):
        """Check unhealthy components to see if they've recovered."""
        with self._lock:
            for name, (checker, component, cb) in self._components.items():
                if component.status != HealthStatus.UNHEALTHY:
                    continue
                if not cb.can_probe():
                    continue

                try:
                    result = checker()
                    if result.status == HealthStatus.HEALTHY:
                        cb.record_success()
                        component.status = HealthStatus.HEALTHY
                        component.consecutive_failures = 0
                        component.error_message = ""
                        component.latency_ms = result.latency_ms
                        component.last_checked = result.last_checked
                        self.logger.info(f"[恢复] {name} 已恢复健康")
                        self._health_cache[name] = component
                    else:
                        cb.record_failure()
                        component.last_failure_time = time.time()
                        self.logger.debug(f"[恢复] {name} 仍不可用: {result.error_message}")
                except Exception as e:
                    cb.record_failure()
                    component.last_failure_time = time.time()
                    self.logger.debug(f"[恢复] {name} 检查异常: {e}")

    async def close(self):
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
