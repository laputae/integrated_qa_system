"""
Health check types: enums, data classes, and circuit breaker.

Extracted from health.py to keep each module under 300 lines.
"""

import time
from dataclasses import dataclass
from enum import Enum, IntEnum

# ============================================================
# Enums
# ============================================================


class HealthStatus(Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DegradationLevel(IntEnum):
    LEVEL0_FULL = 0  # all healthy
    LEVEL1_NO_REDIS = 1  # Redis down, no caching
    LEVEL2_NO_MILVUS = 2  # Milvus/embedding/reranker down, BM25 only
    LEVEL3_NO_LLM = 3  # LLM down, BM25 or raw context
    LEVEL4_NO_MYSQL = 4  # MySQL down, 503


# Maps component name to the degradation level it triggers when unhealthy
_COMPONENT_DEGRADATION_MAP = {
    "mysql": DegradationLevel.LEVEL4_NO_MYSQL,
    "llm": DegradationLevel.LEVEL3_NO_LLM,
    "milvus": DegradationLevel.LEVEL2_NO_MILVUS,
    "embedding": DegradationLevel.LEVEL2_NO_MILVUS,
    "reranker": DegradationLevel.LEVEL2_NO_MILVUS,
    "classifier": DegradationLevel.LEVEL2_NO_MILVUS,
    "llm_reranker": DegradationLevel.LEVEL2_NO_MILVUS,
    "hallucination_guard": DegradationLevel.LEVEL2_NO_MILVUS,
    "redis": DegradationLevel.LEVEL1_NO_REDIS,
}

# Ordered for display
_COMPONENT_ORDER = [
    "mysql",
    "redis",
    "milvus",
    "llm",
    "embedding",
    "reranker",
    "classifier",
    "llm_reranker",
    "hallucination_guard",
    "eval_quality",
]


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
