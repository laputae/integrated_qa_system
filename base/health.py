"""
Health check + multi-level degradation system — orchestration layer.

Provides:
  - SystemHealth: orchestrates health checks, degradation level, circuit breakers, auto-recovery.

Types (HealthStatus, DegradationLevel, ComponentHealth, CircuitBreaker) are in health_types.py.
HealthChecker (per-dependency probes) is in health_checker.py.
"""
import asyncio
import threading
import time
from collections.abc import Callable

from base import Config, logger
from base.health_checker import HealthChecker
from base.health_types import (
    _COMPONENT_DEGRADATION_MAP,
    _COMPONENT_ORDER,
    CircuitBreaker,
    ComponentHealth,
    DegradationLevel,
    HealthStatus,
)
from base.metrics import qa_component_health, qa_degradation_level

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
        self._components: dict[str, tuple] = {}
        self._health_cache: dict[str, ComponentHealth] = {}

        self._background_task: asyncio.Task | None = None

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
            elif result.status == HealthStatus.DEGRADED:
                # Degraded means operational but at reduced capacity (e.g. disabled
                # features). Do NOT promote to UNHEALTHY — that falsely triggers
                # degradation levels and blocks the RAG pipeline.
                component.status = HealthStatus.DEGRADED
                component.error_message = result.error_message
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

    def check_all(self) -> dict[str, ComponentHealth]:
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

    def _get_cached_or_refresh(self) -> dict[str, ComponentHealth]:
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


# ============================================================
# Re-exports for backward compatibility
# ============================================================

__all__ = [
    "HealthStatus",
    "DegradationLevel",
    "ComponentHealth",
    "CircuitBreaker",
    "HealthChecker",
    "SystemHealth",
]
