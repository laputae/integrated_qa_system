"""降级路由测试 — _resolve_pipeline_action 门控与组件健康状态查询。

覆盖修复点：
1. Level 3（LLM 故障）且 Milvus 组健康 → degraded 降级检索路径（原代码为死分支）
2. Level 3 且 Milvus 组不健康 → not_found，避免进入必然失败的降级检索
3. Level 2/0/4 原有路由回归守卫
"""

import time

from base import Config, logger
from base.health import SystemHealth
from base.health_types import ComponentHealth, HealthStatus
from base.health_types import DegradationLevel
from main import IntegratedQASystem


def _make_system(rag_system=object()):
    """跳过 __init__（避免拉起 Redis/MySQL/Milvus），仅保留门控所需属性。"""
    sys = IntegratedQASystem.__new__(IntegratedQASystem)
    sys.rag_system = rag_system
    return sys


class _FakeHealth:
    """替身 health 对象：仅提供 get_component_status 映射。"""

    def __init__(self, statuses):
        self._statuses = statuses

    def get_component_status(self, name):
        return self._statuses.get(name)


# ---------- _resolve_pipeline_action ----------


def test_level3_with_healthy_retrieval_routes_to_degraded():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL3_NO_LLM, answer=None, need_rag=True, rag_retrieval_available=True
    )
    assert action == "degraded"
    assert payload is None


def test_level3_with_unhealthy_retrieval_returns_not_found():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL3_NO_LLM, answer=None, need_rag=True, rag_retrieval_available=False
    )
    assert action == "not_found"
    assert "RAG 不可用" in payload


def test_level2_returns_not_found_even_when_retrieval_healthy():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL2_NO_MILVUS, answer=None, need_rag=True, rag_retrieval_available=True
    )
    assert action == "not_found"
    assert "RAG 不可用" in payload


def test_level4_returns_not_found_defensively():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL4_NO_MYSQL, answer=None, need_rag=True, rag_retrieval_available=True
    )
    assert action == "not_found"
    assert "RAG 不可用" in payload


def test_bm25_hit_short_circuits_before_level_checks():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL3_NO_LLM, answer="命中答案", need_rag=True, rag_retrieval_available=False
    )
    assert action == "bm25_hit"
    assert payload == "命中答案"


def test_level0_need_rag_returns_full_rag():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL0_FULL, answer=None, need_rag=True, rag_retrieval_available=True
    )
    assert action == "full_rag"


def test_need_rag_false_returns_not_found():
    sys = _make_system()
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL0_FULL, answer=None, need_rag=False, rag_retrieval_available=True
    )
    assert action == "not_found"
    assert payload == "未找到答案"


def test_no_rag_system_returns_not_found():
    sys = _make_system(rag_system=None)
    action, payload = sys._resolve_pipeline_action(
        DegradationLevel.LEVEL0_FULL, answer=None, need_rag=True, rag_retrieval_available=True
    )
    assert action == "not_found"
    assert payload == "未找到答案"


# ---------- _rag_retrieval_available ----------


def test_rag_retrieval_available_requires_all_three_components_healthy():
    sys = _make_system()
    sys.health = _FakeHealth(
        {
            "milvus": HealthStatus.HEALTHY,
            "embedding": HealthStatus.HEALTHY,
            "reranker": HealthStatus.HEALTHY,
        }
    )
    assert sys._rag_retrieval_available() is True


def test_rag_retrieval_available_false_when_reranker_unhealthy():
    sys = _make_system()
    sys.health = _FakeHealth(
        {
            "milvus": HealthStatus.HEALTHY,
            "embedding": HealthStatus.HEALTHY,
            "reranker": HealthStatus.UNHEALTHY,
        }
    )
    assert sys._rag_retrieval_available() is False


# ---------- SystemHealth.get_component_status ----------


def _healthy_checker(name):
    def _check():
        return ComponentHealth(name=name, status=HealthStatus.HEALTHY, last_checked=time.time())

    return _check


def test_system_health_get_component_status_returns_cached_status():
    health = SystemHealth(Config(), logger)
    health.register_component("milvus", _healthy_checker("milvus"))
    health.register_component("redis", _healthy_checker("redis"))
    assert health.get_component_status("milvus") == HealthStatus.HEALTHY
    assert health.get_component_status("redis") == HealthStatus.HEALTHY


def test_system_health_get_component_status_unknown_name_returns_none():
    health = SystemHealth(Config(), logger)
    health.register_component("milvus", _healthy_checker("milvus"))
    assert health.get_component_status("nonexistent") is None
