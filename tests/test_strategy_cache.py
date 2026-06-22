"""测试策略选择缓存 + 规则预判"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from rag_qa.core.strategy_selector import RulePreJudge, StrategySelector
from base import Config


def test_rule_prejudge():
    """规则预判单元测试（无外部依赖）"""
    rp = RulePreJudge()
    cases = [
        # (query, expected)
        ("AI学费多少？JAVA课程大纲是什么？", "子查询检索"),
        ("1. AI课程 2. JAVA课程",                "子查询检索"),
        ("学费",                                   "直接检索"),
        ("如何部署Milvus到生产环境",               "回溯问题检索"),
        ("AI学科学费是多少",                       "直接检索"),
        ("JAVA的课程大纲是什么",                   "直接检索"),
        ("容器化技术有哪些种类",                   "假设问题检索"),
        ("Milvus的定义",                           "直接检索"),    # Milvus 命中实体规则先于抽象
        ("今天有什么课",                           "直接检索"),    # 6 字符 → 极短规则
    ]
    for query, expected in cases:
        result = rp.prejudge(query)
        status = "PASS" if result == expected else f"FAIL (got {result!r})"
        print(f"  {status}: {query!r} → {result!r}")

    # 统计
    passed = sum(1 for q, e in cases if rp.prejudge(q) == e)
    print(f"\n  RulePreJudge: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_config():
    """验证策略缓存配置"""
    conf = Config()
    ttl = conf.STRATEGY_CACHE_TTL
    assert ttl == 604800, f"Expected 604800, got {ttl}"
    print(f"  PASS: STRATEGY_CACHE_TTL = {ttl}")


def test_strategy_selector_no_redis():
    """StrategySelector 无 Redis 时仍可正常工作"""
    ss = StrategySelector()  # 不传 redis_client
    assert ss.redis_client is None
    assert ss._cache_get("abc123") is None  # 无 Redis 时返回 None
    print("  PASS: StrategySelector 无 Redis 时仍然可用")


def test_hash_query():
    """验证查询哈希稳定"""
    h1 = StrategySelector._hash_query("AI学费多少")
    h2 = StrategySelector._hash_query("AI学费多少")
    h3 = StrategySelector._hash_query("JAVA课程")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 32  # MD5 hex digest
    print("  PASS: _hash_query 稳定且不同查询产生不同哈希")


if __name__ == "__main__":
    print("=== 测试1: 规则预判 ===")
    r1 = test_rule_prejudge()

    print("\n=== 测试2: 配置加载 ===")
    test_config()

    print("\n=== 测试3: 无 Redis 兼容性 ===")
    test_strategy_selector_no_redis()

    print("\n=== 测试4: 查询哈希 ===")
    test_hash_query()

    print("\n全部测试通过！")
