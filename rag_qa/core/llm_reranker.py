"""
LLM listwise reranker for RAG document ranking.

Extracted from rag_system.py to keep each module under 300 lines.
"""

import json
import re
import time

from base import logger
from base.metrics import qa_llm_rerank_latency_seconds, qa_llm_rerank_total


def rerank_with_llm(llm_callable, query: str, docs: list, config) -> list:
    """Use LLM to do listwise re-ranking of documents.

    One LLM call sorts all candidate docs. Falls back to original order on failure.

    Args:
        llm_callable: Callable that takes (prompt) and yields/generates tokens.
        query: User query string.
        docs: List of langchain Document objects.
        config: Config object (for prompt template).

    Returns:
        Reordered list of documents (with llm_rerank_position metadata).
    """
    if len(docs) < 2:
        return docs

    start_time = time.time()
    log = logger

    # Format documents as numbered list, truncating each to 800 chars
    doc_lines = []
    for i, doc in enumerate(docs, start=1):
        content_preview = doc.page_content[:800]
        doc_lines.append(f"[{i}] {content_preview}")
    documents_str = "\n---\n".join(doc_lines)

    # Import prompt template locally to avoid circular import at module level
    from prompts import RAGPrompts

    prompt = RAGPrompts.llm_reranker_prompt().format(
        query=query,
        documents=documents_str,
    )

    try:
        response = "".join(llm_callable(prompt)).strip()

        # Extract last JSON array
        json_match = re.search(r"\[[\d,\s]+\]", response)
        if not json_match:
            log.warning(f"LLM reranker: 响应中无 JSON 数组，回退。Response: {response[:200]}")
            qa_llm_rerank_total.labels(status="parse_failure").inc()
            return docs

        indices = json.loads(json_match.group())

        # Validate index count
        if not isinstance(indices, list) or len(indices) != len(docs):
            log.warning(f"LLM reranker: 索引数量不匹配 (got {len(indices)}, expected {len(docs)})，回退")
            qa_llm_rerank_total.labels(status="invalid_indices").inc()
            return docs

        seen = set()
        reordered = []
        for idx in indices:
            zero_idx = int(idx) - 1
            if zero_idx < 0 or zero_idx >= len(docs):
                log.warning(f"LLM reranker: 索引 {idx} 越界 [1, {len(docs)}]，回退")
                qa_llm_rerank_total.labels(status="out_of_range").inc()
                return docs
            if zero_idx in seen:
                log.warning(f"LLM reranker: 重复索引 {idx}，回退")
                qa_llm_rerank_total.labels(status="duplicate_index").inc()
                return docs
            seen.add(zero_idx)
            reordered.append(docs[zero_idx])

        # Annotate positions
        for rank, doc in enumerate(reordered):
            doc.metadata["llm_rerank_position"] = rank
            doc.metadata["llm_rerank_source"] = "llm"

        qa_llm_rerank_total.labels(status="success").inc()
        latency = time.time() - start_time
        qa_llm_rerank_latency_seconds.observe(latency)
        log.info(f"LLM reranker 完成: {len(docs)} 文档重排序, 新顺序: {indices}, 耗时 {latency:.2f}s")
        return reordered

    except json.JSONDecodeError as e:
        log.warning(f"LLM reranker: JSON 解析错误 ({e})，回退")
        qa_llm_rerank_total.labels(status="json_error").inc()
        return docs
    except Exception as e:
        log.warning(f"LLM reranker: 异常 ({e})，回退")
        qa_llm_rerank_total.labels(status="error").inc()
        return docs


def is_critical_query(query: str, strategy: str, config) -> bool:
    """判断查询是否需要 LLM 重排序。

    同时满足以下条件时视为'关键查询'：
    1. LLM 重排序功能已在配置中启用
    2. 策略属于配置的复杂策略列表（HyDE / 回溯 / 子查询）
    3. 查询长度达到配置的最低阈值
    """
    if not config.LLM_RERANKER_ENABLED:
        return False

    if strategy not in config.LLM_RERANKER_CRITICAL_STRATEGIES:
        return False

    if len(query.strip()) < config.LLM_RERANKER_CRITICAL_MIN_LENGTH:
        return False

    return True


def check_force_rag_keywords(query: str) -> bool:
    """领域关键词预检 — 命中则强制走 RAG，不受分类器结果影响"""
    force_rag_patterns = [
        "课程",
        "大纲",
        "教案",
        "讲义",
        "课件",
        "实训",
        "实验",
        "项目",
        "案例",
        "作业",
        "考试",
        "考核",
        "认证",
        "培训",
        "教学",
        "师资",
        "老师",
        "教师",
        "讲师",
        "架构",
        "框架",
        "算法",
        "模型",
        "原理",
        "实现",
        "部署",
        "优化",
        "调优",
    ]
    for pattern in force_rag_patterns:
        if pattern in query:
            return True
    return False
