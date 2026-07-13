"""
RAG retrieval strategies: backtracking, subqueries, HyDE.

Extracted from rag_system.py to keep each module under 300 lines.
Each strategy is a standalone function that takes its dependencies explicitly.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from base import logger as _default_logger

from .prompts import RAGPrompts


def retrieve_with_backtracking(
    vector_store,
    llm_callable,
    redis_client,
    config,
    query,
    source_filter=None,
    top_k=None,
    log=None,
):
    """使用回溯问题进行检索。LLM 生成简化查询 → 检索原始文档。"""
    if log is None:
        log = _default_logger
    log.info(f"使用回溯问题策略进行检索 (查询: '{query}')")

    backtrack_prompt_template = RAGPrompts.backtracking_prompt()

    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
    cache_key = f"bt:{query_hash}"
    simplified_query = None

    if redis_client:
        try:
            simplified_query = redis_client.get_data(cache_key)
            if simplified_query:
                log.info(f"回溯问题缓存命中: '{simplified_query}' (原始查询: '{query}')")
        except Exception as e:
            log.warning(f"读取回溯问题缓存失败: {e}")

    try:
        if not simplified_query:
            simplified_query = "".join(llm_callable(backtrack_prompt_template.format(query=query))).strip()
            log.info(f"生成的回溯问题: '{simplified_query}'")
            if redis_client:
                try:
                    redis_client.set_data(cache_key, simplified_query, ttl=config.EMBEDDING_CACHE_TTL)
                except Exception as e:
                    log.warning(f"写入回溯问题缓存失败: {e}")

        return vector_store.hybrid_search_with_rerank(
            simplified_query,
            k=config.RETRIEVAL_K,
            source_filter=source_filter,
            top_k=top_k,
        )
    except Exception as e:
        log.error(f"回溯问题策略执行失败: {e}")
        return []


def retrieve_with_subqueries(
    vector_store,
    llm_callable,
    redis_client,
    config,
    query,
    source_filter=None,
    top_k=None,
    log=None,
):
    """使用子查询进行检索。LLM 分解查询 → 并行检索 → 去重合并。"""
    if log is None:
        log = _default_logger
    log.info(f"使用子查询策略进行检索 (查询: '{query}')")

    subquery_prompt_template = RAGPrompts.subquery_prompt()

    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
    cache_key = f"sq:{query_hash}"
    subqueries = None

    if redis_client:
        try:
            subqueries = redis_client.get_data(cache_key)
            if subqueries:
                log.info(f"子查询缓存命中: {subqueries} (原始查询: '{query}')")
        except Exception as e:
            log.warning(f"读取子查询缓存失败: {e}")

    try:
        if not subqueries:
            subqueries_text = "".join(llm_callable(subquery_prompt_template.format(query=query))).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            log.info(f"生成的子查询: {subqueries}")
            if redis_client:
                try:
                    redis_client.set_data(cache_key, subqueries, ttl=config.EMBEDDING_CACHE_TTL)
                except Exception as e:
                    log.warning(f"写入子查询缓存失败: {e}")
        if not subqueries:
            log.warning("未能生成有效的子查询")
            return []

        all_docs = []
        max_workers = min(len(subqueries), config.RETRIEVAL_MAX_WORKERS)
        sub_k = top_k if top_k is not None else config.CANDIDATE_M // 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_subq = {
                executor.submit(vector_store.hybrid_search_with_rerank, sub_q, sub_k, source_filter): sub_q
                for sub_q in subqueries
            }
            for future in as_completed(future_to_subq):
                sub_q = future_to_subq[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    log.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
                except Exception as e:
                    log.error(f"子查询 '{sub_q}' 检索失败: {e}")

        unique_docs_dict = {doc.page_content: doc for doc in all_docs}
        unique_docs = list(unique_docs_dict.values())

        log.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
        return unique_docs
    except Exception as e:
        log.error(f"子查询存在错误：{e}")
        return []


def retrieve_with_hyde(
    vector_store,
    llm_callable,
    redis_client,
    config,
    query,
    source_filter=None,
    top_k=None,
    log=None,
):
    """使用 HyDE (假设文档) 进行检索。LLM 生成假设答案 → 以答案为 query 检索。"""
    if log is None:
        log = _default_logger
    log.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")

    hyde_prompt_template = RAGPrompts.hyde_prompt()

    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
    cache_key = f"hyde:{query_hash}"
    hypo_answer = None

    if redis_client:
        try:
            hypo_answer = redis_client.get_data(cache_key)
            if hypo_answer:
                log.info(f"HyDE 假设答案缓存命中 (原始查询: '{query}')")
        except Exception as e:
            log.warning(f"读取 HyDE 缓存失败: {e}")

    try:
        if not hypo_answer:
            hypo_answer = "".join(llm_callable(hyde_prompt_template.format(query=query))).strip()
            log.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            if redis_client:
                try:
                    redis_client.set_data(cache_key, hypo_answer, ttl=config.EMBEDDING_CACHE_TTL)
                except Exception as e:
                    log.warning(f"写入 HyDE 缓存失败: {e}")

        return vector_store.hybrid_search_with_rerank(
            hypo_answer,
            k=config.RETRIEVAL_K,
            source_filter=source_filter,
            top_k=top_k,
        )
    except Exception as e:
        log.error(f"HyDE 策略执行失败: {e}")
        return []
