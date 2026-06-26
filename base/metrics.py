# -*- coding:utf-8 -*-
"""
Prometheus metrics for the integrated QA system.

All metrics use the default registry. Import this module and call the
metric methods directly — they are module-level singletons.

HTTP-level metrics (http_requests_total, http_request_duration_seconds)
are handled automatically by prometheus-fastapi-instrumentator in app.py.
"""

from prometheus_client import Counter, Histogram, Gauge

# ---- Business query metrics ----

qa_query_total = Counter(
    'qa_query_total',
    'Total number of user queries processed',
    ['degradation_level', 'source'],
)

qa_query_latency_seconds = Histogram(
    'qa_query_latency_seconds',
    'End-to-end query processing latency',
    ['degradation_level'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')),
)

# ---- LLM call metrics ----

qa_llm_call_total = Counter(
    'qa_llm_call_total',
    'LLM API call results',
    ['status'],  # success | failure | retry_exhausted
)

# ---- BM25 metrics ----

qa_bm25_hit_total = Counter(
    'qa_bm25_hit_total',
    'BM25 search cache hits',
)

# ---- RAG retrieval metrics ----

qa_rag_retrieval_latency_seconds = Histogram(
    'qa_rag_retrieval_latency_seconds',
    'RAG document retrieval latency',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')),
)

# ---- LLM Reranker metrics ----

qa_llm_rerank_total = Counter(
    'qa_llm_rerank_total',
    'LLM reranking call results',
    ['status'],  # success, parse_failure, invalid_indices, out_of_range, duplicate_index, json_error, error
)

qa_llm_rerank_latency_seconds = Histogram(
    'qa_llm_rerank_latency_seconds',
    'LLM reranking latency (listwise call)',
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf')),
)

# ---- HallucinationGuard metrics ----

qa_hallucination_guard_total = Counter(
    'qa_hallucination_guard_total',
    'HallucinationGuard verification results',
    ['result'],  # passed | flagged | no_claims | error
)

qa_hallucination_guard_latency_seconds = Histogram(
    'qa_hallucination_guard_latency_seconds',
    'HallucinationGuard per-query NLI verification latency',
    buckets=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float('inf')),
)

# ---- Health / degradation metrics ----

qa_component_health = Gauge(
    'qa_component_health',
    'Per-component health status (1 = healthy, 0 = unhealthy)',
    ['component'],
)

qa_degradation_level = Gauge(
    'qa_degradation_level',
    'Current system degradation level (0 = full, 4 = no MySQL)',
)
