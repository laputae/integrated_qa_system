"""
Query embedding cache — Redis-backed cache for dense/sparse query embeddings.

Extracted from vector_store.py to keep each module under 300 lines.
"""
import hashlib

import numpy as np

from base import logger


def get_query_embedding_cached(query, embedding_function, redis_client, config,
                               log=None):
    """Get query embeddings, preferring Redis cache over recomputation.

    Cache key: emb:{md5(query)}
    Falls back to direct computation on cache miss or Redis unavailability.

    Args:
        query: The query string to embed.
        embedding_function: Callable that takes [text] and returns {"dense": [...], "sparse": [...]}.
        redis_client: RedisClient instance for cache read/write.
        config: Config object for cache TTL.
        log: Optional logger (defaults to module-level logger).

    Returns:
        dict with "dense" (list[np.ndarray]) and "sparse" (list[dict]).
    """
    if log is None:
        log = logger

    cache_ttl = config.EMBEDDING_CACHE_TTL
    cache_key = f"emb:{hashlib.md5(query.encode('utf-8')).hexdigest()}"

    # Try cache read
    if redis_client is not None:
        try:
            cached = redis_client.get_data(cache_key)
            if cached is not None:
                log.info(f"查询嵌入缓存命中: {cache_key}")
                dense = np.array(cached["dense"], dtype=np.float32)
                sparse = cached["sparse"]
                return {"dense": [dense], "sparse": [sparse]}
        except Exception as e:
            log.warning(f"Redis 缓存查询失败，降级为直接计算: {e}")

    # Compute
    query_embeddings = embedding_function([query])

    # Ensure dense vectors are float32
    dense_vec = query_embeddings["dense"][0]
    if hasattr(dense_vec, 'dtype') and dense_vec.dtype != np.float32:
        query_embeddings["dense"][0] = dense_vec.astype(np.float32)

    # Write cache
    if redis_client is not None:
        try:
            cache_value = {
                "dense": query_embeddings["dense"][0].tolist(),
                "sparse": _sparse_to_dict(query_embeddings["sparse"][0]),
            }
            redis_client.set_data(cache_key, cache_value, ttl=cache_ttl)
            log.info(f"查询嵌入已缓存: {cache_key}")
        except Exception as e:
            log.warning(f"缓存查询嵌入失败: {e}")

    return query_embeddings


def _sparse_to_dict(sparse_row) -> dict:
    """Convert sparse vector to dict, handling csr_matrix, dict, and empty formats."""
    if hasattr(sparse_row, 'indices'):
        indices = sparse_row.indices if hasattr(sparse_row, 'indices') else sparse_row.col
        return dict(zip(indices, sparse_row.data))
    elif isinstance(sparse_row, dict):
        if sparse_row and not isinstance(next(iter(sparse_row.keys())), int):
            return {int(k): float(v) for k, v in sparse_row.items()}
        return sparse_row
    else:
        return {}
