# -*-coding:utf-8-*-
"""
Embedding model registry for A/B comparison.

Allows runtime switching between embedding models via config.ini [embedding] model.
Each registered model provides:
  - dense_dim: embedding vector dimension
  - supports_sparse: whether model produces sparse vectors (for hybrid search)
  - create_milvus_model(path, device) -> callable compatible with BGEM3EmbeddingFunction interface
  - create_llamaindex_model(path, device) -> BaseEmbedding compatible with LlamaIndex
"""
import json
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current_dir)
sys.path.insert(0, _rag_qa_path)
_project_root = os.path.dirname(_rag_qa_path)
sys.path.insert(0, _project_root)

from base import logger

MODEL_DIR = os.path.join(_rag_qa_path, 'models')

_registry: dict = {}


def register(name: str, *, dense_dim: int, supports_sparse: bool = False,
             milvus_factory=None, llamaindex_factory=None):
    """Register an embedding model."""
    _registry[name] = {
        "dense_dim": dense_dim,
        "supports_sparse": supports_sparse,
        "milvus_factory": milvus_factory,
        "llamaindex_factory": llamaindex_factory,
    }
    logger.info(f"Registered embedding model: {name} (dim={dense_dim}, sparse={supports_sparse})")


def get_model_info(name: str) -> dict:
    if name not in _registry:
        available = ", ".join(_registry.keys())
        raise ValueError(f"Unknown embedding model: '{name}'. Available: [{available}]")
    return _registry[name]


def get_dense_dim(name: str) -> int:
    return get_model_info(name)["dense_dim"]


def supports_sparse(name: str) -> bool:
    return get_model_info(name)["supports_sparse"]


def list_models() -> list:
    return list(_registry.keys())


def create_milvus_model(name: str, model_path: str = None,
                        device: str = None):
    """Create a model callable for pymilvus/MilvusClient.

    Returns callable with signature: __call__(texts: list[str]) -> dict
    Output: {"dense": list[np.ndarray], "sparse": csr_matrix or list[dict]}
    """
    info = get_model_info(name)
    if info["milvus_factory"] is None:
        raise ValueError(f"Model '{name}' does not support milvus interface.")
    if device is None:
        device = 'cpu'
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(model_path):
        logger.warning(
            f"Model directory not found: {model_path}. "
            f"Will be downloaded on first use (requires internet)."
        )
    return info["milvus_factory"](model_path=model_path, device=device)


def create_llamaindex_model(name: str, model_path: str = None,
                            device: str = None):
    """Create an embedding model for LlamaIndex."""
    info = get_model_info(name)
    if info["llamaindex_factory"] is None:
        raise ValueError(f"Model '{name}' does not support llama_index interface.")
    if device is None:
        device = 'cpu'
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(model_path):
        logger.warning(
            f"Model directory not found: {model_path}. "
            f"Will be downloaded on first use (requires internet)."
        )
    return info["llamaindex_factory"](model_path=model_path, device=device)


# ============================================================
# Built-in model registrations
# ============================================================

# --- bge-m3 (default, dense+sparse hybrid) ---
try:
    from milvus_model.hybrid import BGEM3EmbeddingFunction as _BGEM3

    def _bge_m3_milvus_factory(model_path, device, **kwargs):
        return _BGEM3(
            model_name_or_path=model_path,
            use_fp16=(device == 'cuda'),
            device=device,
        )

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding as _HFEmb

        def _bge_m3_llamaindex_factory(model_path, device, **kwargs):
            return _HFEmb(model_name=model_path, device=device)

        _llamaindex_bge_m3 = _bge_m3_llamaindex_factory
    except ImportError:
        _llamaindex_bge_m3 = None

    register(
        "bge-m3",
        dense_dim=1024,
        supports_sparse=True,
        milvus_factory=_bge_m3_milvus_factory,
        llamaindex_factory=_llamaindex_bge_m3,
    )
except ImportError as e:
    logger.warning(f"Cannot register bge-m3: {e}")


# --- bge-large-zh (dense-only via sentence-transformers) ---
try:
    from sentence_transformers import SentenceTransformer as _ST

    def _bge_large_zh_milvus_factory(model_path, device, **kwargs):
        model = _ST(model_path, device=device)

        def _embed(texts):
            import numpy as np
            dense = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return {
                "dense": [np.array(vec, dtype=np.float32) for vec in dense],
                "sparse": [{} for _ in texts],
            }
        return _embed

    try:
        def _bge_large_zh_llamaindex_factory(model_path, device, **kwargs):
            return _HFEmb(model_name=model_path, device=device)
        _llamaindex_bge_large_zh = _bge_large_zh_llamaindex_factory
    except ImportError:
        _llamaindex_bge_large_zh = None

    register(
        "bge-large-zh",
        dense_dim=1024,
        supports_sparse=False,
        milvus_factory=_bge_large_zh_milvus_factory,
        llamaindex_factory=_llamaindex_bge_large_zh,
    )
except ImportError as e:
    logger.warning(f"Cannot register bge-large-zh: {e}")


# --- text2vec-large-chinese (dense-only via sentence-transformers) ---
try:
    def _text2vec_milvus_factory(model_path, device, **kwargs):
        model = _ST(model_path, device=device)

        def _embed(texts):
            import numpy as np
            dense = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return {
                "dense": [np.array(vec, dtype=np.float32) for vec in dense],
                "sparse": [{} for _ in texts],
            }
        return _embed

    try:
        def _text2vec_llamaindex_factory(model_path, device, **kwargs):
            return _HFEmb(model_name=model_path, device=device)
        _llamaindex_text2vec = _text2vec_llamaindex_factory
    except ImportError:
        _llamaindex_text2vec = None

    register(
        "text2vec-large-chinese",
        dense_dim=768,
        supports_sparse=False,
        milvus_factory=_text2vec_milvus_factory,
        llamaindex_factory=_llamaindex_text2vec,
    )
except ImportError as e:
    logger.warning(f"Cannot register text2vec-large-chinese: {e}")


# ============================================================
# batch_embed(): batched embedding with progress bar + checkpoint/resume
# ============================================================
from typing import List, Optional
from tqdm import tqdm


def batch_embed(
    embedding_fn,
    texts: List[str],
    batch_size: int = 32,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    desc: str = "Embedding",
) -> dict:
    """Embed texts in batches with tqdm progress and optional checkpoint/resume.

    Returns:
        dict with "dense" (list[np.ndarray]) and "sparse" (list[dict]) keys.
    """
    import numpy as np

    if not texts:
        logger.warning("batch_embed called with empty texts list")
        return {"dense": [], "sparse": []}

    completed_batches: set = set()
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            completed_batches = set(ckpt.get("completed_batches", []))
            logger.info(
                f"Resuming from checkpoint: {len(completed_batches)} batches done "
                f"({len(completed_batches) * batch_size} texts)"
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Checkpoint unreadable, starting fresh: {e}")

    batches = [
        (idx, texts[i:i + batch_size])
        for idx, i in enumerate(range(0, len(texts), batch_size))
    ]

    all_dense = []
    all_sparse = []

    for batch_idx, batch_texts in tqdm(batches, desc=desc, unit="batch"):
        if batch_idx in completed_batches:
            continue

        try:
            embeddings = embedding_fn(batch_texts)
            sparse_batch = []
            for s in embeddings["sparse"]:
                if hasattr(s, 'indices'):
                    indices = s.indices if hasattr(s, 'indices') else s.col
                    sparse_batch.append(dict(zip(indices, s.data)))
                elif isinstance(s, dict):
                    sparse_batch.append(s)
                else:
                    sparse_batch.append({})

            for vec in embeddings["dense"]:
                if isinstance(vec, np.ndarray):
                    all_dense.append(vec if vec.dtype == np.float32 else vec.astype(np.float32))
                else:
                    all_dense.append(np.array(vec, dtype=np.float32))
            all_sparse.extend(sparse_batch)

            if checkpoint_path:
                completed_batches.add(batch_idx)
                _save_checkpoint(checkpoint_path, {
                    "completed_batches": sorted(completed_batches),
                    "total_texts": len(texts),
                })

        except Exception as e:
            logger.error(
                f"Batch {batch_idx + 1}/{len(batches)} failed "
                f"(texts {batch_idx * batch_size}-{batch_idx * batch_size + len(batch_texts)}): {e}"
            )
            raise RuntimeError(
                f"Embedding failed at batch {batch_idx}. "
                f"Set resume=True and rerun to skip completed batches."
            ) from e

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint removed after completion: {checkpoint_path}")
        except OSError:
            pass

    return {"dense": all_dense, "sparse": all_sparse}


def _save_checkpoint(path: str, data: dict):
    import tempfile
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
