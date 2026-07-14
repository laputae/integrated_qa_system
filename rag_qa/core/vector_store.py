"""
向量存储 — 基于 LlamaIndex MilvusVectorStore。

读路径：hybrid_search_with_rerank() 使用 LlamaIndex VectorStoreQuery(HYBRID)
写路径：add_documents() 通过 MilvusVectorStore.add() 插入 TextNode
BGE-Reranker 作为后处理重排序，查询嵌入缓存委托给 embedding_cache 模块。
"""

import hashlib
import os
import time

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from base import Config, logger
from base.metrics import qa_rag_retrieval_latency_seconds

from .embedding_cache import get_query_embedding_cached
from .embedding_registry import (
    batch_embed,
    create_milvus_model,
    create_sparse_embedding_function,
    get_dense_dim,
    supports_sparse,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)

conf = Config()


class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(
        self,
        collection_name=conf.MILVUS_COLLECTION_NAME,
        host=conf.MILVUS_HOST,
        port=conf.MILVUS_PORT,
        database=conf.MILVUS_DATABASE_NAME,
        redis_client=None,
    ):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 注入共享 Redis 客户端（可选，为 None 时回退到工厂函数）
        self._redis_client = redis_client
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设备：{self.device}")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = os.path.join(rag_qa_path, "models", "bge-reranker-large")
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        if self.device == "cuda":
            self.reranker.model.half()
        self.reranker_score_threshold = conf.RERANKER_SCORE_THRESHOLD
        # 通过注册表获取嵌入模型（用于查询缓存）
        model_name = conf.EMBEDDING_MODEL
        self.logger.info(f"使用嵌入模型: {model_name}")
        self.embedding_function = create_milvus_model(
            model_name,
            model_path=os.path.join(rag_qa_path, "models", model_name),
            device=self.device,
        )
        self.dense_dim = get_dense_dim(model_name)
        if not supports_sparse(model_name):
            self.logger.warning(f"模型 '{model_name}' 不支持稀疏向量，混合检索将降级为纯稠密检索")
        # 初始化 LlamaIndex MilvusVectorStore（替代原 pymilvus MilvusClient）
        self._init_vector_store()

    def _init_vector_store(self):
        """初始化 LlamaIndex MilvusVectorStore。"""
        from llama_index.vector_stores.milvus import MilvusVectorStore

        model_name = conf.EMBEDDING_MODEL
        model_path = os.path.join(rag_qa_path, "models", model_name)

        self._sparse_embed_fn = create_sparse_embedding_function(
            model_name, model_path=model_path, device=self.device
        )

        self.milvus_vector_store = MilvusVectorStore(
            uri=f"http://{self.host}:{self.port}",
            collection_name=self.collection_name,
            db_name=self.database,
            dim=self.dense_dim,
            embedding_field="dense_vector",
            sparse_embedding_field="sparse_vector",
            text_key="text",
            enable_sparse=self._sparse_embed_fn is not None,
            sparse_embedding_function=self._sparse_embed_fn,
            overwrite=False,
            hybrid_ranker="WeightedRanker",
            hybrid_ranker_params={"weights": [1.0, 0.7]},
        )
        logger.info(f"已加载集合 {self.collection_name}")

    @property
    def client(self):
        """暴露 MilvusClient，供健康检查和 chunk_sweep 使用。"""
        if self.milvus_vector_store is not None:
            return self.milvus_vector_store.client
        return None

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents, batch_size=None, use_checkpoint=True):
        texts = [doc.page_content for doc in documents]

        if not texts:
            self.logger.warning("add_documents: 文档列表为空")
            return

        batch_size = batch_size or conf.EMBEDDING_BATCH_SIZE

        checkpoint_path = None
        if use_checkpoint:
            ckpt_dir = conf.EMBEDDING_CHECKPOINT_DIR
            content_hash = hashlib.md5("".join(texts).encode("utf-8")).hexdigest()[:16]
            checkpoint_path = os.path.join(ckpt_dir, f"add_docs_{content_hash}.json")

        embeddings = batch_embed(
            self.embedding_function,
            texts,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            resume=True,
            desc="Embedding documents",
        )

        from llama_index.core.schema import TextNode

        nodes = []
        for i, doc in enumerate(documents):
            node = TextNode(
                text=doc.page_content,
                metadata={
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "parent_content": doc.metadata.get("parent_content", ""),
                    "source": doc.metadata.get("source", "unknown"),
                    "timestamp": doc.metadata.get("timestamp", "unknown"),
                },
            )
            node.embedding = embeddings["dense"][i]
            nodes.append(node)

        if nodes:
            self.milvus_vector_store.add(nodes)
            logger.info(f"已插入或更新 {len(nodes)} 个文档")

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None, top_k=None):
        start = time.time()
        # 使用带缓存的查询嵌入（委托给 embedding_cache 模块）
        query_embeddings = get_query_embedding_cached(
            query, self.embedding_function, self._redis_client, conf, self.logger
        )
        dense_query_vector = query_embeddings["dense"][0].tolist()
        sparse_query_vector = query_embeddings["sparse"][0]

        # 构建 LlamaIndex 混合检索查询
        from llama_index.core.vector_stores import VectorStoreQuery
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQueryMode

        vs_query = VectorStoreQuery(
            query_str=query,
            query_embedding=dense_query_vector,
            similarity_top_k=k,
            mode=VectorStoreQueryMode.HYBRID,
        )

        if source_filter:
            vs_query.filters = MetadataFilters(
                filters=[MetadataFilter(key="source", value=source_filter)]
            )

        # 通过 LlamaIndex MilvusVectorStore 执行混合检索
        result = self.milvus_vector_store.query(vs_query)

        sub_chunks = [self._doc_from_node(node) for node in result.nodes]
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        if not parent_docs:
            qa_rag_retrieval_latency_seconds.observe(time.time() - start)
            return []
        if len(parent_docs) < 2:
            qa_rag_retrieval_latency_seconds.observe(time.time() - start)
            limit = top_k if top_k is not None else conf.CANDIDATE_M
            return parent_docs[:limit]

        # BGE-Reranker 重排序
        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs)

        sorted_pairs = sorted(zip(scores, parent_docs), key=lambda x: x[0], reverse=True)
        ranked_parent_docs = []
        for score, doc in sorted_pairs:
            doc.metadata["rerank_score"] = float(score)
            ranked_parent_docs.append(doc)

        # Reranker 分数阈值过滤
        threshold = self.reranker_score_threshold
        if threshold > 0.0:
            kept = [doc for doc in ranked_parent_docs if doc.metadata.get("rerank_score", 0.0) >= threshold]
            filtered_count = len(ranked_parent_docs) - len(kept)
            if filtered_count > 0:
                self.logger.info(
                    f"Reranker 阈值={threshold}: 过滤掉 {filtered_count}/{len(ranked_parent_docs)} 个低分文档"
                )
            ranked_parent_docs = kept

        qa_rag_retrieval_latency_seconds.observe(time.time() - start)
        limit = top_k if top_k is not None else conf.CANDIDATE_M
        return ranked_parent_docs[:limit]

    def _get_unique_parent_docs(self, sub_chunks):
        parent_contents = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)
        return unique_docs

    def _doc_from_node(self, node):
        return Document(
            page_content=node.text,
            metadata={
                "parent_id": node.metadata.get("parent_id", ""),
                "parent_content": node.metadata.get("parent_content", ""),
                "source": node.metadata.get("source", ""),
                "timestamp": node.metadata.get("timestamp", ""),
            },
        )
