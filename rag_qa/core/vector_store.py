import hashlib
import os
import time

import torch
from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from sentence_transformers import CrossEncoder

from base import Config, logger
from base.metrics import qa_rag_retrieval_latency_seconds

from .document_processor import *
from .embedding_cache import get_query_embedding_cached
from .embedding_registry import batch_embed, create_milvus_model, get_dense_dim, supports_sparse

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)

conf = Config()


def _sparse_to_dict(sparse_row) -> dict:
    """Convert sparse vector to dict, handling csr_matrix, dict, and empty formats."""
    if hasattr(sparse_row, 'indices'):
        # csr_matrix (BGEM3 output)
        indices = sparse_row.indices if hasattr(sparse_row, 'indices') else sparse_row.col
        return dict(zip(indices, sparse_row.data))
    elif isinstance(sparse_row, dict):
        # Redis JSON round-trip converts int keys to strings — convert them back
        if sparse_row and not isinstance(next(iter(sparse_row.keys())), int):
            return {int(k): float(v) for k, v in sparse_row.items()}
        return sparse_row
    else:
        return {}

# Alias for backward compatibility within this module
_sparse_row_to_dict = _sparse_to_dict


class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME,
                 redis_client=None):
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
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        if self.device == "cuda":
            self.reranker.model.half()
        self.reranker_score_threshold = conf.RERANKER_SCORE_THRESHOLD
        # 通过注册表获取嵌入模型
        model_name = conf.EMBEDDING_MODEL
        self.logger.info(f"使用嵌入模型: {model_name}")
        self.embedding_function = create_milvus_model(
            model_name,
            model_path=os.path.join(rag_qa_path, 'models', model_name),
            device=self.device,
        )
        self.dense_dim = get_dense_dim(model_name)
        if not supports_sparse(model_name):
            self.logger.warning(
                f"模型 '{model_name}' 不支持稀疏向量，混合检索将降级为纯稠密检索"
            )
        # 初始化 Milvus 客户端
        self.client = MilvusClient(
            uri=f"http://{self.host}:{self.port}",
            db_name=self.database,
            timeout=conf.MILVUS_TIMEOUT,
        )
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 类私有化方法
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"已创建集合 {self.collection_name}")
        else:
            logger.info(f"已加载集合 {self.collection_name}")
        self.client.load_collection(self.collection_name)

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

        data = []
        for i, doc in enumerate(documents):
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            sparse_vector = embeddings["sparse"][i]
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f"已插入或更新 {len(data)} 个文档")

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None, top_k=None):
        start = time.time()
        # 使用带缓存的查询嵌入（委托给 embedding_cache 模块）
        query_embeddings = get_query_embedding_cached(
            query, self.embedding_function, self._redis_client, conf, self.logger
        )
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = _sparse_row_to_dict(query_embeddings["sparse"][0])
        # 初始化过滤表达式
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        if not sparse_query_vector:
            # 稀疏向量为空时降级为纯稠密检索
            self.logger.warning("稀疏查询向量为空，降级为纯稠密检索")
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query_vector],
                anns_field="dense_vector",
                search_params={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=k,
                filter=filter_expr,
                output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
            )[0]
        else:
            # 创建稠密向量搜索请求
            dense_request = AnnSearchRequest(
                data=[dense_query_vector],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=k,
                expr=filter_expr
            )
            # 创建稀疏向量搜索请求
            sparse_request = AnnSearchRequest(
                data=[sparse_query_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=k,
                expr=filter_expr
            )
            ranker = WeightedRanker(1.0, 0.7)
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=ranker,
                limit=k,
                output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
            )[0]

        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
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
            kept = [doc for doc in ranked_parent_docs
                    if doc.metadata.get("rerank_score", 0.0) >= threshold]
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

    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )
