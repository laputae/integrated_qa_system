# -*- coding:utf-8 -*-
import torch.cuda
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
core_path = os.path.join(rag_qa_path, 'core')
sys.path.insert(0, core_path)
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from document_processor import *
from base import logger, Config
from embedding_registry import (
    create_milvus_model, get_dense_dim, supports_sparse, batch_embed
)


conf = Config()


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
def _sparse_to_dict(sparse_row) -> dict:
    """Convert sparse vector to dict, handling csr_matrix, dict, and empty formats."""
    if hasattr(sparse_row, 'indices'):
        # csr_matrix (BGEM3 output)
        indices = sparse_row.indices if hasattr(sparse_row, 'indices') else sparse_row.col
        return dict(zip(indices, sparse_row.data))
    elif isinstance(sparse_row, dict):
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
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
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
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
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
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
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

    def _get_query_embedding_cached(self, query, cache_ttl=None):
        """获取查询嵌入，优先从 Redis 缓存读取。
        缓存 key: emb:{md5(query)}
        缓存未命中或 Redis 不可用时降级为直接计算。
        """
        import hashlib
        import numpy as np
        from mysql_qa import RedisClient

        if cache_ttl is None:
            cache_ttl = conf.EMBEDDING_CACHE_TTL

        cache_key = f"emb:{hashlib.md5(query.encode('utf-8')).hexdigest()}"

        try:
            redis_client = RedisClient()
            cached = redis_client.get_data(cache_key)
            if cached is not None:
                self.logger.info(f"查询嵌入缓存命中: {cache_key}")
                dense = np.array(cached["dense"], dtype=np.float32)
                sparse = cached["sparse"]
                return {"dense": [dense], "sparse": [sparse]}
        except Exception as e:
            self.logger.warning(f"Redis 缓存查询失败，降级为直接计算: {e}")

        query_embeddings = self.embedding_function([query])

        try:
            cache_value = {
                "dense": query_embeddings["dense"][0].tolist(),
                "sparse": _sparse_to_dict(query_embeddings["sparse"][0]),
            }
            redis_client = RedisClient()
            redis_client.set_data(cache_key, cache_value, ttl=cache_ttl)
            self.logger.info(f"查询嵌入已缓存: {cache_key}")
        except Exception as e:
            self.logger.warning(f"缓存查询嵌入失败: {e}")

        return query_embeddings

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # 使用带缓存的查询嵌入
        query_embeddings = self._get_query_embedding_cached(query)
        # 获取查询的稠密向量
        # print(f'query_embeddings---》{query_embeddings}')
        dense_query_vector = query_embeddings["dense"][0]
        # print(f'dense_query_vector--》{dense_query_vector.shape}')
        sparse_query_vector = _sparse_row_to_dict(query_embeddings["sparse"][0])
        # print(f'sparse_query_vector-->{sparse_query_vector}')
        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # print(f'filter_expr--》{filter_expr}')
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

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(1.0, 0.7)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]
        # print(f'results--》{results}')
        # print(f'results--》{type(results)}')
        # print(f'results--》{len(results)}')
        # 将上述搜索到的结果进行Document对象封装，便于查询使用
        sub_chunks = [self._doc_from_hit(hit["entity"])for hit in results]
        # print(f'sub_chunks--》{len(sub_chunks)}')
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')
        # # 如果只有1个文档或者没有，直接返回跳过重排序
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # print(f'scores--》{scores}')
            # 根据得分从高到低排序文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # 如果没有父文档，返回空列表
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 m 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
            # 返回去重后的父文档列表
        return unique_docs

    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )
if __name__ == "__main__":
    vector_store = VectorStore()
    # directory_path = '/Users/ligang/Desktop/EduRAG课堂资料/codes/integrated_qa_system/rag_qa/data/ai_data'
    # print(f"embedding_function.dim--》{vector_store.embedding_function.dim}")
    # documents = process_documents(directory_path)
    # vector_store.add_documents(documents)
    query = "AI学科的课程内容是什么"
    results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    print(f'results-->{results}')
    print(f'results-->{len(results)}')