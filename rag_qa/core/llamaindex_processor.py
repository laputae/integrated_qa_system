# -*-coding:utf-8-*-
"""
LlamaIndex 文档处理器 - 混合模式
- 文档加载：保留原始 OCR 加载器（OCRPDFLoader/OCRDOCLoader/OCRPPTLoader/OCRIMGLoader）
- 文本切分：保留原始 ChineseRecursiveTextSplitter + MarkdownTextSplitter
- 索引构建：使用 LlamaIndex VectorStoreIndex 实现增量更新
- 增量追踪：SQLite IngestionTracker + LlamaIndex ref_doc_id
"""

import os
from datetime import datetime

from langchain_community.document_loaders import TextLoader

try:
    from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
except ImportError:
    UnstructuredMarkdownLoader = None

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.vector_stores.milvus import MilvusVectorStore

from base import Config, logger
from base.chunk_config import ChunkConfigManager

from ..edu_document_loaders import OCRDOCLoader, OCRIMGLoader, OCRPDFLoader, OCRPPTLoader
from ..edu_text_spliter.chunk_strategy import (
    MARKDOWN,
    RECURSIVE,
    create_child_splitter,
    create_parent_splitter,
)
from .document_quality import (
    clean_document_text,
    estimate_document_quality,
)
from .embedding_registry import create_llamaindex_model, get_dense_dim
from .ingestion_tracker import IngestionTracker

_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current_dir)

DATA_DIR = os.path.join(_rag_qa_path, "data")
MODEL_DIR = os.path.join(_rag_qa_path, "models")
USE_CUDA = False

conf = Config()

# 原始文档加载器映射（保持不变）
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".jpeg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader if UnstructuredMarkdownLoader is not None else TextLoader,
}


class LlamaIndexProcessor:
    """
    混合模式处理器：
    - load_documents: 使用原始 OCR 加载器
    - process_documents: 使用原始 ChineseRecursiveTextSplitter + MarkdownTextSplitter
    - incremental_process_and_index: 使用 LlamaIndex 索引 + SQLite 增量追踪
    """

    def __init__(self, tracker_db_path: str | None = None):
        self.logger = logger
        self.storage_dir = os.path.join(DATA_DIR, "llamaindex_storage")
        os.makedirs(self.storage_dir, exist_ok=True)

        if tracker_db_path is None:
            tracker_db_path = os.path.join(DATA_DIR, "ingestion_tracker.db")
        self.tracker = IngestionTracker(tracker_db_path)

        self._init_embedding()
        self._init_vector_store()
        self._init_index()

    def _init_embedding(self):
        """初始化嵌入模型（通过注册表支持多模型切换）"""
        model_name = conf.EMBEDDING_MODEL
        model_path = os.path.join(MODEL_DIR, model_name)
        self.embed_model = create_llamaindex_model(
            model_name,
            model_path=model_path,
            device="cuda" if USE_CUDA else "cpu",
        )
        self.logger.info(f"嵌入模型初始化完成: {model_path} (model={model_name})")

    def _init_vector_store(self):
        """初始化 Milvus 向量存储（字段名与 VectorStore 对齐，共享同一集合）。"""
        from .embedding_registry import create_sparse_embedding_function

        model_name = conf.EMBEDDING_MODEL
        model_path = os.path.join(MODEL_DIR, model_name)
        device = "cuda" if USE_CUDA else "cpu"

        self.sparse_embed_fn = create_sparse_embedding_function(
            model_name, model_path=model_path, device=device
        )

        self.llamaindex_collection = conf.MILVUS_COLLECTION_NAME
        self.vector_store = MilvusVectorStore(
            uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
            collection_name=self.llamaindex_collection,
            db_name=conf.MILVUS_DATABASE_NAME,
            dim=get_dense_dim(model_name),
            embedding_field="dense_vector",
            sparse_embedding_field="sparse_vector",
            text_key="text",
            enable_sparse=self.sparse_embed_fn is not None,
            sparse_embedding_function=self.sparse_embed_fn,
            overwrite=False,
            hybrid_ranker="WeightedRanker",
            hybrid_ranker_params={"weights": [1.0, 0.7]},
        )
        self.logger.info(f"Milvus 向量存储初始化完成: {self.llamaindex_collection}")

    def _init_index(self):
        """初始化或加载 LlamaIndex 索引"""
        try:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store, persist_dir=self.storage_dir)
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
            self.logger.info("从存储加载索引成功")
        except Exception as e:
            self.logger.warning(f"加载索引失败，创建新索引: {e}")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_documents(
                [], storage_context=storage_context, embed_model=self.embed_model
            )

    def _load_single_file(self, file_path, source):
        """加载单个文件：dispatch loader → 清洗文本 → 设置元数据 → 质量评估"""
        file_extension = os.path.splitext(file_path)[1].lower()
        loader_class = document_loaders[file_extension]
        loader = loader_class(file_path, encoding="utf-8") if file_extension == ".txt" else loader_class(file_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.page_content = clean_document_text(doc.page_content)
            doc.metadata["source"] = source
            doc.metadata["file_path"] = file_path
            doc.metadata["timestamp"] = datetime.now().isoformat()
            estimate_document_quality(doc)
        return loaded_docs

    def _load_files(self, file_paths: list[str]) -> list:
        """核心文件加载逻辑：遍历路径列表 → 过滤扩展名 → OCR加载 → 清洗 → 质量评估。

        source 元数据按每个文件的父目录名独立推导（修复旧 _load_selected_files
        只取首个文件 source 的 bug）。
        """
        documents = []
        supported_extensions = document_loaders.keys()

        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension not in supported_extensions:
                self.logger.warning(f"不支持的文件类型: {file_path}")
                continue

            parent_dir = os.path.basename(os.path.dirname(file_path))
            source = parent_dir.replace("_data", "")

            try:
                documents.extend(self._load_single_file(file_path, source))
                self.logger.info(f"成功加载文件: {file_path}")
            except Exception as e:
                self.logger.error(f"加载文件 {file_path} 失败: {str(e)}")

        return documents

    def load_documents(self, directory_path):
        """使用原始 OCR 加载器加载目录下所有文档"""
        supported_extensions = document_loaders.keys()
        file_paths = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension not in supported_extensions:
                    self.logger.warning(f"不支持的文件类型: {file_path}")
                    continue
                file_paths.append(file_path)

        return self._load_files(file_paths)

    def process_documents(self, directory_path, parent_chunk_size=None, child_chunk_size=None, chunk_overlap=None):
        """使用原始切分器进行两级切分（保持与原有代码完全一致）"""
        parent_chunk_size = parent_chunk_size or conf.PARENT_CHUNK_SIZE
        child_chunk_size = child_chunk_size or conf.CHILD_CHUNK_SIZE
        chunk_overlap = chunk_overlap or conf.CHUNK_OVERLAP

        documents = self.load_documents(directory_path)
        self.logger.info(f"加载的文档数量: {len(documents)}")

        child_chunks = self._split_documents(documents, parent_chunk_size, child_chunk_size, chunk_overlap)
        self.logger.info(f"子块数量: {len(child_chunks)}")
        return child_chunks

    def _split_documents(self, documents, parent_chunk_size, child_chunk_size, chunk_overlap):
        """两级切分：父块→子块，支持自适应策略选择。

        策略来源: ChunkConfigManager 单例（支持运行时热更新）。
        Markdown 文件始终使用 MarkdownTextSplitter 以保持结构感知能力。
        语义策略（AliTextSplitter）只作用于 parent 级别；child 级别始终递归切分。
        """
        config_mgr = ChunkConfigManager()
        cfg = config_mgr.get_config()
        semantic_model_path = cfg.get("semantic_model_path") or None

        child_chunks = []
        for i, doc in enumerate(documents):
            file_path = doc.metadata.get("file_path", "")
            file_extension = os.path.splitext(file_path)[1].lower()
            is_markdown = file_extension == ".md"

            parent_strategy = MARKDOWN if is_markdown else config_mgr.get_strategy(file_extension)
            child_strategy = MARKDOWN if is_markdown else parent_strategy

            try:
                parent_splitter = create_parent_splitter(
                    parent_strategy,
                    parent_chunk_size,
                    chunk_overlap,
                    semantic_model_path=semantic_model_path,
                )
            except Exception:
                fallback = cfg.get("semantic_fallback_strategy", RECURSIVE)
                self.logger.warning("语义切分失败，回退到 %s 策略: %s", fallback, file_path)
                parent_strategy = fallback
                parent_splitter = create_parent_splitter(
                    parent_strategy,
                    parent_chunk_size,
                    chunk_overlap,
                )

            child_splitter = create_child_splitter(
                child_strategy,
                child_chunk_size,
                chunk_overlap,
            )

            self.logger.info("处理文档: %s, 策略: %s", file_path, parent_strategy)

            parent_docs = parent_splitter.split_documents([doc])

            for j, parent_doc in enumerate(parent_docs):
                parent_id = f"doc_{i}_parent_{j}"
                sub_chunks = child_splitter.split_documents([parent_doc])

                for k, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata["parent_id"] = parent_id
                    sub_chunk.metadata["parent_content"] = parent_doc.page_content
                    sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                    child_chunks.append(sub_chunk)

        return child_chunks

    def _load_selected_files(self, file_paths: list[str]) -> list:
        """只加载指定文件列表（复用 _load_files 核心逻辑）"""
        return self._load_files(file_paths)

    def incremental_process_and_index(
        self,
        directory_path: str,
        parent_chunk_size: int | None = None,
        child_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, int]:
        """增量处理目录并将文档添加到索引。

        1. 扫描目录，哈希对比 SQLite → 分类为 new/modified/unchanged/deleted
        2. DELETED → 从 Milvus 删除, MODIFIED → 先删旧块
        3. 只对 NEW + MODIFIED 文件做 OCR + 切分
        4. 用 ref_doc_id 将 chunks 关联到源文件，batch insert
        5. 更新 SQLite 记录

        Returns:
            dict: {"new", "modified", "deleted", "unchanged", "total_chunks"}
        """
        self.logger.info(f"开始增量处理: {directory_path}")

        # Step 1: 扫描目录
        scan_result = self.tracker.scan_directory(directory_path)

        new_count = len(scan_result["new"])
        modified_count = len(scan_result["modified"])
        deleted_count = len(scan_result["deleted"])
        unchanged_count = len(scan_result["unchanged"])

        self.logger.info(
            f"扫描: {new_count} 新增, {modified_count} 修改, {deleted_count} 已删除, {unchanged_count} 未变"
        )

        # Step 2: 处理 DELETED 文件
        for entry in scan_result["deleted"]:
            doc_id = entry["doc_id"]
            try:
                self.index.delete_ref_doc(doc_id)
                self.tracker.mark_deleted(doc_id)
                self.logger.info(f"从索引删除: {entry['file_path']}")
            except Exception as e:
                self.logger.error(f"删除失败 {entry['file_path']}: {e}")

        # Step 3: 处理 MODIFIED 文件 — 先清除旧块
        for entry in scan_result["modified"]:
            doc_id = entry["doc_id"]
            try:
                self.index.delete_ref_doc(doc_id)
                self.logger.info(f"清除旧块: {entry['file_path']}")
            except Exception as e:
                self.logger.error(f"清除旧块失败 {entry['file_path']}: {e}")

        # Step 4: 只加载 NEW + MODIFIED 文件
        files_to_process = scan_result["new"] + scan_result["modified"]

        if not files_to_process:
            self.logger.info("无文件需要处理，跳过。")
            self.index.storage_context.persist(persist_dir=self.storage_dir)
            return {
                "new": 0,
                "modified": 0,
                "deleted": deleted_count,
                "unchanged": unchanged_count,
                "total_chunks": 0,
            }

        parent_chunk_size = parent_chunk_size or conf.PARENT_CHUNK_SIZE
        child_chunk_size = child_chunk_size or conf.CHILD_CHUNK_SIZE
        chunk_overlap = chunk_overlap or conf.CHUNK_OVERLAP

        # Step 4a: 加载文档
        loaded_docs = self._load_selected_files([e["file_path"] for e in files_to_process])
        self.logger.info(f"加载了 {len(loaded_docs)} 个文档，来自 {len(files_to_process)} 个文件")

        # Step 4b: 切分
        child_chunks = self._split_documents(loaded_docs, parent_chunk_size, child_chunk_size, chunk_overlap)
        self.logger.info(f"生成了 {len(child_chunks)} 个子块")

        # Step 5: 按源文件分组，设置 ref_doc_id，批量插入
        chunks_by_file: dict[str, list] = {}
        for chunk in child_chunks:
            fp = chunk.metadata.get("file_path", "")
            if fp not in chunks_by_file:
                chunks_by_file[fp] = []
            chunks_by_file[fp].append(chunk)

        total_chunks = 0
        for entry in files_to_process:
            file_path = entry["file_path"]
            doc_id = entry["doc_id"]
            file_chunks = chunks_by_file.get(file_path, [])

            if not file_chunks:
                self.logger.warning(f"文件无块: {file_path}")
                continue

            nodes = []
            for idx, chunk in enumerate(file_chunks):
                node = TextNode(
                    text=chunk.page_content,
                    metadata={**chunk.metadata, "source_doc_id": doc_id},
                    id_=f"{doc_id}_chunk_{idx}",
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
                nodes.append(node)

            try:
                self.index.insert_nodes(nodes)
                self.tracker.mark_ingested(
                    file_path=file_path,
                    content_hash=entry["content_hash"],
                    doc_id=doc_id,
                    file_size=entry.get("file_size", 0),
                    file_mtime=entry.get("file_mtime", 0.0),
                    chunk_count=len(nodes),
                )
                total_chunks += len(nodes)
                self.logger.info(f"插入 {len(nodes)} 个块: {file_path}")
            except Exception as e:
                self.logger.error(f"插入失败 {file_path}: {e}")

        # Persist
        self.index.storage_context.persist(persist_dir=self.storage_dir)

        self.logger.info(
            f"增量处理完成. "
            f"新增: {new_count}, 修改: {modified_count}, "
            f"删除: {deleted_count}, 未变: {unchanged_count}, "
            f"总块数: {total_chunks}"
        )

        return {
            "new": new_count,
            "modified": modified_count,
            "deleted": deleted_count,
            "unchanged": unchanged_count,
            "total_chunks": total_chunks,
        }


# 保持与原有 API 兼容
def load_documents_from_directory(directory_path):
    """兼容原有函数名"""
    processor = LlamaIndexProcessor()
    return processor.load_documents(directory_path)


def process_documents(directory_path, parent_chunk_size=None, child_chunk_size=None, chunk_overlap=None):
    """兼容原有函数签名"""
    processor = LlamaIndexProcessor()
    return processor.process_documents(
        directory_path,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )


def incremental_process_and_index(
    directory_path,
    parent_chunk_size=None,
    child_chunk_size=None,
    chunk_overlap=None,
):
    """便捷函数：创建处理器并运行增量管线"""
    processor = LlamaIndexProcessor()
    return processor.incremental_process_and_index(
        directory_path,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )


def discover_data_dirs(base_dir: str | None = None) -> list[str]:
    """自动发现 base_dir 下所有 *_data 子目录，按名称排序返回绝对路径列表。"""
    if base_dir is None:
        base_dir = DATA_DIR
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.endswith("_data"):
            dirs.append(full)
    return dirs


def batch_process_directories(
    directories: list[str],
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, dict]:
    """用单个 LlamaIndexProcessor 实例批量处理多个学科目录。

    避免 shell 循环中反复初始化嵌入模型和 Milvus 连接，一次加载、顺序处理。
    返回 {目录名: 处理结果} 的字典。
    """
    if not directories:
        print("没有需要处理的目录。")
        return {}

    processor = LlamaIndexProcessor()
    results: dict[str, dict] = {}

    for i, dir_path in enumerate(directories, 1):
        abs_path = os.path.abspath(dir_path)
        subject = os.path.basename(abs_path).replace("_data", "")
        print(f"\n{'='*60}")
        print(f"[{i}/{len(directories)}] 开始处理学科: {subject}")
        print(f"     目录: {abs_path}")
        print(f"{'='*60}")

        if not os.path.isdir(abs_path):
            print(f"  [跳过] 目录不存在: {abs_path}")
            continue

        try:
            result = processor.incremental_process_and_index(
                abs_path,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results[subject] = result
            print(f"  [完成] {subject}: {result}")
        except Exception as e:
            print(f"  [失败] {subject}: {e}")
            results[subject] = {"error": str(e)}

    # 汇总
    print(f"\n{'='*60}")
    print("批量处理汇总")
    print(f"{'='*60}")
    total_new = total_modified = total_deleted = total_chunks = 0
    for subject, r in results.items():
        if "error" in r:
            print(f"  {subject}: ❌ {r['error']}")
        else:
            print(
                f"  {subject}: "
                f"新增={r.get('new', 0)}, "
                f"修改={r.get('modified', 0)}, "
                f"删除={r.get('deleted', 0)}, "
                f"未变={r.get('unchanged', 0)}, "
                f"块数={r.get('total_chunks', 0)}"
            )
            total_new += r.get("new", 0)
            total_modified += r.get("modified", 0)
            total_deleted += r.get("deleted", 0)
            total_chunks += r.get("total_chunks", 0)
    print(
        f"  合计: "
        f"新增={total_new}, 修改={total_modified}, "
        f"删除={total_deleted}, 总块数={total_chunks}"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LlamaIndex 增量文档处理 — OCR加载 → 切分 → 嵌入 → 写入 Milvus",
    )
    parser.add_argument(
        "directories",
        nargs="*",
        help="文档目录路径（可指定多个，空格分隔；不指定则默认处理 rag_qa/data/ai_data）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="自动发现 rag_qa/data/ 下所有 *_data 目录并批量处理",
    )
    parser.add_argument("--parent-chunk-size", type=int, default=None, help="父块大小（默认使用 config.ini 配置）")
    parser.add_argument("--child-chunk-size", type=int, default=None, help="子块大小（默认使用 config.ini 配置）")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="块重叠大小（默认使用 config.ini 配置）")
    args = parser.parse_args()

    # 确定要处理的目录列表
    if args.all:
        directories = discover_data_dirs()
        if not directories:
            print(f"未在 {DATA_DIR} 下找到任何 *_data 目录。")
            exit(1)
        print(f"自动发现 {len(directories)} 个学科目录: {[os.path.basename(d) for d in directories]}")
    elif args.directories:
        directories = args.directories
    else:
        # 默认：单目录兼容旧行为
        directories = [os.path.join(DATA_DIR, "ai_data")]

    # 校验目录存在性（提前报错，避免跑到一半才发现）
    valid_dirs = []
    for d in directories:
        abs_d = os.path.abspath(d)
        if os.path.isdir(abs_d):
            valid_dirs.append(abs_d)
        else:
            print(f"警告: 目录不存在，跳过 — {abs_d}")
    if not valid_dirs:
        print("错误: 没有有效的目录可供处理。")
        exit(1)

    results = batch_process_directories(
        valid_dirs,
        parent_chunk_size=args.parent_chunk_size,
        child_chunk_size=args.child_chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # 检查是否有失败的
    failures = {k: v for k, v in results.items() if "error" in v}
    if failures:
        print(f"\n{failures}")
        exit(1)
    print("\n全部处理完成。")
