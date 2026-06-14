# -*-coding:utf-8-*-
"""
LlamaIndex 文档处理器 - 混合模式
- 文档加载：保留原始 OCR 加载器（OCRPDFLoader/OCRDOCLoader/OCRPPTLoader/OCRIMGLoader）
- 文本切分：保留原始 ChineseRecursiveTextSplitter + MarkdownTextSplitter
- 索引构建：使用 LlamaIndex VectorStoreIndex 实现增量更新
"""
import os
import sys
import torch
from datetime import datetime

# 路径推导
_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current_dir)
sys.path.insert(0, _rag_qa_path)
_project_root = os.path.dirname(_rag_qa_path)
sys.path.insert(0, _project_root)

DATA_DIR = os.path.join(_rag_qa_path, 'data')
MODEL_DIR = os.path.join(_rag_qa_path, 'models')
USE_CUDA = torch.cuda.is_available()

# 原始加载器（保留 OCR 能力）
from langchain_community.document_loaders import TextLoader
try:
    from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
except ImportError:
    UnstructuredMarkdownLoader = None
from edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader

# 原始切分器（保留中文递归切分 + Markdown 支持）
from edu_text_spliter import ChineseRecursiveTextSplitter
from langchain_text_splitters import MarkdownTextSplitter

# LlamaIndex 核心（仅用于索引构建）
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document as LlamaDocument
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.documents import Document as LangchainDocument
from base import logger, Config

conf = Config()

# 原始文档加载器映射（保持不变）
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader if UnstructuredMarkdownLoader is not None else TextLoader
}
class LlamaIndexProcessor:
    """
    混合模式处理器：
    - load_documents: 使用原始 OCR 加载器
    - process_documents: 使用原始 ChineseRecursiveTextSplitter + MarkdownTextSplitter
    - add_documents: 使用 LlamaIndex 索引（支持增量更新）
    """

    def __init__(self):
        self.logger = logger
        self.storage_dir = os.path.join(DATA_DIR, "llamaindex_storage")
        os.makedirs(self.storage_dir, exist_ok=True)
        self._init_embedding()
        self._init_vector_store()
        self._init_index()

    def _init_embedding(self):
        """初始化嵌入模型"""
        model_path = os.path.join(MODEL_DIR, "bge-m3")
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            device="cuda" if USE_CUDA else "cpu"
        )
        self.logger.info(f"嵌入模型初始化完成: {model_path}")

    def _init_vector_store(self):
        """初始化 Milvus 向量存储"""
        self.vector_store = MilvusVectorStore(
            uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
            collection_name=conf.MILVUS_COLLECTION_NAME,
            db_name=conf.MILVUS_DATABASE_NAME,
            overwrite=False
        )
        self.logger.info("Milvus 向量存储初始化完成")

    def _init_index(self):
        """初始化或加载 LlamaIndex 索引"""
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=self.storage_dir
            )
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.embed_model
            )
            self.logger.info("从存储加载索引成功")
        except Exception as e:
            self.logger.warning(f"加载索引失败，创建新索引: {e}")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context,
                embed_model=self.embed_model
            )

    def load_documents(self, directory_path):
        """
        使用原始 OCR 加载器加载文档（保持与原有代码完全一致）
        """
        documents = []
        supported_extensions = document_loaders.keys()
        source = os.path.basename(directory_path).replace("_data", "")

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension in supported_extensions:
                    try:
                        loader_class = document_loaders[file_extension]
                        if file_extension == ".txt":
                            loader = loader_class(file_path, encoding="utf-8")
                        else:
                            loader = loader_class(file_path)
                        loaded_docs = loader.load()

                        for doc in loaded_docs:
                            doc.metadata["source"] = source
                            doc.metadata["file_path"] = file_path
                            doc.metadata["timestamp"] = datetime.now().isoformat()

                        documents.extend(loaded_docs)
                        self.logger.info(f"成功加载文件: {file_path}")
                    except Exception as e:
                        self.logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                else:
                    self.logger.warning(f"不支持的文件类型: {file_path}")

        return documents

    def process_documents(self, directory_path, parent_chunk_size=None,
                          child_chunk_size=None, chunk_overlap=None):
        """
        使用原始切分器进行两级切分（保持与原有代码完全一致）
        """
        parent_chunk_size = parent_chunk_size or conf.PARENT_CHUNK_SIZE
        child_chunk_size = child_chunk_size or conf.CHILD_CHUNK_SIZE
        chunk_overlap = chunk_overlap or conf.CHUNK_OVERLAP

        documents = self.load_documents(directory_path)
        self.logger.info(f"加载的文档数量: {len(documents)}")

        # 初始化原始切分器
        parent_splitter = ChineseRecursiveTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
        )
        child_splitter = ChineseRecursiveTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
        )
        markdown_parent_splitter = MarkdownTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
        )
        markdown_child_splitter = MarkdownTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
        )

        child_chunks = []
        for i, doc in enumerate(documents):
            file_extension = os.path.splitext(
                doc.metadata.get("file_path", '')
            )[1].lower()
            is_markdown = (file_extension == '.md')

            parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
            child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter

            self.logger.info(
                f"处理文档: {doc.metadata['file_path']}, "
                f"使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}"
            )

            parent_docs = parent_splitter_to_use.split_documents([doc])

            for j, parent_doc in enumerate(parent_docs):
                parent_id = f"doc_{i}_parent_{j}"
                sub_chunks = child_splitter_to_use.split_documents([parent_doc])

                for k, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata["parent_id"] = parent_id
                    sub_chunk.metadata["parent_content"] = parent_doc.page_content
                    sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                    child_chunks.append(sub_chunk)

        self.logger.info(f"子块数量: {len(child_chunks)}")
        return child_chunks

    def add_documents(self, documents):
        """
        使用 LlamaIndex 索引添加文档（支持增量更新）
        documents: list[langchain_core.documents.Document]
        """
        llama_docs = [
            LlamaDocument(text=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]

        for doc in llama_docs:
            self.index.insert(doc)

        self.index.storage_context.persist(persist_dir=self.storage_dir)
        self.logger.info(f"成功添加 {len(documents)} 个文档到索引")

    def query(self, query_str, k=5):
        """查询索引"""
        query_engine = self.index.as_query_engine(similarity_top_k=k)
        return query_engine.query(query_str)


# 保持与原有 API 兼容
def load_documents_from_directory(directory_path):
    """兼容原有函数名"""
    processor = LlamaIndexProcessor()
    return processor.load_documents(directory_path)


def process_documents(directory_path, parent_chunk_size=None,
                     child_chunk_size=None, chunk_overlap=None):
    """兼容原有函数签名"""
    processor = LlamaIndexProcessor()
    return processor.process_documents(
        directory_path,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap
    )


if __name__ == "__main__":
    processor = LlamaIndexProcessor()
    directory_path = os.path.join(DATA_DIR, "ai_data")

    # 处理文档
    docs = processor.process_documents(directory_path)
    print(f"处理了 {len(docs)} 个子块")

    # 添加到索引
    processor.add_documents(docs)

    # 查询
    response = processor.query("AI学科的课程内容是什么")
    print(response)