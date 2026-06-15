# -*-coding:utf-8-*-
"""
LlamaIndex 文档处理器 - 混合模式
- 文档加载：保留原始 OCR 加载器（OCRPDFLoader/OCRDOCLoader/OCRPPTLoader/OCRIMGLoader）
- 文本切分：保留原始 ChineseRecursiveTextSplitter + MarkdownTextSplitter
- 索引构建：使用 LlamaIndex VectorStoreIndex 实现增量更新
"""
import os
import re
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
def clean_document_text(text: str) -> str:
    """OCR文本预处理管道：去除噪音、规范化空白、统一标点"""
    if not text:
        return text

    # 1. 去除零宽字符（软连字符、各种空格、BOM、方向标记等不可见字符）
    text = re.sub(
        r'[\u00ad\u2000-\u200f\u2028\u2029\u205f\u2060-\u2064\u3000\ufeff\ufffe\uffff]',
        '', text
    )

    # 2. 规范化换行 → 单 \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 3. 统一中英文标点
    text = text.replace('，', ',')
    text = text.replace('；', ';')
    text = text.replace('：', ':')
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('【', '[').replace('】', ']')
    text = text.replace('《', '<').replace('》', '>')
    text = text.replace('！', '!')
    text = text.replace('？', '?')
    text = text.replace('～', '~')

    # 4. 去除页码/页眉/页脚噪音
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\(（]?\d{1,4}[\)）]?\s*$', '', text, flags=re.MULTILINE)

    # 5. 压缩多余空白
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)

    # 6. 清理连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# ---- 文档质量评估 ----

_CJK_START = 0x4E00        # CJK统一表意文字起始
_CJK_END = 0x9FFF          # CJK统一表意文字结尾
_CJK_EXT_A_START = 0x3400  # CJK扩展A起始
_CJK_EXT_A_END = 0x4DBF    # CJK扩展A结尾
_STANDARD_PUNCT = set(',.;:!?"\'()[]{}<>-+/\\| \t\n\r@#$%^&*~`=')

LOW_QUALITY_THRESHOLD = 0.3


def _is_content_char(c: str) -> bool:
    """是否为内容字符（中文、拉丁字母、数字）"""
    cp = ord(c)
    if _CJK_START <= cp <= _CJK_END:
        return True
    if _CJK_EXT_A_START <= cp <= _CJK_EXT_A_END:
        return True
    return c.isascii() and (c.isalpha() or c.isdigit())


def estimate_document_quality(doc) -> float:
    """评估 OCR 文档质量，返回 0-1 分数。

    基于已清洗文本（clean_document_text 之后）评估：
      - 文本长度充足度（权重 0.30）
      - 有效字符占比（权重 0.40）
      - OCR 噪音伪影（权重 0.30）

    同时设置 doc.metadata["quality_score"] 和 doc.metadata["is_low_quality"]。
    """
    text = doc.page_content
    if not text or not text.strip():
        doc.metadata["quality_score"] = 0.0
        doc.metadata["is_low_quality"] = True
        return 0.0

    total = len(text)

    # 1. 文本长度分数（权重 0.30）
    if total < 50:
        length_score = 0.0
    elif total < 200:
        length_score = (total - 50) / 150 * 0.5
    elif total < 500:
        length_score = 0.5 + (total - 200) / 300 * 0.35
    else:
        length_score = 1.0

    # 2. 有效字符占比（权重 0.40）
    content_chars = sum(1 for c in text if _is_content_char(c))
    content_ratio = content_chars / total

    # 3. OCR 噪音分数（权重 0.30）
    # 3a. 连续重复字符（同一字符连续 6 次以上）
    repeat_count = len(re.findall(r'(.)\1{5,}', text))
    repeat_penalty = min(repeat_count * 0.1, 0.30)

    # 3b. 非标准字符惩罚
    non_standard = sum(
        1 for c in text
        if not _is_content_char(c) and c not in _STANDARD_PUNCT and not c.isspace()
    )
    ns_ratio = non_standard / total
    ns_penalty = min(ns_ratio * 2.0, 0.40)

    # 3c. 行结构一致性
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len < 15:
            line_penalty = (15 - avg_line_len) / 15 * 0.30
        elif avg_line_len > 300:
            line_penalty = min((avg_line_len - 300) / 300, 1.0) * 0.30
        else:
            line_penalty = 0.0
    else:
        line_penalty = 0.30

    noise_penalty = min(repeat_penalty + ns_penalty + line_penalty, 0.80)
    noise_score = 1.0 - noise_penalty

    # 综合评分
    quality = (0.30 * length_score +
               0.40 * content_ratio +
               0.30 * noise_score)
    quality = max(0.0, min(1.0, quality))
    # 几乎没有有效内容时硬封顶
    if content_ratio < 0.1:
        quality = min(quality, 0.15)

    doc.metadata["quality_score"] = round(quality, 4)
    doc.metadata["is_low_quality"] = quality < LOW_QUALITY_THRESHOLD

    return quality


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
                            doc.page_content = clean_document_text(doc.page_content)
                            doc.metadata["source"] = source
                            doc.metadata["file_path"] = file_path
                            doc.metadata["timestamp"] = datetime.now().isoformat()
                            estimate_document_quality(doc)

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
    # ---- 质量评估冒烟测试 ----
    print("=" * 50)
    print("estimate_document_quality 冒烟测试")
    print("=" * 50)

    test_cases = [
        ("", "空文本"),
        ("   \n\n  ", "仅空白"),
        ("机器学习概述 监督学习 无监督学习", "短中文"),
        ("!" * 100, "全是标点"),
        ("人工智能" * 200, "长干净中文（800字）"),
        ("Hello World! This is a test document with some English text.", "短英文"),
        ("机器学习!!概述学!!习!!", "混合噪音中文"),
    ]

    for text, label in test_cases:
        doc = LangchainDocument(page_content=text, metadata={})
        score = estimate_document_quality(doc)
        print(f"[{label}] score={score:.4f}, is_low={doc.metadata['is_low_quality']}, "
              f"len={len(text)}")

    print()

    # ---- 完整流程 ----
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