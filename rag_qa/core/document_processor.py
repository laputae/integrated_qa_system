"""
文档处理器 - 使用 LlamaIndex 实现
保持与原有 API 兼容
"""
import os
from datetime import datetime
from typing import List

# 导入必要的类型检查
try:
    from langchain_core.documents import Document
except ImportError:
    Document = None

# 导入 LlamaIndex 处理器
from llamaindex_processor import (
    LlamaIndexProcessor,
    load_documents_from_directory as llamaindex_load,
    process_documents as llamaindex_process
)

# 保持原有全局变量（如果其他模块引用）
document_loaders = {
    ".txt": None,
    ".pdf": None,
    ".docx": None,
    ".ppt": None,
    ".pptx": None,
    ".jpg": None,
    ".png": None,
    ".md": None
}


def load_documents_from_directory(directory_path):
    """
    从目录加载文档（保持原有函数签名）

    Args:
        directory_path: 文档目录路径

    Returns:
        list[Document]: 加载的文档列表
    """
    return llamaindex_load(directory_path)


def process_documents(directory_path, parent_chunk_size=None,
                      child_chunk_size=None, chunk_overlap=None):
    """
    处理文档并进行分层切分（保持原有函数签名）

    Args:
        directory_path: 文档目录路径
        parent_chunk_size: 父块大小（可选，使用配置默认值）
        child_chunk_size: 子块大小（可选，使用配置默认值）
        chunk_overlap: 块重叠大小（可选，使用配置默认值）

    Returns:
        list[Document]: 子块文档列表
    """
    # 参数已在 LlamaIndexProcessor 中通过配置处理
    return llamaindex_process(directory_path)


def validate_document_format(documents: List) -> bool:
    """
    验证文档格式是否符合预期

    Args:
        documents: 文档列表

    Returns:
        bool: 验证是否通过
    """
    if not documents:
        print("[验证] ❌ 文档列表为空")
        return False

    # 检查返回类型
    if not isinstance(documents, list):
        print("[验证] ❌ 返回值不是列表类型")
        return False

    # 检查第一个文档
    first_doc = documents[0]

    # 检查文档类型（兼容 langchain Document）
    if Document and not isinstance(first_doc, Document):
        print(f"[验证] ⚠️ 文档类型不是 langchain Document，而是: {type(first_doc)}")
    else:
        print("[验证] ✓ 文档类型正确")

    # 检查必要的属性
    required_attrs = ['page_content', 'metadata']
    for attr in required_attrs:
        if not hasattr(first_doc, attr):
            print(f"[验证] ❌ 缺少必要属性: {attr}")
            return False
    print("[验证] ✓ 文档属性完整")

    # 检查元数据字段
    required_metadata = ['id', 'parent_id', 'parent_content', 'source', 'file_path', 'timestamp']
    missing_fields = []

    if hasattr(first_doc, 'metadata') and isinstance(first_doc.metadata, dict):
        for field in required_metadata:
            if field not in first_doc.metadata:
                missing_fields.append(field)

        if missing_fields:
            print(f"[验证] ⚠️ 缺少元数据字段: {missing_fields}")
        else:
            print("[验证] ✓ 元数据字段完整")
    else:
        print("[验证] ❌ 元数据不是字典类型")
        return False

    # 检查内容非空
    if first_doc.page_content.strip():
        print("[验证] ✓ 文档内容非空")
    else:
        print("[验证] ⚠️ 文档内容为空")

    # 检查 ID 唯一性
    doc_ids = [doc.metadata.get('id') for doc in documents if hasattr(doc, 'metadata')]
    unique_ids = set(doc_ids)
    if len(doc_ids) == len(unique_ids):
        print("[验证] ✓ 文档ID唯一")
    else:
        print("[验证] ⚠️ 存在重复的文档ID")

    print(f"\n[验证] 共处理 {len(documents)} 个文档")
    return True


def check_dependencies():
    """检查项目依赖是否完整"""
    print("\n" + "="*50)
    print("📦 依赖检查")
    print("="*50)

    dependencies = [
        ('llama_index.core', 'LlamaIndex 核心模块'),
        ('llama_index.vector_stores.milvus', 'Milvus 向量存储'),
        ('llama_index.embeddings.huggingface', 'HuggingFace 嵌入'),
        ('pymilvus', 'Milvus Python SDK'),
        ('langchain_core', 'LangChain 核心'),
        ('torch', 'PyTorch'),
    ]

    missing_deps = []
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✓ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            missing_deps.append(module)

    if missing_deps:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing_deps)}")
        print("请运行: uv sync")
        return False
    else:
        print("\n✅ 所有依赖检查通过")
        return True


# 保持原有入口（如果有脚本直接运行）
if __name__ == '__main__':
    print("🚀 文档处理器验证测试")
    print("="*50)

    # 1. 依赖检查
    if not check_dependencies():
        exit(1)

    # 2. 加载和处理文档
    print("\n📄 开始处理文档...")
    directory_path = 'D:\\PythonProjects\\integrated_qa_system\\rag_qa\\data\\ai_data'

    # 确保路径存在
    if not os.path.exists(directory_path):
        print(f"❌ 目录不存在: {directory_path}")
        exit(1)

    try:
        # 测试 load_documents_from_directory
        print("\n--- 测试 load_documents_from_directory ---")
        docs = load_documents_from_directory(directory_path)
        print(f"加载文档数量: {len(docs)}")

        # 测试 process_documents
        print("\n--- 测试 process_documents ---")
        child_chunks = process_documents(directory_path)
        print(f"生成子块数量: {len(child_chunks)}")

        # 3. 格式验证
        print("\n--- 格式验证 ---")
        validate_document_format(child_chunks)

        # 4. 打印第一个文档示例
        if child_chunks:
            print("\n--- 文档示例 ---")
            first_chunk = child_chunks[0]
            print(f"文档ID: {first_chunk.metadata.get('id', 'N/A')}")
            print(f"父块ID: {first_chunk.metadata.get('parent_id', 'N/A')}")
            print(f"来源: {first_chunk.metadata.get('source', 'N/A')}")
            print(f"文件路径: {first_chunk.metadata.get('file_path', 'N/A')}")
            print(f"时间戳: {first_chunk.metadata.get('timestamp', 'N/A')}")
            print(f"内容预览: {first_chunk.page_content[:100]}...")

        print("\n🎉 验证测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)