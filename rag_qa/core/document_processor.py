"""
文档处理器 — 公开 API 外观层

统一从 llamaindex_processor 导出所有管线函数，保持向后兼容的同时
提供单一导入入口。
"""

import os

from .llamaindex_processor import (
    LlamaIndexProcessor,
    batch_process_directories,
    discover_data_dirs,
    incremental_process_and_index,
    load_documents_from_directory,
    process_documents,
)

# ---- 所有公开符号均直接从 llamaindex_processor 导出 ----
__all__ = [
    "load_documents_from_directory",
    "process_documents",
    "incremental_process_and_index",
    "batch_process_directories",
    "discover_data_dirs",
    "LlamaIndexProcessor",
    "validate_document_format",
]


def validate_document_format(documents: list) -> dict:
    """验证文档格式是否符合预期，自动识别原始文档/切分后文档。

    Args:
        documents: 文档列表（原始加载或切分后的均可）

    Returns:
        dict: {"valid": bool, "doc_type": "raw"|"chunked"|"unknown",
               "count": int, "issues": list[str]}
    """
    issues: list[str] = []

    if not documents:
        issues.append("文档列表为空")
        return {"valid": False, "doc_type": "unknown", "count": 0, "issues": issues}

    first_doc = documents[0]

    # 检查基本属性
    if not hasattr(first_doc, "page_content"):
        issues.append("缺少 page_content 属性")
        return {"valid": False, "doc_type": "unknown", "count": len(documents), "issues": issues}
    if not hasattr(first_doc, "metadata") or not isinstance(first_doc.metadata, dict):
        issues.append("metadata 缺失或不是字典类型")
        return {"valid": False, "doc_type": "unknown", "count": len(documents), "issues": issues}

    metadata = first_doc.metadata

    # 自动识别文档类型：切分后的文档有 parent_id 字段
    chunk_fields = {"id", "parent_id", "parent_content"}
    is_chunked = chunk_fields.issubset(metadata.keys())
    doc_type = "chunked" if is_chunked else "raw"

    # 按文档类型选择必要的元数据字段
    raw_required = {"source", "file_path"}
    chunk_required = {"source", "file_path", "timestamp", "id", "parent_id", "parent_content"}

    required = chunk_required if is_chunked else raw_required
    missing = [f for f in required if f not in metadata]
    if missing:
        issues.append(f"缺少元数据字段 ({doc_type} 模式): {missing}")
    else:
        issues.append(f"元数据字段完整 ({doc_type} 模式)")

    # 内容非空检查
    if first_doc.page_content.strip():
        issues.append("文档内容非空")
    else:
        issues.append("文档内容为空")

    # ID 唯一性检查（仅切分后文档有 id 字段）
    if is_chunked:
        doc_ids = [doc.metadata.get("id") for doc in documents if hasattr(doc, "metadata")]
        unique_ids = set(doc_ids)
        if len(doc_ids) == len(unique_ids):
            issues.append("文档ID唯一")
        else:
            issues.append("存在重复的文档ID")

    valid = not any("缺少" in i or "缺失" in i for i in issues)
    return {"valid": valid, "doc_type": doc_type, "count": len(documents), "issues": issues}


# 保持原有入口（如果有脚本直接运行）
if __name__ == "__main__":
    print("🚀 文档处理器验证测试")
    print("=" * 50)

    # 1. 加载和处理文档
    print("\n📄 开始处理文档...")
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _rag_qa_path = os.path.dirname(_current_dir)
    directory_path = os.path.join(_rag_qa_path, "data", "ai_data")

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

        # 2. 格式验证
        print("\n--- 格式验证 (原始文档) ---")
        raw_result = validate_document_format(docs)
        for issue in raw_result["issues"]:
            print(f"  {'✓' if raw_result['valid'] else '⚠️'} {issue}")

        print("\n--- 格式验证 (切分后文档) ---")
        chunked_result = validate_document_format(child_chunks)
        for issue in chunked_result["issues"]:
            print(f"  {'✓' if chunked_result['valid'] else '⚠️'} {issue}")

        # 3. 打印第一个文档示例
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
