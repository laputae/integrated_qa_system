"""Chunking strategy factory — returns the correct LangChain-compatible text splitter
for a given strategy name."""

from typing import Optional

from langchain_text_splitters import MarkdownTextSplitter, TextSplitter

from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import (
    ChineseRecursiveTextSplitter,
)
from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter

SEMANTIC = "semantic"
RECURSIVE = "recursive"
MARKDOWN = "markdown"

VALID_STRATEGIES = {SEMANTIC, RECURSIVE, MARKDOWN}


def create_parent_splitter(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    semantic_model_path: Optional[str] = None,
) -> TextSplitter:
    """Create a parent-level text splitter for the given strategy."""
    if strategy == SEMANTIC:
        return AliTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_path=semantic_model_path,
        )
    elif strategy == MARKDOWN:
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:  # RECURSIVE (default)
        return ChineseRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def create_child_splitter(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> TextSplitter:
    """Create a child-level text splitter.

    Child splitting always uses recursive splitting when the parent strategy is
    'recursive' or 'semantic' — semantic segmentation only operates at the parent
    (coarse) level. Markdown files keep their structure-aware splitter at both levels.
    """
    if strategy == MARKDOWN:
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
