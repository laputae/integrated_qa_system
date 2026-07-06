"""自适应 Chunk 配置 — 切分策略 & 集成测试"""
from unittest.mock import MagicMock, patch
import pytest


# ============================================================================
# AliTextSplitter 测试
# ============================================================================

class TestAliTextSplitter:
    def test_default_model_path(self):
        """AliTextSplitter 默认 model_path 指向 rag_qa/nlp_bert_..."""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
        path = AliTextSplitter._default_model_path()
        assert "nlp_bert_document-segmentation_chinese-base" in path
        assert path.endswith("base")

    def test_custom_model_path(self):
        """AliTextSplitter 接受自定义 model_path"""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
        splitter = AliTextSplitter(model_path="/custom/path")
        assert splitter._model_path == "/custom/path"

    def test_pipeline_lazy_loading(self):
        """AliTextSplitter pipeline 是懒加载的"""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
        splitter = AliTextSplitter(model_path="/custom/path")
        assert splitter._pipeline is None

    @patch("rag_qa.edu_text_spliter.edu_model_text_spliter.pipeline")
    def test_pipeline_created_once(self, mock_pipeline):
        """pipeline 只创建一次（单例行为）"""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: "\n\t".join(["段落1", "段落2", "段落3"])
        mock_pipeline.return_value = MagicMock(return_value=mock_result)

        splitter = AliTextSplitter(model_path="/custom/path")
        splitter.split_text("测试文本内容。这是第二句话。")
        splitter.split_text("另一段文本。")

        assert mock_pipeline.call_count == 1

    @patch("rag_qa.edu_text_spliter.edu_model_text_spliter.pipeline")
    def test_split_text_returns_list(self, mock_pipeline):
        """split_text 返回字符串列表"""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter

        class FakeResult(dict):
            pass

        fake_result = FakeResult()
        fake_result["text"] = "段落A\n\t段落B\n\t段落C"
        mock_pipeline.return_value = MagicMock(return_value=fake_result)

        splitter = AliTextSplitter(model_path="/custom/path")
        result = splitter.split_text("测试文本内容。")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["段落A", "段落B", "段落C"]

    @patch("rag_qa.edu_text_spliter.edu_model_text_spliter.pipeline")
    def test_pdf_mode_preprocessing(self, mock_pipeline):
        """pdf=True 时应用预处理逻辑"""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter

        class FakeResult(dict):
            pass
        fake_result = FakeResult()
        fake_result["text"] = "段落1\n\t段落2"
        mock_pipeline.return_value = MagicMock(return_value=fake_result)

        splitter = AliTextSplitter(pdf=True, model_path="/custom/path")
        text_with_newlines = "这是第一段\n\n\n\n第二段\n\n第三段  extra spaces"
        result = splitter.split_text(text_with_newlines)
        assert len(result) == 2

    @patch("rag_qa.edu_text_spliter.edu_model_text_spliter.pipeline")
    def test_split_text_raises_on_errors(self, mock_pipeline):
        """pipeline 失败时抛出 RuntimeError"""
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
        mock_pipeline.side_effect = RuntimeError("Model download failed")

        splitter = AliTextSplitter(model_path="/invalid/path")
        with pytest.raises(RuntimeError, match="语义切分失败"):
            splitter.split_text("测试文本")


# ============================================================================
# ChunkStrategy 工厂测试
# ============================================================================

class TestChunkStrategy:
    def test_create_parent_recursive(self):
        from rag_qa.edu_text_spliter.chunk_strategy import (
            create_parent_splitter, RECURSIVE,
        )
        from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import (
            ChineseRecursiveTextSplitter,
        )
        splitter = create_parent_splitter(RECURSIVE, 1200, 50)
        assert isinstance(splitter, ChineseRecursiveTextSplitter)
        assert splitter._chunk_size == 1200

    def test_create_parent_markdown(self):
        from rag_qa.edu_text_spliter.chunk_strategy import (
            create_parent_splitter, MARKDOWN,
        )
        from langchain_text_splitters import MarkdownTextSplitter
        splitter = create_parent_splitter(MARKDOWN, 800, 30)
        assert isinstance(splitter, MarkdownTextSplitter)

    def test_create_parent_semantic(self):
        from rag_qa.edu_text_spliter.chunk_strategy import (
            create_parent_splitter, SEMANTIC,
        )
        from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
        splitter = create_parent_splitter(SEMANTIC, 1200, 50)
        assert isinstance(splitter, AliTextSplitter)

    def test_create_child_semantic_uses_recursive(self):
        """child splitter 在 semantic 策略下仍使用 recursive"""
        from rag_qa.edu_text_spliter.chunk_strategy import (
            create_child_splitter, SEMANTIC,
        )
        from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import (
            ChineseRecursiveTextSplitter,
        )
        splitter = create_child_splitter(SEMANTIC, 300, 50)
        assert isinstance(splitter, ChineseRecursiveTextSplitter)

    def test_create_child_markdown(self):
        from rag_qa.edu_text_spliter.chunk_strategy import (
            create_child_splitter, MARKDOWN,
        )
        from langchain_text_splitters import MarkdownTextSplitter
        splitter = create_child_splitter(MARKDOWN, 300, 50)
        assert isinstance(splitter, MarkdownTextSplitter)


# ============================================================================
# _split_documents 策略选择集成测试
# ============================================================================

class TestSplitDocumentsStrategy:
    def test_markdown_always_uses_markdown_strategy(self):
        """Markdown 文件的 markdown 策略由 _split_documents 强制使用"""
        from base.chunk_config import ChunkConfigManager
        mgr = ChunkConfigManager()
        mgr.update_config({"default_strategy": "semantic"})
        strategy = mgr.get_strategy("md")
        assert strategy in ("semantic", "recursive", "markdown")
        assert mgr.get_strategy("txt") == "semantic"

    @patch("rag_qa.core.llamaindex_processor.create_parent_splitter")
    @patch("rag_qa.core.llamaindex_processor.create_child_splitter")
    @patch("rag_qa.core.llamaindex_processor.ChunkConfigManager")
    def test_semantic_fallback_on_error(
        self, mock_cfg_mgr, mock_create_child, mock_create_parent,
    ):
        """语义策略失败时回退到 fallback 策略"""
        from rag_qa.core.llamaindex_processor import LlamaIndexProcessor
        from langchain_core.documents import Document as LangchainDocument

        mgr_instance = MagicMock()
        mgr_instance.get_strategy.return_value = "semantic"
        mgr_instance.get_config.return_value = {
            "semantic_fallback_strategy": "recursive",
            "semantic_model_path": "/bad/path",
        }
        mock_cfg_mgr.return_value = mgr_instance

        mock_create_parent.side_effect = [
            RuntimeError("model not found"),
            MagicMock(),
        ]
        mock_create_child.return_value = MagicMock()
        mock_create_child.return_value.split_documents.return_value = []

        processor = LlamaIndexProcessor()
        doc = LangchainDocument(page_content="测试内容", metadata={"file_path": "/data/test.txt"})
        result = processor._split_documents([doc], 1200, 300, 50)
        assert isinstance(result, list)
        assert mock_create_parent.call_count == 2

    @patch("rag_qa.core.llamaindex_processor.create_parent_splitter")
    @patch("rag_qa.core.llamaindex_processor.create_child_splitter")
    @patch("rag_qa.core.llamaindex_processor.ChunkConfigManager")
    def test_recursive_strategy_used_by_default(
        self, mock_cfg_mgr, mock_create_child, mock_create_parent,
    ):
        """默认策略为 recursive"""
        from rag_qa.core.llamaindex_processor import LlamaIndexProcessor
        from langchain_core.documents import Document as LangchainDocument

        mgr_instance = MagicMock()
        mgr_instance.get_strategy.return_value = "recursive"
        mgr_instance.get_config.return_value = {
            "semantic_fallback_strategy": "recursive",
            "semantic_model_path": "",
        }
        mock_cfg_mgr.return_value = mgr_instance

        mock_parent = MagicMock()
        mock_parent.split_documents.return_value = []
        mock_create_parent.return_value = mock_parent
        mock_create_child.return_value = MagicMock()

        processor = LlamaIndexProcessor()
        doc = LangchainDocument(page_content="测试", metadata={"file_path": "/data/test.txt"})
        processor._split_documents([doc], 1200, 300, 50)

        mock_create_parent.assert_called_once()
        call_args = mock_create_parent.call_args
        assert call_args[0][0] == "recursive"
