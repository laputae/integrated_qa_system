"""自适应 Chunk 配置 — ChunkConfigManager 单例测试"""

import threading


class TestChunkConfigManager:
    def test_singleton(self):
        """ChunkConfigManager 是线程安全的单例"""
        from base.chunk_config import ChunkConfigManager

        m1 = ChunkConfigManager()
        m2 = ChunkConfigManager()
        assert m1 is m2

    def test_singleton_thread_safety(self):
        """多线程并发获取实例返回同一个对象"""
        from base.chunk_config import ChunkConfigManager

        instances = []

        def get_instance():
            instances.append(ChunkConfigManager())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        for inst in instances[1:]:
            assert inst is first

    def test_get_config_returns_expected_keys(self):
        """get_config 返回预期的配置键"""
        from base.chunk_config import ChunkConfigManager

        mgr = ChunkConfigManager()
        cfg = mgr.get_config()
        expected_keys = {
            "default_strategy",
            "doc_type_strategies",
            "semantic_model_path",
            "semantic_device",
            "semantic_fallback_strategy",
            "parent_chunk_size",
            "child_chunk_size",
            "chunk_overlap",
        }
        assert set(cfg.keys()) == expected_keys
        assert cfg["default_strategy"] == "recursive"

    def test_get_strategy_with_mapping(self):
        """get_strategy 根据扩展名映射返回正确策略"""
        from base.chunk_config import ChunkConfigManager

        mgr = ChunkConfigManager()
        mgr.update_config({"doc_type_strategies": {"pdf": "semantic", "txt": "recursive"}})
        assert mgr.get_strategy("pdf") == "semantic"
        assert mgr.get_strategy("txt") == "recursive"
        assert mgr.get_strategy("docx") == "recursive"

    def test_get_strategy_strips_dot(self):
        """get_strategy 自动去除扩展名前导点"""
        from base.chunk_config import ChunkConfigManager

        mgr = ChunkConfigManager()
        mgr.update_config({"doc_type_strategies": {"md": "markdown"}})
        assert mgr.get_strategy(".md") == "markdown"
        assert mgr.get_strategy("md") == "markdown"

    def test_update_config_preserves_other_keys(self):
        """update_config 仅更新传入的键，不影响其他配置"""
        from base.chunk_config import ChunkConfigManager

        mgr = ChunkConfigManager()
        before = mgr.get_config()
        mgr.update_config({"parent_chunk_size": 9999})
        after = mgr.get_config()
        assert after["parent_chunk_size"] == 9999
        assert after["child_chunk_size"] == before["child_chunk_size"]
        assert after["default_strategy"] == before["default_strategy"]

    def test_reload_resets_to_ini_values(self):
        """reload 从 config.ini 重载，覆盖运行时修改"""
        from base.chunk_config import ChunkConfigManager

        mgr = ChunkConfigManager()
        mgr.update_config({"parent_chunk_size": 500})
        assert mgr.get_config()["parent_chunk_size"] == 500
        mgr.reload()
        assert mgr.get_config()["parent_chunk_size"] == 1200

    def test_get_strategy_is_case_insensitive(self):
        """扩展名匹配大小写不敏感"""
        from base.chunk_config import ChunkConfigManager

        mgr = ChunkConfigManager()
        mgr.update_config({"doc_type_strategies": {"PDF": "semantic"}})
        assert mgr.get_strategy("PDF") == "semantic"
        assert mgr.get_strategy("pdf") == "semantic"
        assert mgr.get_strategy("Pdf") == "semantic"
