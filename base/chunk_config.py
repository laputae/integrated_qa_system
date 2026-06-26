"""Chunk configuration manager — thread-safe singleton with hot-reload support.

Reads [chunking] section from config.ini on init/reload.
Runtime updates via update_config() are kept in memory only (not persisted to disk).
"""

import json
import threading
from typing import Dict, Optional

from base import Config
from base import logger


class ChunkConfigManager:
    """Singleton managing chunking configuration at runtime."""

    _instance: Optional["ChunkConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChunkConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config_lock = threading.RLock()
        self._config: Dict = {}
        self._initialized = True
        self.reload()

    def _parse_doc_type_strategies(self, raw: str) -> Dict[str, str]:
        if not raw or raw == "{}":
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k.lower(): v.lower() for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid doc_type_strategies in [chunking]: %s", raw)
        return {}

    def reload(self):
        """Reload configuration from config.ini."""
        conf = Config()
        raw_doc_type = conf.CHUNK_DOC_TYPE_STRATEGIES
        with self._config_lock:
            self._config = {
                "default_strategy": conf.CHUNK_DEFAULT_STRATEGY,
                "doc_type_strategies": self._parse_doc_type_strategies(raw_doc_type),
                "semantic_model_path": conf.CHUNK_SEMANTIC_MODEL_PATH or "",
                "semantic_device": conf.CHUNK_SEMANTIC_DEVICE,
                "semantic_fallback_strategy": conf.CHUNK_SEMANTIC_FALLBACK_STRATEGY,
                "parent_chunk_size": conf.PARENT_CHUNK_SIZE,
                "child_chunk_size": conf.CHILD_CHUNK_SIZE,
                "chunk_overlap": conf.CHUNK_OVERLAP,
            }
        logger.info("ChunkConfigManager reloaded from config.ini")

    def get_config(self) -> Dict:
        """Return a copy of the current configuration."""
        with self._config_lock:
            return dict(self._config)

    def update_config(self, updates: Dict):
        """Update runtime configuration (in-memory only, does not persist to disk)."""
        allowed_keys = {
            "default_strategy", "doc_type_strategies", "semantic_model_path",
            "semantic_device", "semantic_fallback_strategy",
            "parent_chunk_size", "child_chunk_size", "chunk_overlap",
        }
        with self._config_lock:
            for key, value in updates.items():
                if key in allowed_keys:
                    if key == "doc_type_strategies" and isinstance(value, dict):
                        value = {k.lower(): v.lower() for k, v in value.items()}
                    self._config[key] = value
                else:
                    logger.warning("Ignoring unknown chunk config key: %s", key)

    def get_strategy(self, file_extension: str) -> str:
        """Determine which strategy to use for a given file extension."""
        ext = file_extension.lstrip(".").lower()
        with self._config_lock:
            doc_map = self._config.get("doc_type_strategies", {})
            if isinstance(doc_map, dict) and ext in doc_map:
                return doc_map[ext]
            return self._config.get("default_strategy", "recursive")
