# -*- coding:utf-8 -*-
import logging
import logging.handlers
import os
import json
import contextvars
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

from config import Config

# ---- Request context (contextvars — async-safe) ----

_request_id_var = contextvars.ContextVar('request_id', default=None)
_user_id_var = contextvars.ContextVar('user_id', default=None)
_tenant_id_var = contextvars.ContextVar('tenant_id', default=None)


class RequestContext:
    """Async-safe request-scoped context for structured log enrichment."""

    @staticmethod
    def set(request_id: Optional[str] = None,
            user_id: Optional[int] = None,
            tenant_id: Optional[int] = None):
        if request_id is not None:
            _request_id_var.set(request_id)
        if user_id is not None:
            _user_id_var.set(user_id)
        if tenant_id is not None:
            _tenant_id_var.set(tenant_id)

    @staticmethod
    def get() -> dict:
        return {
            'request_id': _request_id_var.get(),
            'user_id': _user_id_var.get(),
            'tenant_id': _tenant_id_var.get(),
        }

    @staticmethod
    def clear():
        _request_id_var.set(None)
        _user_id_var.set(None)
        _tenant_id_var.set(None)

    @staticmethod
    @contextmanager
    def ctx(request_id: Optional[str] = None,
            user_id: Optional[int] = None,
            tenant_id: Optional[int] = None):
        """Context manager that saves context, sets new values, restores on exit."""
        old = RequestContext.get()
        try:
            RequestContext.set(
                request_id=request_id, user_id=user_id, tenant_id=tenant_id
            )
            yield
        finally:
            RequestContext.set(**old)


# ---- JSON formatter for structured file logs ----

class JsonFormatter(logging.Formatter):
    def format(self, record):
        ctx = RequestContext.get()
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if ctx['request_id']:
            entry['request_id'] = ctx['request_id']
        if ctx['user_id']:
            entry['user_id'] = ctx['user_id']
        if ctx['tenant_id']:
            entry['tenant_id'] = ctx['tenant_id']
        if record.exc_info and record.exc_info[1]:
            entry['exception'] = str(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False)


# ---- Setup ----

_current_file_path = os.path.abspath(__file__)
_current_dir_path = os.path.dirname(_current_file_path)
_project_root = os.path.dirname(_current_dir_path)


def setup_logging(log_file=None):
    config = Config()
    log_file = log_file or os.path.join(_project_root, config.LOG_FILE)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("EduRAG")
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        # File handler — JSON, with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding='utf-8',
        )
        file_handler.setLevel(level)
        if config.LOG_FORMAT == 'json':
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        # Console handler — human-readable
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger that inherits EduRAG handlers via propagation."""
    return logging.getLogger(f"EduRAG.{name}")


logger = setup_logging()
