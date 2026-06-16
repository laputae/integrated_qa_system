import json
from datetime import datetime
from enum import Enum
from typing import Optional


class AuditEventType(str, Enum):
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    REGISTER_SUCCESS = "REGISTER_SUCCESS"
    TOKEN_REFRESH = "TOKEN_REFRESH"
    LOGOUT = "LOGOUT"
    SQL_INJECTION_ATTEMPT = "SQL_INJECTION_ATTEMPT"
    XSS_ATTEMPT = "XSS_ATTEMPT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    INVALID_INPUT = "INVALID_INPUT"
    HISTORY_CLEARED = "HISTORY_CLEARED"


class AuditLogger:
    def __init__(self):
        self._repo = None

    @property
    def repo(self):
        if self._repo is None:
            from repositories.audit_repo import AuditRepository
            from models.base import SessionLocal
            self._repo = AuditRepository(SessionLocal)
        return self._repo

    def log(self, event_type: AuditEventType, user_id: Optional[int] = None,
            ip_address: Optional[str] = None, user_agent: Optional[str] = None,
            detail: Optional[dict] = None):
        try:
            self.repo.insert(
                user_id=user_id,
                event_type=event_type.value,
                ip_address=ip_address,
                user_agent=user_agent,
                detail=json.dumps(detail, ensure_ascii=False) if detail else None,
            )
        except Exception:
            pass


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
