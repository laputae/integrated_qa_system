import json
from enum import Enum


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
    HISTORY_DELETED = "HISTORY_DELETED"


class AuditLogger:
    def __init__(self):
        self._repo = None

    @property
    def repo(self):
        if self._repo is None:
            from db_models.base import SessionLocal
            from repositories.audit_repo import AuditRepository

            self._repo = AuditRepository(SessionLocal)
        return self._repo

    def log(
        self,
        event_type: AuditEventType,
        user_id: int | None = None,
        tenant_id: int | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        detail: dict | None = None,
    ):
        try:
            self.repo.insert(
                user_id=user_id,
                tenant_id=tenant_id,
                event_type=event_type.value,
                ip_address=ip_address,
                user_agent=user_agent,
                detail=json.dumps(detail, ensure_ascii=False) if detail else None,
            )
        except Exception:
            pass


_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
