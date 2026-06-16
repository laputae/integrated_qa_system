from typing import Optional

from models.audit_log import AuditLog


class AuditRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def insert(self, user_id: Optional[int], event_type: str,
               ip_address: Optional[str] = None,
               user_agent: Optional[str] = None,
               detail: Optional[str] = None):
        with self.session_factory() as session:
            entry = AuditLog(
                user_id=user_id,
                event_type=event_type,
                ip_address=ip_address,
                user_agent=user_agent,
                detail=detail,
            )
            session.add(entry)
            session.commit()
