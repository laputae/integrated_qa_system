from typing import Optional

from sqlalchemy.orm import Session

from db_models.tenant import Tenant


class TenantRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def get_by_name(self, name: str) -> Optional[Tenant]:
        with self.session_factory() as session:
            return session.query(Tenant).filter(Tenant.name == name).first()

    def get_by_id(self, tenant_id: int) -> Optional[Tenant]:
        with self.session_factory() as session:
            return session.query(Tenant).filter(Tenant.id == tenant_id).first()

    def create(self, name: str) -> Tenant:
        with self.session_factory() as session:
            tenant = Tenant(name=name)
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            return tenant

    def get_or_create(self, name: str) -> Tenant:
        tenant = self.get_by_name(name)
        if tenant:
            return tenant
        try:
            return self.create(name)
        except Exception:
            return self.get_by_name(name)
