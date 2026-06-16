from typing import Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from db_models.user import User


class UserRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, username: str, password_hash: str, tenant_id: int) -> User:
        with self.session_factory() as session:
            user = User(
                username=username,
                password_hash=password_hash,
                tenant_id=tenant_id,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def get_by_username(self, username: str, tenant_id: int) -> Optional[User]:
        with self.session_factory() as session:
            return session.query(User).filter(
                and_(User.username == username, User.tenant_id == tenant_id)
            ).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        with self.session_factory() as session:
            return session.query(User).filter(User.id == user_id).first()

    def username_exists(self, username: str, tenant_id: int) -> bool:
        with self.session_factory() as session:
            return session.query(User).filter(
                and_(User.username == username, User.tenant_id == tenant_id)
            ).first() is not None
