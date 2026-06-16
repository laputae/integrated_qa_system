from typing import Optional

from sqlalchemy.orm import Session

from models.user import User


class UserRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, username: str, password_hash: str) -> User:
        with self.session_factory() as session:
            user = User(username=username, password_hash=password_hash)
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def get_by_username(self, username: str) -> Optional[User]:
        with self.session_factory() as session:
            return session.query(User).filter(User.username == username).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        with self.session_factory() as session:
            return session.query(User).filter(User.id == user_id).first()

    def username_exists(self, username: str) -> bool:
        with self.session_factory() as session:
            return session.query(User).filter(User.username == username).first() is not None
