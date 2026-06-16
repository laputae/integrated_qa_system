from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from models.conversation import Conversation


class ConversationRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def insert(self, session_id: str, user_id: int, question: str, answer: str):
        with self.session_factory() as session:
            conv = Conversation(
                user_id=user_id,
                session_id=session_id,
                question=question,
                answer=answer,
            )
            session.add(conv)
            session.commit()

    def get_recent_history(self, session_id: str, user_id: int, limit: int = 5) -> list[dict]:
        with self.session_factory() as session:
            rows = (
                session.query(Conversation)
                .filter(
                    Conversation.session_id == session_id,
                    Conversation.user_id == user_id,
                )
                .order_by(desc(Conversation.timestamp))
                .limit(limit)
                .all()
            )
            history = [{"question": r.question, "answer": r.answer} for r in rows]
            return history[::-1]

    def get_session_history(self, session_id: str, user_id: int) -> list[dict]:
        with self.session_factory() as session:
            rows = (
                session.query(Conversation)
                .filter(
                    Conversation.session_id == session_id,
                    Conversation.user_id == user_id,
                )
                .order_by(Conversation.timestamp)
                .all()
            )
            return [{"question": r.question, "answer": r.answer} for r in rows]

    def delete_session(self, session_id: str, user_id: int) -> bool:
        with self.session_factory() as session:
            deleted = (
                session.query(Conversation)
                .filter(
                    Conversation.session_id == session_id,
                    Conversation.user_id == user_id,
                )
                .delete()
            )
            session.commit()
            return deleted > 0

    def prune_old_records(self, session_id: str, user_id: int, keep: int = 5):
        with self.session_factory() as session:
            subq = (
                session.query(Conversation.id)
                .filter(
                    Conversation.session_id == session_id,
                    Conversation.user_id == user_id,
                )
                .order_by(desc(Conversation.timestamp))
                .limit(keep)
                .subquery()
            )
            session.query(Conversation).filter(
                Conversation.session_id == session_id,
                Conversation.user_id == user_id,
                Conversation.id.notin_(subq),
            ).delete(synchronize_session=False)
            session.commit()

    def get_user_sessions(self, user_id: int) -> list[dict]:
        with self.session_factory() as session:
            from sqlalchemy import func
            rows = (
                session.query(
                    Conversation.session_id,
                    func.count(Conversation.id).label("count"),
                    func.max(Conversation.timestamp).label("last_time"),
                )
                .filter(Conversation.user_id == user_id)
                .group_by(Conversation.session_id)
                .order_by(func.max(Conversation.timestamp).desc())
                .all()
            )
            return [
                {"session_id": r.session_id, "count": r.count, "last_time": str(r.last_time)}
                for r in rows
            ]
