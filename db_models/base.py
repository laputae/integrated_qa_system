from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from base import Config

_config = Config()

DATABASE_URL = (
    f"mysql+pymysql://{_config.MYSQL_USER}:{_config.MYSQL_PASSWORD}"
    f"@{_config.MYSQL_HOST}:3306/{_config.MYSQL_DATABASE}"
    f"?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def init_db(config=None):
    return engine, SessionLocal


def get_session() -> Session:
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
