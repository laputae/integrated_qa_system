"""Add is_deleted column to conversations table."""
import sys
sys.path.insert(0, '.')

from sqlalchemy import text
from db_models.base import engine, SessionLocal


def migrate():
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'conversations'
              AND COLUMN_NAME = 'is_deleted'
        """))
        if result.scalar() > 0:
            print("Column 'is_deleted' already exists, skipping.")
            return

        conn.execute(text("""
            ALTER TABLE conversations
            ADD COLUMN is_deleted TINYINT(1) NOT NULL DEFAULT 0
        """))
        conn.execute(text("""
            ALTER TABLE conversations
            ADD INDEX idx_is_deleted (is_deleted)
        """))
        conn.commit()
        print("Migration successful: added is_deleted column + index.")


if __name__ == "__main__":
    migrate()
