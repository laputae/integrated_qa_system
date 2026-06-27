"""Add chunk_config_snapshot column to eval_runs table."""
import sys
sys.path.insert(0, '.')

from sqlalchemy import text
from db_models.base import engine


def migrate():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'eval_runs'
              AND COLUMN_NAME = 'chunk_config_snapshot'
        """))
        if result.scalar() > 0:
            print("Column 'chunk_config_snapshot' already exists, skipping.")
            return

        conn.execute(text("""
            ALTER TABLE eval_runs
            ADD COLUMN chunk_config_snapshot JSON NULL
        """))
        conn.commit()
        print("Migration successful: added chunk_config_snapshot column.")


if __name__ == "__main__":
    migrate()
