"""One-time script: create default tenant and migrate existing users.

Run this script before starting the app after upgrading to multi-tenant schema:
    uv run scripts/seed_default_tenant.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, update
from db_models.base import SessionLocal, init_db, Base, engine
from db_models.tenant import Tenant


def run_ddl():
    """Apply schema changes: create tenants table, add tenant_id columns to existing tables."""
    with engine.connect() as conn:
        # Create tenants table (safe to run even if it exists when using IF NOT EXISTS)
        Base.metadata.create_all(engine)

        # Check and add tenant_id column to existing tables (create_all won't alter existing tables)
        existing_cols = {
            row[0]
            for row in conn.execute(
                text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'users'")
            ).fetchall()
        }

        ddl_statements = []
        if "tenant_id" not in existing_cols:
            ddl_statements.extend([
                "ALTER TABLE users ADD COLUMN tenant_id INT DEFAULT NULL",
                "ALTER TABLE users ADD INDEX idx_users_tenant (tenant_id)",
            ])
        existing_cols = {
            row[0]
            for row in conn.execute(
                text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'conversations'")
            ).fetchall()
        }
        if "tenant_id" not in existing_cols:
            ddl_statements.extend([
                "ALTER TABLE conversations ADD COLUMN tenant_id INT DEFAULT NULL",
                "ALTER TABLE conversations ADD INDEX idx_conv_tenant (tenant_id)",
            ])
        existing_cols = {
            row[0]
            for row in conn.execute(
                text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'refresh_tokens'")
            ).fetchall()
        }
        if "tenant_id" not in existing_cols:
            ddl_statements.extend([
                "ALTER TABLE refresh_tokens ADD COLUMN tenant_id INT DEFAULT NULL",
                "ALTER TABLE refresh_tokens ADD INDEX idx_rt_tenant (tenant_id)",
            ])
        existing_cols = {
            row[0]
            for row in conn.execute(
                text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'audit_logs'")
            ).fetchall()
        }
        if "tenant_id" not in existing_cols:
            ddl_statements.extend([
                "ALTER TABLE audit_logs ADD COLUMN tenant_id INT DEFAULT NULL",
                "ALTER TABLE audit_logs ADD INDEX idx_audit_tenant (tenant_id)",
            ])

        for stmt in ddl_statements:
            print(f"DDL: {stmt}")
            conn.execute(text(stmt))
        conn.commit()

        if ddl_statements:
            print(f"Applied {len(ddl_statements)} schema changes.")
        else:
            print("Schema is already up to date.")


def seed():
    run_ddl()

    with SessionLocal() as session:
        t = session.query(Tenant).filter(Tenant.name == "default").first()

        if not t:
            t = Tenant(name="default")
            session.add(t)
            session.flush()
            tenant_id = t.id
            print(f"Created default tenant (id={tenant_id})")
        else:
            t = session.query(Tenant).filter(Tenant.name == "default").first()
            tenant_id = t.id
            print(f"Default tenant already exists (id={tenant_id})")

        # Migrate NULL tenant_id rows to default tenant
        from db_models.user import User
        from db_models.conversation import Conversation
        from db_models.refresh_token import RefreshToken

        for model, label in [
            (User, "users"),
            (Conversation, "conversations"),
            (RefreshToken, "refresh_tokens"),
        ]:
            result = session.execute(
                update(model)
                .where(model.tenant_id.is_(None))
                .values(tenant_id=tenant_id)
            )
            if result.rowcount > 0:
                print(f"Migrated {result.rowcount} {label} to default tenant")

        session.commit()
        print("Seed completed successfully.")


if __name__ == "__main__":
    seed()
