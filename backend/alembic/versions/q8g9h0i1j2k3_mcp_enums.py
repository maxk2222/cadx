"""add 'code' + 'meshgen' to jobtype + versionkind enums

The MCP server's LLM-free job types persist Version rows with
kind='code'/'meshgen' and Job rows with type='code'/'meshgen' — same
pattern as the 'organic'/'refine' additions (Postgres needs
ALTER TYPE ... ADD VALUE in an autocommit block; SQLite ignores it).

Revision ID: q8g9h0i1j2k3
Revises: p7a8b9c0d1e2
Create Date: 2026-10-20 09:00:00.000000

"""

from alembic import op

revision = "q8g9h0i1j2k3"
down_revision = "p7a8b9c0d1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'code'")
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'meshgen'")
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'code'")
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'meshgen'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
