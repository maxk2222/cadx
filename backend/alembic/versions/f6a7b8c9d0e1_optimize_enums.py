"""add 'optimize' to versionkind and jobtype enums

The optimize/TopOpt code paths insert kind='optimize' / type='optimize' rows
but the enum values were never added to the Postgres types (SQLite ignores
enums, so tests did not catch this). ALTER TYPE ... ADD VALUE requires an
autocommit block on Postgres.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-10-11 20:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'optimize'")
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'optimize'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
