"""add 'resample' to jobtype enum

The topopt threshold-resample endpoint creates Job rows with
type='resample' — same pattern as the 'optimize' enum addition (Postgres
needs ALTER TYPE ... ADD VALUE in an autocommit block; SQLite ignores it).

Revision ID: l3d4e5f6a7b8
Revises: k2c3d4e5f6a7
Create Date: 2026-10-20 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "l3d4e5f6a7b8"
down_revision = "k2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'resample'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
