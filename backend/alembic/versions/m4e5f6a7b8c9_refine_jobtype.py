"""add 'refine' to jobtype + versionkind enums

Post-TopOpt feature finishing creates Job rows with type='refine' and
Version rows with kind='refine' — same pattern as the 'resample' addition
(Postgres needs ALTER TYPE ... ADD VALUE in an autocommit block; SQLite
ignores it).

Revision ID: m4e5f6a7b8c9
Revises: l3d4e5f6a7b8
Create Date: 2026-10-14 07:30:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "m4e5f6a7b8c9"
down_revision = "l3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'refine'")
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'refine'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
