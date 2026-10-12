"""add 'mesh_edit' + 'fem' enums for the GUI mesh/structural editors

The visual mesh editor creates kind='mesh_edit' versions via
JobType.mesh_edit; the standalone FEM check uses JobType.fem.
ALTER TYPE ... ADD VALUE requires an autocommit block on Postgres.

Revision ID: i9c0d1e2f3a4
Revises: h8c9d0e1f2a3
Create Date: 2026-10-12 12:00:00.000000

"""

from alembic import op

revision = "i9c0d1e2f3a4"
down_revision = "h8c9d0e1f2a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'mesh_edit'")
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'mesh_edit'")
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'fem'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
