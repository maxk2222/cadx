"""add 'organic' to jobtype + versionkind enums

Procedural pseudo-topology organic restyling creates Job rows with
type='organic' and Version rows with kind='organic' — same pattern as the
'refine'/'resample' additions (Postgres needs ALTER TYPE ... ADD VALUE in
an autocommit block; SQLite ignores it).

Revision ID: n5f6a7b8c9d0
Revises: m4e5f6a7b8c9
Create Date: 2026-10-15 09:00:00.000000

"""

from alembic import op

revision = "n5f6a7b8c9d0"
down_revision = "m4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'organic'")
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'organic'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
