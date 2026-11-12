"""add 'guide' job/version kinds + 'pdf'/'cad_glb' asset kinds

Anleitungsbuilder: a durable guide-building job (JobType.guide) hangs on a
companion Version (VersionKind.guide) that also carries the exported
unity_guide.json artifacts; uploads accept the instruction PDF and GLB CAD
parts.

ALTER TYPE ... ADD VALUE requires an autocommit block on Postgres.

Revision ID: v3l4m5n6o7p8
Revises: u2k3l4m5n6o7
Create Date: 2026-11-20 10:00:00.000000

"""

from alembic import op

revision = "v3l4m5n6o7p8"
down_revision = "u2k3l4m5n6o7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'guide'")
        op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'guide'")
        op.execute("ALTER TYPE assetkind ADD VALUE IF NOT EXISTS 'pdf'")
        op.execute("ALTER TYPE assetkind ADD VALUE IF NOT EXISTS 'cad_glb'")
        op.execute("ALTER TYPE artifactkind ADD VALUE IF NOT EXISTS 'unity_guide'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
