"""add 'note' asset kind for masternotes

Markdown notes the guide agent (or the user via chat) writes about an
assembly guide — tips, assumptions, extra info — surfaced as project
assets.

Revision ID: w4m5n6o7p8q9
Revises: v3l4m5n6o7p8
Create Date: 2026-11-12 20:00:00.000000

"""

from alembic import op

revision = "w4m5n6o7p8q9"
down_revision = "v3l4m5n6o7p8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE assetkind ADD VALUE IF NOT EXISTS 'note'")


def downgrade() -> None:
    pass
