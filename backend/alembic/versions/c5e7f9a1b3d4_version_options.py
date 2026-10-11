"""version options column (GUI options panel provenance)

Revision ID: c5e7f9a1b3d4
Revises: b7c8d9e0f1a2
Create Date: 2026-10-12 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "c5e7f9a1b3d4"
down_revision = "b7c8d9e0f1a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "version",
        sa.Column("options", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("version", "options")
