"""version.accepted_at: user-verified master version (max one per part)

Revision ID: u2k3l4m5n6o7
Revises: t1j2k3l4m5n6
Create Date: 2026-10-24 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "u2k3l4m5n6o7"
down_revision = "t1j2k3l4m5n6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "version",
        sa.Column("accepted_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("version", "accepted_at")
