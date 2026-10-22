"""version.name: optional user-assigned label

Revision ID: r9h0i1j2k3l4
Revises: q8g9h0i1j2k3
Create Date: 2026-10-22 18:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "r9h0i1j2k3l4"
down_revision = "q8g9h0i1j2k3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "version",
        sa.Column("name", sa.String(), server_default="", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("version", "name")
