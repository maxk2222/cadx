"""asset.derived_from for edited image copies

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-16 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("asset", sa.Column("derived_from", sa.Uuid(), nullable=True))


def downgrade() -> None:
    op.drop_column("asset", "derived_from")
