"""mesh modes: gen_mode/resolved_mode/mesh columns + mesh_raw artifact kind

Revision ID: a1b2c3d4e5f6
Revises: 3f9d012667b4
Create Date: 2026-10-10 18:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "3f9d012667b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "version",
        sa.Column("gen_mode", sa.String(), nullable=False, server_default="auto"),
    )
    op.add_column("version", sa.Column("resolved_mode", sa.String(), nullable=True))
    op.add_column("version", sa.Column("mesh", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("version", "mesh")
    op.drop_column("version", "resolved_mode")
    op.drop_column("version", "gen_mode")
