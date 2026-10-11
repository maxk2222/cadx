"""version.issues: structured issue list from the agent loop

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-10-11 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b7c8d9e0f1a2"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("version", sa.Column("issues", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("version", "issues")
