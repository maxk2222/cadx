"""job.live — structured live progress state (iterations, scores, step timings)

Revision ID: k2c3d4e5f6a7
Revises: j1c2d3e4f5a6
Create Date: 2026-10-13
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "k2c3d4e5f6a7"
down_revision = "j1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("job", sa.Column("live", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("job", "live")
