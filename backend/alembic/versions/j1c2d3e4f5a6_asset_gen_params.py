"""add Asset.gen_params JSON column for generated-image provenance

Stores prompt/model/seed/settings so generated images are reproducible and
their origin is inspectable. NULL for non-generated assets.

Revision ID: j1c2d3e4f5a6
Revises: j0b1c2d3e4f5
Create Date: 2026-10-14 12:01:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "j1c2d3e4f5a6"
down_revision = "j0b1c2d3e4f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("asset", sa.Column("gen_params", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("asset", "gen_params")
