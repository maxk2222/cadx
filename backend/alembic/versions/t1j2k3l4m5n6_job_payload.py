"""job.payload: verbatim arq envelope {func,args} for queue-loss replay

Revision ID: t1j2k3l4m5n6
Revises: s0i1j2k3l4m5
Create Date: 2026-11-01 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "t1j2k3l4m5n6"
down_revision = "s0i1j2k3l4m5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("job", sa.Column("payload", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("job", "payload")
