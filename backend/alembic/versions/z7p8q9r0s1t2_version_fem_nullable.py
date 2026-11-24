"""add version.fem JSON + align JSON/timestamp nullability with the model

Version.fem (standalone FEM report payload) was added to the model without
a migration — every ORM query selecting version.fem 500'd on Postgres.
Same drift class for guide.doc / version.options (model Column(JSON) is
nullable, schema had them NOT NULL) and guide.created_at/updated_at
(schema allowed NULL, model does not).

Revision ID: z7p8q9r0s1t2
Revises: y6o7p8q9r0s1
Create Date: 2026-11-24 16:30:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "z7p8q9r0s1t2"
down_revision = "y6o7p8q9r0s1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("version", sa.Column("fem", sa.JSON(), nullable=True))
    op.alter_column("guide", "doc", existing_type=sa.JSON(), nullable=True)
    op.alter_column("version", "options", existing_type=sa.JSON(), nullable=True)
    op.execute("UPDATE guide SET created_at = now() WHERE created_at IS NULL")
    op.execute("UPDATE guide SET updated_at = now() WHERE updated_at IS NULL")
    op.alter_column("guide", "created_at", existing_type=sa.TIMESTAMP(timezone=True), nullable=False)
    op.alter_column("guide", "updated_at", existing_type=sa.TIMESTAMP(timezone=True), nullable=False)


def downgrade() -> None:
    op.alter_column("guide", "updated_at", existing_type=sa.TIMESTAMP(timezone=True), nullable=True)
    op.alter_column("guide", "created_at", existing_type=sa.TIMESTAMP(timezone=True), nullable=True)
    op.alter_column("version", "options", existing_type=sa.JSON(), nullable=False)
    op.alter_column("guide", "doc", existing_type=sa.JSON(), nullable=False)
    op.drop_column("version", "fem")
