"""librarypart table + version.library_part_ids

Revision ID: d4e5f6a7b8c9
Revises: c5e7f9a1b3d4
Create Date: 2026-10-14 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "d4e5f6a7b8c9"
down_revision = "c5e7f9a1b3d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "librarypart",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("code", sa.String(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("caption", sa.String(), nullable=False),
        sa.Column("source_version_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.add_column("version", sa.Column("library_part_ids", sa.JSON(), nullable=True))
    op.add_column("version", sa.Column("bom", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("version", "bom")
    op.drop_column("version", "library_part_ids")
    op.drop_table("librarypart")
