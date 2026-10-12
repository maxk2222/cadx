"""add 'mesh_raw' and 'fem' to artifactkind enum

Mesh-mode versions insert artifact kind='mesh_raw' (added with the mesh
pipeline, enum value never migrated) and optimize/FEM artifacts use
kind='fem'. Both crashed Postgres inserts at finalize time, leaving jobs
stuck in 'running'. SQLite ignores enums, so tests did not catch this.

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-10-11 23:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "g7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE artifactkind ADD VALUE IF NOT EXISTS 'mesh_raw'")
        op.execute("ALTER TYPE artifactkind ADD VALUE IF NOT EXISTS 'fem'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
