"""add 'generated_image' to assetkind for local text-to-image assets

Generated images (FLUX/SDXL via the mesh-server image adapters) are stored
as regular image assets with kind='generated_image' so they can feed the
LLM context and image->mesh pipeline like uploaded images while staying
distinguishable in the UI (ai badge) and in provenance (gen_params JSON).

ALTER TYPE ... ADD VALUE requires an autocommit block on Postgres.

Revision ID: j0b1c2d3e4f5
Revises: i9c0d1e2f3a4
Create Date: 2026-10-14 12:00:00.000000

"""

from alembic import op

revision = "j0b1c2d3e4f5"
down_revision = "i9c0d1e2f3a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # sqlite tests: enums are plain VARCHAR
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE assetkind ADD VALUE IF NOT EXISTS 'generated_image'")


def downgrade() -> None:
    # Postgres cannot drop enum values; no-op.
    pass
