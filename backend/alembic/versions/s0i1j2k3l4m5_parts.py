"""parts: new `part` table + `version.part_id` with backfill

A Part groups one or more root version trees inside a project. Every
existing root becomes its own part (named from the root's name or
prompt); children inherit the root's part. Batch mode keeps this working
on SQLite and Postgres alike.

Revision ID: s0i1j2k3l4m5
Revises: r9h0i1j2k3l4
Create Date: 2026-10-24 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "s0i1j2k3l4m5"
down_revision = "r9h0i1j2k3l4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "part",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("project.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("project_id", "number", name="uq_part_project_number"),
    )
    op.create_index("ix_part_project_id", "part", ["project_id"])

    op.add_column("version", sa.Column("part_id", sa.Uuid(), nullable=True))
    with op.batch_alter_table("version") as b:
        b.create_index("ix_version_part_id", ["part_id"])
        b.create_foreign_key("fk_version_part", "part", ["part_id"], ["id"])

    from cadx.db.backfill import backfill_parts

    backfill_parts(op.get_bind())

    with op.batch_alter_table("version") as b:
        b.alter_column("part_id", existing_type=sa.Uuid(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("version") as b:
        b.drop_constraint("fk_version_part", type_="foreignkey")
        b.drop_index("ix_version_part_id")
        b.drop_column("part_id")
    op.drop_index("ix_part_project_id", table_name="part")
    op.drop_table("part")
