"""renumber versions per part

Version numbers are per-part sequences (v1, v2, … within each part) —
previously they were allocated project-wide, so a part's first version
could show up as V9. Existing rows are renumbered by creation order
inside each part, preserving every part's internal newest-first order.

Revision ID: x5n6o7p8q9r0
Revises: w4m5n6o7p8q9
Create Date: 2026-11-12 21:00:00.000000

"""

from alembic import op

revision = "x5n6o7p8q9r0"
down_revision = "w4m5n6o7p8q9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE version
        SET number = sub.rn
        FROM (
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY part_id ORDER BY created_at, id) AS rn
            FROM version
        ) sub
        WHERE version.id = sub.id AND version.number <> sub.rn
        """
    )


def downgrade() -> None:
    pass  # renumbering is not reversible — old project-wide numbers are lost by design
