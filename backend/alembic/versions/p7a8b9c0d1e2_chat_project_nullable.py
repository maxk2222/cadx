"""chat.project_id nullable — chats may exist before a project is bound

Revision ID: p7a8b9c0d1e2
Revises: o6f7a8b9c0d1
Create Date: 2026-10-16 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "p7a8b9c0d1e2"
down_revision = "o6f7a8b9c0d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch mode keeps the migration working on SQLite too
    with op.batch_alter_table("chat") as b:
        b.alter_column("project_id", existing_type=sa.Uuid(), nullable=True)


def downgrade() -> None:
    # delete leftover project-less chats before tightening (nothing to rebind to)
    op.execute(
        "DELETE FROM chatmessage WHERE chat_id IN (SELECT id FROM chat WHERE project_id IS NULL)"
    )
    op.execute("DELETE FROM chat WHERE project_id IS NULL")
    with op.batch_alter_table("chat") as b:
        b.alter_column("project_id", existing_type=sa.Uuid(), nullable=False)
