"""chat + chatmessage tables for the conversational agent UI

Revision ID: o6f7a8b9c0d1
Revises: n5f6a7b8c9d0
Create Date: 2026-10-15 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "o6f7a8b9c0d1"
down_revision = "n5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("project.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("llm_messages", sa.JSON(), nullable=True),
        sa.Column("turn_running", sa.Boolean(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_project_id", "chat", ["project_id"])
    op.create_table(
        "chatmessage",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("chat_id", sa.Uuid(), sa.ForeignKey("chat.id"), nullable=False),
        sa.Column("role", sa.Enum("user", "assistant", "system", name="chatrole"), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("asset_ids", sa.JSON(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("pending", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chatmessage_chat_id", "chatmessage", ["chat_id"])


def downgrade() -> None:
    op.drop_index("ix_chatmessage_chat_id", table_name="chatmessage")
    op.drop_table("chatmessage")
    op.drop_index("ix_chat_project_id", table_name="chat")
    op.drop_table("chat")
