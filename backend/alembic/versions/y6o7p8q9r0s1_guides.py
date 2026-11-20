"""guide table — AR-guide documents (KIARA web-suite parity)

The `doc` JSON column stores the normalized GuideSchema document verbatim;
the scalar columns are denormalized mirrors for locking (`FOR UPDATE`),
listing, and the project↔guide publish coupling.

Revision ID: y6o7p8q9r0s1
Revises: x5n6o7p8q9r0
Create Date: 2026-11-21 00:00:00.000000

"""

from alembic import op

revision = "y6o7p8q9r0s1"
down_revision = "x5n6o7p8q9r0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS guide (
            id UUID PRIMARY KEY,
            guide_id VARCHAR NOT NULL,
            database_id VARCHAR NOT NULL DEFAULT '',
            title VARCHAR NOT NULL DEFAULT '',
            doc JSON NOT NULL DEFAULT '{}',
            version INTEGER NOT NULL DEFAULT 1,
            base_version INTEGER,
            content_hash VARCHAR,
            format_version INTEGER,
            project_id UUID REFERENCES project(id),
            source_version_id UUID,
            source_file_id VARCHAR,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_guide_guide_id ON guide (guide_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_guide_database_id ON guide (database_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_guide_project_id ON guide (project_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS guide")
