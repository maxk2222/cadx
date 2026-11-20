"""Data backfills for alembic migrations — plain SQLAlchemy Core so they
stay independent of the ORM models (which evolve after the migration).

`backfill_parts(conn)` (parts feature): every version gets a part — one
part per ROOT version (parent_id IS NULL), the whole subtree follows the
root's part. Orphaned versions whose parent row is missing become their
own part. Names come from `version.name`, else the prompt, else "Part n".
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

import sqlalchemy as sa


def _derive_name(prompt: str, number: int) -> str:
    flat = re.sub(r"\s+", " ", prompt or "").strip()
    if not flat:
        return f"Part {number}"
    if len(flat) > 48:
        cut = flat[:48]
        flat = cut.rsplit(" ", 1)[0].strip() if " " in cut[:-1] else cut
    return flat or f"Part {number}"


def backfill_parts(conn: sa.Connection) -> int:
    """Assign `version.part_id` for every existing row. Returns part count.

    Assumes the `part` table exists and `version.part_id` is present
    (nullable). Idempotent-ish: skips versions already assigned."""
    version = sa.table(
        "version",
        sa.column("id", sa.Uuid),
        sa.column("project_id", sa.Uuid),
        sa.column("part_id", sa.Uuid),
        sa.column("parent_id", sa.Uuid),
        sa.column("number", sa.Integer),
        sa.column("name", sa.String),
        sa.column("prompt", sa.String),
    )
    part = sa.table(
        "part",
        sa.column("id", sa.Uuid),
        sa.column("project_id", sa.Uuid),
        sa.column("number", sa.Integer),
        sa.column("name", sa.String),
        sa.column("created_at", sa.DateTime),
        sa.column("updated_at", sa.DateTime),
    )
    all_rows = conn.execute(
        sa.select(
            version.c.id,
            version.c.project_id,
            version.c.part_id,
            version.c.parent_id,
            version.c.number,
            version.c.name,
            version.c.prompt,
        ).order_by(version.c.number)
    ).all()
    rows = [r for r in all_rows if r.part_id is None]
    if not rows:
        return 0

    vs = {r.id: r for r in all_rows}  # full tree map for root walks

    def part_for(vid: uuid.UUID) -> tuple[uuid.UUID | None, uuid.UUID]:
        """(existing part_id, top version id) — an assigned ancestor wins;
        else the unassigned top becomes a new part."""
        seen: set[uuid.UUID] = set()
        cur = vid
        while True:
            r = vs.get(cur)
            if r is None or cur in seen:
                return None, cur
            if r.part_id is not None:
                return r.part_id, cur
            if r.parent_id is None or r.parent_id not in vs:
                return None, cur
            seen.add(cur)
            cur = r.parent_id

    # group unassigned versions under their resolved part (existing or new)
    reuse: dict[uuid.UUID, list[uuid.UUID]] = {}
    by_top: dict[uuid.UUID, list[uuid.UUID]] = {}
    for r in rows:
        existing, top = part_for(r.id)
        (reuse if existing else by_top).setdefault(existing or top, []).append(r.id)

    # part numbers continue after existing maxima per project
    max_num: dict[uuid.UUID, int] = {}
    for pid_row in conn.execute(
        sa.select(part.c.project_id, sa.func.coalesce(sa.func.max(part.c.number), 0)).group_by(
            part.c.project_id
        )
    ).all():
        max_num[pid_row[0]] = int(pid_row[1])

    now = datetime.now(UTC)
    for pid, members in reuse.items():
        conn.execute(version.update().where(version.c.id.in_(members)).values(part_id=pid))

    created = 0
    for top_id, members in by_top.items():
        root = vs[top_id]
        pid = root.project_id
        num = max_num.get(pid, 0) + 1
        max_num[pid] = num
        part_id = uuid.uuid4()
        conn.execute(
            part.insert().values(
                id=part_id,
                project_id=pid,
                number=num,
                name=(root.name or "").strip() or _derive_name(root.prompt, num),
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(version.update().where(version.c.id.in_(members)).values(part_id=part_id))
        created += 1
    return created
