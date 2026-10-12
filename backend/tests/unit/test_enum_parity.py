"""Static parity check: Python StrEnum members vs. Postgres enum migrations.

SQLAlchemy persists enum member *names*; Postgres enum labels must cover all
of them. Missing labels crash INSERTs at finalize time (observed: 'optimize'
missing from jobtype/versionkind, 'mesh_raw'/'fem' from artifactkind). SQLite
tests don't enforce enums, so drift was invisible — this test parses the
alembic migrations and fails when a member has no label.
"""

from __future__ import annotations

import ast
import enum
import re
from pathlib import Path

import cadx.db.models as models

MIGRATIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"

_ADD_VALUE = re.compile(
    r"ALTER\s+TYPE\s+(\w+)\s+ADD\s+VALUE\s+(?:IF\s+NOT\s+EXISTS\s+)?'(\w+)'",
    re.IGNORECASE,
)


def _snake(name: str) -> str:
    # pg enum names follow the class name lowercased: ArtifactKind -> artifactkind
    return name.lower()


def _declared_labels() -> dict[str, set[str]]:
    """{pg_enum_name: {label,...}} from initial sa.Enum(...) + ALTER TYPE adds."""
    labels: dict[str, set[str]] = {}
    for path in sorted(MIGRATIONS.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            # sa.Enum("a", "b", name="kind") in create_table calls
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Enum"
            ):
                enum_name = next(
                    (
                        kw.value.value
                        for kw in node.keywords
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant)
                    ),
                    None,
                )
                if enum_name:
                    labels.setdefault(enum_name, set()).update(
                        arg.value
                        for arg in node.args
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    )
        for m in _ADD_VALUE.finditer(src):
            labels.setdefault(m.group(1).lower(), set()).add(m.group(2))
    return labels


def _python_enums() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(models):
        obj = getattr(models, name)
        if isinstance(obj, type) and issubclass(obj, enum.StrEnum):
            out[_snake(name)] = set(obj.__members__)
    return out


def test_every_str_enum_member_has_a_pg_label() -> None:
    pg = _declared_labels()
    assert pg, "no enum labels parsed from migrations — parser broken?"
    missing: list[str] = []
    for enum_name, members in sorted(_python_enums().items()):
        declared = pg.get(enum_name, set())
        assert declared, f"pg enum '{enum_name}' not found in any migration"
        for member in sorted(members - declared):
            missing.append(f"{enum_name}.{member}")
    assert not missing, (
        "enum members without a migration label (INSERTs will crash on "
        f"Postgres): {missing} — add an 'ALTER TYPE ... ADD VALUE' migration"
    )
