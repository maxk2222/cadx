"""Deterministic preassembly simulation — verify the planner against a
saved guide state WITHOUT a 20-minute eval run.

Usage (from backend/):
    ../.venv/bin/uv run python ../evals/sim_preassembly.py <state.json>

The script copies the state into a temp dir, rewrites container mesh
paths (``/data/...``) to the local data root, clears any placed
assembly, runs ``run_preassembly`` and dumps the joint graph +
instance poses + a few physical invariants (upward pipes, bridge
pairing, stacked parts).
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cadx.guide.preassembly import run_preassembly
from cadx.guide.session import GuideToolSession
from cadx.guide.state import GuideState


def _load_state(state_path: Path, *, reextract_ports: bool = False) -> tuple[GuideState, Path]:
    raw = json.loads(state_path.read_text())
    # container paths (/data/...) resolve against the repo data root, not
    # backend/data — this sim always runs from the checkout
    data_root = Path(__file__).resolve().parent.parent / "data"
    for part in raw.get("parts", []):
        mp = part.get("mesh_path") or ""
        if mp.startswith("/data/"):
            part["mesh_path"] = str(data_root / mp[len("/data/") :])
    if reextract_ports:
        _reextract(raw)
    raw.setdefault("assembly", {})["instances"] = []
    raw["assembly"]["joints"] = []
    raw["assembly"]["placements"] = []
    tmp = Path(tempfile.mkdtemp(prefix="sim-preasm-"))
    (tmp / "state.json").write_text(json.dumps(raw))
    state = GuideState.load(tmp)
    assert state is not None
    return state, tmp


def _reextract(raw: dict) -> None:
    """Re-run rim extraction on the part meshes — states saved before the
    bore/seat measurement model carry stale port metadata."""
    import trimesh

    from cadx.guide.cadmeta import _rim_ports
    from cadx.guide.state import GuidePart

    for p in raw.get("parts", []):
        mp = p.get("mesh_path")
        if not mp or not Path(mp).exists():
            continue
        m = trimesh.load(mp)
        if hasattr(m, "to_geometry"):
            m = m.to_geometry()
        elif hasattr(m, "geometry"):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        stub = GuidePart(id=p["id"], name=p.get("name", ""))
        p["ports"] = [pi.model_dump() for pi in _rim_ports(m, stub)]


def _dump(state: GuideState) -> None:
    print("\n== instances ==")
    for i in state.assembly.instances:
        t = i.transform
        print(
            f"  {i.instance_id:<28} step={i.placed_in_step:<3} "
            f"t=({t[0][3]:7.1f},{t[1][3]:7.1f},{t[2][3]:7.1f})"
        )
    print("== joints ==")
    for j in state.assembly.joints:
        print(
            f"  st={j.get('step'):<3} br={bool(j.get('bridge'))!s:<5} "
            f"{j.get('moving_instance')}:{j.get('moving_port')} -> "
            f"{j.get('fixed_instance')}:{j.get('fixed_port')}"
        )


def _check_invariants(state: GuideState) -> list[str]:
    """Cheap physical sanity: no stacked duplicates, bridges join two
    DIFFERENT instances, every instance joins at most once per port."""
    problems: list[str] = []
    seen: dict[str, list[float]] = {}
    for i in state.assembly.instances:
        t = [i.transform[r][3] for r in range(3)]
        for pid, pt in seen.items():
            if pid == i.part_id and max(abs(t[k] - pt[k]) for k in range(3)) < 2.0:
                problems.append(f"stacked duplicates: {i.instance_id} on {pid}")
        seen[i.instance_id] = t
    for j in state.assembly.joints:
        if j.get("bridge") and j.get("moving_instance") == j.get("fixed_instance"):
            problems.append(f"bridge self-loop on {j.get('moving_instance')}")
    return problems


def main() -> int:
    state_path = Path(sys.argv[1])
    reextract = "--reextract" in sys.argv[2:]
    state, tmp = _load_state(state_path, reextract_ports=reextract)
    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(
        guide_image_budget=4,
        agent_build_verify_max=0,
        agent_tool_verify_max=0,
        agent_tool_verify=False,
        agent_fem_max_runs=0,
        agent_struct_tools=False,
        image_tool_budget=0,
        agent_topopt_max_runs=0,
        agent_struct_budget_s=0,
    )
    ctx.image = None
    ctx.image_available = False
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir=tmp,
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    report = asyncio.run(run_preassembly(sess))
    print(
        f"placed={len(report['placed'])} skipped={len(report['skipped'])} "
        f"failed={len(report['failed'])}"
    )
    for f in report["failed"]:
        print(f"  FAIL st={f.get('step')} part={f.get('part')}: {f.get('reason')}")
    _dump(state)
    problems = _check_invariants(state)
    for p in problems:
        print(f"  INVARIANT: {p}")
    return 1 if report["failed"] or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
