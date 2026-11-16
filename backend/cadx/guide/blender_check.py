"""Blender-mode scene gate — deterministic checks on the synced scene.

Runs after the build mission declares done (and again after every
lint/reopen round-trip): re-reads the tagged Blender objects, rebuilds
the instance list, then verifies what the emit contract actually needs
— completeness, transform sanity, and a bounding-box overlap scan.

Findings:
- ``blockers`` reopen the owning mission (build) with the text as notes;
  they name facts the scene did not provide, never LLM intent.
- overlaps land in ``state.blender_manifest`` as warnings — assembled
  joints legitimately interpenetrate, so only a mostly-buried instance
  escalates to a blocker (>50% of its world AABB inside a neighbor's).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from cadx.guide import blender_tools
from cadx.guide.blender_client import BlenderError
from cadx.logging import get_logger

log = get_logger(__name__)

_BURY_FRACTION = 0.5


def _world_aabb(inst: Any, part: Any) -> tuple[np.ndarray, np.ndarray]:
    """Part local bbox -> world-space axis-aligned box via the instance
    transform (8-corner hull — conservative for rotations)."""
    lo, hi = part.bbox_min, part.bbox_max
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )
    m = np.asarray(inst.transform, dtype=float)
    world = (m @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _overlap_fraction(a: tuple, b: tuple) -> float:
    """Intersection volume / smaller box volume — 1.0 = fully buried."""
    inter_lo = np.maximum(a[0], b[0])
    inter_hi = np.minimum(a[1], b[1])
    inter = np.maximum(inter_hi - inter_lo, 0.0)
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0.0:
        return 0.0
    va = float(np.prod(np.maximum(a[1] - a[0], 0.0)))
    vb = float(np.prod(np.maximum(b[1] - b[0], 0.0)))
    smaller = min(va, vb)
    return inter_vol / smaller if smaller > 1e-9 else 0.0


def _transform_blockers(instances: list) -> list[str]:
    """Finite 4x4, near-orthonormal scale — mirrored or zero-scaled
    placements are placement errors, not modeling choices."""
    out: list[str] = []
    for inst in instances:
        m = np.asarray(inst.transform, dtype=float)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            out.append(f"{inst.instance_id}: non-finite or malformed matrix_world")
            continue
        det = float(abs(np.linalg.det(m[:3, :3])))
        if not (0.3 <= det <= 3.0):
            out.append(
                f"{inst.instance_id}: implausible scale |det|={det:.3g} — "
                "expected a rigid/near-uniform transform"
            )
    return out


def _duplicate_warnings(instances: list) -> list[str]:
    seen: dict[tuple, str] = {}
    out: list[str] = []
    for inst in instances:
        key = (
            inst.part_id,
            tuple(round(float(v), 2) for row in inst.transform for v in row),
        )
        if key in seen:
            out.append(
                f"{inst.instance_id} and {seen[key]} share an identical "
                "transform — duplicate placement?"
            )
        else:
            seen[key] = inst.instance_id
    return out


def _overlap_scan(instances: list, parts: dict, manifest: dict) -> list[str]:
    """World-AABB overlap scan between instances of DIFFERENT parts —
    joint interpenetration is normal (warning); a mostly-buried
    non-premounted instance is a placement error (blocker)."""
    insts = {i.instance_id: i for i in instances}
    aabbs: dict[str, tuple] = {}
    for iid, inst in insts.items():
        part = parts.get(inst.part_id)
        if part is None or part.bbox_min == part.bbox_max:
            continue
        try:
            aabbs[iid] = _world_aabb(inst, part)
        except Exception:
            continue
    blockers: list[str] = []
    ids = sorted(aabbs)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ia, ib = insts[ids[i]], insts[ids[j]]
            if ia.part_id == ib.part_id:
                continue
            frac = _overlap_fraction(aabbs[ids[i]], aabbs[ids[j]])
            if frac <= 0.0:
                continue
            manifest["overlaps"].append(
                {"a": ids[i], "b": ids[j], "fraction": round(frac, 3)}
            )
            if frac >= _BURY_FRACTION:
                va = float(np.prod(np.maximum(aabbs[ids[i]][1] - aabbs[ids[i]][0], 0.0)))
                vb = float(np.prod(np.maximum(aabbs[ids[j]][1] - aabbs[ids[j]][0], 0.0)))
                buried = ia if va <= vb else ib
                if not buried.premounted:
                    blockers.append(
                        f"{buried.instance_id} is {frac * 100:.0f}% inside "
                        f"{ids[j] if buried is ia else ids[i]}'s bounding box "
                        "— reposition the buried object in Blender."
                    )
    return blockers


async def run_scene_gate(s: Any) -> tuple[list[str], dict[str, Any]]:
    """Return (blockers, manifest). The manifest is written into
    state.blender_manifest by the caller."""
    try:
        objects = await blender_tools._read_scene_objects(s)
    except BlenderError as exc:
        return [
            f"scene read-back failed — Blender unreachable ({exc}). The "
            "scene cannot be verified; restore Blender and retry."
        ], {"checked_at": time.time(), "reachable": False}

    instances, step_instances, _counts, sync_errors = blender_tools._build_instances(
        s, objects
    )
    s.state.assembly.instances = instances
    s.state.assembly.step_instances = step_instances
    s.state.assembly.status = "complete" if instances else "empty"

    manifest: dict[str, Any] = {
        "checked_at": time.time(),
        "reachable": True,
        "objects": {
            name: {
                "part": rec.get("part"),
                "step": rec.get("step"),
                "premount": rec.get("premount"),
            }
            for name, rec in objects.items()
        },
        "instances": {
            i.instance_id: {
                "part": i.part_id,
                "step": i.placed_in_step,
                "premounted": i.premounted,
                "object": i.blender_object,
            }
            for i in instances
        },
        "warnings": [],
        "overlaps": [],
        "blockers": [],
    }

    blockers = list(sync_errors)
    if not instances:
        blockers.append(
            "the Blender scene has no objects tagged cadx_part — nothing was "
            "placed yet. Model the parts, place them, tag every placed "
            "object (cadx_part/cadx_step), then call blender_sync_scene."
        )

    placed_pids = {i.part_id for i in instances}
    blockers += [
        f"registered part '{part.id}' ({part.name}) has no tagged "
        "instance in the scene — place and tag at least one, or "
        "drop the registration"
        for part in s.state.parts
        if part.id not in placed_pids
    ]
    blockers += _transform_blockers(instances)
    manifest["warnings"] += _duplicate_warnings(instances)
    blockers += _overlap_scan(instances, {p.id: p for p in s.state.parts}, manifest)

    manifest["warnings"].extend(
        f"{o['a']} vs {o['b']}: {o['fraction'] * 100:.0f}% bbox overlap"
        for o in manifest["overlaps"]
        if o["fraction"] < _BURY_FRACTION
    )
    manifest["blockers"] = blockers
    manifest["renders"] = sorted(
        p.name for p in s.state_dir.glob("media/step_*_blender.png")
    )
    manifest["blend"] = (
        str(s.state_dir / "blender" / "guide.blend")
        if (s.state_dir / "blender" / "guide.blend").exists()
        else ""
    )
    return blockers, manifest
