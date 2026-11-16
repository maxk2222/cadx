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
# solid-level confirmation: the bbox scan is only a cheap filter — a
# DN22 pipe socketed into an elbow can score >90% AABB overlap while
# interpenetrating almost none of the elbow's wall material. "Buried"
# means this share of the smaller part's VOLUME inside the other's
# solid — measured, not inferred.
_SOLID_BURY_FRACTION = 0.30
_BURY_SAMPLES = 400


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


def _load_part_mesh(part: Any) -> Any:
    """part.mesh_path -> trimesh (cached by the caller's dict)."""
    import trimesh

    m = trimesh.load(part.mesh_path)
    if hasattr(m, "to_geometry"):
        m = m.to_geometry()
    return m


def _solid_buried_fraction(small: Any, big: Any) -> float | None:
    """Share of ``small``'s volume inside ``big``'s solid — measured via
    boolean intersection volume, with interior point sampling as the
    fallback when the boolean engine refuses (non-watertight meshes).
    ``None`` = unmeasurable — callers never block on an unmeasured
    suspicion, they warn."""
    try:
        v_small = float(small.volume)
        if v_small > 1e-9:
            inter = small.intersection(big)
            if inter is not None:
                return max(0.0, float(inter.volume) / v_small)
    except Exception:
        pass
    try:
        lo, hi = small.bounds
        rng = np.random.default_rng(0)
        pts = rng.uniform(lo, hi, (_BURY_SAMPLES * 4, 3))
        inside_small = np.asarray(small.contains(pts), dtype=bool)
        core = pts[inside_small]
        if len(core) < 20:
            return None
        return float(np.mean(np.asarray(big.contains(core), dtype=bool)))
    except Exception:
        return None


def _unbury_hint(buried: tuple, other: tuple) -> str:
    """Least-escape translation for a buried instance's world AABB — the
    axis of smallest overlap, signed away from the other's center. Repair
    sessions stall on open-ended 'reposition' findings (re-measurement
    loops observed: bbox dumps re-emitted 5-6x instead of a move); a
    concrete delta converts the finding into an action. AABB clearance
    bounds but does not guarantee solid clearance — the agent must
    re-verify after moving.

    Frame: the AABBs live in glTF y-up (instance transforms are stored
    via _to_gltf_frame), the agent moves objects in Blender z-up —
    glTF (x,y,z) maps to Blender (x,z,-y)."""
    lo_b, hi_b = buried
    lo_o, hi_o = other
    best_axis, best_d = -1, float("inf")
    for ax in range(3):
        d = min(hi_b[ax], hi_o[ax]) - max(lo_b[ax], lo_o[ax])
        if 0 < d < best_d:
            best_axis, best_d = ax, d
    if best_axis < 0:
        return ""
    sign = (
        1
        if (lo_b[best_axis] + hi_b[best_axis]) >= (lo_o[best_axis] + hi_o[best_axis])
        else -1
    )
    b_axis, b_sign = ((("X", 1), ("Z", 1), ("Y", -1)))[best_axis]
    s = "+" if sign * b_sign > 0 else "-"
    return (
        f"; suggested delta: obj.location.{b_axis.lower()} {s}= "
        f"{best_d + 1.0:.0f} (Blender world axes, then re-verify)"
    )


def _overlap_scan(instances: list, parts: dict, manifest: dict) -> list[str]:
    """World-AABB overlap scan between instances of DIFFERENT parts —
    joint interpenetration is normal (warning); a mostly-buried
    non-premounted instance is a placement error (blocker). Bbox only
    SUSPECTS burial — the verdict needs the measured solid overlap,
    because socketed joints legitimately fill a neighbor's bounding box."""
    insts = {i.instance_id: i for i in instances}
    aabbs: dict[str, tuple] = {}
    meshes: dict[str, Any] = {}
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
            entry = {"a": ids[i], "b": ids[j], "fraction": round(frac, 3)}
            manifest["overlaps"].append(entry)
            if frac >= _BURY_FRACTION:
                va = float(np.prod(np.maximum(aabbs[ids[i]][1] - aabbs[ids[i]][0], 0.0)))
                vb = float(np.prod(np.maximum(aabbs[ids[j]][1] - aabbs[ids[j]][0], 0.0)))
                buried = ia if va <= vb else ib
                other = ib if buried is ia else ia
                if buried.premounted:
                    continue
                solid = None
                try:
                    for inst in (buried, other):
                        if inst.instance_id not in meshes:
                            part = parts[inst.part_id]
                            m = _load_part_mesh(part).copy()
                            m.apply_transform(np.asarray(inst.transform, dtype=float))
                            meshes[inst.instance_id] = m
                    solid = _solid_buried_fraction(
                        meshes[buried.instance_id], meshes[other.instance_id]
                    )
                except Exception:
                    solid = None
                entry["solid_fraction"] = round(solid, 3) if solid is not None else None
                if solid is not None and solid >= _SOLID_BURY_FRACTION:
                    blockers.append(
                        f"{buried.instance_id}: {solid * 100:.0f}% of its "
                        f"volume sits inside {other.instance_id}'s solid "
                        f"(bbox overlap {frac * 100:.0f}%) — reposition the "
                        "buried object in Blender"
                        + _unbury_hint(aabbs[buried.instance_id], aabbs[other.instance_id])
                        + "."
                    )
                elif solid is None:
                    manifest["warnings"].append(
                        f"{buried.instance_id} vs {other.instance_id}: "
                        f"{frac * 100:.0f}% bbox overlap but the solid-level "
                        "check could not measure — verify the placement "
                        "visually in the render."
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
        + (
            f" (solid check: only {o['solid_fraction'] * 100:.0f}% real "
            "interpenetration — joint, not burial)"
            if o.get("solid_fraction") is not None
            else ""
        )
        for o in manifest["overlaps"]
        if o["fraction"] < _BURY_FRACTION
        or (
            o.get("solid_fraction") is not None
            and o["solid_fraction"] < _SOLID_BURY_FRACTION
        )
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
