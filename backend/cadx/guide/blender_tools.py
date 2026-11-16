"""Blender-mode guide tools — the free-build mission's Blender surface.

The agent models and places geometry with full bpy freedom
(``blender_exec``), verifies visually (``blender_screenshot``), and
declares durable facts through thin tools that translate between the
Blender scene and GuideState:

- ``blender_register_part`` exports named objects -> normalized mm GLB ->
  GuidePart(source="blender") — no port logic, bbox only.
- ``blender_sync_scene`` reads objects tagged ``cadx_part``/``cadx_step``/
  ``cadx_premount`` -> state.assembly.instances (scene = placement truth).
- ``blender_render_step`` renders a deterministic cumulative step view
  (hides objects tagged beyond step N and untagged MESH clutter).
- ``blender_screenshot`` / ``blender_save`` / ``blender_export_scene`` /
  ``blender_ping`` round out the surface.

Unit contract: 1 Blender unit = 1 mm. ``blender_exec`` enforces
``scene.unit_settings.scale_length = 0.001`` before every call so the
glTF exporter (meters) and normalize_mesh_units (m->mm) compose into an
exact mm pipeline, and matrix_world values ARE millimeters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as _np

from cadx.guide import blender_client
from cadx.guide.blender_client import BlenderError
from cadx.guide.part_tools import _int_or_none, _slug
from cadx.guide.state import AssemblyInstance, GuidePart
from cadx.guide.tools import _arr, _fn, _num, _s
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

log = get_logger(__name__)

_EXEC_STDOUT_CAP = 4000
_CADX_TAG = "__CADX__"

# Frame contract: Blender is Z-up, glTF (and thus every mesh_path trimesh
# loads) is Y-up. Instance transforms, bboxes and the solver all share
# the glTF frame, so Blender matrices enter state only after basis
# conjugation M_gltf = C @ M_blender @ C.T (C is orthogonal -> C^-1 = C.T).
_BLENDER_TO_GLTF = _np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _to_gltf_frame(matrix_blender) -> list[list[float]]:
    m = _np.asarray(matrix_blender, dtype=float)
    return (_BLENDER_TO_GLTF @ m @ _BLENDER_TO_GLTF.T).tolist()


_CHECKPOINT_TOOLS = {"blender_sync_scene", "blender_register_part", "blender_save"}
_CHECKPOINT_WARN = 40
_CHECKPOINT_ESCALATE = 80


def _calls_since_checkpoint(s: GuideToolSession) -> int:
    """Tool calls since the last durable checkpoint — scans the
    append-only tool log backwards until a register/sync/save call. The
    current in-flight call is not yet logged, so it counts too."""
    n = 0
    for entry in reversed(s.tool_log):
        if entry.get("name") in _CHECKPOINT_TOOLS:
            break
        n += 1
    return n


def _checkpoint_nudge(s: GuideToolSession) -> str | None:
    """Escalating reminder appended to blender_exec results: unsynced
    modeling work survives only in Blender's memory — a session that
    dies at budget loses it all, so checkpoint early and often."""
    n = _calls_since_checkpoint(s)
    if n < _CHECKPOINT_WARN:
        return None
    if n >= _CHECKPOINT_ESCALATE:
        return (
            f"URGENT: {n} tool calls with NO durable checkpoint — everything "
            "you modeled exists only in Blender's memory and will be LOST if "
            "this session ends. STOP refining: tag every placed object "
            "(cadx_part/cadx_step) and call blender_sync_scene NOW — it "
            "auto-registers tagged parts into durable state."
        )
    return (
        f"checkpoint reminder: {n} tool calls without syncing. Call "
        "blender_sync_scene soon — it auto-registers tagged parts and "
        "rebuilds durable instance state from the live scene."
    )


def _wait_for_file(
    path: Path, expected_bytes: int | None = None, timeout_s: float = 15.0
) -> bool:
    """Blender writes on the HOST; the worker reads through the shared
    mount, where visibility is eventually consistent — exists() can pass
    while only the file header has synced (trimesh then dies on a
    'chunk was not expected length' partial read). Wait until the size
    matches the exporter-reported byte count, or stabilizes nonzero when
    no count is known (renders/screenshots/saves)."""
    deadline = time.monotonic() + timeout_s
    last = -1
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        if expected_bytes is not None:
            if size == expected_bytes:
                return True
        elif size > 0 and size == last:
            return True
        last = size
        time.sleep(0.15)
    try:
        return path.stat().st_size > 0
    except OSError:
        return False

# enforced before every agent exec — the mm contract is a gate, not a
# prompt wish: 1 BU = 1 mm, scale_length makes the glTF exporter emit
# meters so normalize_mesh_units lands on exact mm
_UNIT_PREAMBLE = (
    "import bpy\n"
    "bpy.context.scene.unit_settings.system = 'METRIC'\n"
    "bpy.context.scene.unit_settings.scale_length = 0.001\n"
)

_SCENE_SYNC_CODE = (
    "import bpy, json\n"
    "out = {}\n"
    "for o in bpy.context.scene.objects:\n"
    "    tag = o.get('cadx_part')\n"
    "    if not tag:\n"
    "        continue\n"
    "    m = o.matrix_world\n"
    "    out[o.name] = {\n"
    "        'part': str(tag),\n"
    "        'inst': str(o.get('cadx_inst') or ''),\n"
    "        'step': int(o.get('cadx_step') or 0),\n"
    "        'premount': bool(o.get('cadx_premount') or False),\n"
    "        'matrix': [[float(m[i][j]) for j in range(4)] for i in range(4)],\n"
    "    }\n"
    f"print({_CADX_TAG!r} + json.dumps(out))\n"
)

# world bbox + matrix_world for the named objects — the scale guard and
# the local-frame transform bake both need scene truth, not file claims
_OBJ_INFO_TMPL = (
    "import bpy, json\n"
    "from mathutils import Vector\n"
    "names = %s\n"
    "mins=[1e30]*3; maxs=[-1e30]*3; found={}\n"
    "for n in names:\n"
    "    o = bpy.data.objects.get(n)\n"
    "    if o is None:\n"
    "        continue\n"
    "    m = o.matrix_world\n"
    "    found[n] = [[float(m[i][j]) for j in range(4)] for i in range(4)]\n"
    "    for c in o.bound_box:\n"
    "        w = m @ Vector(c)\n"
    "        for i in range(3):\n"
    "            mins[i] = min(mins[i], w[i]); maxs[i] = max(maxs[i], w[i])\n"
    f"print({_CADX_TAG!r} + json.dumps(dict(found=found, min=mins, max=maxs)))\n"
)


def _blender_dir(s: GuideToolSession) -> Path:
    d = s.state_dir / "blender"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cadx_payload(stdout: str) -> dict[str, Any]:
    """Parse the __CADX__<json> line a bpy script printed."""
    for line in reversed(str(stdout).splitlines()):
        line = line.strip()
        if line.startswith(_CADX_TAG):
            try:
                return json.loads(line[len(_CADX_TAG) :])
            except json.JSONDecodeError:
                return {}
    return {}


def _err(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)[:1500]}


async def blender_ping(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    client = blender_client.probe_client()
    if client is None:
        return {
            "ok": False,
            "error": (
                f"Blender MCP not reachable on port {blender_client.blender_port()} "
                "— start Blender with the BlenderMCP addon enabled (the socket "
                "server auto-starts when a .blend loads), then retry. You may "
                "ask_user to notify the operator while parked."
            ),
        }
    try:
        info = client.get_scene_info()
    except BlenderError as exc:
        return _err(exc)
    return {
        "ok": True,
        "host": client.host,
        "port": client.port,
        "scene": {k: info[k] for k in list(info)[:12]},
    }


async def blender_exec(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    code = str(a.get("code") or "").strip()
    if not code:
        return {"ok": False, "error": "code is required"}
    try:
        client = blender_client.get_client()
        timeout = min(float(a.get("timeout_s") or 120.0), 300.0)
        out = client.execute_code(_UNIT_PREAMBLE + code, timeout_s=timeout)
    except BlenderError as exc:
        return _err(exc)
    truncated = len(out) > _EXEC_STDOUT_CAP
    result: dict[str, Any] = {
        "ok": True,
        "output": out[:_EXEC_STDOUT_CAP],
        "truncated": truncated,
    }
    nudge = _checkpoint_nudge(s)
    if nudge:
        result["checkpoint"] = nudge
    return result


async def blender_screenshot(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    try:
        client = blender_client.get_client()
        dst = _blender_dir(s) / f"shot_{int(time.time())}.png"
        res = client.viewport_screenshot(blender_client.to_host_path(dst))
    except BlenderError as exc:
        return _err(exc)
    expected = res.get("bytes") if isinstance(res, dict) else None
    if not _wait_for_file(dst, expected_bytes=expected):
        return {
            "ok": False,
            "error": f"Blender reported success but {dst.name} did not appear "
            f"in the shared data dir (host write: {res!r}) — check "
            "CADX_DATA_HOST_ROOT",
        }
    return s._images_result([str(dst)], path=str(dst))


async def blender_register_part(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    """Export named Blender objects -> normalized GLB -> GuidePart.

    The objects' world transform is baked out: the mesh is stored in the
    master's LOCAL frame (inverse matrix_world applied), so placed
    instance transforms stay absolute scene truth.
    """
    pid = _slug(str(a.get("part_id", "")))
    if not pid:
        return {"ok": False, "error": "part_id must be a non-empty slug"}
    name = str(a.get("name") or "").strip() or pid
    quantity = _int_or_none(a.get("quantity"))
    names = [str(n) for n in (a.get("object_names") or []) if str(n).strip()]
    if not names:
        return {"ok": False, "error": "object_names (Blender objects to export) is required"}
    existing = next((p for p in s.state.parts if p.id == pid), None)
    if existing is not None:
        return {
            "ok": False,
            "error": f"part_id '{pid}' is already registered ({existing.name}) "
            "— alias_part name variants instead of re-registering",
        }
    part, missing, err = await _export_register(s, pid, name, quantity, names)
    if err is not None:
        return err
    return {
        "ok": True,
        "part_id": pid,
        "mesh_path": part.mesh_path,
        "bbox": [part.bbox_min, part.bbox_max],
        "missing_objects": missing or None,
        "note": "tag every placed instance object with "
        "obj['cadx_part']='{pid}' (plus cadx_step/cadx_premount) so "
        "blender_sync_scene can bind them".replace("{pid}", pid),
    }


async def _export_register(
    s: GuideToolSession, pid: str, name: str, quantity: int | None, names: list[str]
) -> tuple[GuidePart | None, list[str], dict[str, Any] | None]:
    """Export Blender objects -> normalized GLB -> appended GuidePart.

    Returns (part, missing_object_names, error_dict) — exactly one of
    part / error is set. Shared by the explicit register tool and the
    sync auto-registration so both bake frames, wait for the synced file
    and run the scale guard identically.
    """
    try:
        client = blender_client.get_client()
        info_out = client.execute_code(_OBJ_INFO_TMPL % json.dumps(names))
        info = _cadx_payload(info_out)
        found = info.get("found") or {}
        if not found:
            return None, [], {
                "ok": False,
                "error": f"none of {names} exists in the Blender scene "
                "(bpy.data.objects has no such object) — check the exact "
                "object names via blender_exec print(list(bpy.data.objects))",
            }
        missing = [n for n in names if n not in found]
        glb_path = _blender_dir(s) / f"{pid}.glb"
        res = client.export_scene(
            blender_client.to_host_path(glb_path),
            object_names=list(found),
        )
    except BlenderError as exc:
        return None, [], _err(exc)
    expected = res.get("bytes") if isinstance(res, dict) else None
    if not _wait_for_file(glb_path, expected_bytes=expected):
        return None, missing, {
            "ok": False,
            "error": "export_scene returned success but the GLB never "
            "reached the shared data dir — check CADX_DATA_HOST_ROOT",
        }

    import trimesh

    from cadx.guide.part_tools import _guide_mesh

    try:
        m = trimesh.load(glb_path)
        if hasattr(m, "to_geometry"):
            m = m.to_geometry()
        # bake inverse of the FIRST object's matrix_world -> local frame;
        # the mesh lives in glTF coords, so the matrix converts first
        first_matrix = _np.asarray(next(iter(found.values())), dtype=float)
        m.apply_transform(_np.linalg.inv(_np.asarray(_to_gltf_frame(first_matrix))))
        m.export(str(glb_path))
    except Exception as exc:
        return None, missing, {
            "ok": False,
            "error": f"exported GLB failed to load: {exc!r}"[:400],
        }

    # scale guard — the exported mesh's extents must agree with the
    # scene-side world extents within 15%; a m/mm mixup blows past that
    scene_min, scene_max = info.get("min"), info.get("max")
    if scene_min and scene_max:
        scene_dims = [float(scene_max[i]) - float(scene_min[i]) for i in range(3)]
        glb_dims = [float(v) for v in (m.bounds[1] - m.bounds[0])]
        for sd, gd in zip(scene_dims, glb_dims, strict=True):
            if sd > 1e-6 and not (sd / 1.15 <= gd <= sd * 1.15):
                return None, missing, {
                    "ok": False,
                    "error": f"scale guard: Blender extents {scene_dims} vs "
                    f"exported GLB extents {glb_dims} disagree — verify the "
                    "1 BU = 1 mm contract",
                }
    mesh_path = _guide_mesh(s, glb_path)
    part = GuidePart(
        id=pid,
        name=name,
        source="blender",
        mesh_path=mesh_path,
        ports=[],
        quantity=quantity,
    )
    # bbox only — blender mode has no port pipeline; organic meshes must
    # not die on a cylinder heuristic they never needed
    try:
        tm = trimesh.load(mesh_path)
        if hasattr(tm, "to_geometry"):
            tm = tm.to_geometry()
        part.bbox_min = [float(v) for v in tm.bounds[0]]
        part.bbox_max = [float(v) for v in tm.bounds[1]]
        part.status = "analyzed"
    except Exception as exc:
        part.status = "error"
        part.error = str(exc)[:300]
    s.state.parts.append(part)
    return part, missing, None


async def _read_scene_objects(s: GuideToolSession) -> dict[str, dict]:
    """Tagged Blender objects -> {object_name: {part, inst, step, premount,
    matrix}}. Shared by the sync tool and the scene gate so both bind the
    same scene truth. Raises BlenderError on transport failures."""
    client = blender_client.get_client()
    out = client.execute_code(_SCENE_SYNC_CODE)
    return _cadx_payload(out)


def _build_instances(
    s: GuideToolSession, objects: dict[str, dict]
) -> tuple[list[AssemblyInstance], dict[str, list[str]], dict[str, int], list[str]]:
    """Scene objects -> (instances, step_instances, per_part_counts, errors).

    Objects keep their Blender name in ``blender_object`` so the manifest
    records which scene entity each instance came from.
    """
    part_ids = {p.id for p in s.state.parts}
    instances: list[AssemblyInstance] = []
    step_instances: dict[str, list[str]] = {}
    errors: list[str] = []
    counts: dict[str, int] = {}
    for obj_name in sorted(objects):
        rec = objects[obj_name]
        pid = _slug(str(rec.get("part") or ""))
        if pid not in part_ids:
            errors.append(
                f"object '{obj_name}' is tagged cadx_part='{rec.get('part')}' "
                f"but no such registered part exists — register it first "
                f"(blender_register_part) or fix the tag"
            )
            continue
        matrix = rec.get("matrix")
        if not (isinstance(matrix, list) and len(matrix) == 4):
            errors.append(f"object '{obj_name}' returned no usable matrix_world")
            continue
        counts[pid] = counts.get(pid, 0) + 1
        iid = str(rec.get("inst") or "").strip() or f"{pid}#{counts[pid]}"
        step = int(rec.get("step") or 0)
        inst = AssemblyInstance(
            part_id=pid,
            instance_id=iid,
            transform=_to_gltf_frame(matrix),
            placed_in_step=step,
            premounted=bool(rec.get("premount") or step == 0),
            blender_object=obj_name,
        )
        instances.append(inst)
        if step > 0:
            step_instances.setdefault(str(step), []).append(iid)
    return instances, step_instances, counts, errors


async def blender_sync_scene(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    """Read tagged scene objects -> state.assembly.instances.

    Idempotent: the whole instance list is rebuilt from scene truth each
    call — stale entries the agent deleted in Blender disappear here.
    Tagged cadx_part slugs that are not registered yet get auto-registered
    (the tag is the binding contract — registration follows from it, not
    from a separately remembered tool call): objects sharing one
    ``cadx_inst`` export together as that part's master geometry.
    """
    try:
        objects = await _read_scene_objects(s)
    except BlenderError as exc:
        return _err(exc)

    from cadx.guide.part_tools import _is_context

    auto: list[str] = []
    auto_errors: list[str] = []
    part_ids = {p.id for p in s.state.parts}
    tagged: dict[str, dict[str, list[str]]] = {}
    for obj_name, rec in objects.items():
        pid = _slug(str(rec.get("part") or ""))
        if pid:
            inst_key = str(rec.get("inst") or obj_name)
            tagged.setdefault(pid, {}).setdefault(inst_key, []).append(obj_name)
    for pid in sorted(tagged):
        if pid in part_ids:
            continue
        groups = tagged[pid]
        if _is_context(s, pid):
            auto_errors.append(
                f"'{pid}' is declared context but {sum(len(g) for g in groups.values())} "
                "object(s) are tagged cadx_part with it — retag with a real "
                "part_id or undeclare the context name"
            )
            continue
        # first instance group (sorted) supplies the master geometry —
        # multi-object instances (shared cadx_inst) export together
        first_key = sorted(groups)[0]
        qty = len(groups)
        _part_out, _missing, err = await _export_register(
            s, pid, pid.replace("_", " "), qty, sorted(groups[first_key])
        )
        if err is not None:
            auto_errors.append(f"auto-register '{pid}' failed: {err.get('error')}")
            continue
        auto.append(pid)
        part_ids.add(pid)

    instances, step_instances, counts, errors = _build_instances(s, objects)
    s.state.assembly.instances = instances
    s.state.assembly.step_instances = step_instances
    s.state.assembly.status = "complete" if instances else "empty"
    errors = auto_errors + errors
    if errors:
        return {
            "ok": False,
            "error": "; ".join(errors[:8]),
            "synced": len(instances),
            "auto_registered": auto or None,
            "note": "these bindings failed — the scene tag must name a "
            "registered part_id and carry a finite matrix_world",
        }
    return {
        "ok": True,
        "instances": len(instances),
        "per_part": counts,
        "per_step": {k: len(v) for k, v in sorted(step_instances.items())},
        "auto_registered": auto or None,
    }


async def blender_render_step(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    """Deterministic cumulative step render: every tagged object with
    cadx_step <= N renders; untagged MESH objects and later steps hide."""
    step_no = int(a.get("step") or 0)
    if step_no <= 0:
        return {"ok": False, "error": "step must be >= 1"}
    try:
        client = blender_client.get_client()
        media = s.state_dir / "media"
        media.mkdir(parents=True, exist_ok=True)
        dst = media / f"step_{step_no:02d}_blender.png"
        host_dst = blender_client.to_host_path(dst)
        code = (
            "import bpy\n"
            f"step_n = {step_no}\n"
            "if bpy.context.scene.camera is None:\n"
            "    raise RuntimeError('no camera in the scene — create one and "
            "frame the assembly first')\n"
            "for o in bpy.context.scene.objects:\n"
            "    if o.type == 'MESH':\n"
            "        tag = o.get('cadx_part')\n"
            "        st = int(o.get('cadx_step') or 0)\n"
            "        o.hide_render = tag is None or st > step_n\n"
            f"bpy.context.scene.render.filepath = {host_dst!r}\n"
            "bpy.ops.render.render(write_still=True)\n"
        )
        client.execute_code(code, timeout_s=300.0)
    except BlenderError as exc:
        return _err(exc)
    if not _wait_for_file(dst):
        return {
            "ok": False,
            "error": "render reported success but the PNG never reached "
            "the shared data dir — check CADX_DATA_HOST_ROOT",
        }
    step = next((x for x in s.state.steps if x.number == step_no), None)
    if step is not None:
        step.media_image = str(dst)
    return s._images_result([str(dst)], path=str(dst), step=step_no)


async def blender_save(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    try:
        client = blender_client.get_client()
        dst = _blender_dir(s) / "guide.blend"
        host_dst = blender_client.to_host_path(dst)
        client.execute_code(
            f"import bpy\nbpy.ops.wm.save_as_mainfile(filepath={host_dst!r})\n",
            timeout_s=120.0,
        )
    except BlenderError as exc:
        return _err(exc)
    if not _wait_for_file(dst):
        return {
            "ok": False,
            "error": "save_as_mainfile reported success but the .blend "
            "never reached the shared data dir — check CADX_DATA_HOST_ROOT",
        }
    return {"ok": True, "blend_path": str(dst)}


async def blender_export_scene(s: GuideToolSession, a: dict[str, Any]) -> dict[str, Any]:
    """Whole-scene GLB — the assembled model as one deliverable artifact."""
    try:
        client = blender_client.get_client()
        dst = _blender_dir(s) / "assembly_blender.glb"
        res = client.export_scene(blender_client.to_host_path(dst))
    except BlenderError as exc:
        return _err(exc)
    expected = res.get("bytes") if isinstance(res, dict) else None
    if not _wait_for_file(dst, expected_bytes=expected):
        return {
            "ok": False,
            "error": "export returned success but the GLB never reached "
            "the shared data dir — check CADX_DATA_HOST_ROOT",
        }
    return {"ok": True, "glb_path": str(dst)}


SCHEMAS: dict[str, dict[str, Any]] = {
    "blender_ping": _fn(
        "blender_ping",
        "Check whether the Blender MCP socket is reachable and return "
        "basic scene info. Call before relying on other blender tools; "
        "an error means Blender must be started first.",
        {},
        [],
    ),
    "blender_exec": _fn(
        "blender_exec",
        "Run arbitrary Blender Python (bpy) — your free modeling channel. "
        "stdout is captured and returned (print() to see values). The mm "
        "unit contract (1 BU = 1 mm) is enforced automatically. Tag every "
        "placed object with obj['cadx_part']='part_id', obj['cadx_step']="
        "the step number placing it (0 = already mounted at start), and "
        "obj['cadx_premount']=True for initial-state hardware.",
        {"code": _s("Blender Python source (bpy available)"),
         "timeout_s": _num("max seconds to wait, <=300")},
        ["code"],
    ),
    "blender_screenshot": _fn(
        "blender_screenshot",
        "Render the current 3D viewport offscreen and return the image — "
        "your visual self-check after every modeling/placement change.",
        {},
        [],
    ),
    "blender_register_part": _fn(
        "blender_register_part",
        "Export named Blender objects as a guide part: GLB is written to "
        "the guide dir, mm-normalized, and registered with its bounding "
        "box. The first object's world transform is baked OUT so the mesh "
        "lands in local (origin) space. One call per distinct part type, "
        "not per instance.",
        {
            "part_id": _s("short slug id, e.g. 'pipe_l70'"),
            "name": _s("display name for the guide"),
            "quantity": _num("total copies the kit ships (Stückliste Menge)"),
            "object_names": _arr(
                _s("object name"), "exact bpy.data.objects names to export"
            ),
        },
        ["part_id", "object_names"],
    ),
    "blender_sync_scene": _fn(
        "blender_sync_scene",
        "Read back every object tagged cadx_part and rebuild the guide's "
        "instance list from scene truth (instance_id, transform, step, "
        "premount). Tagged part ids that are not registered yet are "
        "auto-registered here (objects sharing a cadx_inst export as one "
        "part; quantity = distinct instance count) — use "
        "blender_register_part explicitly only for custom names/quantities "
        "or a specific master. Call after tagging/placing and before "
        "done_mission — the manifest gate re-reads the same data.",
        {},
        [],
    ),
    "blender_render_step": _fn(
        "blender_render_step",
        "Render the cumulative assembly state at a step: tagged objects "
        "visible up to that step, later-step and untagged MESH objects "
        "hidden. Requires a camera in the scene. The PNG becomes the "
        "step's media image.",
        {"step": _num("step number to render")},
        ["step"],
    ),
    "blender_save": _fn(
        "blender_save",
        "Save the current scene as guide.blend inside the guide dir — "
        "durable crash recovery and a downloadable artifact.",
        {},
        [],
    ),
    "blender_export_scene": _fn(
        "blender_export_scene",
        "Export the entire Blender scene as assembly_blender.glb — the "
        "single-file assembled model artifact.",
        {},
        [],
    ),
}

HANDLERS: dict[str, Any] = {
    "blender_ping": blender_ping,
    "blender_exec": blender_exec,
    "blender_screenshot": blender_screenshot,
    "blender_register_part": blender_register_part,
    "blender_sync_scene": blender_sync_scene,
    "blender_render_step": blender_render_step,
    "blender_save": blender_save,
    "blender_export_scene": blender_export_scene,
}
