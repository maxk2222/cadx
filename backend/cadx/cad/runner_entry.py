"""Executed inside the sandboxed subprocess: runs user model.py and exports a BREP.

Usage: python -I runner_entry.py <workdir>
Reads:   <workdir>/model.py, <workdir>/input.json
Writes:  <workdir>/result.json, <workdir>/model.brep, optional <workdir>/flat_profile.brep
Always exits 0 — the parent detects failures via result.json / missing files.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import io
import json
import re
import sys
import traceback
from pathlib import Path

# Resolve `cadx` against THIS tree before any cadx import: under `python -I`
# PYTHONPATH is ignored and a stale site-packages/editable install would
# otherwise shadow the live code (mixed-mount eval containers hit exactly that)
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

ALLOWED_MODULES = {
    "build123d",
    "bd_warehouse",
    "cadlib",
    "math",
    "numpy",
    "np",
    "typing",
    "dataclasses",
}

# In-namespace measurement primitives (measure/find_holes/cross_sections/...)
# for generated code — a trusted module with no cadx imports, same pattern as
# mesh_boolean: imported here (module top runs from a runner frame, or before
# the guard exists at all) and injected into the exec namespace below. The
# cold-script path (`python -I runner_entry.py`) may lack `cadx` on sys.path
# until main() re-adds the cad/ dir — resolved there as fallback.
try:
    from cadx.cad import inspect_tools as _inspect
except Exception:
    # any failure here must not take the runner down — resolved/retried in
    # main(); generated code then gets self-explaining stubs instead
    _inspect = None


class _ImportGuard(importlib.abc.MetaPathFinder):
    """Second line of defence: block imports issued *by generated code* outside
    the allow-list. Imports triggered internally by build123d/bd_warehouse are
    untouched (the AST check in allowlist.py is the primary barrier)."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in ALLOWED_MODULES or fullname in sys.modules:
            return
        frame = sys._getframe()
        try:
            while frame is not None:
                fname = frame.f_code.co_filename
                if fname == "model.py":
                    raise ImportError(
                        f"module '{fullname}' is not allowed in generated model code "
                        f"(allowed: {sorted(ALLOWED_MODULES)})"
                    )
                if "runner_entry.py" in fname:
                    break
                frame = frame.f_back
        finally:
            del frame
        return


def _unavailable_mesh_solid():
    raise RuntimeError("load_mesh_solid() is only available in hybrid mode")


def _geometry_snapshot(namespace: dict) -> list[dict]:
    """Best-effort geometry inventory of the failed namespace — where the
    built solids actually ENDED UP. This is the single most valuable signal
    for a repair loop that can't see the render: `Pos * Box` builder traps
    leave a primitive at the origin while the model believes it moved."""
    try:
        from build123d import BuildPart, Compound, Part, Shape
    except Exception:
        return []
    out: list[dict] = []
    for name, v in namespace.items():
        if len(out) >= 6 or name.startswith("_"):
            continue
        try:
            shape = None
            if isinstance(v, BuildPart):
                shape = v.part
            elif isinstance(v, Part | Compound | Shape):
                shape = v
            if shape is None:
                continue
            bb = shape.bounding_box()
            out.append(
                {
                    "name": name,
                    "bbox_min": [round(bb.min.X, 1), round(bb.min.Y, 1), round(bb.min.Z, 1)],
                    "bbox_max": [round(bb.max.X, 1), round(bb.max.Y, 1), round(bb.max.Z, 1)],
                    "solids": len(shape.solids()),
                    "volume_mm3": round(float(shape.volume), 1),
                }
            )
        except Exception:
            continue  # a degenerate shape must never mask the real error
    return out


def _json_records(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    try:
        serialised = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return []
    return [row for row in serialised if isinstance(row, dict)]


def _mesh_profile(path: Path, bins: int = 13) -> dict[str, list[list[float]]]:
    """Sample the mesh solid's silhouette per axis: for each bin along x/y
    return [axis_pos, min_a, max_a, min_b, max_b] of the two remaining axes —
    where material actually is, unlike the bbox which hides asymmetry and
    curvature (a handle spine can sweep several mm off the bbox centerline).
    Vertex sampling is approximate but dense enough for feature placement."""
    if not path.exists():
        return {}
    try:
        from build123d import import_brep

        body = import_brep(path)
        verts = [(v.X, v.Y, v.Z) for v in body.vertices()]
    except Exception:
        return {}
    if not verts:
        return {}
    out: dict[str, list[list[float]]] = {}
    for axis in range(2):  # x and y are the placement axes for through-features
        others = [i for i in range(3) if i != axis]
        lo = min(v[axis] for v in verts)
        hi = max(v[axis] for v in verts)
        width = (hi - lo) / bins or 1.0
        rows: list[list[float]] = []
        for b in range(bins):
            b_lo = lo + b * width
            b_hi = b_lo + width
            sel = [v for v in verts if b_lo <= v[axis] < (b_hi if b < bins - 1 else hi + 1e-9)]
            if not sel:
                continue
            rows.append(
                [
                    round((b_lo + b_hi) / 2, 2),
                    round(min(v[others[0]] for v in sel), 2),
                    round(max(v[others[0]] for v in sel), 2),
                    round(min(v[others[1]] for v in sel), 2),
                    round(max(v[others[1]] for v in sel), 2),
                ]
            )
        out["xy"[axis]] = rows
    return out


def main(workdir: Path | None = None) -> int:  # noqa: C901
    """Run <workdir>/model.py. `workdir` may be passed directly (warm fork-server)
    or via sys.argv[1] (cold `python -I runner_entry.py <workdir>`)."""
    workdir = workdir or Path(sys.argv[1])
    input_path = workdir / "input.json"
    result_path = workdir / "result.json"

    try:
        spec = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        result_path.write_text(
            json.dumps({"ok": False, "error": f"bad input.json: {e}"}), encoding="utf-8"
        )
        return 0

    params_override = spec.get("params_override") or {}
    detail_mode = str(spec.get("detail_mode") or "preview")
    if detail_mode not in ("preview", "export"):
        result_path.write_text(
            json.dumps({"ok": False, "error": f"invalid detail_mode '{detail_mode}'"}),
            encoding="utf-8",
        )
        return 0
    references = {r["asset_id"]: r["path"] for r in spec.get("references", [])}
    mesh_solid_path = spec.get("mesh_solid")
    mesh_bbox = spec.get("mesh_bbox")
    anchors = spec.get("anchors") or {}
    anchor_holes_list = anchors.get("holes") or []
    image_profile_pts = anchors.get("image_outline") or []

    sys.meta_path.insert(0, _ImportGuard())
    # python -I strips the script dir; add it back so `import cadlib` resolves.
    # Other modules on this path are still blocked by _ImportGuard.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # the package root too: `import cadx.cad.inspect_tools` must resolve to the
    # same tree this script lives in — under -I, PYTHONPATH is ignored and a
    # stale site-packages/editable install would otherwise shadow it
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # tell cadlib which detail tier to build (preview proxies vs real helix) —
    # imported from this runner frame so the import guard permits it
    try:
        import cadlib.ops as _cops_detail

        _cops_detail.set_detail_mode(detail_mode)
    except Exception:
        pass

    # module-top import may have failed when `cadx` isn't installed (bare
    # sandbox path) — resolve the sibling module now that cad/ is on sys.path.
    # The import runs from this runner frame, so the guard permits its
    # build123d/OCP imports.
    global _inspect
    if _inspect is None:
        try:
            from cadx.cad import inspect_tools as _inspect_pkg

            _inspect = _inspect_pkg
        except Exception:
            try:
                import inspect_tools as _inspect_local  # type: ignore[no-redef]

                _inspect = _inspect_local
            except Exception:
                _inspect = None

    # mesh-native boolean backend (manifold3d) — trusted module, imported from
    # this runner frame so the import guard permits trimesh/OCP. Injected into
    # cadlib.ops (the top-level `cadlib` model.py resolves) as the rescue lane
    # for mesh_cut/mesh_fuse/mesh_intersect/trim_below; also powers the
    # deterministic anchor re-cut below. Any failure leaves OCC-only behavior.
    mesh_boolean = None
    if mesh_solid_path or anchor_holes_list:
        try:
            import cadlib.ops as _cops

            try:
                from cadx.cad import mesh_boolean as _mb  # installed package
            except ImportError:
                import mesh_boolean as _mb  # type: ignore[no-redef]  sandbox path

            _mb._warmup()  # resolves lazy imports before model.py frames exist
            _cops.set_boolean_backend(_mb.boolean)
            mesh_boolean = _mb
        except Exception:
            mesh_boolean = None

    # source trimesh for the hybrid base solid — attached to the BREP import so
    # every backend boolean reuses it instead of re-tessellating the sewn
    # solid (OCC tessellation at the face cap costs minutes, the mesh is free)
    _base_mesh = None
    if mesh_boolean is not None and mesh_solid_path:
        try:
            stl_sib = Path(mesh_solid_path).resolve().with_name("model.stl")
            if not stl_sib.exists():
                # edit/params runs ship the cleaned baseline mesh under its
                # own name — without it the manifold rescue lane and the
                # deterministic anchor re-cut silently lose their operand
                stl_sib = Path(mesh_solid_path).resolve().with_name("mesh_clean.stl")
            if stl_sib.exists():
                import trimesh

                _base_mesh = trimesh.load(str(stl_sib), force="mesh", process=False)
                _base_mesh.merge_vertices()
                _base_mesh.remove_unreferenced_vertices()
        except Exception:
            _base_mesh = None

    from build123d import (
        Compound,
        Face,
        Part,
        Polyline,
        Shape,
        Solid,
        Spline,
        Vector,
        export_brep,
        import_brep,
        import_step,
        import_stl,
        make_face,
    )

    def load_reference(asset_id: str) -> Shape:
        """Load a user-provided reference STEP/STL body."""
        ref = Path(references[asset_id]).resolve()
        if not ref.exists():
            raise FileNotFoundError(f"reference asset {asset_id} not found")
        if ref.suffix.lower() in {".step", ".stp"}:
            return import_step(ref)
        if ref.suffix.lower() == ".stl":
            obj = import_stl(ref)
            if isinstance(obj, Shape):
                return obj
            return Shape(obj)
        raise ValueError(f"unsupported reference format: {ref.suffix}")

    def load_mesh_solid() -> Shape:
        """Load the version's generated mesh as a (faceted) B-Rep solid.

        Available in hybrid mode only: the mesh is pre-scaled/oriented to mm,
        rests on the XY plane and is centred at the origin. Use it for boolean
        operations with exact CAD features — not for face/edge selection.
        """
        if not mesh_solid_path:
            raise RuntimeError("load_mesh_solid() is only available in hybrid mode")
        ref = Path(mesh_solid_path).resolve()
        if not ref.exists():
            raise FileNotFoundError("mesh solid not found")
        shp = import_brep(ref)
        if _base_mesh is not None:
            with contextlib.suppress(Exception):
                mesh_boolean.attach_mesh(shp, _base_mesh)
        return shp

    mesh_profile: dict[str, list[list[float]]] = {}
    if mesh_solid_path:
        mesh_profile = _mesh_profile(Path(mesh_solid_path).resolve())

    def _spine(axis: str, pos: float) -> float:
        """Local body centreline along `axis` at `pos` (linear interpolation
        over MESH_PROFILE rows). Fallback: 0 when no profile exists."""
        rows = mesh_profile.get(axis) or []
        if not rows:
            return 0.0
        pos = min(max(pos, rows[0][0]), rows[-1][0])
        for i in range(1, len(rows)):
            if pos <= rows[i][0]:
                x0, x1 = rows[i - 1][0], rows[i][0]
                c0 = (rows[i - 1][1] + rows[i - 1][2]) / 2
                c1 = (rows[i][1] + rows[i][2]) / 2
                t = (pos - x0) / (x1 - x0) if x1 > x0 else 0.0
                return c0 + t * (c1 - c0)
        return (rows[-1][1] + rows[-1][2]) / 2

    def spine_y(x: float) -> float:
        """Y of the body's material centreline at the given x."""
        return _spine("x", x)

    def spine_x(y: float) -> float:
        """X of the body's material centreline at the given y."""
        return _spine("y", y)

    _top_z_body = None
    if mesh_solid_path:
        try:
            _top_z_body = import_brep(Path(mesh_solid_path).resolve())
        except Exception:
            _top_z_body = None

    def top_z(x: float, y: float) -> float:
        """Z of the body's top surface directly above (x, y) — the local
        surface height for anchoring countersinks, pockets and recesses.
        Falls back to MESH_BBOX z outside the silhouette."""
        z_max = mesh_bbox[2] if mesh_bbox else 0.0
        if _top_z_body is None:
            return z_max
        try:
            from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
            from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

            inter = BRepIntCurveSurface_Inter()
            inter.Init(
                _top_z_body.wrapped,
                gp_Lin(
                    gp_Pnt(float(x), float(y), z_max + 50.0),
                    gp_Dir(0.0, 0.0, -1.0),
                ),
                1e-6,
            )
            hits = []
            while inter.More():
                hits.append(inter.Pnt().Z())
                inter.Next()
            return max(hits) if hits else z_max
        except Exception:
            return z_max

    def anchor_holes() -> list[dict[str, object]]:
        """Measured hole anchors from the plan view: [{x, y, dia, index, ...}].

        When entries exist, EVERY positioned hole MUST use these coordinates —
        never guess positions from the bbox."""
        return [dict(h) for h in anchor_holes_list]

    def _smooth_ring(pts_xy: list[list[float]], window: int = 5) -> list[list[float]]:
        """Circular moving-average smoothing on a closed outline ring.

        Measured outlines carry pixel noise; a periodic Spline through every
        raw point keeps the jitter as visible faceting. A small wrap-around
        moving average (~window * point spacing ≈ 4-5 mm) smooths the noise
        while preserving real features like a finger choil.
        """
        n = len(pts_xy)
        if n <= window:
            return pts_xy
        half = window // 2
        out = []
        for i in range(n):
            xs = sum(pts_xy[(i + k) % n][0] for k in range(-half, half + 1)) / window
            ys = sum(pts_xy[(i + k) % n][1] for k in range(-half, half + 1)) / window
            out.append([xs, ys])
        return out

    def load_image_profile() -> Face:
        """Measured plan-view outline as a planar Face on the XY plane (Frame
        coords). RuntimeError when no measured top view is available."""
        if len(image_profile_pts) < 3:
            raise RuntimeError("no measured plan view available")
        pts_xy = [list(map(float, p)) for p in image_profile_pts]
        if pts_xy[0] == pts_xy[-1]:
            pts_xy = pts_xy[:-1]  # closed ring stored with repeated first point
        pts_xy = _smooth_ring(pts_xy)
        pts = [Vector(x, y, 0.0) for x, y in pts_xy]
        try:
            wire = Spline(*pts, periodic=True)
            return make_face(wire)
        except Exception:
            wire = Polyline(*pts, close=True)
            return make_face(wire)

    namespace: dict[str, object] = {
        "__name__": "__model__",
        "__PARAMS_OVERRIDE__": params_override,
        "load_reference": load_reference,
        "load_mesh_solid": load_mesh_solid if mesh_solid_path else _unavailable_mesh_solid,
        "MESH_BBOX": mesh_bbox or [],
        "MESH_PROFILE": mesh_profile,
        "spine_y": spine_y,
        "spine_x": spine_x,
        "top_z": top_z,
        "ANCHOR_HOLES": anchor_holes_list,
        "anchor_holes": anchor_holes,
        "IMAGE_PROFILE": image_profile_pts,
        "load_image_profile": load_image_profile,
        "CADX_DETAIL_MODE": detail_mode,
    }

    # measurement/inspection primitives — bare names in generated code, e.g.
    # `print(measure(part))`. A failed import degrades to stubs that explain
    # themselves instead of a bare NameError.
    if _inspect is not None:
        namespace.update(
            {
                "measure": _inspect.measure,
                "find_holes": _inspect.find_holes,
                "find_bosses": _inspect.find_bosses,
                "find_countersinks": _inspect.find_countersinks,
                "cross_sections": _inspect.cross_sections,
                "clearance": _inspect.clearance,
                "bbox_dims": _inspect.bbox_dims,
                "list_edges": _inspect.list_edges,
            }
        )
    else:

        def _unavailable_inspect(*_args, **_kwargs):
            raise RuntimeError("inspection helpers failed to load in the runner")

        for _name in (
            "measure",
            "find_holes",
            "find_bosses",
            "find_countersinks",
            "cross_sections",
            "clearance",
            "bbox_dims",
            "list_edges",
        ):
            namespace[_name] = _unavailable_inspect

    code = (workdir / "model.py").read_text(encoding="utf-8")
    stdout_buf = io.StringIO()
    result: dict[str, object] = {"ok": False}
    try:
        sys.stdout = stdout_buf  # capture prints
        compiled = compile(code, "model.py", "exec")
        exec(compiled, namespace)
    except Exception:
        result["error"] = traceback.format_exc(limit=30)
        # Trim to frames inside model.py + the exception line
        tb_lines = [ln for ln in str(result["error"]).splitlines() if "runner_entry" not in ln]
        result["traceback"] = "\n".join(tb_lines[-30:])
        result["geometry_at_failure"] = _geometry_snapshot(namespace)
    finally:
        sys.stdout = sys.__stdout__

    result["stdout"] = stdout_buf.getvalue()[-8000:]

    # `error` holds the real failure even when every traceback frame was in
    # runner_entry.py itself (helper raised) — never mask it as "no result"
    if result.get("traceback") or result.get("error"):
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    shape = namespace.get("result")
    if shape is None:
        if "result" not in namespace:
            result["error"] = "no `result` assigned"
        else:
            # `result = <something>` executed but evaluated to None — in hybrid
            # mode the classic cause is `body = body & tool` (or .common() /
            # .intersect()): OCC boolean ops on a faceted mesh return None
            # instead of raising when they produce nothing.
            suspects = [
                f"line {i}: {ln.strip()}"
                for i, ln in enumerate(code.splitlines(), 1)
                if "&" in ln or ".common(" in ln or ".intersect(" in ln
            ]
            hint = (
                " `result` is None — a boolean op likely returned an empty/None result. "
                "On faceted mesh solids `a & b`, `a.common(b)` and `a.intersect(b)` "
                "return None instead of raising; use `body -= tool` / `body += tool` "
                "(cut/fuse are mesh-safe) or check for None explicitly."
            )
            result["error"] = (
                "`result` evaluated to None"
                + hint
                + (" suspects: " + "; ".join(suspects[:4]) if suspects else "")
            )
        result["geometry_at_failure"] = _geometry_snapshot(namespace)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    try:
        if isinstance(shape, Solid):
            shape = Part(shape.wrapped)
        if not isinstance(shape, Compound | Part | Shape):
            raise TypeError(f"result is {type(shape).__name__}, expected a build123d Shape")
        solids = shape.solids()
        if len(solids) == 0:
            raise ValueError("result contains no solids")

        # hybrid mode: boolean ops on a faceted mesh solid shed sub-mm sliver
        # solids (observed: 28-solid compound, largest fragment ~94% of the
        # volume). The debris isn't a feature — collapse to the dominant
        # solid so the exported part is the intended single body, and flag
        # it so the improve loop learns to use mesh_cut(). But when even the
        # largest fragment lost most of the mesh body's volume the boolean
        # shattered the part — that is a failure, not debris.
        if mesh_solid_path and len(solids) > 1:
            from cadx.cad.cadlib.ops import largest_solid

            n = len(solids)
            dominant = largest_solid(shape)
            base_vol = getattr(_top_z_body, "volume", 0) or 0
            if base_vol > 0 and dominant.volume < base_vol * 0.3:
                raise ValueError(
                    f"boolean ops shattered the mesh solid: {n} fragments, "
                    f"largest keeps only {dominant.volume:.1f} mm^3 of the "
                    f"{base_vol:.1f} mm^3 body — wrap cuts/fuses in "
                    "cadlib.mesh_cut()/mesh_fuse() so the op falls back to "
                    "the un-cut body instead of fragmenting it"
                )
            shape = dominant
            solids = shape.solids()
            result.setdefault("warnings", []).append(
                f"result had {n} disconnected solids — collapsed to the largest; "
                "wrap mesh booleans in cadlib.mesh_cut()/mesh_fuse() to drop slivers"
            )

        # deterministic anchor re-cut: bores measured on the reference image
        # are authoritative — when generated code missed or blocked one, cut
        # it here instead of burning improve iterations on it. In pure CAD
        # mode (no mesh solid) only image-measured anchors qualify — spec
        # and mesh-derived anchors are declared intent the codegen owns;
        # cutting them blind drills through features it placed deliberately.
        recut_anchors = (
            anchor_holes_list
            if mesh_solid_path
            else [
                h
                for h in anchor_holes_list
                if isinstance(h, dict) and str(h.get("source") or "") in ("image", "image+mesh")
            ]
        )
        if mesh_boolean is not None and recut_anchors:
            try:
                shape2, applied = mesh_boolean.recut_missing_holes(shape, recut_anchors)
                if applied:
                    shape = shape2
                    solids = shape.solids()
                    result.setdefault("warnings", []).append(
                        "anchor re-cut: measured holes missing/blocked in the "
                        f"generated code were cut deterministically ({', '.join(applied)})"
                    )
            except Exception as e:
                result.setdefault("warnings", []).append(f"anchor re-cut failed: {e}")

        # hybrid result cleanup: merge coplanar facets into planar faces —
        # cheaper downstream ops, cleaner STEP; validated, keeps input on drift
        if mesh_boolean is not None and mesh_solid_path:
            shape = mesh_boolean.unify_checked(shape)

        export_brep(shape, str(workdir / "model.brep"))

        # faceted hybrid result: the attached manifold mesh IS the exact
        # geometry — ship it next to the brep so the pipeline re-attaches it
        # instead of re-tessellating tens of thousands of OCC faces (minutes)
        if mesh_boolean is not None:
            _res_mesh = getattr(shape, mesh_boolean.MESH_ATTR, None)
            if _res_mesh is not None:
                with contextlib.suppress(Exception):
                    _res_mesh.export(str(workdir / "result_mesh.glb"))

        flat = namespace.get("flat_profile")
        if isinstance(flat, Shape):
            export_brep(flat, str(workdir / "flat_profile.brep"))

        # count real (non-simple) thread objects present in the namespace —
        # bd_warehouse thread/fastener objects, cadlib cutters tagged
        # `_cadx_thread`, or a `Helix` path a declared thread was built from
        design_features = _json_records(namespace.get("CADX_FEATURES"))
        declared_threads = sum(1 for feature in design_features if feature.get("kind") == "thread")
        # loop-built threads land inside a container (`taps.append(tap)`) —
        # flatten one level so every tagged cutter counts; dedupe by id so
        # the loop variable aliasing the last one isn't double-counted
        flat_objs: list[tuple[str, object]] = []
        seen_ids: set[int] = set()
        for name, v in namespace.items():
            items = v if isinstance(v, list | tuple | set | frozenset) else (v,)
            for obj in items:
                if id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    flat_objs.append((name, obj))
        helix_paths = sum(
            1 for _name, v in flat_objs if type(v).__name__ == "Helix" and not isinstance(v, type)
        )
        thread_count = 0
        bom: dict[tuple, dict] = {}
        for name, v in flat_objs:
            cls_name = type(v).__name__
            mod = type(v).__module__ or ""
            if name != "result" and (
                (
                    not isinstance(v, type)
                    and mod.startswith("bd_warehouse.thread")
                    and cls_name.endswith("Thread")
                    and not getattr(v, "simple", True)
                )
                or (
                    not isinstance(v, type)
                    and cls_name.endswith(("Screw", "Nut"))
                    and getattr(v, "simple", True) is False
                    and getattr(v, "thread_diameter", None) is not None
                    and getattr(v, "thread_pitch", None) is not None
                )
                or (not isinstance(v, type) and getattr(v, "_cadx_thread", None) is not None)
            ):
                thread_count += 1
            if not mod.startswith("bd_warehouse"):
                continue
            # BOM row: one per distinct (class, type, size); count reflects
            # recorded hole placements when the part drove hole operations.
            # `.size` is a numeric OCP-internal on many classes — the human
            # size string lives in *_size attributes.
            size = next(
                (
                    getattr(v, a)
                    for a in (
                        "screw_size",
                        "nut_size",
                        "washer_size",
                        "insert_size",
                        "bearing_size",
                        "ring_size",
                        "key_size",
                        "size",
                    )
                    if isinstance(getattr(v, a, None), str)
                ),
                None,
            )
            part_type = next(
                (
                    getattr(v, a)
                    for a in ("fastener_type", "bearing_type", "thread_type", "part_type")
                    if isinstance(getattr(v, a, None), str)
                ),
                None,
            )
            key = (cls_name, part_type, size)
            count = len(getattr(v, "hole_locations", None) or []) or 1
            if key in bom:
                bom[key]["count"] += count
            else:
                bom[key] = {"part_class": cls_name, "type": key[1], "size": key[2], "count": count}

        if declared_threads and thread_count > declared_threads:
            # un-consumed temporaries inflate the flat scan — e.g. an
            # un-placed cutter left bound to the loop variable next to the
            # `taps` list holding all four moved copies. The part carries at
            # most the declared set; extra bookkeeping objects add none.
            thread_count = declared_threads
        if thread_count == 0:
            v = namespace.get("result")
            cls_name = type(v).__name__
            mod = type(v).__module__ or ""
            if not isinstance(v, type) and (
                (
                    mod.startswith("bd_warehouse.thread")
                    and cls_name.endswith("Thread")
                    and not getattr(v, "simple", True)
                )
                or (
                    cls_name.endswith(("Screw", "Nut"))
                    and getattr(v, "simple", True) is False
                    and getattr(v, "thread_diameter", None) is not None
                    and getattr(v, "thread_pitch", None) is not None
                )
            ):
                thread_count = 1
        if thread_count == 0 and declared_threads and helix_paths:
            # hand-rolled real thread: declared feature + Helix path left in
            # the namespace (the IsoThread-free pattern models fall back to)
            thread_count = min(declared_threads, helix_paths)
        if thread_count < declared_threads and detail_mode == "export":
            # the tag is bookkeeping, not the thread itself: a cutter used as
            # an unnamed temporary (`part - Pos(...) * tapped_hole(...)`) still
            # cut the real helical groove into the brep — count the calls
            helper_calls = len(re.findall(r"\btapped_hole\s*\(", code))
            helper_calls += len(
                re.findall(r"ThreadedHole\s*\([^)]*simple\s*=\s*False", code, re.S)
            )
            if helper_calls:
                thread_count = max(
                    thread_count, min(declared_threads, thread_count + helper_calls)
                )
        if thread_count and declared_threads and detail_mode == "export":
            # intent is not geometry: a tagged or source-counted cutter that
            # never touched the part (disjoint placement) still counts — but
            # an ISO groove always leaves BSPLINE wall faces; none means the
            # export carries no real thread whatever the bookkeeping says
            result_shape = namespace.get("result")
            try:
                has_bspline = result_shape is not None and any(
                    str(f.geom_type).endswith("BSPLINE") for f in result_shape.faces()
                )
            except Exception:
                has_bspline = True  # inspection failed — keep the counted value
            if not has_bspline:
                thread_count = 0

        params_used = namespace.get("PARAMS", {})
        try:
            params_used = json.loads(json.dumps(params_used, default=str))
        except Exception:
            params_used = {}

        declared_bom = _json_records(namespace.get("CADX_BOM"))
        if any(
            not isinstance(row.get("part_class"), str) or not isinstance(row.get("count"), int)
            for row in declared_bom
        ):
            declared_bom = []
        preview_thread_count = sum(
            1 for feature in design_features if feature.get("kind") == "thread"
        )
        result.update(
            {
                "ok": True,
                "params_used": params_used,
                "thread_count": thread_count,
                "preview_thread_count": preview_thread_count,
                "detail_mode": detail_mode,
                "design_features": design_features,
                "solid_count": len(solids),
                "bom": sorted(
                    declared_bom or bom.values(),
                    key=lambda e: (e["part_class"], e["type"] or "", e["size"] or ""),
                ),
            }
        )
    except Exception:
        result["error"] = traceback.format_exc(limit=20)
        result["traceback"] = "\n".join(
            ln for ln in str(result["error"]).splitlines() if "runner_entry" not in ln
        )[-4000:]
        result["geometry_at_failure"] = _geometry_snapshot(namespace)

    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
