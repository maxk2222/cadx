"""Executed inside the sandboxed subprocess: runs user model.py and exports a BREP.

Usage: python -I runner_entry.py <workdir>
Reads:   <workdir>/model.py, <workdir>/input.json
Writes:  <workdir>/result.json, <workdir>/model.brep, optional <workdir>/flat_profile.brep
Always exits 0 — the parent detects failures via result.json / missing files.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import io
import json
import sys
import traceback
from pathlib import Path

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


def main() -> int:  # noqa: C901
    workdir = Path(sys.argv[1])
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
        return import_brep(ref)

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
    }

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
    finally:
        sys.stdout = sys.__stdout__

    result["stdout"] = stdout_buf.getvalue()[-4000:]

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

        export_brep(shape, str(workdir / "model.brep"))

        flat = namespace.get("flat_profile")
        if isinstance(flat, Shape):
            export_brep(flat, str(workdir / "flat_profile.brep"))

        # count real (non-simple) thread objects present in the namespace
        thread_count = 0
        bom: dict[tuple, dict] = {}
        for v in namespace.values():
            cls_name = type(v).__name__
            mod = type(v).__module__ or ""
            if (
                mod.startswith("bd_warehouse.thread")
                and cls_name.endswith("Thread")
                and not getattr(v, "simple", True)
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

        params_used = namespace.get("PARAMS", {})
        try:
            params_used = json.loads(json.dumps(params_used, default=str))
        except Exception:
            params_used = {}

        result.update(
            {
                "ok": True,
                "params_used": params_used,
                "thread_count": thread_count,
                "solid_count": len(solids),
                "bom": sorted(
                    bom.values(),
                    key=lambda e: (e["part_class"], e["type"] or "", e["size"] or ""),
                ),
            }
        )
    except Exception:
        result["error"] = traceback.format_exc(limit=20)
        result["traceback"] = "\n".join(
            ln for ln in str(result["error"]).splitlines() if "runner_entry" not in ln
        )[-4000:]

    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
