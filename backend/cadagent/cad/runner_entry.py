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

    sys.meta_path.insert(0, _ImportGuard())
    # python -I strips the script dir; add it back so `import cadlib` resolves.
    # Other modules on this path are still blocked by _ImportGuard.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from build123d import (
        Compound,
        Part,
        Shape,
        Solid,
        export_brep,
        import_brep,
        import_step,
        import_stl,
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

    namespace: dict[str, object] = {
        "__name__": "__model__",
        "__PARAMS_OVERRIDE__": params_override,
        "load_reference": load_reference,
        "load_mesh_solid": load_mesh_solid if mesh_solid_path else _unavailable_mesh_solid,
        "MESH_BBOX": mesh_bbox or [],
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

    if result.get("traceback"):
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    shape = namespace.get("result")
    if shape is None:
        result["error"] = "no `result` assigned"
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
