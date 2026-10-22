"""Stage gates — declarative pass/fail checks on a StageOutcome.

Gate specs live in the suite JSON (`stages[].gates`) as
``{"type": <name>, ...kwargs}``. Two families:

- **outcome gates** — status/score/bbox/features/watertight/verify_spec/
  fem/topopt/tokens/wall time, read off the normalized outcome;
- **mesh gates** — `radial_peaks`, `volume(_fraction)`, `contains_point`,
  `bodies` — computed with trimesh on the stage's first mesh artifact
  (STL/GLB/3MF), skipped with detail when no mesh is available.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cadx.evalkit.models import GateResult, StageOutcome


def _feature_entry_ok(  # noqa: C901
    ef: dict[str, Any], feats: dict[str, Any], groups: list[Any]
) -> bool:
    """Ported feature check — hole/slot/countersink/fillet/chamfer matching
    against the measured feature dict (identical semantics to the old
    harness, incl. axis/positions/pcd/blocked handling)."""
    # elongated voids are slots — they must not satisfy a round-hole
    # expectation (a 20x6 slot would otherwise read as a big bore)
    round_groups = [g for g in groups if not g.get("elongated")]

    ftype = str(ef.get("type", "")).lower()
    want = int(ef.get("count") or 1)

    def _axis_ok(g: dict[str, Any], want_axis: str | None) -> bool:
        if want_axis is None:
            return True
        got = str(g.get("axis") or "").lstrip("+-").lower()
        return got == want_axis

    def _positions_ok(centers: list[list[float]], positions: list[Any]) -> bool:
        tol = 1.5
        for ex in positions:
            if not isinstance(ex, list | tuple) or len(ex) < 2:
                continue
            if not any(
                len(c) >= 2
                and abs(float(c[0]) - float(ex[0])) <= tol
                and abs(float(c[1]) - float(ex[1])) <= tol
                for c in centers
            ):
                return False
        return True

    def _pcd_ok(centers: list[list[float]], pcd: float) -> bool:
        if len(centers) < 3:
            return False
        cx = sum(float(c[0]) for c in centers) / len(centers)
        cy = sum(float(c[1]) for c in centers) / len(centers)
        radii = [math.hypot(float(c[0]) - cx, float(c[1]) - cy) for c in centers if len(c) >= 2]
        if len(radii) < 3:
            return False
        tol = max(1.5, pcd * 0.03)
        if any(abs(r - radii[0]) > tol for r in radii):
            return False
        return abs(2 * sum(radii) / len(radii) - pcd) <= tol

    if ftype == "tunnel":
        axis = str(ef.get("axis") or "").lower() or None
        dia = ef.get("dia")
        dia_tol = max(0.5, float(dia) * 0.15) if dia is not None else None
        mesh_got = sum(
            1
            for h in feats.get("holes") or []
            if h.get("kind") == "tunnel"
            and (axis is None or h.get("axis") == axis)
            and (dia_tol is None or abs(float(h.get("dia") or 0) - float(dia)) <= dia_tol)
        )
        cad_got = sum(
            max(0, int(g.get("count") or 0) - int(g.get("blocked") or 0))
            for g in round_groups
            if g.get("depth_class") == "through"
            and _axis_ok(g, axis)
            and (dia_tol is None or abs(float(g.get("diameter") or 0) - float(dia)) <= dia_tol)
        )
        return mesh_got + cad_got >= want
    if ftype == "slot":
        return int(feats.get("slots") or 0) >= want
    if ftype in ("hole", "holes", "bore", "through-hole"):
        want_axis = str(ef.get("axis") or "").lstrip("+-").lower() or None
        matching = [g for g in round_groups if _axis_ok(g, want_axis)]
        dia = ef.get("dia")
        if dia is not None:
            tol = max(0.5, float(dia) * 0.1)
            matching = [
                g for g in matching if abs(float(g.get("diameter") or 0) - float(dia)) <= tol
            ]
        centers = [c for g in matching for c in (g.get("centers") or [])]
        n_matching = sum(int(g.get("count") or 0) for g in matching)
        if n_matching < want:
            return False
        positions = ef.get("positions")
        if positions and not _positions_ok(centers, positions):
            return False
        pcd = ef.get("pcd")
        return pcd is None or _pcd_ok(centers, float(pcd))
    if ftype == "countersink":
        return int(feats.get("countersinks") or 0) >= want
    if ftype == "counterbore":
        return int(feats.get("counterbores") or 0) >= want
    if ftype in ("fillet", "round"):
        return bool(feats.get("rounds") or [])
    if ftype == "chamfer":
        return int(feats.get("chamfer_faces") or 0) >= 1
    return True


def features_ok(expected: list[dict[str, Any]], features: dict[str, Any] | None) -> bool | None:
    """All expected feature entries must hold; None when nothing expected."""
    if not expected:
        return None
    feats = features or {}
    groups = feats.get("hole_groups") or []
    return all(_feature_entry_ok(ef, feats, groups) for ef in expected)


def verify_spec_report(source: dict[str, Any] | None, issues: list[dict[str, Any]]) -> dict | None:
    """Structured PASS/FAIL/UNVERIFIED report — topopt verify_spec nested in
    mesh first, then the issue list's verify_spec entries."""
    mesh = (source or {}).get("mesh") if isinstance(source, dict) else None
    topopt = mesh.get("topopt") if isinstance(mesh, dict) else None
    report = topopt.get("verify_spec") if isinstance(topopt, dict) else None
    if isinstance(report, dict) and report.get("status") in ("PASS", "FAIL", "UNVERIFIED"):
        return report
    for issue in issues or []:
        if not isinstance(issue, dict) or issue.get("source") != "verify_spec":
            continue
        report = issue.get("report")
        if isinstance(report, dict) and report.get("status") in ("PASS", "FAIL", "UNVERIFIED"):
            return report
    return None


# ---------------------------------------------------------------------------
# trimesh gates
# ---------------------------------------------------------------------------


def _load_mesh(outcome: StageOutcome) -> Any | None:
    """First loadable mesh artifact (stl/glb/3mf/obj) or None."""
    import trimesh

    for p in outcome.artifact_files:
        path = Path(p)
        if path.suffix.lower() not in (".stl", ".glb", ".gltf", ".3mf", ".obj"):
            continue
        if not path.exists():
            continue
        try:
            m = trimesh.load(path)
            if isinstance(m, trimesh.Scene):
                geoms = list(m.dump())
                m = trimesh.util.concatenate(geoms) if geoms else None
            if m is not None and getattr(m, "vertices", None) is not None and len(m.vertices):
                return m
        except Exception:
            continue
    return None


def _radial_peak_count(mesh: Any, axis: str = "z", bins: int = 720) -> tuple[int, str]:
    """Count teeth/lobes from a polar max-radius profile around `axis`."""
    import numpy as np

    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    a1, a2 = axes.get(axis.lower(), (0, 1))
    v = np.asarray(mesh.vertices, dtype=float)
    c1, c2 = (v[:, a1].min() + v[:, a1].max()) / 2, (v[:, a2].min() + v[:, a2].max()) / 2
    r = np.hypot(v[:, a1] - c1, v[:, a2] - c2)
    th = np.arctan2(v[:, a2] - c2, v[:, a1] - c1)
    idx = ((th + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    prof = np.zeros(bins)
    np.maximum.at(prof, idx, r)
    # fill empty bins by neighbour interpolation, light smoothing
    empty = prof == 0
    if empty.all():
        return 0, "empty profile"
    if empty.any():
        filled = np.interp(np.arange(bins), np.flatnonzero(~empty), prof[~empty], period=bins)
        prof = filled
    k = max(1, bins // 360)
    prof = np.convolve(np.r_[prof[-k:], prof, prof[:k]], np.ones(2 * k + 1) / (2 * k + 1), "same")[
        k:-k
    ]
    lo, hi = float(prof.min()), float(prof.max())
    if hi - lo < 0.5:
        return 0, f"flat profile r∈[{lo:.1f},{hi:.1f}]"
    thr = lo + 0.35 * (hi - lo)
    # count rising crossings of the threshold — one per tooth
    above = prof > thr
    crossings = int(np.count_nonzero(~above[:-1] & above[1:]) + (above[-1] < above[0]))
    return crossings, f"r∈[{lo:.1f},{hi:.1f}] thr={thr:.1f}"


def _gate_mesh(g: dict[str, Any], outcome: StageOutcome) -> GateResult:
    name = str(g.get("type"))
    mesh = _load_mesh(outcome)
    if mesh is None:
        return GateResult(name, False, expected=g, detail="no loadable mesh artifact")

    if name == "radial_peaks":
        want = int(g.get("count") or 0)
        tol = int(g.get("tol") or 0)
        n, info = _radial_peak_count(mesh, str(g.get("axis") or "z"))
        return GateResult(
            name, want - tol <= n <= want + tol, measured=n, expected=want, detail=info
        )

    if name == "volume":
        vol = float(mesh.volume) if mesh.is_volume else abs(float(mesh.volume or 0))
        lo, hi = g.get("min_mm3"), g.get("max_mm3")
        ok = (lo is None or vol >= float(lo)) and (hi is None or vol <= float(hi))
        return GateResult(name, ok, measured=round(vol, 1), expected=[lo, hi])

    if name == "volume_fraction":
        vol = abs(float(mesh.volume or 0))
        ext = mesh.bounding_box.extents
        mode = str(g.get("of") or "bbox")
        if mode == "disk":
            r = max(float(ext[0]), float(ext[1])) / 2
            denom = math.pi * r * r * float(ext[2])
        else:
            denom = float(ext[0]) * float(ext[1]) * float(ext[2])
        frac = vol / denom if denom > 0 else 0.0
        lo, hi = g.get("min"), g.get("max")
        ok = (lo is None or frac >= float(lo)) and (hi is None or frac <= float(hi))
        return GateResult(name, ok, measured=round(frac, 3), expected=[lo, hi])

    if name == "contains_point":
        xyz = g.get("xyz")
        if not isinstance(xyz, list | tuple) or len(xyz) != 3:
            return GateResult(name, False, detail="contains_point needs xyz [x,y,z]")
        try:
            inside = bool(mesh.contains([[float(x) for x in xyz]])[0])
        except Exception as e:
            return GateResult(name, False, detail=f"contains failed: {e}")
        return GateResult(name, inside, measured="inside" if inside else "outside", expected=xyz)

    if name == "bodies":
        try:
            n = len(mesh.split(only_watertight=False))
        except Exception:
            n = 1
        maxb = int(g.get("max") or 1)
        return GateResult(name, n <= maxb, measured=n, expected=f"<= {maxb}")

    return GateResult(name, True, detail="unhandled mesh gate (skipped)")


# ---------------------------------------------------------------------------
# outcome gates
# ---------------------------------------------------------------------------


def _topopt(outcome: StageOutcome) -> dict[str, Any]:
    mesh = outcome.mesh or {}
    return mesh.get("topopt") or {} if isinstance(mesh, dict) else {}


def _fem_field(outcome: StageOutcome, field: str) -> Any:
    fem = outcome.fem or {}
    if field in fem:
        return fem[field]
    # sf / worst_safety_factor resolve across the known verdict fields —
    # explicit None checks so a solved 0.0 stays a valid verdict
    if field in ("sf", "worst_safety_factor", "safety_factor"):
        top = _topopt(outcome)
        sf = fem.get("worst_safety_factor")
        if sf is None:
            sf = fem.get("safety_factor")
        if sf is None:
            sf = top.get("sf_verified")
        if sf is None:
            sf = top.get("sf_export")
        if sf is None:
            cases = fem.get("cases") or []
            sfs = [c.get("safety_factor") for c in cases if c.get("safety_factor") is not None]
            sf = min(sfs) if sfs else None
        return sf
    cases = fem.get("cases") or []
    for c in cases:
        if field in c:
            return c[field]
    return None


def _gate_outcome(  # noqa: C901
    g: dict[str, Any], outcome: StageOutcome, wall_s: float
) -> GateResult:
    name = str(g.get("type"))
    analysis = outcome.analysis or {}
    feats = outcome.features or {}

    if name == "status":
        expect = g.get("expect", "succeeded")
        allowed = expect if isinstance(expect, list) else [expect]
        return GateResult(
            name, outcome.status in allowed, measured=outcome.status, expected=allowed
        )

    if name in ("score", "score_min"):
        mn = float(g.get("min", g.get("value", 0)))
        return GateResult(name, (outcome.score or 0) >= mn, measured=outcome.score, expected=mn)

    if name == "bbox":
        dims = g.get("dims") or []
        actual = analysis.get("bbox_size")
        if not actual or len(dims) != 3:
            return GateResult(name, False, measured=actual, expected=dims, detail="no bbox")
        exp, act = sorted(float(x) for x in dims), sorted(float(x) for x in actual)
        tol = float(g.get("tol_mm", 1.0))
        mode = str(g.get("match") or "match")
        if mode == "within":
            ok = all(a <= e + max(tol, e * 0.02) for a, e in zip(act, exp, strict=True))
        else:
            ok = all(abs(a - e) <= max(tol, e * 0.02) for a, e in zip(act, exp, strict=True))
        return GateResult(name, ok, measured=[round(x, 2) for x in actual], expected=dims)

    if name in (
        "holes",
        "hole",
        "tunnel",
        "slot",
        "countersink",
        "counterbore",
        "fillet",
        "chamfer",
        "bore",
        "through-hole",
        "features",
    ):
        entries = g.get("entries") if name == "features" else [{**g, "type": name}]
        ok = features_ok(entries or [], feats)
        return GateResult(
            name,
            bool(ok),
            measured=feats.get("hole_groups") or feats,
            expected=entries,
        )

    if name == "watertight":
        wt = analysis.get("watertight_after_repair")
        if wt is None:
            wt = analysis.get("watertight")
        want = bool(g.get("expect", True))
        return GateResult(name, wt is want, measured=wt, expected=want)

    if name == "verify_spec":
        report = verify_spec_report({"mesh": outcome.mesh or {}}, outcome.issues)
        want = str(g.get("expect") or "PASS")
        got = str(report["status"]) if report else None
        fails = [
            c.get("name")
            for c in (report or {}).get("checks") or []
            if isinstance(c, dict) and c.get("status") == "FAIL"
        ]
        return GateResult(name, got == want, measured=got, expected=want, detail=str(fails[:6]))

    if name == "dfm":
        dfm = outcome.dfm or {}
        warns = [w for w in (dfm.get("warnings") or []) if isinstance(w, dict)]
        counts = {
            sev: sum(1 for w in warns if w.get("severity") == sev)
            for sev in ("high", "medium", "low")
        }
        proc = dfm.get("process")
        proc_ok = g.get("process") in (None, proc)
        ok = (
            bool(dfm)
            and proc_ok
            and all(counts[s] <= int(g.get(f"max_{s}", 1 << 30)) for s in counts)
        )
        return GateResult(
            name,
            ok,
            measured={
                "process": proc,
                "counts": counts,
                "warnings": [w.get("code") for w in warns[:8]],
            },
            expected={k: v for k, v in g.items() if k != "type"},
            detail="" if dfm else "no dfm report on the version",
        )

    if name == "issue_contains":
        needle = str(g.get("substr") or "").lower()
        texts = [
            f"{i.get('what', '')} {i.get('fix_hint', '')}".strip() for i in outcome.issues or []
        ]
        return GateResult(name, any(needle in t.lower() for t in texts), measured=texts[:6])

    if name == "code":
        want = bool(g.get("expect", True))
        return GateResult(name, (outcome.code is not None) is want, measured=bool(outcome.code))

    if name == "execs_min":
        n = int(g.get("n") or 1)
        return GateResult(
            name, int(outcome.extras.get("execs") or 0) >= n, measured=outcome.extras.get("execs")
        )

    if name == "max_wall_s":
        return GateResult(name, wall_s <= float(g.get("s") or g.get("max") or 0), measured=wall_s)

    if name == "max_prompt_tokens":
        pt = int((outcome.usage or {}).get("prompt_tokens") or 0)
        return GateResult(name, pt <= int(g.get("n") or g.get("max") or 0), measured=pt)

    if name == "mode":
        want = str(g.get("expect") or "")
        return GateResult(name, outcome.resolved_mode == want, measured=outcome.resolved_mode)

    if name == "fem":
        field = str(g.get("field") or "sf")
        val = _fem_field(outcome, field)
        lo, hi = g.get("min"), g.get("max")
        ok = (
            val is not None
            and (lo is None or float(val) >= float(lo))
            and (hi is None or float(val) <= float(hi))
        )
        return GateResult(name, ok, measured=val, expected={field: [lo, hi]})

    if name in ("topopt", "volfrac", "envelope_within"):
        top = _topopt(outcome)
        if name == "volfrac":
            val = top.get("volfrac_achieved")
            if val is None:
                val = top.get("volfrac")
            lo, hi = g.get("min"), g.get("max")
            ok = (
                val is not None
                and (lo is None or float(val) >= float(lo))
                and (hi is None or float(val) <= float(hi))
            )
            return GateResult(name, ok, measured=val, expected=[lo, hi])
        if name == "envelope_within":
            dims = g.get("dims") or []
            actual = analysis.get("bbox_size")
            if not actual or len(dims) != 3:
                return GateResult(name, False, measured=actual, expected=dims)
            tol = float(g.get("tol_mm", 2.0))
            ok = all(
                a <= e + max(tol, e * 0.02)
                for a, e in zip(
                    sorted(float(x) for x in actual), sorted(float(x) for x in dims), strict=True
                )
            )
            return GateResult(name, ok, measured=actual, expected=dims)
        field = str(g.get("field") or "sf_export")
        val = top.get(field)
        lo = g.get("min")
        ok = val is not None and (lo is None or float(val) >= float(lo))
        return GateResult(name, ok, measured=val, expected={field: [lo, g.get("max")]})

    if name == "asset_image":
        want = int(g.get("min") or 1)
        return GateResult(name, len(outcome.asset_ids) >= want, measured=len(outcome.asset_ids))

    return GateResult(name, True, detail=f"unknown gate type {name!r} (skipped)")


MESH_GATES = {"radial_peaks", "volume", "volume_fraction", "contains_point", "bodies"}


def run_gates(
    specs: list[dict[str, Any]], outcome: StageOutcome, wall_s: float
) -> list[GateResult]:
    """Evaluate a stage's gate list — mesh gates need an artifact file,
    everything else reads the normalized outcome."""
    out: list[GateResult] = []
    for g in specs or []:
        if not isinstance(g, dict) or "type" not in g:
            continue
        if str(g.get("type")) in MESH_GATES:
            out.append(_gate_mesh(g, outcome))
        else:
            out.append(_gate_outcome(g, outcome, wall_s))
    return out
