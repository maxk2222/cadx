"""Stage gates — declarative pass/fail checks on a StageOutcome.

Gate specs live in the suite JSON (`stages[].gates`) as
``{"type": <name>, ...kwargs}``. Two families:

- **outcome gates** — status/score/bbox/features/watertight/verify_spec/
  fem/topopt/tokens/wall time, read off the normalized outcome;
- **mesh gates** — `radial_peaks`, `radial_envelope`, `volume(_fraction)`,
  `contains_point`, `bodies`, `threads` — computed with trimesh on the stage's
  final committed mesh
  (STL/GLB/3MF, authoritative-first via `_mesh_artifact_files`), skipped
  with detail when no mesh is available.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

from cadx.cad.radial import radial_peak_count
from cadx.cad.spec_features import feature_drop_count, feature_drop_records
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
        depth_class = ef.get("depth_class")
        if depth_class:
            matching = [g for g in matching if str(g.get("depth_class") or "") == str(depth_class)]
        centers = []
        ax_idx = {"x": 0, "y": 1, "z": 2}.get(str(want_axis or ""))
        for g in matching:
            xyz_rows = g.get("centers_xyz") or []
            if xyz_rows:
                # transverse coords only — `centers` mixes the through-axis
                # coordinate in for non-Z holes (legacy projection pair)
                for row in xyz_rows:
                    if ax_idx is not None and len(row) == 3:
                        centers.append(
                            [c for i, c in enumerate(row) if i != ax_idx and c is not None]
                        )
                    else:
                        centers.append([c for c in row if c is not None])
            else:
                centers.extend(g.get("centers") or [])
        must_open = ftype == "through-hole" or str(depth_class or "") == "through"
        n_matching = sum(
            max(0, int(g.get("count") or 0) - (int(g.get("blocked") or 0) if must_open else 0))
            for g in matching
        )
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


def _gate_mesh(g: dict[str, Any], outcome: StageOutcome) -> GateResult:
    name = str(g.get("type"))
    mesh = _load_mesh(outcome)
    if mesh is None:
        return GateResult(name, False, expected=g, detail="no loadable mesh artifact")

    if name == "radial_peaks":
        want = int(g.get("count") or 0)
        tol = int(g.get("tol") or 0)
        n, info = radial_peak_count(mesh, str(g.get("axis") or "z"))
        return GateResult(
            name, want - tol <= n <= want + tol, measured=n, expected=want, detail=info
        )

    if name == "radial_envelope":
        # max radial reach about an axis — the honest rotor/impeller check:
        # an axis-aligned bbox can never reach the tip circle on odd blade
        # counts (a 5-blade pentagon tops out ~4% short of its Ø)
        import numpy as np

        ax = str(g.get("axis") or "z").lstrip("+-").lower()
        a1, a2 = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}.get(ax, (0, 1))
        v = np.asarray(mesh.vertices, dtype=float)
        # the rotation axis is the vertex centroid — for odd-fold symmetry
        # (5 blades) the bbox centre is NOT on the axis and inflates r_max
        c1, c2 = float(v[:, a1].mean()), float(v[:, a2].mean())
        r_max = float(np.hypot(v[:, a1] - c1, v[:, a2] - c2).max())
        want = float(g.get("radius_mm") or 0.0)
        tol = float(g.get("tol_mm", 1.0))
        return GateResult(
            name,
            abs(r_max - want) <= tol,
            measured=round(r_max, 2),
            expected=want,
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
            from cadx.cad.mesh_ops import solid_body_count

            n = solid_body_count(mesh)
        except Exception:
            n = 1
        maxb = int(g.get("max") or 1)
        return GateResult(name, n <= maxb, measured=n, expected=f"<= {maxb}")

    if name in ("threads", "thread"):
        # Helical-groove evidence on the shipped mesh. verify_spec only counts
        # declared thread metadata and refine_verify re-probes bore tunnels —
        # both passed runs whose final STL carried smooth pilot bores
        # (r19/r29). Bores are located via scan_mesh_anchors (blind taps are
        # invisible to hole_groups); a groove shows as wall verts past the
        # pilot radius covering >=4 of 8 azimuth sectors across >=35% of the
        # bore depth (a helix rotates through all sectors; a mouth chamfer or
        # a neighbouring offset channel cannot produce that signature).
        found = _thread_grooves(mesh)
        external = _external_threads(mesh)
        want = int(g.get("min", 1))
        n = sum(1 for t in found if t["groove"]) + sum(1 for t in external if t["thread"])
        return GateResult(
            name,
            n >= want,
            measured={
                "grooves": n,
                "bores": len(found),
                "detail": found,
                "external": external,
            },
            expected={"min": want},
        )

    return GateResult(name, True, detail="unhandled mesh gate (skipped)")


_AXIS_IDX = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}


def _thread_grooves(mesh: Any) -> list[dict[str, Any]]:
    """Per-bore helical-groove evidence for the `threads` gate."""
    import numpy as np

    from cadx.cad.fem.anchors3d import scan_mesh_anchors

    try:
        anchors = scan_mesh_anchors(mesh)
    except Exception:
        return []
    verts = np.asarray(mesh.vertices, dtype=float)
    out: list[dict[str, Any]] = []
    for a in anchors:
        dia = float(a.get("dia") or 0.0)
        axis = str(a.get("axis") or "")
        if a.get("kind") not in ("hole", "window", "dimple") or axis not in _AXIS_IDX:
            continue
        if not 2.0 <= dia <= 24.0:  # thread scale only — not bores/bearings
            continue
        ax = _AXIS_IDX[axis]
        pos = [float(x) for x in (a.get("pos") or [0.0, 0.0, 0.0])]
        depth = float(a.get("depth_mm") or 0.0) or 5.0
        iu, iv = (i for i in range(3) if i != ax)
        rel_u = verts[:, iu] - pos[iu]
        rel_v = verts[:, iv] - pos[iv]
        r = np.hypot(rel_u, rel_v)
        axial = np.abs(verts[:, ax] - pos[ax])
        sel = (r > dia / 2 + 0.3) & (r < dia / 2 + 1.8) & (axial <= depth + 1.0)
        idx = np.nonzero(sel)[0]
        entry: dict[str, Any] = {
            "pos": [round(x, 1) for x in pos],
            "axis": axis,
            "dia": round(dia, 2),
            "groove": False,
        }
        if len(idx) >= 10:
            axial_cov = float(axial[idx].max() - axial[idx].min()) / depth
            sectors = int(
                np.unique(
                    ((np.arctan2(rel_v[idx], rel_u[idx]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
                ).size
            )
            # groove flanks end at the major radius in EVERY height slice of a
            # helix; lattice voids or stray verts only spike single slices.
            # Taking the per-bin max and then the median keeps the crest radius
            # while verts at/near the 1.8 mm band ceiling (a bigger lattice
            # void, not a thread) can no longer veto a real groove.
            edges = np.linspace(0.0, depth + 1.0, 12)
            in_band = r[idx]
            in_axial = axial[idx]
            bin_reach = [
                float(in_band[(in_axial >= lo) & (in_axial < hi)].max() - dia / 2)
                for lo, hi in itertools.pairwise(edges)
                if int(((in_axial >= lo) & (in_axial < hi)).sum()) >= 3
            ]
            reach = float(np.median(bin_reach)) if bin_reach else 0.0
            entry.update(
                groove=sectors >= 4 and axial_cov >= 0.35 and 0.4 <= reach <= 1.7,
                sectors=sectors,
                axial_cov=round(axial_cov, 2),
                reach=round(reach, 2),
            )
        out.append(entry)
    return out


def _external_threads(mesh: Any) -> list[dict[str, Any]]:
    """External-thread evidence: a screw shank carries no bore anchor, so the
    bore probe can never see it. A helix shows as a contiguous axial run of
    slices whose radial spread sits in thread-depth range (0.25-2.5 mm) while
    the azimuth of the crest verts rotates along the axis. A smooth or
    stepped shaft has ~0 spread; straight flutes/lobes don't rotate."""
    import numpy as np

    verts = np.asarray(mesh.vertices, dtype=float)
    if len(verts) < 50:
        return []
    out: list[dict[str, Any]] = []
    for ax in range(3):
        iu, iv = (i for i in range(3) if i != ax)
        cu = (verts[:, iu].min() + verts[:, iu].max()) / 2.0
        cv = (verts[:, iv].min() + verts[:, iv].max()) / 2.0
        r = np.hypot(verts[:, iu] - cu, verts[:, iv] - cv)
        az = np.arctan2(verts[:, iv] - cv, verts[:, iu] - cu)
        axial = verts[:, ax]
        lo, hi = float(axial.min()), float(axial.max())
        span = hi - lo
        if span < 3.0:
            continue
        step = 1.5
        edges = np.arange(lo, hi + step * 0.5, step)
        spreads: list[float] = []
        z_at: list[float] = []
        sector_of = ((az + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        for a, b in itertools.pairwise(edges):
            sel = (axial >= a) & (axial < b)
            if int(sel.sum()) < 8:
                spreads.append(0.0)
                z_at.append((a + b) / 2.0)
                continue
            rr = r[sel]
            spreads.append(float(np.percentile(rr, 90) - np.percentile(rr, 10)))
            z_at.append((a + b) / 2.0)
        if len(spreads) < 3:
            continue
        # longest contiguous run of thread-depth spreads
        best: list[int] = []
        run: list[int] = []
        for i, s in enumerate(spreads):
            if 0.25 <= s <= 2.5:
                run.append(i)
            else:
                if len(run) > len(best):
                    best = run
                run = []
        if len(run) > len(best):
            best = run
        run_mm = len(best) * step
        if run_mm < 4.0:
            continue
        # helix phase test: the thread's extreme-radius verts sit at a
        # different height in each 45° azimuth sector — mean crest/valley-z
        # spans ~one pitch across sectors. A sphere belly, box corner or
        # smooth shaft holds the same z in every sector. The crest is used
        # when the land is narrow (cut threads) and the valley when the
        # groove floor is the thin helix (ground threads) — either proves it.
        z_lo = min(z_at[i] for i in best) - step / 2
        z_hi = max(z_at[i] for i in best) + step / 2
        sel = (axial >= z_lo) & (axial < z_hi)
        r_run = r[sel]
        phase = 0.0
        for pct in (80, 20):
            band = (
                r_run >= np.percentile(r_run, pct)
                if pct == 80
                else r_run <= np.percentile(r_run, pct)
            )
            z_mean = [
                float(axial[sel][band & (sector_of[sel] == s)].mean())
                for s in range(8)
                if int((band & (sector_of[sel] == s)).sum()) >= 3
            ]
            if len(z_mean) >= 6:
                phase = max(phase, float(max(z_mean) - min(z_mean)))
        out.append(
            {
                "axis": "xyz"[ax],
                "span_mm": round(run_mm, 1),
                "median_spread_mm": round(float(np.median([spreads[i] for i in best])), 2),
                "crest_phase_mm": round(phase, 2),
                "thread": run_mm >= 4.0 and phase >= 0.5,
            }
        )
    return out


# ---------------------------------------------------------------------------
# outcome gates
# ---------------------------------------------------------------------------


def _v3(v: Any) -> list[float] | None:
    if isinstance(v, list | tuple) and len(v) == 3:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def _axis_index(v: Any) -> int | None:
    """ "+x"/"x" → 0, "-y"/"y" → 1, "+z"/"z" → 2."""
    s = str(v or "").strip().lower().lstrip("+-")
    return {"x": 0, "y": 1, "z": 2}.get(s)


def _unit(v: list[float]) -> list[float] | None:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-9 else None


def _fem_case(fem: dict[str, Any], key: Any) -> dict[str, Any] | None:
    """Match a check's `case` selector — case name, 0-based index, or the
    first case when unset."""
    cases = fem.get("cases") or []
    if not cases:
        return None
    if key is None:
        return cases[0]
    if isinstance(key, int):
        return cases[key] if 0 <= key < len(cases) else None
    for c in cases:
        if isinstance(c, dict) and str(c.get("name")) == str(key):
            return c
    return None


def _load_centroid(applied: list[dict[str, Any]]) -> list[float] | None:
    """Load-weighted centroid of the applied-load records — where the load
    actually landed on the part (region centroid weighted by the resolved
    force it carries; a pure-couple record weights by its moment, so
    moment-only cases still get a landing point)."""
    num = [0.0, 0.0, 0.0]
    den = 0.0
    for a in applied:
        c = _v3(a.get("centroid_mm"))
        w = float(a.get("force_mag_n") or 0.0) + float(a.get("couple_mag_nm") or 0.0)
        if c is not None and w > 1e-9:
            for i in range(3):
                num[i] += w * c[i]
            den += w
    return [x / den for x in num] if den > 1e-9 else None


def _declared_resultant(spec: dict[str, Any] | None, case_name: str) -> list[float] | None:
    """Sum the declared `force_n` vectors of the spec's matching load case —
    what the agent *said* the case should apply (non-vector load kinds like
    pressure/bearing can't be summed without mesh normals and are skipped)."""
    for lc in (spec or {}).get("load_cases") or []:
        if not isinstance(lc, dict) or str(lc.get("name")) != case_name:
            continue
        res = [0.0, 0.0, 0.0]
        hit = False
        for ld in lc.get("loads") or []:
            f = _v3((ld or {}).get("force_n"))
            if f is not None:
                for i in range(3):
                    res[i] += f[i]
                hit = True
        return res if hit else None
    return None


def _gate_fem_loadcases(g: dict[str, Any], outcome: StageOutcome) -> GateResult:  # noqa: C901
    """Semantic boundary-condition gate — re-checks the *resolved* load
    cases rather than trusting the request: resultant magnitude/direction,
    where the load landed (load centroid) vs. where the supports sit
    (fixed centroid), which load mechanisms were applied, warnings, and
    the SF band. Missing evidence (no fem report, no reactions, no
    declared spec case) makes a check inconclusive, not a failure."""
    name = "fem_loadcases"
    fem = outcome.fem or {}
    cases = fem.get("cases") or []
    problems: list[str] = []
    inconclusive: list[str] = []
    measured: dict[str, Any] = {"n_cases": len(cases)}

    if not fem:
        return GateResult(name, None, measured=measured, detail="no fem report — inconclusive")

    min_cases = g.get("min_cases")
    if min_cases is not None and len(cases) < int(min_cases):
        problems.append(f"cases={len(cases)} < min_cases={min_cases}")

    src_expect = g.get("source")
    if src_expect and src_expect != "any":
        src = outcome.extras.get("load_cases_source")
        measured["load_cases_source"] = src
        if src is None:
            inconclusive.append("load_cases_source not recorded")
        elif src != src_expect:
            problems.append(f"load_cases_source={src} != expected {src_expect}")

    # a declared region resolving to a boundary-patch fallback means the
    # declared BC was NOT applied (phantom anchors, sealed holes) — the
    # solved SF describes different physics; count across all cases so the
    # gate also fires when no explicit check names the affected case
    fb_total = 0
    for c in cases:
        fb = [w for w in (c.get("warnings") or []) if "region_fallback" in str(w)]
        if fb:
            fb_total += len(fb)
            tag = str(c.get("name") or "?")
            problems.append(
                f"{tag}: {len(fb)} declared region(s) unresolved — "
                "fallback boundary patch applied instead"
            )
    if fb_total:
        measured["unresolved_regions"] = fb_total

    for chk in g.get("checks") or []:
        c = _fem_case(fem, chk.get("case"))
        if c is None:
            problems.append(f"case {chk.get('case')!r} not resolved")
            continue
        tag = str(c.get("name") or chk.get("case"))
        rxn = c.get("reactions") or {}
        applied = [a for a in (c.get("applied") or []) if isinstance(a, dict)]
        warnings = [str(w) for w in (c.get("warnings") or [])]
        warnings += [str(w) for w in (fem.get("warnings") or [])]
        tol_mm = float(chk.get("tol_mm", 5.0))

        if "resultant_n" in chk:
            band = chk["resultant_n"] or {}
            mag = rxn.get("resultant_mag_n")
            measured[f"{tag}.resultant_n"] = rxn.get("resultant_n")
            if mag is None:
                inconclusive.append(f"{tag}: no reactions record")
            else:
                lo, hi = band.get("min"), band.get("max")
                if (lo is not None and float(mag) < float(lo)) or (
                    hi is not None and float(mag) > float(hi)
                ):
                    problems.append(f"{tag}: resultant {float(mag):.1f} N outside [{lo},{hi}]")
        if "moment_mag_nm" in chk:
            band = chk["moment_mag_nm"] or {}
            mag = rxn.get("moment_mag_nm")
            measured[f"{tag}.moment_mag_nm"] = mag
            if mag is None:
                inconclusive.append(f"{tag}: no moment reaction record")
            else:
                lo, hi = band.get("min"), band.get("max")
                if (lo is not None and float(mag) < float(lo)) or (
                    hi is not None and float(mag) > float(hi)
                ):
                    problems.append(f"{tag}: moment {float(mag):.1f} N·m outside [{lo},{hi}]")
        if "resultant_dir" in chk:
            res, want = _v3(rxn.get("resultant_n")), _v3(chk["resultant_dir"])
            if res is None or want is None:
                inconclusive.append(f"{tag}: resultant direction unavailable")
            else:
                u, w = _unit(res), _unit(want)
                cos_tol = math.cos(math.radians(float(chk.get("dir_tol_deg", 20.0))))
                if u is None or w is None:
                    inconclusive.append(f"{tag}: zero resultant — no direction to check")
                elif sum(a * b for a, b in zip(u, w, strict=True)) < cos_tol:
                    problems.append(f"{tag}: resultant dir {res} deviates from {want}")
        if "load_centroid_mm" in chk:
            want = _v3(chk["load_centroid_mm"])
            got = _load_centroid(applied)
            measured[f"{tag}.load_centroid_mm"] = got
            if want is None or got is None:
                inconclusive.append(f"{tag}: load centroid unavailable")
            elif max(abs(a - b) for a, b in zip(got, want, strict=True)) > tol_mm:
                problems.append(
                    f"{tag}: load centroid {[round(x, 1) for x in got]} "
                    f"deviates >{tol_mm} mm from {want}"
                )
        if "fixed_centroid_mm" in chk:
            want = _v3(chk["fixed_centroid_mm"])
            got = _v3(rxn.get("fixed_centroid_mm"))
            measured[f"{tag}.fixed_centroid_mm"] = got
            if want is None or got is None:
                inconclusive.append(f"{tag}: fixed centroid unavailable")
            elif max(abs(a - b) for a, b in zip(got, want, strict=True)) > tol_mm:
                problems.append(
                    f"{tag}: fixed centroid {[round(x, 1) for x in got]} "
                    f"deviates >{tol_mm} mm from {want}"
                )
        # axis-extremum variants — "fixed at the -X end" / "load on the +Z
        # face" without pinning absolute coordinates (the part need not be
        # centred for the physics to be right)
        for key, label in (("fixed_at_axis", "fixed"), ("load_at_axis", "load")):
            if key not in chk:
                continue
            got = _v3(rxn.get("fixed_centroid_mm")) if label == "fixed" else _load_centroid(applied)
            ax = _axis_index(chk[key])
            lo = _v3((outcome.analysis or {}).get("bbox_min"))
            hi = _v3((outcome.analysis or {}).get("bbox_max"))
            if lo is None or hi is None:
                # fem stages carry no geometry analysis — the FEM report's
                # solid-cell bbox is the honest part-bounds evidence
                grid = fem.get("grid") or {}
                lo = lo or _v3(grid.get("bbox_min_mm"))
                hi = hi or _v3(grid.get("bbox_max_mm"))
            measured[f"{tag}.{key}"] = {"centroid": got, "axis": chk[key]}
            if got is None or ax is None or lo is None or hi is None:
                inconclusive.append(f"{tag}: {label} centroid/bbox unavailable")
                continue
            extreme = hi[ax] if str(chk[key]).startswith("+") else lo[ax]
            if abs(got[ax] - extreme) > tol_mm:
                problems.append(
                    f"{tag}: {label} centroid {got[ax]:.1f} is "
                    f"{abs(got[ax] - extreme):.1f} mm off the {chk[key]} extreme "
                    f"({extreme:.1f})"
                )
        if "mechanisms" in chk:
            have = {str(m) for a in applied for m in (a.get("mechanisms") or [])}
            want = {str(m) for m in (chk["mechanisms"] or [])}
            missing = sorted(want - have)
            measured[f"{tag}.mechanisms"] = sorted(have)
            if not applied:
                inconclusive.append(f"{tag}: no applied-load records")
            elif missing:
                problems.append(f"{tag}: load mechanisms missing: {missing}")
        if "warnings_absent" in chk:
            hits = [
                w
                for w in warnings
                for needle in chk["warnings_absent"]
                if str(needle).lower() in w.lower()
            ]
            if hits:
                problems.append(f"{tag}: forbidden warnings present: {hits[:3]}")
        if "sf_min" in chk or "sf_max" in chk:
            sf = c.get("safety_factor")
            measured[f"{tag}.safety_factor"] = sf
            if sf is None:
                inconclusive.append(f"{tag}: no safety factor solved")
            else:
                if "sf_min" in chk and float(sf) < float(chk["sf_min"]):
                    problems.append(f"{tag}: sf {float(sf):.2f} < {chk['sf_min']}")
                if "sf_max" in chk and float(sf) > float(chk["sf_max"]):
                    problems.append(f"{tag}: sf {float(sf):.2f} > {chk['sf_max']}")
        if "declared_resultant_rel_tol" in chk:
            # the agent's declared load vs. what actually resolved — catches
            # 'asked for 100 N down, got 40 N sideways' misconfiguration
            declared = _declared_resultant(outcome.spec, str(c.get("name") or ""))
            res = _v3(rxn.get("resultant_n"))
            if declared is None or res is None:
                inconclusive.append(f"{tag}: no declared/resolved resultant to compare")
            else:
                dn = math.sqrt(sum(x * x for x in declared))
                if dn < 1e-9:
                    inconclusive.append(f"{tag}: declared resultant is zero")
                else:
                    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(res, declared, strict=True)))
                    rel = err / dn
                    measured[f"{tag}.declared_vs_resultant"] = round(rel, 3)
                    if rel > float(chk["declared_resultant_rel_tol"]):
                        problems.append(
                            f"{tag}: resolved resultant {res} deviates "
                            f"{rel:.0%} from declared {declared}"
                        )

    if problems:
        return GateResult(name, False, measured=measured, detail="; ".join(problems[:8]))
    if inconclusive:
        return GateResult(name, None, measured=measured, detail="; ".join(inconclusive[:8]))
    return GateResult(name, True, measured=measured)


def _topopt(outcome: StageOutcome) -> dict[str, Any]:
    mesh = outcome.mesh or {}
    top = mesh.get("topopt") if isinstance(mesh, dict) else None
    if top:
        return top
    # keep-base: the agent declined every candidate — measure the best
    # rejected run so gates see the real verdict (sf/volfrac/load_path)
    # instead of a bare "no topopt happened" null
    kept = mesh.get("topopt_kept_base") if isinstance(mesh, dict) else None
    rejected = (kept or {}).get("rejected_runs") or []
    best = max(
        (r for r in rejected if isinstance(r, dict)),
        key=lambda r: float(r.get("worst_sf") or 0.0),
        default=None,
    )
    if best is None:
        return {}
    ver = best.get("verification") or {}
    # a solved sf_verified of 0.0 IS a verdict — never or-chain it away
    sf_verified = ver.get("sf_verified")
    if sf_verified is None:
        sf_verified = best.get("worst_sf")
    return {
        "kept_base": True,
        "volfrac_achieved": best.get("achieved"),
        "sf_verified": sf_verified,
        "sf_export": best.get("sf_export"),
        "n_bodies_export": best.get("n_bodies_export"),
        "load_path_ok": best.get("load_path_ok"),
        # sealed-void telemetry of the best rejected run — same "measure the
        # closest produced candidate" semantics as sf/volfrac on kept base
        "enclosed_void_mm3": ver.get("enclosed_void_mm3"),
        "n_sealed_cavities": ver.get("n_sealed_cavities"),
        "largest_void_mm3": ver.get("largest_void_mm3"),
        "sealed_cavities_mm3": ver.get("sealed_cavities_mm3"),
        "void_frac_of_solid": ver.get("void_frac_of_solid"),
        "export_grid": ver.get("export_grid"),
    }


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
        if not entries:
            return GateResult(name, True, expected=entries, detail="no feature assertions")
        if outcome.features is None or outcome.features_available is False:
            reason = outcome.features_error or "no measurable artifact or feature evidence"
            return GateResult(
                name,
                None,
                measured=None,
                expected=entries,
                detail=f"feature measurement unavailable — inconclusive: {reason}",
            )
        ok = features_ok(entries or [], feats)
        if name in ("fillet", "round"):
            meas: Any = {
                "rounds": feats.get("rounds") or [],
                "fillets": feats.get("fillets") or [],
            }
        elif name == "chamfer":
            meas = {
                "chamfer_faces": feats.get("chamfer_faces") or 0,
                "chamfers": feats.get("chamfers") or [],
            }
        elif name == "slot":
            meas = {
                "slots": feats.get("slots") or 0,
                "hole_groups": [g for g in (feats.get("hole_groups") or []) if g.get("elongated")],
            }
        elif name in ("countersink", "counterbore"):
            key = "countersinks" if name == "countersink" else "counterbores"
            meas = {key: feats.get(key) or 0}
        else:
            meas = feats.get("hole_groups") or feats
        return GateResult(
            name,
            ok,
            measured=meas,
            expected=entries,
        )

    if name == "refine_verify":
        # the refine job's own post-cut verification: applied feature entries
        # are re-scanned on the shipped mesh (tunnel/dimple probes at the
        # declared positions). This is the authoritative evidence for
        # recut features — extract_features cannot see blind holes on a
        # heavily faceted solid at all (facet-cap skips the cylinder scan;
        # the mesh fallback detects through-tunnels only).
        rep = (outcome.mesh or {}).get("refine") or {}
        misses = rep.get("verify_misses") or []
        probes = rep.get("probes") or []
        applied = rep.get("applied") or []
        unverified = [p for p in probes if not p.get("verified")]
        min_applied = int(g.get("min_applied", 0) or 0)
        ok = bool(rep) and not misses and not unverified and len(applied) >= min_applied
        return GateResult(
            name,
            ok if rep else None,
            measured={
                "applied": len(applied),
                "verify_misses": len(misses),
                "probes_verified": sum(1 for p in probes if p.get("verified")),
                "probes_total": len(probes),
            },
            expected={"verify_misses": 0, "min_applied": min_applied},
            detail=""
            if ok
            else f"misses={[str(m.get('entry', {}).get('name')) for m in misses[:5]]} "
            f"unverified={[str(p.get('name')) for p in unverified[:5]]}",
        )

    if name == "watertight":
        wt = analysis.get("watertight_after_repair")
        if wt is None:
            wt = analysis.get("watertight")
        want = bool(g.get("expect", True))
        return GateResult(name, wt is want, measured=wt, expected=want)

    if name == "sits_on_xy":
        # frame contract: the part must rest on the XY plane (min z = 0).
        # bbox/verify_spec only prove dims — a mirrored or floating part
        # slips through them (observed: wing built at z=-12..0).
        sits = analysis.get("sits_on_xy")
        bb_min = analysis.get("bbox_min")
        if sits is None and isinstance(bb_min, list | tuple) and len(bb_min) == 3:
            sits = abs(float(bb_min[2])) < 0.1
        if sits is None:
            return GateResult(name, None, measured=None, detail="no frame measurement")
        want = bool(g.get("expect", True))
        return GateResult(name, bool(sits) is want, measured=sits, expected=want)

    if name == "verify_spec":
        report = verify_spec_report({"mesh": outcome.mesh or {}}, outcome.issues)
        want = str(g.get("expect") or "PASS")
        # "ignore": ["check", ...] — recompute the verdict with named checks
        # dropped (e.g. the DfM hint tier on a shape whose manufacturability
        # warning is intrinsic, like a flat-printed airfoil)
        checks = [c for c in (report or {}).get("checks") or [] if isinstance(c, dict)]
        ignore = {str(x) for x in (g.get("ignore") or [])}
        if ignore:
            checks = [c for c in checks if c.get("name") not in ignore]
            n_fail = sum(1 for c in checks if c.get("status") == "FAIL")
            n_pass = sum(1 for c in checks if c.get("status") == "PASS")
            got = "FAIL" if n_fail else ("PASS" if n_pass else "UNVERIFIED")
        else:
            got = str(report["status"]) if report else None
        fails = [c.get("name") for c in checks if c.get("status") == "FAIL"]
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

    if name == "spec_features_dropped":
        if outcome.spec is None:
            return GateResult(name, None, detail="spec/drop audit is unavailable")
        count = feature_drop_count(outcome.spec)
        expected = int(g.get("max", g.get("max_count", 0)))
        details = feature_drop_records(outcome.spec)
        return GateResult(
            name,
            count <= expected,
            measured={"count": count, "drops": details},
            expected={"max": expected},
        )

    if name == "spec_contains":
        # substring match over the serialized PartSpec — the auditable check
        # for pipeline-side evidence (research facts landing in assumptions,
        # a declared process, …) that geometry gates cannot see
        if outcome.spec is None:
            return GateResult(name, None, detail="spec unavailable")
        import json

        dump = getattr(outcome.spec, "model_dump", None)
        text = json.dumps(dump(mode="json") if callable(dump) else outcome.spec).lower()
        need = [str(s).lower() for s in g.get("all") or []]
        miss = [s for s in need if s not in text]
        return GateResult(
            name,
            not miss,
            measured=miss or None,
            expected={"all": need},
            detail=f"missing {miss[:4]}" if miss else "",
        )

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
        # expect = exact match for non-numeric fields (e.g. recommended_mode);
        # min/max keep the numeric range semantics
        if "expect" in g:
            return GateResult(name, val == g.get("expect"), measured=val, expected=g.get("expect"))
        lo, hi = g.get("min"), g.get("max")
        ok = (
            val is not None
            and (lo is None or float(val) >= float(lo))
            and (hi is None or float(val) <= float(hi))
        )
        return GateResult(name, ok, measured=val, expected={field: [lo, hi]})

    if name == "fem_loadcases":
        return _gate_fem_loadcases(g, outcome)

    if name == "sealed_voids":
        top = _topopt(outcome)
        measured: dict[str, Any] = {
            key: top.get(key)
            for key in (
                "n_sealed_cavities",
                "enclosed_void_mm3",
                "largest_void_mm3",
            )
            if top.get(key) is not None
        }
        limits = {
            "max_count": g.get("max_count"),
            "max_mm3": g.get("max_mm3"),
            "largest_mm3": g.get("largest_mm3"),
        }
        checks: list[bool] = []
        missing: list[str] = []
        field_for_limit = {
            "max_count": "n_sealed_cavities",
            "max_mm3": "enclosed_void_mm3",
            "largest_mm3": "largest_void_mm3",
        }
        for limit_name, limit in limits.items():
            if limit is None:
                continue
            field = field_for_limit[limit_name]
            value = top.get(field)
            if value is None:
                missing.append(field)
            else:
                checks.append(float(value) <= float(limit))
        max_frac = g.get("max_frac")
        if max_frac is not None:
            value = top.get("enclosed_void_mm3")
            grid = top.get("export_grid") or {}
            domain_volume = grid.get("domain_volume_mm3")
            if value is None or not domain_volume or float(domain_volume) <= 0:
                missing.append("enclosed_void_mm3/export_grid.domain_volume_mm3")
            else:
                fraction = float(value) / float(domain_volume)
                measured["void_fraction"] = round(fraction, 6)
                checks.append(fraction <= float(max_frac))
        # material-relative share — comparable across parts whose bounding
        # grid is mostly air (max_frac uses the whole grid as denominator)
        max_solid_frac = g.get("max_solid_frac")
        if max_solid_frac is not None:
            value = top.get("void_frac_of_solid")
            if value is None:
                missing.append("void_frac_of_solid")
            else:
                measured["void_frac_of_solid"] = float(value)
                checks.append(float(value) <= float(max_solid_frac))
        if (
            not any(limit is not None for limit in limits.values())
            and max_frac is None
            and max_solid_frac is None
        ):
            checks.append(top.get("enclosed_void_mm3") == 0)
            if top.get("enclosed_void_mm3") is None:
                missing.append("enclosed_void_mm3")
        expected = {key: val for key, val in limits.items() if val is not None}
        if max_frac is not None:
            expected["max_frac"] = max_frac
        if max_solid_frac is not None:
            expected["max_solid_frac"] = max_solid_frac
        if any(check is False for check in checks):
            return GateResult(name, False, measured=measured, expected=expected)
        if missing:
            return GateResult(
                name,
                None,
                measured=measured,
                expected=expected,
                detail=f"sealed-void telemetry unavailable: {', '.join(missing)}",
            )
        return GateResult(name, True, measured=measured, expected=expected)

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
        if "expect" in g:
            return GateResult(
                name, val == g.get("expect"), measured=val, expected={field: g.get("expect")}
            )
        lo, hi = g.get("min"), g.get("max")
        if lo is None and hi is None:
            return GateResult(name, bool(val), measured=val, expected={field: "truthy"})
        ok = (
            val is not None
            and (lo is None or float(val) >= float(lo))
            and (hi is None or float(val) <= float(hi))
        )
        return GateResult(name, ok, measured=val, expected={field: [lo, hi]})

    if name == "load_path":
        val = _topopt(outcome).get("load_path_ok")
        want = bool(g.get("expect", True))
        ok = None if val is None else bool(val) is want
        return GateResult(
            name,
            ok,
            measured=val,
            expected=want,
            detail="load-path certificate unavailable" if val is None else "",
        )

    if name == "asset_image":
        want = int(g.get("min") or 1)
        return GateResult(name, len(outcome.asset_ids) >= want, measured=len(outcome.asset_ids))

    if name == "guide_joints":
        # authoritative mate graph — pairs of (moving part, fixed part)
        # that must exist in the emitted joints; synonym lists tolerated
        doc = outcome.extras.get("guide")
        if not isinstance(doc, dict):
            return GateResult(
                name,
                None,
                detail="no unity_guide.json on the outcome — guide emit never ran",
            )
        joints = [j for j in doc.get("joints") or [] if isinstance(j, dict)]
        pairs = [
            (f"{j.get('moving_part', '')}".lower(), f"{j.get('fixed_part', '')}".lower())
            for j in joints
        ]
        missing = []
        for pair in g.get("all") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            mov_grp = [str(v).lower() for v in (pair[0] if isinstance(pair[0], list) else [pair[0]])]
            fix_grp = [str(v).lower() for v in (pair[1] if isinstance(pair[1], list) else [pair[1]])]
            hit = any(
                any(m in mp for m in mov_grp) and any(f in fp for f in fix_grp)
                for mp, fp in pairs
            )
            if not hit:
                missing.append(f"{mov_grp[0]}→{fix_grp[0]}")
        ok = not missing and len(joints) >= int(g.get("min") or 0)
        return GateResult(
            name,
            ok,
            measured={"n_joints": len(joints), "missing_pairs": missing or None},
            expected={"all": g.get("all"), "min": g.get("min")},
        )

    if name in ("guide_steps", "guide_parts", "guide_placed", "guide_score"):
        # anleitungsbuilder evidence — parsed unity_guide.json the driver
        # stored on extras["guide"]; absent → inconclusive, never silently ok
        doc = outcome.extras.get("guide")
        if not isinstance(doc, dict):
            return GateResult(
                name,
                None,
                detail="no unity_guide.json on the outcome — guide emit never ran",
            )
        steps = [s for s in doc.get("steps") or [] if isinstance(s, dict)]
        if name == "guide_steps":
            n = len(steps)
            expected = {k: g.get(k) for k in ("count", "min", "max") if g.get(k) is not None}
            ok = True
            if g.get("count") is not None:
                ok = ok and n == int(g["count"])
            if g.get("min") is not None:
                ok = ok and n >= int(g["min"])
            if g.get("max") is not None:
                ok = ok and n <= int(g["max"])
            needles = [
                [str(v).lower() for v in (x if isinstance(x, list) else [x])]
                for x in g.get("all") or []
            ]
            texts = [
                f"{s.get('display_text', '')} {s.get('internal_text', '')}".lower() for s in steps
            ]
            missing = [grp[0] for grp in needles if not any(x in t for x in grp for t in texts)]
            ok = ok and not missing
            measured: dict[str, Any] = {"n_steps": n}
            if missing:
                measured["missing_text"] = missing
            return GateResult(name, ok, measured=measured, expected={**expected, "all": needles})
        if name == "guide_parts":
            parts = [p for p in doc.get("parts") or [] if isinstance(p, dict)]
            hay = [f"{p.get('part_id', '')} {p.get('name', '')}".lower() for p in parts]
            # `all` entries may be synonym lists — a German manual yields
            # "rohr_*"/"bogen_*" part ids, so expected kinds must match ANY
            # listed alternative per group
            needles = [
                [str(v).lower() for v in (x if isinstance(x, list) else [x])]
                for x in g.get("all") or []
            ]
            missing = [grp[0] for grp in needles if not any(x in h for x in grp for h in hay)]
            n_min = int(g.get("min") or 0)
            ok = not missing and len(parts) >= n_min
            return GateResult(
                name,
                ok,
                measured={"n_parts": len(parts), "missing": missing or None},
                expected={"all": needles, "min": n_min},
            )
        if name == "guide_placed":
            total = sum(len(s.get("placements") or []) for s in steps)
            want = int(g.get("min") or 1)
            return GateResult(name, total >= want, measured=total, expected={"min": want})
        review = doc.get("review") or {}
        val = review.get("score")
        want = float(g.get("min") or 0)
        ok = val is not None and float(val) >= want
        return GateResult(name, ok, measured=val, expected={"min": want})

    return GateResult(name, True, detail=f"unknown gate type {name!r} (skipped)")


MESH_GATES = {
    "radial_peaks",
    "radial_envelope",
    "volume",
    "volume_fraction",
    "contains_point",
    "bodies",
    "threads",
    "thread",
}


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
