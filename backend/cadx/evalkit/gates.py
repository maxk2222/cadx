"""Stage gates — declarative pass/fail checks on a StageOutcome.

Gate specs live in the suite JSON (`stages[].gates`) as
``{"type": <name>, ...kwargs}``. Two families:

- **outcome gates** — status/score/bbox/features/watertight/verify_spec/
  fem/topopt/tokens/wall time, read off the normalized outcome;
- **mesh gates** — `radial_peaks`, `volume(_fraction)`, `contains_point`,
  `bodies` — computed with trimesh on the stage's final committed mesh
  (STL/GLB/3MF, authoritative-first via `_mesh_artifact_files`), skipped
  with detail when no mesh is available.
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


def _radial_peak_count(mesh: Any, axis: str = "z", bins: int = 1440) -> tuple[int, str]:
    """Count teeth/lobes from a polar max-radius profile around `axis`.

    The profile is built from the tightest non-flat outer-radius band
    (interior features like bores/windows otherwise dominate the count), and
    the tooth count is read off the profile autocorrelation — threshold
    crossings over/under-count on uneven teeth and on helical flanks (one
    slanted flank → several maxima).
    """
    import numpy as np

    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    a1, a2 = axes.get(axis.lower(), (0, 1))
    v = np.asarray(mesh.vertices, dtype=float)
    c1, c2 = (v[:, a1].min() + v[:, a1].max()) / 2, (v[:, a2].min() + v[:, a2].max()) / 2
    r_all = np.hypot(v[:, a1] - c1, v[:, a2] - c2)
    r_max = float(r_all.max())
    if r_max <= 0:
        return 0, "degenerate mesh"
    for frac in np.arange(0.94, 0.79, -0.03):
        sel = r_all >= frac * r_max
        if int(sel.sum()) < 24:
            continue
        vo = v[sel]
        r = np.hypot(vo[:, a1] - c1, vo[:, a2] - c2)
        th = np.arctan2(vo[:, a2] - c2, vo[:, a1] - c1)
        idx = ((th + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        prof = np.zeros(bins)
        np.maximum.at(prof, idx, r)
        empty = prof == 0
        if empty.all():
            continue
        if empty.any():
            prof = np.interp(np.arange(bins), np.flatnonzero(~empty), prof[~empty], period=bins)
        lo, hi = float(prof.min()), float(prof.max())
        if hi - lo < 0.4:
            continue
        p = prof - prof.mean()
        ac = np.correlate(p, p, "full")[bins - 1 :] / (p @ p + 1e-12)
        # pitch detection: first strong local maximum of the autocorrelation
        # after the initial decay — global argmax misfires on flat-topped
        # profiles (tiny lags correlate well) and on harmonics (helical
        # flanks inject 2-5x the tooth frequency)
        lags = [
            k
            for k in range(4, bins // 3)
            if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1] and ac[k] >= 0.30
        ]
        if not lags:
            continue
        lag = lags[0]
        n = max(1, round(bins / lag))
        return n, f"band={frac:.2f} r∈[{lo:.1f},{hi:.1f}] lag={lag} ac={ac[lag]:.2f}"
    return 0, f"no non-flat radial band (r_max={r_max:.1f})"


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


def _v3(v: Any) -> list[float] | None:
    if isinstance(v, list | tuple) and len(v) == 3:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


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
    """Force-weighted centroid of the applied-load records — where the load
    actually landed on the part (region centroid weighted by the resolved
    force it carries, so a dominant load outweighs a small one)."""
    num = [0.0, 0.0, 0.0]
    den = 0.0
    for a in applied:
        c = _v3(a.get("centroid_mm"))
        w = float(a.get("force_mag_n") or 0.0)
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
    return {
        "kept_base": True,
        "volfrac_achieved": best.get("achieved"),
        "sf_verified": ver.get("sf_verified") or best.get("worst_sf"),
        "sf_export": best.get("sf_export"),
        "n_bodies_export": best.get("n_bodies_export"),
        "load_path_ok": best.get("load_path_ok"),
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
        return GateResult(
            name,
            ok,
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
        lo, hi = g.get("min"), g.get("max")
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
