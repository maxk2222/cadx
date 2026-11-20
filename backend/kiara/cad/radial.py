"""Radial repetition measurement — count teeth/lobes/blades from a polar
max-radius profile around an axis.

Shared by the evalkit ``radial_peaks`` mesh gate and the executor feature
pipeline (``MeasuredFeatures.radial_peaks``) so verify_spec can compare a
declared N-fold radial feature ("5 blades at 72 deg spacing around Z")
against measured geometry instead of leaving ``type=other`` features
completely unverified.
"""

from __future__ import annotations

from typing import Any


def _ac_harmonic_ok(ac: Any, k: int, k_min: int, bins: int) -> bool:
    """True if lag `k`'s multiples keep a strong autocorrelation peak.

    The true blade pitch repeats at 2k/3k; a sparse-blade profile's
    fragment-spacing peaks (high-pitch blades scattering vertices ~20 bins
    apart) decay at their multiples — dropping those lets deeper bands win.
    """
    for m in (m * k for m in (2, 3) if m * k < int(bins * 0.55)):
        w = max(5, int(0.03 * m))
        if float(ac[max(k_min, m - w) : m + w + 1].max()) < 0.25:
            return False
    return True


def radial_peak_count(mesh: Any, axis: str = "z", bins: int = 1440) -> tuple[int, str]:
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
    # full-mesh silhouette floor: per-bin max radius over ALL vertices,
    # interpolated across its own gaps. Wide empty bands in an outer profile
    # fall back to this (the hub/lower-blade silhouette in open water)
    # instead of being bridged flat tip-to-tip by np.interp.
    th_all = np.arctan2(v[:, a2] - c2, v[:, a1] - c1)
    idx_all = ((th_all + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    prof_all = np.zeros(bins)
    np.maximum.at(prof_all, idx_all, r_all)
    nz_all = np.flatnonzero(prof_all > 0)
    if not nz_all.size:
        return 0, "degenerate mesh"
    floor = np.interp(np.arange(bins), nz_all, prof_all[nz_all], period=bins)
    # tight outer bands first; 0.50 is the deep-hub fallback for rotors
    # whose lobes never reach the usual band
    for frac in [*np.arange(0.94, 0.79, -0.03), 0.50]:
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
            filled = prof.copy()
            short = np.zeros(bins, dtype=bool)
            nz = np.flatnonzero(~empty)
            for i in range(len(nz)):
                a, b = int(nz[i]), int(nz[(i + 1) % len(nz)])
                gap = (b - a - 1) % bins
                if not gap:
                    continue
                ks = (a + 1 + np.arange(gap)) % bins
                if gap <= 12:  # <=3° — tessellation speckle, interpolate
                    short[ks] = True
                else:  # real gap (inter-blade water) — flat hub-floor valley
                    filled[ks] = floor[ks].min()
            if short.any():
                filled[short] = np.interp(
                    np.flatnonzero(short),
                    np.flatnonzero(~short),
                    filled[~short],
                    period=bins,
                )
            prof = filled
        lo, hi = float(prof.min()), float(prof.max())
        if hi - lo < 0.4:
            continue
        p = prof - prof.mean()
        ac = np.correlate(p, p, "full")[bins - 1 :] / (p @ p + 1e-12)
        # pitch detection: first strong local maximum of the autocorrelation
        # AFTER the zero-lag lobe has decayed — a wide-lobe profile keeps ac
        # high for tens of bins and any tessellation ripple inside the lobe
        # reads as a false high-frequency peak. The upper bound leaves head-
        # room for 2-/3-blade rotors (lag 720/480 at bins=1440).
        decayed = np.flatnonzero(ac[4:] <= 0.15)
        k_min = 4 + int(decayed[0]) if decayed.size else int(bins * 0.55)
        # counts above ~64 lags/rotor are implausible for these parts and a
        # short-lag cutoff keeps tessellation ripple out of the candidates
        k_min = max(k_min, int(bins / 64))
        lags = [
            k
            for k in range(k_min, int(bins * 0.55))
            if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1] and ac[k] >= 0.30
        ]

        lags = [k for k in lags if _ac_harmonic_ok(ac, k, k_min, bins)]
        if not lags:
            continue
        # first lag reaching 70% of the strongest repeat — multiples of the
        # true pitch match it in strength, so the earliest "nearly dominant"
        # peak is the pitch (first-max misfires on the bumpy rising flank of
        # a sparse-blade profile; global argmax misfires on equal multiples)
        k_best = max(lags, key=lambda k: ac[k])
        lag = next(k for k in lags if ac[k] >= 0.70 * ac[k_best])
        n = max(1, round(bins / lag))
        return n, f"band={frac:.2f} r∈[{lo:.1f},{hi:.1f}] lag={lag} ac={ac[lag]:.2f}"
    return 0, f"no non-flat radial band (r_max={r_max:.1f})"
