#!/usr/bin/env python3
"""Benchmark every available mesh adapter against a synthetic corpus.

Corpus entries (`bench/corpus/*.json`) describe a part class, a normalized
outline rendered to a silhouette PNG, and shape expectations (aspect ratio,
height ratio, hole count). For each available adapter we generate one mesh
per corpus entry and measure:

    success      generation returned a usable mesh
    latency_s    wall time per generation
    iou          footprint IoU (mesh plan projection vs corpus outline)
    side_iou     elevation IoU vs corpus `side_outline` (only when the
                 corpus entry defines one; catches waist/notch defects
                 the plan footprint cannot see)
    aspect_err   |L/W - expected| / expected on the two largest extents
    height_err   |H/L - expected| / expected on smallest/largest extent
    circularity  mean 4*pi*A/P^2 of cross-sections along the main axis
                 (1.0 = round, ~0.785 = square) — revolver-part probe
    wall_min     coarse minimum wall thickness from surface sampling;
                 absent when it cannot be measured
    watertight   share of results that are watertight single bodies
    faces        mean face count

Usage (in-process, recommended — exercises the real Registry/adapter path):

    cd mesh_server && uv run python scripts/bench_adapters.py \
        --out <DATA_DIR>/mesh_bench.json

    # only some adapters / entries, more seeds:
    uv run python scripts/bench_adapters.py --adapters triposg,hunyuan3d-2.1-mlx \
        --only flat_plate --seeds 2

The CADX backend reads the report via `cadx.agent.model_pick.load_bench`
and ranks adapters per part class.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "bench" / "corpus"

sys.path.insert(0, str(ROOT))  # allow running as a plain script


def _load_corpus(only: set[str] | None) -> list[dict]:
    entries = []
    for p in sorted(CORPUS_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["name"] = p.stem
        if only and d["name"] not in only:
            continue
        entries.append(d)
    return entries


def _render(entry: dict, size: int = 512):
    """Corpus spec -> silhouette PNG (black shape + white hole discs)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    pts = [(float(x) * size, float(y) * size) for x, y in entry["outline"]]
    draw.polygon(pts, fill="black")
    for hx, hy, hr in entry.get("holes") or []:
        cx, cy, r = hx * size, hy * size, hr * size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="white")
    return img


def _silhouette_iou(
    mesh, outline: list[list[float]], axes: list[int], size: int = 256
) -> float:
    """IoU of the mesh's projection onto the plane spanned by `axes`
    against a normalised corpus outline — both rasterised into a unit
    square. Outline space is image-like (v grows downwards): the mesh's
    higher axis-1 coordinates land at v=0, so author outlines with v=0
    at the top of the profile."""
    import numpy as np
    from PIL import Image, ImageDraw

    tris = mesh.triangles  # (f,3,3)
    pts2 = tris[:, :, axes].reshape(-1, 2)
    lo, hi = pts2.min(axis=0), pts2.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)

    def norm(pts):
        q = (pts - lo) / span
        return np.stack([q[:, 0] * size, (1 - q[:, 1]) * size], axis=1)

    img = Image.new("1", (size, size), 0)
    draw = ImageDraw.Draw(img)
    # triangle soup projection: draw each face's 2D triangle
    for f in range(len(tris)):
        draw.polygon([tuple(p) for p in norm(tris[f][:, axes])], fill=1)
    a = np.asarray(img, dtype=bool)

    ol = np.asarray(outline, dtype=float)
    img2 = Image.new("1", (size, size), 0)
    ImageDraw.Draw(img2).polygon(
        [tuple(p) for p in np.stack([ol[:, 0] * size, ol[:, 1] * size], axis=1)],
        fill=1,
    )
    b = np.asarray(img2, dtype=bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def _footprint_iou(mesh, outline: list[list[float]], size: int = 256) -> float:
    """Plan footprint IoU: projection on the two longest extent axes."""
    import numpy as np

    axes = list(np.argsort(mesh.extents)[::-1][:2])
    return _silhouette_iou(mesh, outline, axes, size)


def _side_iou(mesh, outline: list[list[float]], size: int = 256) -> float:
    """Side-elevation IoU: projection onto the plane spanned by the
    longest and the shortest extent axis — i.e. viewed along the median
    axis, the elevation complementary to the plan footprint. This is the
    silhouette that shows waist/profile defects (sharp V-notch vs soft
    waist) which plan-view IoU cannot distinguish."""
    import numpy as np

    order = np.argsort(mesh.extents)  # ascending: [shortest, median, longest]
    axes = [int(order[2]), int(order[0])]  # longest -> u, shortest -> v
    return _silhouette_iou(mesh, outline, axes, size)


def _loop_area_perimeter(pts) -> tuple[float, float]:
    """Shoelace area + perimeter of a closed 2D loop; the first vertex
    may or may not be repeated as the last."""
    import numpy as np

    v = np.asarray(pts, dtype=float)
    if len(v) < 3:
        return 0.0, 0.0
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[:1]])
    seg = np.diff(v, axis=0)
    per = float(np.linalg.norm(seg, axis=1).sum())
    area = 0.5 * abs(
        float(np.dot(v[:-1, 0], v[1:, 1]) - np.dot(v[:-1, 1], v[1:, 0]))
    )
    return area, per


def _cross_circularity(mesh, stations: int = 9) -> float:
    """Mean circularity 4*pi*A/P^2 of cross-sections cut perpendicular
    to the main (longest-extent) axis — the revolver-part probe. Each
    section contributes its largest closed loop (outer boundary), so
    bores do not corrupt the shape measure; 1.0 = perfectly round,
    pi/4 ~ 0.785 = square. Returns 0.0 when no usable section exists.
    Loops are chained from closed section entities — no shapely needed."""
    import numpy as np

    ext = np.asarray(mesh.extents, dtype=float)
    main = int(np.argmax(ext))
    if ext[main] <= 0:
        return 0.0
    lo, hi = float(mesh.bounds[0][main]), float(mesh.bounds[1][main])
    origin = np.asarray(mesh.bounds).mean(axis=0)
    normal = np.zeros(3)
    normal[main] = 1.0
    vals = []
    for t in np.linspace(0.15, 0.85, stations):
        org = origin.copy()
        org[main] = lo + float(t) * (hi - lo)
        try:
            sec = mesh.section(plane_origin=org, plane_normal=normal)
            if sec is None:
                continue
            path2d, _ = sec.to_2D()
            verts = np.asarray(path2d.vertices)
            loops = []
            for ent in path2d.entities:
                if not getattr(ent, "closed", False):
                    continue
                area, per = _loop_area_perimeter(verts[np.asarray(ent.points)])
                if area > 0 and per > 0:
                    loops.append((area, per))
            if loops:
                a, p = max(loops)  # outer boundary = largest enclosed area
                vals.append(4.0 * np.pi * a / (p * p))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else 0.0


def _wall_min(mesh, max_samples: int = 400) -> float | None:
    """Coarse minimum wall thickness: sample up to `max_samples` (hard
    bound 500, keeps the pairwise matrix small) surface points and take
    the minimum distance to a sample whose normal opposes ours — i.e.
    the nearest point on an inward-facing surface, which approximates
    the wall/thickness chord. Semantics: NOT a medial-axis measure; on
    solid bodies it reports the chord to the nearest oppositely-facing
    surface (roughly local thickness), on thin shells the true wall.
    Returns None when sampling fails or no opposing surface is found."""
    import numpy as np

    n = int(min(max_samples, 500))
    try:
        pts, face_idx = mesh.sample(n, return_index=True)
    except Exception:
        return None
    if len(pts) < 8:
        return None
    pts = np.asarray(pts, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)[face_idx]
    scale = float(np.max(mesh.extents)) or 1.0
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    oppose = (normals @ normals.T) < -0.5
    dist = np.where(oppose, dist, np.inf)
    best = dist.min(axis=1)
    best = best[np.isfinite(best) & (best > 1e-4 * scale)]
    return float(best.min()) if len(best) else None


def _edge_roughness_deg(mesh) -> float:
    """Mean dihedral angle (deg) between adjacent non-horizontal faces —
    stair-stepping proxy: voxel/MC artifacts raise it on side surfaces,
    flat tops/floors (|nz| >= 0.95) are excluded so genuinely flat parts
    don't score as 'rough'."""
    import numpy as np

    try:
        angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
        if not len(angles):
            return 0.0
        pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
        normals = np.asarray(mesh.face_normals, dtype=float)
        vertical = np.abs(normals[:, 2]) < 0.95
        mask = vertical[pairs[:, 0]] & vertical[pairs[:, 1]]
        sel = angles[mask]
        return float(np.degrees(np.mean(sel))) if len(sel) else 0.0
    except Exception:
        return 0.0


def _metrics(mesh, entry: dict, latency: float) -> dict:
    import numpy as np

    ext = sorted(float(v) for v in mesh.extents)
    exp_aspect = float(entry.get("expect_aspect") or 0)
    exp_hr = float(entry.get("expect_height_ratio") or 0)
    aspect_err = abs(ext[2] / max(ext[1], 1e-9) - exp_aspect) / exp_aspect if exp_aspect else 0.0
    height_err = abs(ext[0] / max(ext[2], 1e-9) - exp_hr) / exp_hr if exp_hr else 0.0
    try:
        bodies = len(mesh.split(only_watertight=False))
    except Exception:
        bodies = -1
    iou = 0.0
    try:
        iou = _footprint_iou(mesh, entry["outline"])
    except Exception:
        pass
    holes_found = 0
    try:
        # tunnel proxy: a mid-height section has >1 closed polygon when the
        # part has through-holes (outer boundary + one loop per hole)
        mid = (mesh.bounds[0] + mesh.bounds[1]) / 2
        sec = mesh.section(plane_origin=mid, plane_normal=np.array([0, 0, 1]))
        if sec is not None:
            path2d, _ = sec.to_2D()
            holes_found = max(len(path2d.polygons_full) - 1, 0)
    except Exception:
        pass
    try:
        circularity = _cross_circularity(mesh)
    except Exception:
        circularity = 0.0
    try:
        wall_min = _wall_min(mesh)
    except Exception:
        wall_min = None
    row = {
        "success": 1.0,
        "latency_s": round(latency, 2),
        "iou": round(iou, 3),
        "aspect_err": round(aspect_err, 3),
        "height_err": round(height_err, 3),
        "watertight": 1.0 if mesh.is_watertight else 0.0,
        "bodies": bodies,
        "faces": len(mesh.faces),
        "roughness_deg": round(_edge_roughness_deg(mesh), 2),
        "circularity": round(circularity, 3),
        "holes_expected": len(entry.get("holes") or []),
        "holes_found": holes_found,
    }
    if entry.get("side_outline"):
        try:
            row["side_iou"] = round(_side_iou(mesh, entry["side_outline"]), 3)
        except Exception:
            row["side_iou"] = 0.0
    if wall_min is not None:
        row["wall_min"] = round(wall_min, 4)
    return row


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    def mean(k):
        vals = [r[k] for r in rows if k in r]
        return round(float(sum(vals) / len(vals)), 4) if vals else 0.0
    return {
        "n": n,
        "success": mean("success"),
        "latency_s": mean("latency_s"),
        "iou": mean("iou"),
        "aspect_err": mean("aspect_err"),
        "height_err": mean("height_err"),
        "watertight": mean("watertight"),
        "faces": mean("faces"),
        "roughness_deg": mean("roughness_deg"),
        "hole_recall": mean("hole_recall"),
        "side_iou": mean("side_iou"),
        "circularity": mean("circularity"),
        "wall_min": mean("wall_min"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapters", default="", help="csv adapter ids (default: all available)")
    ap.add_argument("--only", default="", help="csv corpus names")
    ap.add_argument("--seeds", type=int, default=1, help="rolls per corpus entry")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="mesh_bench.json")
    ap.add_argument("--fake", action="store_true", help="include the fake adapter (CI)")
    args = ap.parse_args()

    from meshserver.adapters.base import GenOpts
    from meshserver.config import ServerConfig
    from meshserver.registry import Registry

    cfg = ServerConfig()
    if args.fake:
        cfg.fake = True
    reg = Registry(cfg)
    want = {a.strip() for a in args.adapters.split(",") if a.strip()}
    # filter by id BEFORE probing — available() imports each adapter's model
    # stack (torch/mlx pipelines), which is expensive for unrequested models
    classes = [
        c
        for c in reg._classes
        if getattr(c, "kind", "mesh") == "mesh"
        and (not want or c.id in want)
        and (cfg.fake or not str(getattr(c, "id", "")).startswith("fake"))
    ]
    classes = [c for c in classes if reg._probe(c)]
    if not classes:
        print("no available adapters", file=sys.stderr)
        return 2

    corpus = _load_corpus({s.strip() for s in args.only.split(",") if s.strip()} or None)
    if not corpus:
        print("empty corpus", file=sys.stderr)
        return 2

    report: dict = {
        "schema": 1,
        "device": reg.device,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {},
    }
    for cls in classes:
        rows_by_class: dict[str, list[dict]] = {}
        try:
            inst = reg.get(cls)
        except Exception as e:
            report["models"][cls.id] = {"_all": {"n": 0, "load_error": str(e)[:200]}}
            continue
        for entry in corpus:
            img = _render(entry)
            for seed in range(args.seeds):
                t0 = time.monotonic()
                try:
                    res = inst.generate(
                        [img],
                        GenOpts(
                            seed=seed,
                            steps=args.steps,
                            subject=entry.get("subject", ""),
                            image_roles=["front"],
                        ),
                    )
                    row = _metrics(res.mesh, entry, time.monotonic() - t0)
                    exp = row.pop("holes_expected")
                    row["hole_recall"] = (
                        min(row.pop("holes_found") / exp, 1.0) if exp else 1.0
                    )
                except Exception as e:
                    row = {"success": 0.0, "error": str(e)[:200], "latency_s": round(time.monotonic() - t0, 2)}
                    row.setdefault("iou", 0.0)
                    row.setdefault("aspect_err", 1.0)
                    row.setdefault("height_err", 1.0)
                    row.setdefault("watertight", 0.0)
                    row.setdefault("faces", 0)
                    row.setdefault("roughness_deg", 0.0)
                    row.setdefault("hole_recall", 0.0)
                    row.setdefault("circularity", 0.0)
                    if entry.get("side_outline"):
                        row.setdefault("side_iou", 0.0)
                    # wall_min intentionally absent — not measurable on a failed row
                    reg.mark_failed(cls.id, str(e))
                rows_by_class.setdefault(entry["class"], []).append(row)
                rows_by_class.setdefault("_all", []).append(row)
        report["models"][cls.id] = {k: _agg(v) for k, v in rows_by_class.items()}
        print(f"{cls.id}: " + json.dumps(report["models"][cls.id]["_all"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
