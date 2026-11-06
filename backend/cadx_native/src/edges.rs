//! Mesh-side fillet/round-edge evidence on a triangle tessellation —
//! independent of OCC/quiddity, so evidence still lands where the B-Rep
//! recogniser is skipped (faceted/oversized meshes) or times out.
//!
//! Pipeline: undirected edge->face adjacency on the tri soup, signed
//! dihedral per manifold edge (convex/concave via the outward-CCW
//! convention: `sign(dot(cross(n1,n2), e_dir))`, positive = convex on a
//! cube corner — verified by the parity fixture), union-find clustering
//! of same-side fold edges into fillet bands by midpoint proximity +
//! near-parallel direction, then a Kasa circle fit on the band's
//! projected midpoints for the radius. Bands must clear an arc-span,
//! fit-residual and elongation gate — sharp edges, chamfer boundaries
//! and corner blends are NOT reported as fillets.
//!
//! Evidence is estimate-grade: callers must tag it `source="mesh_edge"`
//! and never claim it as OCC-recogniser evidence.

use std::collections::HashMap;

use numpy::{PyReadonlyArray2};
use pyo3::prelude::*;

// an edge folds enough to bound a rounded region when the adjacent face
// normals turn more than ~0.6° across it — fillet strips tessellate at
// 0.5-3° per strip while coplanar pairs sit under ~0.1°
const FOLD_RAD: f64 = 0.01;
// a fillet strip turns GRADUALLY — a per-edge fold above ~34° is a sharp
// edge (chamfer boundary, cube corner), not a rolled surface; a 90°
// fillet must therefore be tessellated into >= 3 strips to register
const MAX_STRIP_RAD: f64 = 0.6;
// bands need at least this many connected fold edges to read as a fillet
// strip (fewer means a sharp edge, not a rolled surface)
const MIN_BAND_EDGES: usize = 3;
// edges merge into a band only when their directions are near-parallel
// (cos ≈ 20°) — fillet strips run along one axis, corner-blend edges do not
const PARALLEL_COS: f64 = 0.94;
// reported fillet candidates are capped so pathological meshes cannot
// bloat the evidence list (same ceiling as quiddity records)
const MAX_FEATURES: usize = 64;

fn v_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn v_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn v_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn v_norm(a: [f64; 3]) -> f64 {
    v_dot(a, a).sqrt()
}

fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    let mut j = i;
    while parent[j] != r {
        let n = parent[j];
        parent[j] = r;
        j = n;
    }
    r
}

#[allow(clippy::type_complexity)]
#[pyfunction]
pub fn feature_edges(
    verts: PyReadonlyArray2<f64>,
    faces: PyReadonlyArray2<i64>,
) -> PyResult<Vec<(f64, f64, f64, f64, String, String, usize, f64)>> {
    let vs = verts.as_slice()?;
    let fs = faces.as_slice()?;
    let ntri = fs.len() / 3;
    let mut normals = vec![[0.0f64; 3]; ntri];
    // canonical edge key -> adjacent face ids (manifold pairs only qualify)
    let mut edges: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut edge_keys: Vec<(i64, i64)> = Vec::new();

    for f in 0..ntri {
        let (a, b, c) = (fs[3 * f] as usize, fs[3 * f + 1] as usize, fs[3 * f + 2] as usize);
        let (pa, pb, pc) = (
            [vs[3 * a], vs[3 * a + 1], vs[3 * a + 2]],
            [vs[3 * b], vs[3 * b + 1], vs[3 * b + 2]],
            [vs[3 * c], vs[3 * c + 1], vs[3 * c + 2]],
        );
        let n = v_cross(v_sub(pb, pa), v_sub(pc, pa));
        let nn = v_norm(n);
        normals[f] = if nn > 1e-12 { [n[0] / nn, n[1] / nn, n[2] / nn] } else { [0.0; 3] };
        let tri = [fs[3 * f], fs[3 * f + 1], fs[3 * f + 2]];
        for k in 0..3 {
            let (u, v) = (tri[k], tri[(k + 1) % 3]);
            let key = if u < v { (u, v) } else { (v, u) };
            match edges.get_mut(&key) {
                Some(adj) => adj.push(f),
                None => {
                    edges.insert(key, vec![f]);
                    edge_keys.push(key);
                }
            }
        }
    }

    // fold edges: (key, convex, midpoint, e_dir, e_len)
    let mut fold: Vec<((i64, i64), bool, [f64; 3], [f64; 3], f64)> = Vec::new();
    for key in edge_keys.iter().copied() {
        let adj = &edges[&key];
        if adj.len() != 2 {
            continue; // boundary / non-manifold — unclassified
        }
        let (fa, fb) = (adj[0], adj[1]);
        let (n1, n2) = (normals[fa], normals[fb]);
        let ct = v_dot(n1, n2).clamp(-1.0, 1.0);
        let theta = ct.acos();
        if !(FOLD_RAD..=MAX_STRIP_RAD).contains(&theta) {
            continue; // coplanar pair, or a sharp edge — not a fillet strip
        }
        // e_dir as wound in face fa (canonical u->v may be reversed for fa)
        let (u, v) = (key.0 as usize, key.1 as usize);
        let mut e_dir = v_sub(
            [vs[3 * v], vs[3 * v + 1], vs[3 * v + 2]],
            [vs[3 * u], vs[3 * u + 1], vs[3 * u + 2]],
        );
        let tri = [fs[3 * fa], fs[3 * fa + 1], fs[3 * fa + 2]];
        let wound = (0..3).any(|k| tri[k] == key.0 && tri[(k + 1) % 3] == key.1);
        if !wound {
            e_dir = [-e_dir[0], -e_dir[1], -e_dir[2]];
        }
        let en = v_norm(e_dir);
        if en < 1e-9 {
            continue;
        }
        let s = v_dot(v_cross(n1, n2), [e_dir[0] / en, e_dir[1] / en, e_dir[2] / en]);
        let convex = s > 0.0;
        let e_dir = [e_dir[0] / en, e_dir[1] / en, e_dir[2] / en];
        let mid = [
            (vs[3 * u] + vs[3 * v]) / 2.0,
            (vs[3 * u + 1] + vs[3 * v + 1]) / 2.0,
            (vs[3 * u + 2] + vs[3 * v + 2]) / 2.0,
        ];
        fold.push((key, convex, mid, e_dir, en));
    }

    // cluster fold edges into fillet bands by SPATIAL proximity of their
    // midpoints — a fillet's strip boundaries are parallel edges that
    // share no vertices, so vertex-sharing alone fragments every fillet.
    // eps scales with median edge length: strips sit ~an edge length
    // apart, distinct fillets stay far wider separated
    let mut elens: Vec<f64> = fold.iter().map(|f| f.4).collect();
    elens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_e = if elens.is_empty() {
        0.0
    } else {
        elens[elens.len() / 2]
    };
    let eps = (3.0 * med_e).max(1e-3);
    let mut parent: Vec<usize> = (0..fold.len()).collect();
    let cell = eps.max(1e-6);
    let mut grid: HashMap<(i64, i64, i64, bool), Vec<usize>> = HashMap::new();
    for (i, (_, convex, mid, _, _)) in fold.iter().enumerate() {
        let g = (
            (mid[0] / cell).floor() as i64,
            (mid[1] / cell).floor() as i64,
            (mid[2] / cell).floor() as i64,
            *convex,
        );
        // check the 26-neighbourhood for same-side fold edges within eps
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                for dz in -1..=1i64 {
                    let key = (g.0 + dx, g.1 + dy, g.2 + dz, g.3);
                    if let Some(cand) = grid.get(&key) {
                        for &j in cand {
                            let mj = fold[j].2;
                            let d = v_norm(v_sub(*mid, mj));
                            // same fillet ⟹ strip boundaries are near-
                            // parallel; corner blends chain perpendicular
                            // fillets together without this guard
                            let par = (0..3)
                                .map(|k| fold[i].3[k] * fold[j].3[k])
                                .sum::<f64>()
                                .abs();
                            if d <= eps && par >= PARALLEL_COS {
                                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                                if a != b {
                                    parent[a] = b;
                                }
                            }
                        }
                    }
                }
            }
        }
        grid.entry(g).or_default().push(i);
    }
    let mut bands: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..fold.len() {
        bands.entry(find(&mut parent, i)).or_default().push(i);
    }

    let mut out = Vec::new();
    for (_, members) in bands {
        if members.len() < MIN_BAND_EDGES {
            continue;
        }
        // band centre/axis from mean midpoint + dominant |direction|
        let mut mid = [0.0f64; 3];
        let mut dir = [0.0f64; 3];
        for &i in &members {
            for k in 0..3 {
                mid[k] += fold[i].2[k] / members.len() as f64;
                dir[k] += fold[i].3[k].abs();
            }
        }
        let (ai, axis) = if dir[0] >= dir[1] && dir[0] >= dir[2] {
            (0, "x")
        } else if dir[1] >= dir[2] {
            (1, "y")
        } else {
            (2, "z")
        };
        // the fold-edge midpoints lie ON the strip boundaries — i.e. on
        // the fillet arc itself. Projected into the cross-section plane
        // they trace the arc; a Kasa least-squares circle gives r directly
        let (u, w) = ((ai + 1) % 3, (ai + 2) % 3);
        let pts: Vec<(f64, f64)> = members.iter().map(|&i| (fold[i].2[u], fold[i].2[w])).collect();
        let Some((h, k, r)) = circle_fit(&pts) else { continue };
        if !(r > 0.0 && r.is_finite()) {
            continue;
        }
        // arc span: point angles around the centre, largest gap excluded
        let mut angs: Vec<f64> = pts
            .iter()
            .map(|&(x, y)| (y - k).atan2(x - h))
            .collect();
        angs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut gap = 0.0f64;
        for j in 0..angs.len() {
            let next = if j + 1 == angs.len() {
                angs[0] + std::f64::consts::TAU
            } else {
                angs[j + 1]
            };
            gap = gap.max(next - angs[j]);
        }
        let span = std::f64::consts::TAU - gap;
        if !(0.25..3.0).contains(&span) {
            continue; // under ~14° no meaningful fillet; over ~172° is not an arc
        }
        // fit residual: midpoints must sit near the circle — kills
        // coincidental clusters of unrelated folds
        let resid = pts
            .iter()
            .map(|&(x, y)| (((x - h).powi(2) + (y - k).powi(2)).sqrt() - r).abs())
            .sum::<f64>()
            / pts.len() as f64;
        if resid > 0.35 * r {
            continue;
        }
        // a fillet is ELONGATED — its extent along the axis far exceeds
        // the arc chord. Measure on edge ENDPOINTS: strip boundaries are
        // long edges spanning the fillet, while their midpoints collapse
        // to a single cross-section — corner blobs stay ~spherical either
        // way and are rejected
        let mut ext_min = f64::INFINITY;
        let mut ext_max = f64::NEG_INFINITY;
        for &i in &members {
            let (eu, ev) = fold[i].0;
            ext_min = ext_min.min(vs[3 * eu as usize + ai]).min(vs[3 * ev as usize + ai]);
            ext_max = ext_max.max(vs[3 * eu as usize + ai]).max(vs[3 * ev as usize + ai]);
        }
        let chord = 2.0 * r * (span / 2.0).sin();
        if ext_max - ext_min < 2.0 * chord {
            continue;
        }
        out.push((
            r,
            mid[0],
            mid[1],
            mid[2],
            axis.to_string(),
            if fold[members[0]].1 { "convex" } else { "concave" }.to_string(),
            members.len(),
            span.to_degrees(),
        ));
        if out.len() >= MAX_FEATURES {
            break;
        }
    }
    Ok(out)
}

/// Kasa algebraic circle fit in the form x²+y² = A·x + B·y + C → 3x3
/// normal equations; circle centre = (A/2, B/2), r² = C + (A²+B²)/4.
fn circle_fit(pts: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let n = pts.len() as f64;
    let (mut sxx, mut sxy, mut syy, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0);
    for &(x, y) in pts {
        let z = x * x + y * y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sx += x;
        sy += y;
        sxz += x * z;
        syz += y * z;
        sz += z;
    }
    // [sxx sxy sx; sxy syy sy; sx sy n] · [A B C]ᵀ = [sxz syz sz]ᵀ
    let m = [
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx, sy, n],
    ];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None; // degenerate — all points coincident
    }
    // symmetric matrix → M⁻¹[i][j] = (-1)^(i+j) · minor(i,j) / det; the
    // minor drops row i and col j, remaining indices in ascending order
    let others = |x: usize| match x {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let inv = |i: usize, j: usize| {
        let (r1, r2) = others(i);
        let (c1, c2) = others(j);
        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        sign * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]) / det
    };
    let rhs = [sxz, syz, sz];
    let a = inv(0, 0) * rhs[0] + inv(1, 0) * rhs[1] + inv(2, 0) * rhs[2];
    let b = inv(0, 1) * rhs[0] + inv(1, 1) * rhs[1] + inv(2, 1) * rhs[2];
    let c = inv(0, 2) * rhs[0] + inv(1, 2) * rhs[1] + inv(2, 2) * rhs[2];
    let (h, k) = (a / 2.0, b / 2.0);
    let r2 = h * h + k * k + c;
    if r2 <= 0.0 {
        return None;
    }
    Some((h, k, r2.sqrt()))
}
