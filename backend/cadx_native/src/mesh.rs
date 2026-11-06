//! BVH-accelerated mesh queries replicating trimesh's exact conventions.
//!
//! Semantics matched to trimesh (see ray_util.contains_points):
//! - `contains`: fixed direction [0.4395, 0.6176, 0.6522], bidirectional
//!   parity (count every triangle hit, no position dedup), both directions
//!   must agree; one-side free-space => outside; disagreement retries once
//!   with a deterministic second direction (trimesh uses a random vector —
//!   same resolution path, ours stays reproducible).
//! - `ray_locs`: all (t>eps) hits per ray, returned sorted by (ray, t).
//! - `scanline_fill`: the parity fill of fem/voxelize._scanline_contains —
//!   eps jitter on the perpendicular grid coords, origins below the mesh
//!   bounds, hit positions deduped <eps, a query point sitting within eps
//!   of a crossing counts as OUTSIDE.

use numpy::{ndarray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::util::par_chunks;

const LEAF_MAX: usize = 8;
const EPS_T: f64 = 1e-10;
// trimesh's fixed "random" probe direction (contains_points default)
const DIR_MAIN: [f64; 3] = [0.4395064455, 0.617598629942, 0.652231566745];
// deterministic second probe direction — replaces trimesh's random retry so
// results stay reproducible run to run (already unit-length)
const DIR_RETRY: [f64; 3] = [0.7071067812, 0.5, 0.5];

#[derive(Clone, Copy)]
struct Node {
    lo: [f64; 3],
    hi: [f64; 3],
    /// >0: index of the right child (left child is always node+1).
    /// <0: leaf, holds -count; `start` indexes `order`.
    right: i32,
    start: u32,
}

struct Bvh {
    verts: Vec<[f64; 3]>,
    tris: Vec<[u32; 3]>,
    nodes: Vec<Node>,
    order: Vec<u32>,
    lo: [f64; 3],
    hi: [f64; 3],
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Möller–Trumbore: hit distance t>t_min, or None. t_min mirrors the
/// caller's convention (contains/scanline use strict EPS_T; ray_locs uses
/// trimesh's `distance > -1e-6` forward test).
#[inline]
fn tri_hit(
    o: [f64; 3],
    d: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    t_min: f64,
) -> Option<f64> {
    let e1 = sub(b, a);
    let e2 = sub(c, a);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = sub(o, a);
    let u = dot(tv, p) * inv;
    if !(-1e-9..=1.0 + 1e-9).contains(&u) {
        return None;
    }
    let q = cross(tv, e1);
    let v = dot(d, q) * inv;
    if v < -1e-9 || u + v > 1.0 + 1e-9 {
        return None;
    }
    let t = dot(e2, q) * inv;
    if t > t_min {
        Some(t)
    } else {
        None
    }
}

/// Slab test: does the ray pass the box? Returns entry/exit t.
#[inline]
fn box_hit(o: [f64; 3], inv: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> bool {
    let mut t0 = f64::NEG_INFINITY;
    let mut t1 = f64::INFINITY;
    for k in 0..3 {
        let mut a = (lo[k] - o[k]) * inv[k];
        let mut b = (hi[k] - o[k]) * inv[k];
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return false;
        }
    }
    t1 > EPS_T
}

impl Bvh {
    fn build(verts: Vec<[f64; 3]>, tris: Vec<[u32; 3]>) -> Bvh {
        let n = tris.len();
        let mut cent = vec![[0.0f64; 3]; n];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (t, f) in tris.iter().enumerate() {
            for k in 0..3 {
                let c =
                    (verts[f[0] as usize][k] + verts[f[1] as usize][k] + verts[f[2] as usize][k])
                        / 3.0;
                cent[t][k] = c;
            }
            for &vi in f {
                let v = verts[vi as usize];
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        let mut bvh = Bvh {
            verts,
            tris,
            nodes: Vec::with_capacity(2 * n.max(1)),
            order: (0..n as u32).collect(),
            lo,
            hi,
        };
        if n > 0 {
            let mut cent = cent;
            bvh.build_node(0, n, &mut cent);
        }
        bvh
    }

    fn tri_bbox(&self, t: usize, lo: &mut [f64; 3], hi: &mut [f64; 3]) {
        let f = self.tris[t];
        *lo = [f64::INFINITY; 3];
        *hi = [f64::NEG_INFINITY; 3];
        for &vi in &f {
            let v = self.verts[vi as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }

    /// Recursive median-split build over `order[start..end]` of triangle ids.
    fn build_node(&mut self, start: usize, end: usize, cent: &mut Vec<[f64; 3]>) -> u32 {
        let idx = self.nodes.len() as u32;
        // node bbox over the range
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &t in &self.order[start..end] {
            let (mut tl, mut th) = ([0.0; 3], [0.0; 3]);
            self.tri_bbox(t as usize, &mut tl, &mut th);
            for k in 0..3 {
                lo[k] = lo[k].min(tl[k]);
                hi[k] = hi[k].max(th[k]);
            }
        }
        self.nodes.push(Node {
            lo,
            hi,
            right: 0,
            start: start as u32,
        });
        let count = end - start;
        if count <= LEAF_MAX {
            self.nodes[idx as usize].right = -(count as i32);
            return idx;
        }
        // longest centroid-bbox axis, median split
        let (mut clo, mut chi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &t in &self.order[start..end] {
            for k in 0..3 {
                clo[k] = clo[k].min(cent[t as usize][k]);
                chi[k] = chi[k].max(cent[t as usize][k]);
            }
        }
        let mut axis = 0;
        for k in 1..3 {
            if chi[k] - clo[k] > chi[axis] - clo[axis] {
                axis = k;
            }
        }
        let mid = start + count / 2;
        self.order[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            cent[a as usize][axis]
                .partial_cmp(&cent[b as usize][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.build_node(start, mid, cent);
        let right = self.build_node(mid, end, cent);
        self.nodes[idx as usize].right = right as i32;
        idx
    }

    /// All triangle hits along the ray as (t, tri) sorted by t.
    fn ray_ts(&self, o: [f64; 3], d: [f64; 3], t_min: f64) -> Vec<(f64, u32)> {
        let mut out: Vec<(f64, u32)> = Vec::new();
        if self.nodes.is_empty() {
            return out;
        }
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        // iterative traversal; a BVH over ~1e6 tris stays under ~64 deep,
        // a heap stack only kicks in for degenerate cases
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        if box_hit(o, inv, self.nodes[0].lo, self.nodes[0].hi) {
            stack.push(0);
        }
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if node.right < 0 {
                let count = (-node.right) as usize;
                let s = node.start as usize;
                for &t in &self.order[s..s + count] {
                    let f = self.tris[t as usize];
                    let (a, b, c) = (
                        self.verts[f[0] as usize],
                        self.verts[f[1] as usize],
                        self.verts[f[2] as usize],
                    );
                    if let Some(tt) = tri_hit(o, d, a, b, c, t_min) {
                        out.push((tt, t));
                    }
                }
            } else {
                let l = ni + 1;
                let r = node.right as u32;
                let lh = box_hit(o, inv, self.nodes[l as usize].lo, self.nodes[l as usize].hi);
                let rh = box_hit(o, inv, self.nodes[r as usize].lo, self.nodes[r as usize].hi);
                // push far first so the near child pops first — hits then
                // arrive roughly front-to-back (still sorted below)
                if rh {
                    stack.push(r);
                }
                if lh {
                    stack.push(l);
                }
            }
        }
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// First-hit test for ray_any.
    fn ray_any(&self, o: [f64; 3], d: [f64; 3]) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        if box_hit(o, inv, self.nodes[0].lo, self.nodes[0].hi) {
            stack.push(0);
        }
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if node.right < 0 {
                let count = (-node.right) as usize;
                let s = node.start as usize;
                for &t in &self.order[s..s + count] {
                    let f = self.tris[t as usize];
                    if tri_hit(
                        o,
                        d,
                        self.verts[f[0] as usize],
                        self.verts[f[1] as usize],
                        self.verts[f[2] as usize],
                        EPS_T,
                    )
                    .is_some()
                    {
                        return true;
                    }
                }
            } else {
                let l = ni + 1;
                let r = node.right as u32;
                if box_hit(o, inv, self.nodes[r as usize].lo, self.nodes[r as usize].hi) {
                    stack.push(r);
                }
                if box_hit(o, inv, self.nodes[l as usize].lo, self.nodes[l as usize].hi) {
                    stack.push(l);
                }
            }
        }
        false
    }

    #[inline]
    fn inside_aabb(&self, p: [f64; 3]) -> bool {
        (0..3).all(|k| p[k] >= self.lo[k] && p[k] <= self.hi[k])
    }

    /// trimesh-contains semantics for one point.
    fn contains_pt(&self, p: [f64; 3]) -> bool {
        if !self.inside_aabb(p) {
            return false;
        }
        let mut d = DIR_MAIN;
        for _ in 0..2 {
            let hf = self.ray_ts(p, d, EPS_T).len();
            let hb = self.ray_ts(p, [-d[0], -d[1], -d[2]], EPS_T).len();
            let (pf, pb) = (hf % 2 == 1, hb % 2 == 1);
            if pf == pb {
                return pf;
            }
            if hf == 0 || hb == 0 {
                return false;
            }
            d = DIR_RETRY; // deterministic retry direction
        }
        false
    }
}

fn default_threads(threads: Option<usize>) -> usize {
    threads
        .filter(|&t| t > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get().min(8)))
}

fn read_pts(a: &PyReadonlyArray2<f64>) -> PyResult<Vec<[f64; 3]>> {
    let arr = a.as_array();
    if arr.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("expected (n,3)"));
    }
    Ok(arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

#[pyclass]
pub struct MeshIndex {
    bvh: Bvh,
}

#[pymethods]
impl MeshIndex {
    #[new]
    fn new(vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<i64>) -> PyResult<Self> {
        let verts = read_pts(&vertices)?;
        let f = faces.as_array();
        if f.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("faces must be (m,3)"));
        }
        let mut tris = Vec::with_capacity(f.shape()[0]);
        for r in f.rows() {
            tris.push([r[0] as u32, r[1] as u32, r[2] as u32]);
        }
        Ok(MeshIndex {
            bvh: Bvh::build(verts, tris),
        })
    }

    /// (2,3) mesh bounds matching mesh.bounds.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let b = &self.bvh;
        PyArray2::from_owned_array(
            py,
            ndarray::array![[b.lo[0], b.lo[1], b.lo[2]], [b.hi[0], b.hi[1], b.hi[2]]],
        )
    }

    /// trimesh.contains parity for (n,3) points.
    #[pyo3(signature = (pts, threads=None))]
    fn contains<'py>(
        &self,
        py: Python<'py>,
        pts: PyReadonlyArray2<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let pts = read_pts(&pts)?;
        let t = default_threads(threads);
        let bvh = &self.bvh;
        let out = py.detach(move || {
            let mut out = vec![false; pts.len()];
            par_chunks(&mut out, t, |chunk, off| {
                for (i, v) in chunk.iter_mut().enumerate() {
                    *v = bvh.contains_pt(pts[off + i]);
                }
            });
            out
        });
        Ok(PyArray1::from_vec(py, out))
    }

    /// trimesh.ray.intersects_any: per-ray first-hit bool.
    #[pyo3(signature = (origins, dirs, threads=None))]
    fn ray_any<'py>(
        &self,
        py: Python<'py>,
        origins: PyReadonlyArray2<f64>,
        dirs: PyReadonlyArray2<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let o = read_pts(&origins)?;
        let d = read_pts(&dirs)?;
        let t = default_threads(threads);
        let bvh = &self.bvh;
        let out = py.detach(move || {
            let mut out = vec![false; o.len()];
            par_chunks(&mut out, t, |chunk, off| {
                for (i, v) in chunk.iter_mut().enumerate() {
                    *v = bvh.ray_any(o[off + i], d[off + i]);
                }
            });
            out
        });
        Ok(PyArray1::from_vec(py, out))
    }

    /// trimesh.ray.intersects_location(multiple_hits=True): hits UNIQUE
    /// per location (intersects_id groups (locations, index_ray) via
    /// unique_rows — shared-edge hits at identical positions collapse to
    /// their first triangle). Returned as (locs (h,3), ray_idx (h,),
    /// tri_idx (h,)) sorted by (ray_idx, t).
    #[pyo3(signature = (origins, dirs, threads=None))]
    fn ray_locs<'py>(
        &self,
        py: Python<'py>,
        origins: PyReadonlyArray2<f64>,
        dirs: PyReadonlyArray2<f64>,
        threads: Option<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let o = read_pts(&origins)?;
        let d = read_pts(&dirs)?;
        let t = default_threads(threads);
        // per-ray hit lists (parallel), then flattened in ray order —
        // o/d are cloned into the released section (tiny) so the originals
        // stay available for the position reconstruction below
        let bvh = &self.bvh;
        let (o2, d2) = (o.clone(), d.clone());
        let per_ray: Vec<Vec<(f64, u32)>> = py.detach(move || {
            let mut per_ray: Vec<Vec<(f64, u32)>> = Vec::new();
            per_ray.resize_with(o2.len(), Vec::new);
            par_chunks(&mut per_ray, t, |chunk, off| {
                for (i, v) in chunk.iter_mut().enumerate() {
                    // trimesh forward test: distance = t*|d|^2 > -1e-6
                    let dd = d2[off + i];
                    let d2l = dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2];
                    let t_min = -1e-6 / d2l.max(1e-30);
                    *v = bvh.ray_ts(o2[off + i], dd, t_min);
                }
            });
            per_ray
        });
        let total: usize = per_ray.iter().map(|v| v.len()).sum();
        let mut locs = Vec::with_capacity(total * 3);
        let mut rids = Vec::with_capacity(total);
        let mut tids = Vec::with_capacity(total);
        for (r, hits) in per_ray.iter().enumerate() {
            // collapse hits at coincident locations — the unique_rows
            // convention merges on ~8 significant digits, so scale the
            // cluster eps to the hit range (adjacent-triangle duplicates
            // share t up to fp jitter)
            let t_max = hits.last().map(|(t, _)| t.abs()).unwrap_or(0.0);
            let eps_t = 1e-9 * t_max.max(1.0);
            let mut prev = f64::NEG_INFINITY;
            for &(tt, tri) in hits {
                if tt - prev < eps_t {
                    prev = tt;
                    continue;
                }
                prev = tt;
                let p = [
                    o[r][0] + tt * d[r][0],
                    o[r][1] + tt * d[r][1],
                    o[r][2] + tt * d[r][2],
                ];
                locs.extend_from_slice(&p);
                rids.push(r as i64);
                tids.push(tri as i64);
            }
        }
        let h = locs.len() / 3; // post-dedup count (total is the pre-dedup capacity hint)
        let locs_arr =
            PyArray2::from_owned_array(py, ndarray::Array2::from_shape_vec((h, 3), locs).unwrap());
        Ok((
            locs_arr,
            PyArray1::from_vec(py, rids),
            PyArray1::from_vec(py, tids),
        ))
    }

    /// fem/voxelize._scanline_contains — full parity fill for a regular
    /// point lattice. `origin` (3,), `dims` (n0,n1,n2), `offset` in [0,1).
    /// Returns the (prod(dims),) inside mask, C-order ravelled.
    #[pyo3(signature = (origin, dims, pitch, offset, threads=None))]
    fn scanline_fill<'py>(
        &self,
        py: Python<'py>,
        origin: PyReadonlyArray1<f64>,
        dims: (i64, i64, i64),
        pitch: f64,
        offset: f64,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let o = origin.as_slice()?;
        if o.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("origin must be (3,)"));
        }
        let n_pts = [dims.0 as usize, dims.1 as usize, dims.2 as usize];
        if n_pts.iter().any(|&n| n == 0) {
            return Ok(PyArray1::from_vec(py, vec![]));
        }
        let t = default_threads(threads);
        let bvh = &self.bvh;
        let origin = [o[0], o[1], o[2]];
        let flat = py.detach(move || scanline(bvh, origin, n_pts, pitch, offset, t));
        Ok(PyArray1::from_vec(py, flat))
    }
}

/// The scanline parity fill — mirrors fem/voxelize._scanline_contains line
/// for line so the conventions stay auditable side by side.
fn scanline(
    bvh: &Bvh,
    origin: [f64; 3],
    n_pts: [usize; 3],
    pitch: f64,
    offset: f64,
    threads: usize,
) -> Vec<bool> {
    let ax = n_pts
        .iter()
        .enumerate()
        .max_by_key(|(_, n)| *n)
        .map(|(i, _)| i)
        .unwrap();
    let (oa, ob) = {
        let mut it = (0..3).filter(|&a| a != ax);
        (it.next().unwrap(), it.next().unwrap())
    };
    let (na, nb, nax) = (n_pts[oa], n_pts[ob], n_pts[ax]);
    let eps = pitch * 1e-3; // same-surface collapse + face-plane jitter
    let q_axis: Vec<f64> = (0..nax)
        .map(|i| origin[ax] + pitch * (i as f64 + offset))
        .collect();
    let o_ax = bvh.lo[ax] - 2.0 * pitch;

    // one bool column per (ia, ib) ray — contiguous per-ray result blocks so
    // threads write disjoint memory
    let n_rays = na * nb;
    let mut cols: Vec<Vec<bool>> = Vec::new();
    cols.resize_with(n_rays, || vec![false; nax]);
    par_chunks(&mut cols, threads, |chunk, off| {
        for (i, col) in chunk.iter_mut().enumerate() {
            let r = off + i;
            let ia = r / nb;
            let ib = r % nb;
            let mut o = [0.0; 3];
            o[oa] = origin[oa] + pitch * (ia as f64 + offset) + eps;
            o[ob] = origin[ob] + pitch * (ib as f64 + offset) + eps;
            o[ax] = o_ax;
            let mut d = [0.0; 3];
            d[ax] = 1.0;
            // hit positions along ax, deduped <eps — element i is a dup when
            // x[i]-x[i-1] < eps (previous ELEMENT, not last kept — a whole
            // cluster chain collapses onto its first hit, matching the
            // np.diff<eps convention in _scanline_contains)
            let ts = bvh.ray_ts(o, d, EPS_T);
            let mut hits: Vec<f64> = Vec::with_capacity(ts.len());
            let mut prev = f64::NEG_INFINITY;
            for (t, _) in ts {
                let x = o[ax] + t;
                if x - prev >= eps {
                    hits.push(x);
                }
                prev = x;
            }
            for (iq, &q) in q_axis.iter().enumerate() {
                // hits strictly below q — a query sitting ON a crossing is
                // boundary and counts as outside
                let pos = hits.partition_point(|&h| h < q);
                let on_hit = pos < hits.len() && hits[pos] - q < eps;
                col[iq] = pos % 2 == 1 && !on_hit;
            }
        }
    });

    // scatter columns into the (n0,n1,n2) C-order grid
    let mut out = vec![false; n_pts[0] * n_pts[1] * n_pts[2]];
    let (n1, n2) = (n_pts[1], n_pts[2]);
    for r in 0..n_rays {
        let ia = r / nb;
        let ib = r % nb;
        let mut ijk = [0usize; 3];
        ijk[oa] = ia;
        ijk[ob] = ib;
        for (iq, v) in cols[r].iter().enumerate() {
            ijk[ax] = iq;
            out[ijk[0] * n1 * n2 + ijk[1] * n2 + ijk[2]] = *v;
        }
    }
    out
}
