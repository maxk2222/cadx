import * as THREE from "three";
import { mergeVertices } from "three-stdlib";

/**
 * Surface-facet segmentation for viewer picking: triangles are welded, then
 * clustered into "facets" — connected regions whose adjacent faces differ by
 * at most a crease angle (dihedral). A planar CAD face becomes one facet;
 * a fillet becomes several. Clustering happens on a welded copy whose
 * positions are transformed into `root` (model/frame) space, so centroids,
 * normals and loop points can be written into RegionSpec points directly.
 */

export interface Facet {
  id: number;
  /** triangle indices into `welded` (same order as the rendered geometry) */
  faces: number[];
  /** area-weighted centroid in model space (mm) */
  centroid: [number, number, number];
  /** area-weighted unit normal in model space */
  normal: [number, number, number];
  /** total triangle area (mm²) */
  area: number;
  /** boundary edges as welded-vertex index pairs (crease + mesh-boundary) */
  boundary: [number, number][];
}

export interface FacetSet {
  /** welded indexed geometry, positions in model space */
  welded: THREE.BufferGeometry;
  faceCount: number;
  /** facet id per triangle (-1 never occurs after clustering) */
  facetOf: Int32Array;
  facets: Facet[];
  /** edge key -> adjacent face indices */
  edgeFaces: Map<number, number[]>;
}

const EDGE_SHIFT = 2 ** 24; // welded vertex index space must stay < 16.7M
const MAX_VERTS = 3_000_000;

const ekey = (a: number, b: number) => (a < b ? a * EDGE_SHIFT + b : b * EDGE_SHIFT + a);

/**
 * Cluster `mesh`'s triangles into facets by crease angle. `root` is the
 * model-space ancestor the returned geometry/points are expressed in
 * (the pick frame's content group). Returns null for empty/huge meshes.
 */
export function buildFacets(
  mesh: THREE.Mesh,
  root: THREE.Object3D,
  angleDeg = 30
): FacetSet | null {
  const src = mesh.geometry as THREE.BufferGeometry | undefined;
  const srcPos = src?.getAttribute("position") as THREE.BufferAttribute | undefined;
  if (!src || !srcPos || srcPos.count < 3 || srcPos.count > MAX_VERTS) return null;

  mesh.updateWorldMatrix(true, false);
  root.updateWorldMatrix(true, false);
  const rel = new THREE.Matrix4()
    .copy(root.matrixWorld)
    .invert()
    .multiply(mesh.matrixWorld);

  // weld positions only — mergeVertices hashes every attribute, so UV/normal
  // seams would stay split and read as fake mesh boundaries
  const posOnly = new THREE.BufferGeometry();
  posOnly.setAttribute("position", srcPos);
  if (src.getIndex()) posOnly.setIndex(src.getIndex()!.clone());
  const welded = mergeVertices(posOnly, 1e-4);
  posOnly.dispose();
  welded.applyMatrix4(rel);
  const index = welded.getIndex();
  const pos = welded.getAttribute("position") as THREE.BufferAttribute | undefined;
  if (!index || !pos || pos.count > MAX_VERTS) {
    welded.dispose();
    return null;
  }
  const idx = index.array as ArrayLike<number>;
  const faceCount = Math.floor(idx.length / 3);
  if (!faceCount) {
    welded.dispose();
    return null;
  }

  // per-face normals + areas (model space)
  const nrm = new Float32Array(faceCount * 3);
  const areas = new Float32Array(faceCount);
  const va = new THREE.Vector3();
  const vb = new THREE.Vector3();
  const vc = new THREE.Vector3();
  for (let f = 0; f < faceCount; f++) {
    va.fromBufferAttribute(pos, idx[f * 3]);
    vb.fromBufferAttribute(pos, idx[f * 3 + 1]).sub(va);
    vc.fromBufferAttribute(pos, idx[f * 3 + 2]).sub(va);
    vb.cross(vc);
    areas[f] = vb.length() / 2;
    vb.normalize();
    nrm[f * 3] = vb.x;
    nrm[f * 3 + 1] = vb.y;
    nrm[f * 3 + 2] = vb.z;
  }

  // edge -> faces adjacency
  const edgeFaces = new Map<number, number[]>();
  const pushEdge = (a: number, b: number, f: number) => {
    const k = ekey(a, b);
    const l = edgeFaces.get(k);
    if (l) l.push(f);
    else edgeFaces.set(k, [f]);
  };
  for (let f = 0; f < faceCount; f++) {
    pushEdge(idx[f * 3], idx[f * 3 + 1], f);
    pushEdge(idx[f * 3 + 1], idx[f * 3 + 2], f);
    pushEdge(idx[f * 3 + 2], idx[f * 3], f);
  }

  // BFS clustering: neighbours merge when their normals differ by <= angleDeg
  const cosTol = Math.cos((angleDeg * Math.PI) / 180);
  const facetOf = new Int32Array(faceCount).fill(-1);
  const facets: Facet[] = [];
  const stack: number[] = [];
  for (let seed = 0; seed < faceCount; seed++) {
    if (facetOf[seed] >= 0) continue;
    const id = facets.length;
    facetOf[seed] = id;
    stack.push(seed);
    const faces: number[] = [];
    while (stack.length) {
      const f = stack.pop()!;
      faces.push(f);
      const l1 = edgeFaces.get(ekey(idx[f * 3], idx[f * 3 + 1]));
      const l2 = edgeFaces.get(ekey(idx[f * 3 + 1], idx[f * 3 + 2]));
      const l3 = edgeFaces.get(ekey(idx[f * 3 + 2], idx[f * 3]));
      for (const l of [l1, l2, l3]) {
        if (!l) continue;
        for (const g of l) {
          if (g === f || facetOf[g] >= 0) continue;
          const dot =
            nrm[f * 3] * nrm[g * 3] +
            nrm[f * 3 + 1] * nrm[g * 3 + 1] +
            nrm[f * 3 + 2] * nrm[g * 3 + 2];
          if (dot >= cosTol) {
            facetOf[g] = id;
            stack.push(g);
          }
        }
      }
    }
    facets.push({ id, faces, centroid: [0, 0, 0], normal: [0, 0, 0], area: 0, boundary: [] });
  }

  // per-facet meta + boundary edges (edge whose adjacent facets differ,
  // including open mesh boundaries with a single adjacent face)
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const v3 = new THREE.Vector3();
  for (const facet of facets) {
    let area = 0;
    const c = new THREE.Vector3();
    const n = new THREE.Vector3();
    for (const f of facet.faces) {
      const A = areas[f];
      area += A;
      v1.fromBufferAttribute(pos, idx[f * 3]);
      v2.fromBufferAttribute(pos, idx[f * 3 + 1]);
      v3.fromBufferAttribute(pos, idx[f * 3 + 2]);
      c.x += ((v1.x + v2.x + v3.x) / 3) * A;
      c.y += ((v1.y + v2.y + v3.y) / 3) * A;
      c.z += ((v1.z + v2.z + v3.z) / 3) * A;
      n.x += nrm[f * 3] * A;
      n.y += nrm[f * 3 + 1] * A;
      n.z += nrm[f * 3 + 2] * A;
    }
    facet.area = area;
    if (area > 0) c.multiplyScalar(1 / area);
    facet.centroid = [c.x, c.y, c.z];
    if (n.lengthSq() > 0) n.normalize();
    else n.set(nrm[facet.faces[0] * 3], nrm[facet.faces[0] * 3 + 1], nrm[facet.faces[0] * 3 + 2]);
    facet.normal = [n.x, n.y, n.z];
  }

  const inFacet = new Set<number>();
  for (const facet of facets) {
    inFacet.clear();
    for (const f of facet.faces) inFacet.add(f);
    for (const f of facet.faces) {
      for (let e = 0; e < 3; e++) {
        const a = idx[f * 3 + e];
        const b = idx[f * 3 + ((e + 1) % 3)];
        const adj = edgeFaces.get(ekey(a, b));
        // interior edge: ≥2 faces and all of them inside this facet — skip.
        // open mesh edges (single face) and crease edges (foreign neighbour)
        // are boundaries.
        if (adj && adj.length > 1 && adj.every((g) => inFacet.has(g))) continue;
        facet.boundary.push([Math.min(a, b), Math.max(a, b)]);
      }
    }
  }

  return { welded, faceCount, facetOf, facets, edgeFaces };
}

/** facet under a raycast hit (faceIndex refers to the rendered geometry —
 *  mergeVertices preserves triangle order, so it indexes `welded` too). */
export function facetAtHit(
  fset: FacetSet | null | undefined,
  faceIndex: number | null | undefined
): Facet | null {
  if (!fset || faceIndex == null || faceIndex < 0 || faceIndex >= fset.faceCount) return null;
  return fset.facets[fset.facetOf[faceIndex]] ?? null;
}

/** highlight geometry: shares the welded position attribute, only the index
 *  is facet-specific — cheap to build per hover. */
export function facetGeometry(fset: FacetSet, facet: Facet): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", fset.welded.getAttribute("position") as THREE.BufferAttribute);
  const idx = fset.welded.getIndex()!.array as ArrayLike<number>;
  const arr = new Uint32Array(facet.faces.length * 3);
  facet.faces.forEach((f, i) => {
    arr[i * 3] = idx[f * 3];
    arr[i * 3 + 1] = idx[f * 3 + 1];
    arr[i * 3 + 2] = idx[f * 3 + 2];
  });
  g.setIndex(new THREE.BufferAttribute(arr, 1));
  return g;
}

/** estimated triangle pitch (mm) of a facet — the radius a `points` region
 *  needs to cover the picked surface without gaps. */
export function facetRadius(facet: Facet): number {
  if (!facet.faces.length || facet.area <= 0) return 2;
  // equilateral triangle pitch ~ sqrt(2*A/n); add a small cover margin
  return Math.max(1.0, Math.sqrt((2 * facet.area) / facet.faces.length) * 1.2);
}

/** triangle centroids of a facet as region points, stride-sampled to maxPts. */
export function facetPickPoints(fset: FacetSet, facet: Facet, maxPts = 1500): number[][] {
  const pos = fset.welded.getAttribute("position") as THREE.BufferAttribute;
  const idx = fset.welded.getIndex()!.array as ArrayLike<number>;
  const step = Math.max(1, Math.ceil(facet.faces.length / maxPts));
  const out: number[][] = [];
  for (let i = 0; i < facet.faces.length; i += step) {
    const f = facet.faces[i];
    const a = idx[f * 3];
    const b = idx[f * 3 + 1];
    const c = idx[f * 3 + 2];
    out.push([
      (pos.getX(a) + pos.getX(b) + pos.getX(c)) / 3,
      (pos.getY(a) + pos.getY(b) + pos.getY(c)) / 3,
      (pos.getZ(a) + pos.getZ(b) + pos.getZ(c)) / 3,
    ]);
  }
  return out;
}

/**
 * Ordered boundary-loop points of `facet`, starting on the boundary edge
 * nearest to `near` (model-space mm). Used by the edge-pick tool: the loop
 * closes or ends at a junction/non-manifold vertex. Returns model-space
 * vertex positions plus `closed` (true when the walk returned to the start
 * vertex), or null when the facet has no boundary.
 */
// boundary adjacency is invariant per facet — cache it (facetLoop runs on
// every throttled hover move)
const boundaryAdjCache = new WeakMap<Facet, Map<number, number[]>>();

export function facetLoop(
  fset: FacetSet,
  facet: Facet,
  near?: [number, number, number]
): { pts: [number, number, number][]; closed: boolean } | null {
  if (!facet.boundary.length) return null;
  const pos = fset.welded.getAttribute("position") as THREE.BufferAttribute;

  // vertex -> neighbouring vertices along the boundary
  let adj = boundaryAdjCache.get(facet);
  if (!adj) {
    adj = new Map<number, number[]>();
    for (const [a, b] of facet.boundary) {
      (adj.get(a) ?? adj.set(a, []).get(a)!).push(b);
      (adj.get(b) ?? adj.set(b, []).get(b)!).push(a);
    }
    boundaryAdjCache.set(facet, adj);
  }

  // start edge = boundary edge nearest to `near`
  let start: [number, number] = facet.boundary[0];
  if (near) {
    let best = Infinity;
    const p = new THREE.Vector3(...near);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const ap = new THREE.Vector3();
    const q = new THREE.Vector3();
    for (const [ia, ib] of facet.boundary) {
      a.fromBufferAttribute(pos, ia);
      b.fromBufferAttribute(pos, ib);
      ab.subVectors(b, a);
      const t = Math.max(0, Math.min(1, ap.subVectors(p, a).dot(ab) / (ab.lengthSq() || 1)));
      const d = q.copy(a).addScaledVector(ab, t).distanceToSquared(p);
      if (d < best) {
        best = d;
        start = [ia, ib];
      }
    }
  }

  // walk the loop containing the start edge (vertex indices first — the
  // nearest edge can sit in the middle of an OPEN chain, in which case the
  // forward walk drops the prefix and we prepend a backward walk)
  const guard = facet.boundary.length * 2 + 16;
  const chain: number[] = [];
  const seen = new Set<number>();
  let closed = false;
  let cur = start[0];
  let nxt = start[1];
  for (let i = 0; i < guard; i++) {
    chain.push(cur);
    seen.add(cur);
    const nbs = adj.get(nxt) ?? [];
    // continue to the neighbour that isn't where we came from
    const next = nbs.find((v) => v !== cur);
    if (next === undefined || next === start[0]) {
      chain.push(nxt);
      closed = next === start[0];
      break; // chain end / loop closed
    }
    cur = nxt;
    nxt = next;
  }
  // open chain — recover the part of the chain before the start edge
  if (!closed) {
    const pre: number[] = [];
    let c = start[0];
    for (let i = 0; i < guard; i++) {
      const back = (adj.get(c) ?? []).find((v) => v !== start[1] && !seen.has(v));
      if (back === undefined) break;
      pre.unshift(back);
      seen.add(back);
      c = back;
    }
    chain.unshift(...pre);
  }
  const out = chain.map(
    (v): [number, number, number] => [pos.getX(v), pos.getY(v), pos.getZ(v)]
  );
  return out.length >= 2 ? { pts: out, closed } : null;
}

/**
 * Resample a (closed) polyline to ~spacing between points. Boundary loops
 * are traced as mesh vertices — long straight edges would otherwise leave
 * only their endpoints in a `points` region and the voxel-domain selection
 * would gap along the edge. The closing segment (last→first) is included.
 */
export function resampleLoop(
  pts: [number, number, number][],
  spacing = 1.5,
  maxPts = 3000,
  closed = true
): number[][] {
  const n = pts.length;
  if (n < 2) return pts.map((p) => [...p]);
  const out: number[][] = [];
  const last = closed ? n : n - 1;
  for (let i = 0; i < last && out.length < maxPts; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dz = b[2] - a[2];
    const len = Math.hypot(dx, dy, dz);
    const segs = Math.max(1, Math.ceil(len / spacing));
    for (let s = 0; s < segs && out.length < maxPts; s++) {
      const t = s / segs;
      out.push([a[0] + dx * t, a[1] + dy * t, a[2] + dz * t]);
    }
  }
  // open chains never emit their endpoint otherwise
  if (!closed && out.length < maxPts) out.push([...pts[n - 1]]);
  return out;
}
