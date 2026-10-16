import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  buildFacets,
  facetAtHit,
  facetGeometry,
  facetLoop,
  facetPickPoints,
  facetRadius,
  resampleLoop,
} from "./facets";

/** wrap a geometry in a mesh inside a root group (the "pick frame" stand-in) */
function meshOf(geom: THREE.BufferGeometry): { mesh: THREE.Mesh; root: THREE.Group } {
  const mesh = new THREE.Mesh(geom);
  const root = new THREE.Group();
  root.add(mesh);
  root.updateMatrixWorld(true);
  return { mesh, root };
}

/** non-indexed geometry from raw triangle soup (9 floats per triangle) */
function tris(...pts: number[]): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return g;
}

describe("buildFacets", () => {
  it("merges coplanar triangles into one facet", () => {
    // a flat 2×2 quad in the XY plane (z=0)
    const { mesh, root } = meshOf(
      tris(0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0)
    );
    const fs = buildFacets(mesh, root)!;
    expect(fs).not.toBeNull();
    expect(fs.faceCount).toBe(2);
    expect(fs.facets).toHaveLength(1);
    const f = fs.facets[0];
    expect(f.faces).toHaveLength(2);
    expect(f.area).toBeCloseTo(4);
    expect(f.normal[2]).toBeCloseTo(1);
    // quad boundary = 4 edges
    expect(f.boundary).toHaveLength(4);
  });

  it("splits triangles across a crease", () => {
    // two tris hinged 90° along the y axis: one in z=0, one in x=0
    const { mesh, root } = meshOf(
      tris(
        0, 0, 0, 0, 2, 0, 2, 0, 0, // flat on XY? no: (0,0,0)(0,2,0)(2,0,0) lies in z=0
        0, 0, 0, 0, 0, 2, 0, 2, 0 // lies in x=0, shares edge (0,0,0)-(0,2,0)
      )
    );
    const fs = buildFacets(mesh, root)!;
    expect(fs.facets).toHaveLength(2);
    // each facet owns exactly one triangle and they are adjacent at the crease
    expect(fs.facets[0].faces).toHaveLength(1);
    expect(fs.facets[1].faces).toHaveLength(1);
  });

  it("clusters a box into 6 facets", () => {
    const { mesh, root } = meshOf(new THREE.BoxGeometry(10, 20, 30));
    const fs = buildFacets(mesh, root)!;
    expect(fs.faceCount).toBe(12);
    expect(fs.facets).toHaveLength(6);
    const areas = fs.facets.map((f) => f.area).sort((a, b) => a - b);
    // 2×(10·20) + 2×(10·30) + 2×(20·30)
    expect(areas).toEqual([
      expect.closeTo(200),
      expect.closeTo(200),
      expect.closeTo(300),
      expect.closeTo(300),
      expect.closeTo(600),
      expect.closeTo(600),
    ]);
    // normals are axis-aligned
    for (const f of fs.facets) {
      const [x, y, z] = f.normal.map(Math.abs);
      expect(Math.max(x, y, z)).toBeCloseTo(1);
    }
  });

  it("expresses facet meta in root space (mesh transform applied)", () => {
    const { mesh, root } = meshOf(new THREE.BoxGeometry(4, 4, 4));
    mesh.position.set(10, 0, 0);
    mesh.rotation.y = Math.PI / 4;
    root.updateMatrixWorld(true);
    const fs = buildFacets(mesh, root)!;
    // each face centroid sits 2 units (half edge) from the shifted centre
    for (const f of fs.facets) {
      const d = Math.hypot(f.centroid[0] - 10, f.centroid[1], f.centroid[2]);
      expect(d).toBeCloseTo(2, 3);
    }
    // and the +x-ish face normal points away from the centre
    const xf = fs.facets.reduce((a, b) => (b.centroid[0] > a.centroid[0] ? b : a));
    const away = xf.normal[0] * (xf.centroid[0] - 10) + xf.normal[2] * xf.centroid[2];
    expect(away).toBeGreaterThan(0);
  });

  it("returns null for empty geometry", () => {
    const { mesh, root } = meshOf(new THREE.BufferGeometry());
    expect(buildFacets(mesh, root)).toBeNull();
  });
});

describe("facetAtHit / facetGeometry / facetPickPoints", () => {
  it("maps a rendered faceIndex to its facet", () => {
    const { mesh, root } = meshOf(new THREE.BoxGeometry(4, 4, 4));
    const fs = buildFacets(mesh, root)!;
    const f = facetAtHit(fs, 0)!;
    expect(f.faces).toContain(0);
    expect(facetAtHit(fs, -1)).toBeNull();
    expect(facetAtHit(fs, 999)).toBeNull();
    expect(facetAtHit(fs, null)).toBeNull();
  });

  it("builds highlight geometry indexing exactly the facet's triangles", () => {
    const { mesh, root } = meshOf(new THREE.BoxGeometry(4, 4, 4));
    const fs = buildFacets(mesh, root)!;
    const f = fs.facets[0];
    const g = facetGeometry(fs, f);
    expect(g.getIndex()!.count).toBe(f.faces.length * 3);
    // shares the welded positions (no copy)
    expect(g.getAttribute("position")).toBe(fs.welded.getAttribute("position"));
  });

  it("produces surface points inside the facet bounds", () => {
    const { mesh, root } = meshOf(new THREE.BoxGeometry(4, 4, 4));
    const fs = buildFacets(mesh, root)!;
    const f = fs.facets[0];
    const pts = facetPickPoints(fs, f);
    expect(pts.length).toBe(f.faces.length);
    // all centroids on the same box face: one coordinate is ±2
    const ax = f.normal.findIndex((v) => Math.abs(v) > 0.5);
    for (const p of pts) expect(Math.abs(p[ax])).toBeCloseTo(2);
  });

  it("facetRadius covers the triangle pitch", () => {
    const { mesh, root } = meshOf(new THREE.BoxGeometry(4, 4, 4));
    const fs = buildFacets(mesh, root)!;
    const r = facetRadius(fs.facets[0]);
    // box face: two 4×4 tris → pitch ~ sqrt(2*16/2)*1.2 = 4.8
    expect(r).toBeCloseTo(4.8, 1);
    expect(facetRadius({ ...fs.facets[0], faces: [], area: 0 })).toBe(2);
  });
});

describe("facetLoop", () => {
  it("traces a closed quad boundary", () => {
    const { mesh, root } = meshOf(
      tris(0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0)
    );
    const fs = buildFacets(mesh, root)!;
    const { pts: loop, closed } = facetLoop(fs, fs.facets[0])!;
    expect(closed).toBe(true);
    expect(loop).toHaveLength(4); // 4 corner verts, closure implied
    const xs = new Set(loop.map((p) => p[0]));
    const ys = new Set(loop.map((p) => p[1]));
    expect(xs).toEqual(new Set([0, 2]));
    expect(ys).toEqual(new Set([0, 2]));
  });

  it("starts the walk on the boundary edge nearest `near`", () => {
    // L-shaped facet, properly tessellated (edge (0,1)-(1,1) shared exactly)
    const { mesh, root } = meshOf(
      tris(
        1, 0, 0, 2, 0, 0, 2, 1, 0,
        1, 0, 0, 2, 1, 0, 1, 1, 0,
        0, 0, 0, 1, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 0, 1, 2, 0,
        0, 1, 0, 1, 2, 0, 0, 2, 0
      )
    );
    const fs = buildFacets(mesh, root)!;
    expect(fs.facets).toHaveLength(1);
    const loop = facetLoop(fs, fs.facets[0], [2, 0.5, 0])!;
    // L outline has 8 verts (the (1,0) subdivision vertex counts)
    expect(loop.closed).toBe(true);
    expect(loop.pts).toHaveLength(8);
    // nearest edge to (2,0.5,0) is the right edge x=2 → loop contains it
    const hasRight = loop.pts.some(
      (p: [number, number, number]) => Math.abs(p[0] - 2) < 1e-4 && Math.abs(p[1] - 1) < 1e-4
    );
    expect(hasRight).toBe(true);
  });

  it("returns null for a facet without boundary", () => {
    const { mesh, root } = meshOf(new THREE.SphereGeometry(5, 8, 6));
    const fs = buildFacets(mesh, root, 180)!; // everything merges → one facet
    const facet = fs.facets[0];
    // sphere has no mesh boundary and at 180° no crease boundary either
    expect(facet.boundary).toHaveLength(0);
    expect(facetLoop(fs, facet)).toBeNull();
  });

  it("recovers the prefix when the nearest edge sits mid open-chain", () => {
    // open boundary chain 0-1-2-3 (e.g. a crease meeting the mesh
    // silhouette): a click nearest the middle edge must still return the
    // whole chain, not just the forward part.
    const welded = new THREE.BufferGeometry();
    welded.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0],
        3
      )
    );
    welded.setIndex([0, 1, 2]);
    const facet = {
      id: 0,
      faces: [0],
      centroid: [1.5, 0, 0] as [number, number, number],
      normal: [0, 0, 1] as [number, number, number],
      area: 1,
      boundary: [
        [0, 1],
        [1, 2],
        [2, 3],
      ] as [number, number][],
    };
    const fset = {
      welded,
      faceCount: 1,
      facetOf: new Int32Array([0]),
      facets: [facet],
      edgeFaces: new Map(),
    };
    const loop = facetLoop(fset, facet, [2.4, 0.1, 0])!; // nearest edge [2,3]
    expect(loop.closed).toBe(false);
    expect(loop.pts.map((p) => p[0])).toEqual([0, 1, 2, 3]);
  });
});

describe("resampleLoop", () => {
  it("densifies long segments so voxel selection has no gaps", () => {
    const square: [number, number, number][] = [
      [0, 0, 0],
      [10, 0, 0],
      [10, 10, 0],
      [0, 10, 0],
    ];
    const pts = resampleLoop(square, 1);
    // 4 edges × 10 segments ≈ 40 points (each edge contributes its start verts)
    expect(pts.length).toBeGreaterThanOrEqual(36);
    // max gap between consecutive points <= spacing
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot(
        pts[i][0] - pts[i - 1][0],
        pts[i][1] - pts[i - 1][1],
        pts[i][2] - pts[i - 1][2]
      );
      expect(d).toBeLessThanOrEqual(1.01);
    }
    // closing edge is included: last point sits ~1 spacing from the start
    const last = pts[pts.length - 1];
    expect(
      Math.hypot(last[0] - pts[0][0], last[1] - pts[0][1], last[2] - pts[0][2])
    ).toBeLessThanOrEqual(1.01);
  });

  it("does not close open chains", () => {
    const chain: [number, number, number][] = [
      [0, 0, 0],
      [5, 0, 0],
    ];
    const pts = resampleLoop(chain, 1, 3000, false);
    // open chain: densified and the endpoint is emitted too
    expect(pts[pts.length - 1][0]).toBe(5);
    expect(pts.length).toBe(6);
  });

  it("respects maxPts", () => {
    const big: [number, number, number][] = [
      [0, 0, 0],
      [1000, 0, 0],
      [1000, 1000, 0],
      [0, 1000, 0],
    ];
    expect(resampleLoop(big, 0.5, 100).length).toBeLessThanOrEqual(100);
  });
});
