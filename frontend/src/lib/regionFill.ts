import * as THREE from "three";
import { positionsAttribute, type PositionsWire } from "./meshArrays";

/**
 * RegionFill compute core — shared between the in-component fallback and
 * viewer.worker.ts so both paths produce identical vertex colors: every
 * vertex within `radius` of a painted point takes the region's colour
 * (first matching entry wins), everything else keeps the material colour.
 */

/** flat points (xyz stride 3) + resolved colour — the wire/compute format. */
export interface FillEntryData {
  points: Float32Array;
  radius: number;
  /** resolved THREE.Color [r,g,b] — hex strings decode to linear space */
  color: [number, number, number];
}

/** one mesh's half of a fill request. */
export interface FillMeshJob {
  pos: PositionsWire;
  /** mesh→model-root transform, column-major 16 (Matrix4.elements) */
  matrix: number[];
  /** material base colour [r,g,b] for vertices outside all fills */
  base: [number, number, number];
}

export interface FillRequest {
  entries: FillEntryData[];
  meshes: FillMeshJob[];
}

/** `number[][]` → flat xyz Float32Array, dropping malformed points
 *  (same guard as the original inline loop). */
export function flattenPoints(pts: number[][]): Float32Array {
  const ok = pts.filter((p) => p.length >= 3);
  const out = new Float32Array(ok.length * 3);
  ok.forEach((p, i) => {
    out[i * 3] = p[0];
    out[i * 3 + 1] = p[1];
    out[i * 3 + 2] = p[2];
  });
  return out;
}

/**
 * Per-vertex colour buffer for one mesh. `pos` may be a live attribute or
 * a worker-rebuilt PositionsWire; `matrix` is the mesh→model transform.
 * Returns `count * 3` RGB floats — identical to the old inline algorithm.
 */
export function fillVertexColors(
  pos: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
  matrix: number[],
  entries: FillEntryData[],
  base: [number, number, number]
): Float32Array {
  const m = new THREE.Matrix4().fromArray(matrix);

  // spatial hash per entry: cellKey -> indices into the flat points array
  const grids = entries.map((e) => {
    const cell = Math.max(e.radius, 0.5);
    const map = new Map<string, number[]>();
    for (let i = 0; i * 3 + 2 < e.points.length; i++) {
      const k = [e.points[i * 3], e.points[i * 3 + 1], e.points[i * 3 + 2]]
        .map((c) => Math.floor(c / cell))
        .join(",");
      (map.get(k) ?? map.set(k, []).get(k)!).push(i);
    }
    return { cell, map, r2: e.radius * e.radius, pts: e.points };
  });

  const out = new Float32Array(pos.count * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos as THREE.BufferAttribute, i).applyMatrix4(m);
    let col: [number, number, number] | null = null;
    for (let e = 0; e < grids.length && !col; e++) {
      const { cell, map, r2, pts } = grids[e];
      const cx = Math.floor(v.x / cell);
      const cy = Math.floor(v.y / cell);
      const cz = Math.floor(v.z / cell);
      outer: for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const l = map.get(`${cx + dx},${cy + dy},${cz + dz}`);
            if (!l) continue;
            for (const pi of l) {
              const px = v.x - pts[pi * 3];
              const py = v.y - pts[pi * 3 + 1];
              const pz = v.z - pts[pi * 3 + 2];
              if (px * px + py * py + pz * pz <= r2) {
                col = entries[e].color;
                break outer;
              }
            }
          }
    }
    const c = col ?? base;
    out[i * 3] = c[0];
    out[i * 3 + 1] = c[1];
    out[i * 3 + 2] = c[2];
  }
  return out;
}

/** worker entry point: rebuild each mesh's attribute and fill it. */
export function runFillRequest(req: FillRequest): (Float32Array | null)[] {
  return req.meshes.map((mj) => {
    try {
      return fillVertexColors(
        positionsAttribute(mj.pos),
        mj.matrix,
        req.entries,
        mj.base
      );
    } catch {
      return null;
    }
  });
}
