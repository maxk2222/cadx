import { useEffect } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { positionsToWire } from "../../lib/meshArrays";
import {
  fillVertexColors,
  flattenPoints,
  type FillEntryData,
} from "../../lib/regionFill";
import { viewerWorker } from "../../lib/viewerWorker";

export interface RegionFillEntry {
  /** painted surface points (model-space mm) */
  points: number[][];
  /** brush radius in mm — approximates the backend's radius_mm+pitch dilation */
  radius: number;
  color: string;
}

const MAX_VERTS = 500_000;

/** below this many vertices the fill is cheap enough to run inline — the
 *  worker round-trip would cost more than it saves. */
const FILL_WORKER_MIN_VERTS = 50_000;

/**
 * Colours the model's surface where painted `points` regions would select:
 * every vertex within `radius` of a painted point gets the region colour —
 * the live approximation of the voxel-domain selection the FEM preview
 * resolves exactly (see `cells_preview`).
 *
 * Recomputes are debounced (strokes stream points) and guarded by a vertex
 * cap. Dense models run the vertex loop in the viewer worker (identical
 * result via lib/regionFill.ts); smaller ones stay synchronous. Geometry
 * and material state are restored on cleanup. Above the cap the fills stay
 * hidden — `onTooDense` lets the editor say why instead of silently
 * no-oping.
 */
export default function RegionFill({
  model,
  entries,
  debounceMs = 140,
  onTooDense,
}: {
  model: THREE.Object3D | null;
  entries: RegionFillEntry[];
  debounceMs?: number;
  /** vertex cap exceeded → fills suppressed (true) / drawable again (false) */
  onTooDense?: (tooDense: boolean) => void;
}) {
  // attribute writes are imperative — under frameloop="demand" nothing else
  // knows they happened
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    if (!model) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      // collect meshes + their transform relative to the model root
      const meshes: { mesh: THREE.Mesh; rel: THREE.Matrix4 }[] = [];
      const walk = (node: THREE.Object3D, parentRel: THREE.Matrix4) => {
        node.updateMatrix();
        const rel = parentRel.clone().multiply(node.matrix);
        if ((node as THREE.Mesh).isMesh) meshes.push({ mesh: node as THREE.Mesh, rel });
        node.children.forEach((c) => walk(c, rel));
      };
      walk(model, new THREE.Matrix4());

      const active = entries.filter((e) => e.points.length && e.radius > 0);
      const totalVerts = meshes.reduce(
        (n, m) => n + (m.mesh.geometry.getAttribute("position")?.count ?? 0),
        0
      );
      const tooDense = totalVerts > MAX_VERTS;
      onTooDense?.(tooDense);
      if (!active.length || tooDense) {
        cleanup(meshes.map((m) => m.mesh));
        invalidate();
        return;
      }

      // resolve colours on this side — THREE.Color decodes hex into the
      // linear working space, which is what the attribute must contain
      const wireEntries: FillEntryData[] = active.map((e) => {
        const c = new THREE.Color(e.color);
        return {
          points: flattenPoints(e.points),
          radius: e.radius,
          color: [c.r, c.g, c.b],
        };
      });
      const jobs = meshes
        .map(({ mesh, rel }) => {
          const attr = mesh.geometry.getAttribute("position") as
            | THREE.BufferAttribute
            | THREE.InterleavedBufferAttribute
            | undefined;
          if (!attr) return null;
          const base = (
            Array.isArray(mesh.material) ? mesh.material[0] : mesh.material
          ) as THREE.MeshStandardMaterial;
          const bc = base?.color ?? new THREE.Color(0.6, 0.6, 0.65);
          return {
            mesh,
            attr,
            matrix: rel.elements.slice(),
            base: [bc.r, bc.g, bc.b] as [number, number, number],
          };
        })
        .filter((x): x is NonNullable<typeof x> => !!x);

      const applyColors = (
        targets: typeof jobs,
        perMesh: (Float32Array | null)[]
      ) => {
        if (cancelled) return;
        targets.forEach((j, i) => {
          const arr = perMesh[i];
          if (!arr) return;
          j.mesh.geometry.setAttribute("color", new THREE.BufferAttribute(arr, 3));
          forEachMat(j.mesh, (m) => {
            if ("vertexColors" in m) {
              m.userData.__rfPrev = m.vertexColors;
              m.vertexColors = true;
              m.needsUpdate = true;
            }
          });
        });
        invalidate();
      };
      const syncPath = () =>
        applyColors(
          jobs,
          jobs.map((j) => fillVertexColors(j.attr, j.matrix, wireEntries, j.base))
        );

      // worker path — only meshes whose attribute serialized stay paired
      const wire = jobs
        .map((j) => ({ j, pos: positionsToWire(j.attr) }))
        .filter(
          (x): x is { j: (typeof jobs)[number]; pos: NonNullable<typeof x.pos> } =>
            !!x.pos
        );
      const req =
        totalVerts >= FILL_WORKER_MIN_VERTS && wire.length
          ? viewerWorker.fills({
              entries: wireEntries,
              meshes: wire.map((x) => ({
                pos: x.pos,
                matrix: x.j.matrix,
                base: x.j.base,
              })),
            })
          : null;
      if (!req) {
        syncPath();
        return;
      }
      req.then((res) => applyColors(wire.map((x) => x.j), res)).catch(syncPath);
    }, debounceMs);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [model, entries, debounceMs, invalidate, onTooDense]);

  // restore on unmount/model change
  useEffect(() => {
    return () => {
      if (!model) return;
      const meshes: THREE.Mesh[] = [];
      model.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
      });
      cleanup(meshes);
      invalidate();
    };
  }, [model, invalidate]);

  return null;
}

function forEachMat(mesh: THREE.Mesh, fn: (m: THREE.Material) => void) {
  (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach(fn);
}

function cleanup(meshes: THREE.Mesh[]) {
  for (const mesh of meshes) {
    mesh.geometry.deleteAttribute("color");
    forEachMat(mesh, (m) => {
      if ("vertexColors" in m && m.userData.__rfPrev !== undefined) {
        m.vertexColors = m.userData.__rfPrev;
        delete m.userData.__rfPrev;
        m.needsUpdate = true;
      }
    });
  }
}
