import { useEffect } from "react";
import * as THREE from "three";

export interface RegionFillEntry {
  /** painted surface points (model-space mm) */
  points: number[][];
  /** brush radius in mm — approximates the backend's radius_mm+pitch dilation */
  radius: number;
  color: string;
}

const MAX_VERTS = 500_000;

/**
 * Colours the model's surface where painted `points` regions would select:
 * every vertex within `radius` of a painted point gets the region colour —
 * the live approximation of the voxel-domain selection the FEM preview
 * resolves exactly (see `cells_preview`).
 *
 * Recomputes are debounced (strokes stream points) and guarded by a vertex
 * cap. Geometry/material state is restored on cleanup.
 */
export default function RegionFill({
  model,
  entries,
  debounceMs = 140,
}: {
  model: THREE.Object3D | null;
  entries: RegionFillEntry[];
  debounceMs?: number;
}) {
  useEffect(() => {
    if (!model) return;
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
      if (!active.length || meshes.reduce((n, m) => n + (m.mesh.geometry.getAttribute("position")?.count ?? 0), 0) > MAX_VERTS) {
        cleanup(meshes.map((m) => m.mesh));
        return;
      }

      // spatial hash: cellKey -> list of (pointIdx within entry, entryIdx)
      type P = { e: number; p: THREE.Vector3 };
      const grids = active.map((entry) => {
        const cell = Math.max(entry.radius, 0.5);
        const map = new Map<string, P[]>();
        entry.points.forEach((p, i) => {
          if (p.length < 3) return;
          const v = new THREE.Vector3(p[0], p[1], p[2]);
          const k = [v.x, v.y, v.z].map((c) => Math.floor(c / cell)).join(",");
          (map.get(k) ?? map.set(k, []).get(k)!).push({ e: i, p: v });
        });
        return { cell, map, r2: entry.radius * entry.radius };
      });

      const colors = active.map((e) => new THREE.Color(e.color));
      const pos = new THREE.Vector3();
      for (const { mesh, rel } of meshes) {
        const attr = mesh.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
        if (!attr) continue;
        const base =
          (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial;
        const baseColor = base?.color ?? new THREE.Color(0.6, 0.6, 0.65);
        const arr = new Float32Array(attr.count * 3);
        for (let i = 0; i < attr.count; i++) {
          pos.fromBufferAttribute(attr, i).applyMatrix4(rel);
          let col: THREE.Color = baseColor;
          for (let e = 0; e < grids.length && col === baseColor; e++) {
            const { cell, map, r2 } = grids[e];
            const cx = Math.floor(pos.x / cell);
            const cy = Math.floor(pos.y / cell);
            const cz = Math.floor(pos.z / cell);
            outer: for (let dx = -1; dx <= 1; dx++)
              for (let dy = -1; dy <= 1; dy++)
                for (let dz = -1; dz <= 1; dz++) {
                  const pts = map.get(`${cx + dx},${cy + dy},${cz + dz}`);
                  if (!pts) continue;
                  for (const { p } of pts) {
                    if (pos.distanceToSquared(p) <= r2) {
                      col = colors[e];
                      break outer;
                    }
                  }
                }
          }
          arr[i * 3] = col.r;
          arr[i * 3 + 1] = col.g;
          arr[i * 3 + 2] = col.b;
        }
        mesh.geometry.setAttribute("color", new THREE.BufferAttribute(arr, 3));
        forEachMat(mesh, (m) => {
          if ("vertexColors" in m) {
            m.userData.__rfPrev = m.vertexColors;
            m.vertexColors = true;
            m.needsUpdate = true;
          }
        });
      }
    }, debounceMs);
    return () => clearTimeout(timer);
  }, [model, entries, debounceMs]);

  // restore on unmount/model change
  useEffect(() => {
    return () => {
      if (!model) return;
      const meshes: THREE.Mesh[] = [];
      model.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
      });
      cleanup(meshes);
    };
  }, [model]);

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
