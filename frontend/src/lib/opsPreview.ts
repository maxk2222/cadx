import * as THREE from "three";
import type { MeshOp } from "./types";

/**
 * Client-side evaluator for the rigid subset of the mesh ops stack
 * (set_up / rotate / mirror / scale). Ports the exact semantics of
 * backend/cadx/cad/mesh_ops.py — orient(), per-op transforms about the
 * bbox centre, and rest_on_xy_center() after EVERY op — so the instant
 * preview matches what the server replays. Returns null as soon as a
 * geometry op appears (those need the server preview).
 */

const RIGID = new Set(["set_up", "rotate", "mirror", "scale"]);
export const isRigidOp = (op: MeshOp) => RIGID.has(op.op);

const num = (v: unknown, fb = 0) => (Number.isFinite(Number(v)) ? Number(v) : fb);

function axisVec(name: string | undefined, fallback: string): THREE.Vector3 {
  const a = (name || fallback).trim().toLowerCase();
  const v = new THREE.Vector3();
  v.setComponent("xyz".indexOf(a.slice(-1)), a.startsWith("-") ? -1 : 1);
  return v;
}

/** Port of mesh_ops.orient(): source `up` -> +Z, source `front` -> +Y. */
function orientMatrix(up?: string, front?: string): THREE.Matrix4 {
  const u = axisVec(up, "+z");
  let f = axisVec(front, "+y");
  if (Math.abs(u.dot(f)) > 0.9999) {
    // front parallel to up — first perpendicular of X/Y/Z (same order as backend)
    for (const c of [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]) {
      if (Math.abs(u.dot(c)) <= 0.9999) {
        f = c.clone();
        break;
      }
    }
  }
  f = f.sub(u.clone().multiplyScalar(u.dot(f))).normalize();
  const n = new THREE.Vector3().crossVectors(f, u); // -> +X
  // columns: src axes mapped to +X,+Y,+Z → rotation = basis^T
  return new THREE.Matrix4().makeBasis(n, f, u).transpose();
}

function opMatrix(o: MeshOp, center: THREE.Vector3): THREE.Matrix4 {
  const a = o.args;
  switch (o.op) {
    case "set_up":
      return orientMatrix(a.up as string | undefined, a.front as string | undefined);
    case "rotate": {
      const axis = axisVec(a.axis as string | undefined, "z");
      const rad = (num(a.deg) * Math.PI) / 180;
      return new THREE.Matrix4()
        .makeTranslation(center.x, center.y, center.z)
        .multiply(new THREE.Matrix4().makeRotationAxis(axis, rad))
        .multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
    }
    case "mirror": {
      const idx = "xyz".indexOf(((a.axis as string) || "x").slice(-1));
      const s = [1, 1, 1];
      s[idx] = -1;
      return new THREE.Matrix4().makeScale(s[0], s[1], s[2]);
    }
    case "scale": {
      const f = [num(a.x ?? a.factor ?? 1, 1), num(a.y ?? a.factor ?? 1, 1), num(a.z ?? a.factor ?? 1, 1)];
      // scaling about the bbox centre (backend: translation c - f*c)
      return new THREE.Matrix4()
        .makeScale(f[0], f[1], f[2])
        .setPosition(
          center.x - f[0] * center.x,
          center.y - f[1] * center.y,
          center.z - f[2] * center.z
        );
    }
    default:
      return new THREE.Matrix4();
  }
}

function groundMatrix(box: THREE.Box3): THREE.Matrix4 {
  return new THREE.Matrix4().makeTranslation(
    -(box.min.x + box.max.x) / 2,
    -(box.min.y + box.max.y) / 2,
    -box.min.z
  );
}

export type AabbEval = (m: THREE.Matrix4) => THREE.Box3;

/**
 * Exact AABB evaluator for a GLB scene: captures each mesh's transform
 * relative to `root` once, then transforms the vertex positions — the
 * per-op bounding boxes match trimesh `mesh.bounds` exactly (unlike a
 * Box3-of-Box3 walk which overshoots for non-orthogonal rotations).
 * Call once when the model loads; safe under preview transforms since it
 * only uses static node-local matrices.
 */
export function meshAabbEvaluator(root: THREE.Object3D): AabbEval {
  const parts: {
    attr: THREE.BufferAttribute | THREE.InterleavedBufferAttribute;
    rel: THREE.Matrix4;
  }[] = [];
  const walk = (node: THREE.Object3D, parentRel: THREE.Matrix4) => {
    node.updateMatrix();
    const rel = parentRel.clone().multiply(node.matrix);
    if ((node as THREE.Mesh).isMesh) {
      const attr = (node as THREE.Mesh).geometry.getAttribute("position");
      if (attr) parts.push({ attr, rel });
    }
    node.children.forEach((c) => walk(c, rel));
  };
  root.updateMatrix();
  walk(root, new THREE.Matrix4());

  const v = new THREE.Vector3();
  const eff = new THREE.Matrix4();
  return (m: THREE.Matrix4) => {
    const box = new THREE.Box3();
    for (const { attr, rel } of parts) {
      eff.copy(m).multiply(rel);
      for (let i = 0; i < attr.count; i++) {
        v.fromBufferAttribute(attr, i).applyMatrix4(eff);
        box.expandByPoint(v);
      }
    }
    return box;
  };
}

/**
 * Compose the matrix of ops[startIndex:] — every op must be rigid or this
 * returns null (caller falls back to the server preview). Re-grounding is
 * applied after each op, exactly like apply_ops().
 */
export function rigidOpsMatrix(ops: MeshOp[], startIndex: number, aabb: AabbEval): THREE.Matrix4 | null {
  const m = new THREE.Matrix4();
  const c = new THREE.Vector3();
  for (let i = startIndex; i < ops.length; i++) {
    const o = ops[i];
    if (!isRigidOp(o)) return null;
    c.copy(aabb(m).getCenter(new THREE.Vector3()));
    m.premultiply(opMatrix(o, c));
    m.premultiply(groundMatrix(aabb(m)));
  }
  return m;
}

/** Longest prefix of `ops` matching `recorded` (element-wise JSON compare). */
export function recordedPrefixLen(ops: MeshOp[], recorded: MeshOp[]): number {
  let i = 0;
  while (
    i < recorded.length &&
    i < ops.length &&
    JSON.stringify(ops[i]) === JSON.stringify(recorded[i])
  ) {
    i++;
  }
  return i;
}
