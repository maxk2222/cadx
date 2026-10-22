import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { applyModelOpacity } from "./modelOpacity";

function scene(): { root: THREE.Group; mat: THREE.MeshStandardMaterial } {
  const root = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial();
  root.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat));
  return { root, mat };
}

describe("applyModelOpacity", () => {
  it("makes materials transparent with depthWrite off below 1", () => {
    const { root, mat } = scene();
    applyModelOpacity(root, 0.5);
    expect(mat.transparent).toBe(true);
    expect(mat.opacity).toBe(0.5);
    expect(mat.depthWrite).toBe(false);
  });

  it("restores the material's original state at 1", () => {
    const { root, mat } = scene();
    applyModelOpacity(root, 0.4);
    applyModelOpacity(root, 1);
    expect(mat.transparent).toBe(false);
    expect(mat.opacity).toBe(1);
    expect(mat.depthWrite).toBe(true);
  });

  it("does not clobber materials that were already transparent", () => {
    const root = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial();
    mat.transparent = true;
    mat.opacity = 0.3;
    root.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat));
    applyModelOpacity(root, 0.8);
    expect(mat.opacity).toBe(0.8);
    applyModelOpacity(root, 1);
    // back to its own original values, not forced to opaque-1
    expect(mat.opacity).toBe(0.3);
    expect(mat.transparent).toBe(true);
  });

  it("covers nested meshes and skips non-mesh children", () => {
    const root = new THREE.Group();
    const inner = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial();
    inner.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat));
    inner.add(new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial()));
    root.add(inner);
    applyModelOpacity(root, 0.6);
    expect(mat.opacity).toBe(0.6);
  });
});
