/** Imperative material-opacity helper for ModelCanvas — the canvas runs
 *  frameloop="demand", so callers must invalidate() after applying. */

import * as THREE from "three";

type Mat = THREE.Material & { opacity: number; depthWrite: boolean };

const ORIG_KEY = "__cadx_orig_opacity";

interface Orig {
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
}

function eachMaterial(root: THREE.Object3D, fn: (m: Mat) => void): void {
  root.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => {
        if ("opacity" in m) fn(m as Mat);
      });
    }
  });
}

/** Apply a transparency factor to every mesh material under `root`.
 *  value >= 1 restores the material's original state — important for GLBs
 *  that ship transparent materials (and for materials shared across
 *  canvases via the loader cache, e.g. compare views of the same url).
 *  transparent + depthWrite=false below 1 gives the x-ray look where
 *  interior geometry (holes, cavities) becomes visible. */
export function applyModelOpacity(root: THREE.Object3D, value: number): void {
  eachMaterial(root, (m) => {
    const rec = m.userData as Record<string, Orig | undefined>;
    if (value >= 1) {
      const orig = rec[ORIG_KEY];
      if (orig) {
        m.opacity = orig.opacity;
        m.transparent = orig.transparent;
        m.depthWrite = orig.depthWrite;
        delete rec[ORIG_KEY];
      }
      return;
    }
    if (!rec[ORIG_KEY]) {
      rec[ORIG_KEY] = {
        opacity: m.opacity,
        transparent: m.transparent,
        depthWrite: m.depthWrite,
      };
    }
    m.transparent = true;
    m.opacity = value;
    m.depthWrite = false;
  });
}
