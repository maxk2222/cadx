import * as THREE from "three";

/** Shared step palette for guide 3D views — viewer and editor must agree:
 *  parts introduced by the CURRENT step glow amber, everything placed
 *  earlier is muted slate. Every GLB mesh gets this uniform flat material
 *  so dark/metallic GLB materials never win (glTF defaults metallicFactor=1
 *  which renders black without an env map). */
export const STEP_CURRENT = new THREE.Color(0xfab005);
export const STEP_EARLIER = new THREE.Color(0x7d8a99);

export function applyStepMaterials(
  obj: THREE.Object3D,
  opts: {
    isNew: boolean;
    /** explicit per-instance color override (editor color picker) */
    colorOverride?: string | null;
    opacity?: number;
    /** selection highlight — defaults to the part's own palette color */
    emissive?: THREE.Color;
    emissiveIntensity?: number;
  },
): void {
  const base = opts.colorOverride
    ? new THREE.Color(opts.colorOverride)
    : opts.isNew
      ? STEP_CURRENT
      : STEP_EARLIER;
  const opacity = opts.opacity ?? 1;
  obj.traverse((c) => {
    const m = c as THREE.Mesh;
    if (!m.isMesh || !m.material) return;
    m.material = new THREE.MeshStandardMaterial({
      color: base,
      roughness: 0.72,
      metalness: 0.12,
      emissive: opts.emissive ?? base,
      emissiveIntensity:
        opts.emissiveIntensity ?? (opts.isNew ? 0.14 : 0.0),
      transparent: opacity < 1,
      opacity,
    });
  });
}
