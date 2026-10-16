import { useMemo } from "react";
import * as THREE from "three";
import type { FeatureAnchor } from "../../lib/types";

/** shared helpers for feature-anchor overlays (structural + organic editors) */

export const ANCHOR_COLOR = "#22d3ee";

/** axis word → unit vector */
export function axisVec(axis: string | undefined): [number, number, number] {
  const a = (axis || "+z").toLowerCase();
  const i = { x: 0, y: 1, z: 2 }[a.replace(/^[+-]/, "")] ?? 2;
  const v: [number, number, number] = [0, 0, 0];
  v[i] = a.startsWith("-") ? -1 : 1;
  return v;
}

/** quaternion aligning +Y onto `dir` (arrows/pins are modelled along +Y) */
export function dirQuat(dir: [number, number, number]): THREE.Quaternion {
  const d = new THREE.Vector3(...dir);
  if (d.lengthSq() < 1e-12) d.set(0, 1, 0);
  d.normalize();
  return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
}

/** feature-anchor → 3D point (model space); null when nothing is known. */
export function anchorPos(
  a: FeatureAnchor,
  box?: THREE.Box3 | null
): [number, number, number] | null {
  const axisIdx = { x: 0, y: 1, z: 2 }[(a.axis || "+z").replace(/^[+-]/, "")] ?? 2;
  const mid = box ? box.getCenter(new THREE.Vector3()).toArray() : [0, 0, 0];
  const p: (number | null)[] = a.pos?.length === 3 ? a.pos : [a.x, a.y, a.z ?? null];
  const out = p.map((c, i) => (c == null ? mid[i]! : c));
  // pos-less legacy anchors: x,y are the in-plane coords of a +z hole
  if (!a.pos && axisIdx === 2) out[2] = a.z ?? mid[2]!;
  if (!a.pos && axisIdx !== 2) {
    // legacy anchors are +z only — trust mid on the unknown axis
    out[axisIdx] = a.z ?? mid[axisIdx]!;
  }
  return out as [number, number, number];
}

/** mouth position for marker/sleeve display: in-plane coords from the
 *  anchor, axial coord at the box face the axis exits through (scanned
 *  anchors don't know their axial coordinate — the silhouette face is a
 *  much better marker anchor than the part centre). */
export function anchorMouth(
  a: FeatureAnchor,
  box?: THREE.Box3 | null
): [number, number, number] | null {
  const p = anchorPos(a, box);
  if (!p || !box) return p;
  const idx = { x: 0, y: 1, z: 2 }[(a.axis || "+z").replace(/^[+-]/, "")] ?? 2;
  const out = [...p] as [number, number, number];
  out[idx] = (a.axis || "+z").startsWith("-")
    ? box.min.getComponent(idx)
    : box.max.getComponent(idx);
  return out;
}

/** feature anchor marker: sphere + short axis pin + index sprite */
export function AnchorMarker({
  a,
  pos,
  selected,
  onSelect,
}: {
  a: FeatureAnchor;
  pos: [number, number, number];
  selected: boolean;
  /** when set, the marker itself is clickable (feature-pick tool armed) */
  onSelect?: () => void;
}) {
  const avec = axisVec(a.axis);
  const tex = useMemo(() => {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 64;
    const g = cv.getContext("2d")!;
    g.beginPath();
    g.arc(32, 32, 28, 0, Math.PI * 2);
    g.fillStyle = selected ? "#f59e0b" : "#164e63";
    g.fill();
    g.strokeStyle = ANCHOR_COLOR;
    g.lineWidth = 4;
    g.stroke();
    g.fillStyle = "#fff";
    g.font = "bold 34px sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(String(a.index), 32, 34);
    const t = new THREE.CanvasTexture(cv);
    t.needsUpdate = true;
    return t;
  }, [a.index, selected]);
  const s = Math.max((a.dia || 4) * 0.25, 2);
  return (
    <group
      position={pos}
      onPointerDown={
        onSelect
          ? (e) => {
              if (e.button !== 0) return;
              e.stopPropagation();
              onSelect();
            }
          : undefined
      }
    >
      <mesh>
        <sphereGeometry args={[s * 0.45, 14, 10]} />
        <meshBasicMaterial
          color={selected ? "#f59e0b" : ANCHOR_COLOR}
          depthTest={false}
          transparent
          opacity={0.95}
        />
      </mesh>
      <mesh quaternion={dirQuat(avec)} position={[avec[0] * s, avec[1] * s, avec[2] * s]}>
        <cylinderGeometry args={[0.35, 0.35, s * 2, 8]} />
        <meshBasicMaterial color={ANCHOR_COLOR} depthTest={false} transparent opacity={0.8} />
      </mesh>
      <sprite scale={[s * 1.6, s * 1.6, 1]} position={[0, s * 1.6, 0]}>
        <spriteMaterial map={tex} depthTest={false} transparent />
      </sprite>
    </group>
  );
}
