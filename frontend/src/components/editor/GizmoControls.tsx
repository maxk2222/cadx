import { TransformControls } from "@react-three/drei";
import type { ReactNode } from "react";
import type * as THREE from "three";

/**
 * Wrapper around drei TransformControls: the children are wrapped in a
 * controlled group whose `position`/`rotation` MUST carry the object's
 * placement — TransformControls reports that group's transform on drag, so
 * `onChange` gets absolute model-space (mm) values, never deltas.
 * OrbitControls are auto-disabled while dragging (makeDefault controls).
 */
export default function GizmoControls({
  children,
  enabled = true,
  mode = "translate",
  axes = "xyz",
  position,
  rotation,
  onChange,
  onDragStart,
  onDragEnd,
}: {
  children: ReactNode;
  enabled?: boolean;
  mode?: "translate" | "rotate";
  /** gizmo axes shown — pick a subset like "z" for slice-plane drags */
  axes?: "x" | "y" | "z" | "xy" | "xyz";
  position?: [number, number, number];
  rotation?: [number, number, number];
  /** fired continuously while dragging — obj is the controlled group;
   *  obj.position/obj.rotation are absolute (model-space mm) */
  onChange?: (obj: THREE.Object3D) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}) {
  const inner = (
    <group position={position} rotation={rotation}>
      {children}
    </group>
  );
  if (!enabled) return inner;
  return (
    <TransformControls
      mode={mode}
      showX={axes.includes("x")}
      showY={axes.includes("y")}
      showZ={axes.includes("z")}
      onMouseDown={onDragStart}
      onMouseUp={onDragEnd}
      onObjectChange={(e) => {
        const o = (e as { target?: { object?: THREE.Object3D } } | undefined)?.target
          ?.object;
        if (o) onChange?.(o);
      }}
    >
      {inner}
    </TransformControls>
  );
}
