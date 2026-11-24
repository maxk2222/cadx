import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { GuideAnnotation, GuideInstance } from "../../lib/types";

/**
 * Shared guide-annotation renderer — arrows with labels + free text
 * placards. Used readonly by GuideScene (GuideTab / GuidePreviewCard)
 * and with selection+editing hooks inside GuideEditor's own scene.
 */

const UP = new THREE.Vector3(0, 1, 0);

export function axisVec(a: GuideAnnotation): THREE.Vector3 {
  const v = a.approach_axis;
  if (v?.length === 3) {
    const out = new THREE.Vector3(v[0], v[1], v[2]);
    if (out.lengthSq() > 1e-9) return out.normalize();
  }
  return UP.clone();
}

/** world position an arrow's tip anchors to: the resolved mating-port
 *  position when the backend wrote one, else the instance origin. */
export function anchorPos(
  a: GuideAnnotation,
  instances: GuideInstance[],
): THREE.Vector3 | null {
  if (a.position?.length === 3) {
    return new THREE.Vector3(a.position[0], a.position[1], a.position[2]);
  }
  const inst = instances.find(
    (i) => i.id === a.part_id || i.part_id === a.part_id,
  );
  if (!inst?.transform?.length) return null;
  const t = inst.transform;
  return new THREE.Vector3(t[0][3], t[1][3], t[2][3]);
}

const SHAFT = 46;
const HEAD = 14;
const GAP = 4;

export function AnnArrow({
  ann,
  origin,
  selected,
  onSelect,
  innerRef,
}: {
  ann: GuideAnnotation;
  origin: THREE.Vector3;
  selected?: boolean;
  onSelect?: () => void;
  innerRef?: (g: THREE.Group | null) => void;
}) {
  const dir = axisVec(ann);
  const scale = ann.size ?? 1;
  const color = ann.color || (selected ? "#2b7cff" : "#f08c00");
  const quat = new THREE.Quaternion().setFromUnitVectors(UP, dir);
  return (
    <group position={origin} quaternion={quat} scale={scale} ref={innerRef}>
      {/* tip ON the part — shaft trails back along -dir (the approach path) */}
      <mesh
        position-y={-(HEAD + SHAFT / 2 + GAP)}
        onClick={(e) => {
          if (!onSelect) return;
          e.stopPropagation();
          onSelect();
        }}
      >
        <cylinderGeometry args={[1.4, 1.4, SHAFT, 10]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <mesh
        position-y={-(HEAD / 2 + GAP)}
        onClick={(e) => {
          if (!onSelect) return;
          e.stopPropagation();
          onSelect();
        }}
      >
        <coneGeometry args={[4.5, HEAD, 12]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {!!ann.label && (
        <Html position={[0, -(SHAFT + HEAD + 8), 0]} center zIndexRange={[30, 0]}>
          <div
            className="pointer-events-none whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] font-medium shadow-lg"
            style={{
              background: "rgba(24,24,27,0.9)",
              borderColor: color,
              color: "#f4f4f5",
            }}
          >
            {ann.label}
          </div>
        </Html>
      )}
    </group>
  );
}

export function TextPlacard({
  ann,
  selected,
  onSelect,
  innerRef,
}: {
  ann: GuideAnnotation;
  selected?: boolean;
  onSelect?: () => void;
  innerRef?: (g: THREE.Group | null) => void;
}) {
  const pos = ann.position?.length === 3 ? ann.position : [0, 0, 0];
  const color = ann.color || (selected ? "#2b7cff" : "#63e6be");
  // invisible hit-proxy so placards stay selectable in 3D space
  return (
    <group position={pos as [number, number, number]} scale={ann.size ?? 1} ref={innerRef}>
      {onSelect && (
        <mesh
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
        >
          <sphereGeometry args={[6, 8, 8]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
      <Html center zIndexRange={[30, 0]}>
        <div
          className={
            "whitespace-pre-wrap rounded-lg border px-2 py-1 text-[12px] leading-snug shadow-lg" +
            (onSelect ? " cursor-pointer" : "")
          }
          style={{
            background: "rgba(24,24,27,0.92)",
            borderColor: color,
            color: "#f4f4f5",
            maxWidth: 220,
          }}
          onClick={onSelect}
        >
          {ann.text || "…"}
        </div>
      </Html>
    </group>
  );
}

/** All annotations of one step, anchored + rendered. */
export function Annotations({
  anns,
  step,
  instances,
  selKey,
  annKey,
  onSelect,
  annRef,
}: {
  anns: GuideAnnotation[];
  step: number;
  instances: GuideInstance[];
  selKey?: string | null;
  /** map annotation → its selection key (editor) */
  annKey?: (a: GuideAnnotation, i: number) => string;
  onSelect?: (a: GuideAnnotation, i: number) => void;
  annRef?: (key: string, g: THREE.Group | null) => void;
}) {
  return (
    <>
      {anns.map((a, i) => {
        if ((a.step ?? 0) !== step) return null;
        const key = annKey ? annKey(a, i) : String(i);
        const sel = selKey != null && selKey === key;
        const selCb = onSelect ? () => onSelect(a, i) : undefined;
        const refCb = annRef ? (g: THREE.Group | null) => annRef(key, g) : undefined;
        if (a.kind === "text") {
          return (
            <TextPlacard
              key={key}
              ann={a}
              selected={sel}
              onSelect={selCb}
              innerRef={refCb}
            />
          );
        }
        const origin = anchorPos(a, instances);
        if (!origin) return null;
        return (
          <AnnArrow
            key={key}
            ann={a}
            origin={origin}
            selected={sel}
            onSelect={selCb}
            innerRef={refCb}
          />
        );
      })}
    </>
  );
}
