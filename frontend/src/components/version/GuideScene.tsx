import { Grid, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useLoader } from "@react-three/fiber";
import { Boxes, Move3d, Rotate3d } from "lucide-react";
import { useMemo, useState, type Ref } from "react";
import { useTranslation } from "react-i18next";
import * as THREE from "three";
import { GLTFLoader } from "three-stdlib";
import type { GuideInstance, GuidePartInfo } from "../../lib/types";
import { IconBtn } from "../ui";

const meshUrl = (vid: string, pid: string) =>
  `/api/versions/${vid}/guide/parts/${encodeURIComponent(pid)}/mesh`;

const HIGHLIGHT = new THREE.Color(0x2f9e44);

function cloneForStep(scene: THREE.Group, isNew: boolean): THREE.Group {
  const obj = scene.clone(true);
  if (!isNew) return obj;
  // parts introduced by the CURRENT step get a green emissive tint so the
  // editor reads "what changed here" at a glance — clone materials first,
  // otherwise the tint leaks into earlier steps' clones.
  obj.traverse((c) => {
    const m = c as THREE.Mesh;
    if (m.isMesh && m.material) {
      const src = m.material as THREE.MeshStandardMaterial;
      const mat = src.clone();
      mat.emissive = HIGHLIGHT;
      mat.emissiveIntensity = 0.35;
      m.material = mat;
    }
  });
  return obj;
}

/** A placed instance: cloned part GLB under its 4x4 world transform. The
 *  outer group's local matrix IS the instance transform, so the gizmo can
 *  edit it directly and commit reads it back. */
function Instance({
  scene,
  inst,
  isNew,
  innerRef,
  onSelect,
}: {
  scene: THREE.Group;
  inst: GuideInstance;
  isNew: boolean;
  innerRef?: Ref<THREE.Group>;
  onSelect: () => void;
}) {
  const obj = useMemo(() => cloneForStep(scene, isNew), [scene, isNew]);
  const { pos, quat, scl } = useMemo(() => {
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const m = new THREE.Matrix4();
    if (inst.transform?.length === 4) m.set(...(inst.transform.flat() as Parameters<THREE.Matrix4["set"]>));
    m.decompose(p, q, s);
    return { pos: p, quat: q, scl: s };
  }, [inst.transform]);
  return (
    <group
      ref={innerRef}
      position={pos}
      quaternion={quat}
      scale={scl}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
    >
      <primitive object={obj} />
    </group>
  );
}

/** Kiara-style editor canvas: cumulative assembly up to `step`, parts added
 *  by that step highlighted, click-to-select + translate/rotate gizmo whose
 *  result persists via the parent's onTransformed (PATCH). */
export default function GuideScene({
  versionId,
  parts,
  instances,
  step,
  sel,
  onSelect,
  onTransformed,
}: {
  versionId: string;
  parts: GuidePartInfo[];
  instances: GuideInstance[];
  step: number;
  sel: string | null;
  onSelect: (iid: string | null) => void;
  onTransformed: (iid: string, m: number[][]) => void;
}) {
  const { t } = useTranslation("version");
  const meshParts = useMemo(() => parts.filter((p) => p.has_mesh), [parts]);
  const urls = useMemo(
    () => meshParts.map((p) => meshUrl(versionId, p.id)),
    [versionId, meshParts],
  );
  if (!urls.length)
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-faint">
        <Boxes className="h-8 w-8" />
        {t("guide_no_meshes")}
      </div>
    );
  return (
    <SceneInner
      key={urls.join("|")}
      urls={urls}
      meshParts={meshParts}
      instances={instances}
      step={step}
      sel={sel}
      onSelect={onSelect}
      onTransformed={onTransformed}
      t={t}
    />
  );
}

function SceneInner({
  urls,
  meshParts,
  instances,
  step,
  sel,
  onSelect,
  onTransformed,
  t,
}: {
  urls: string[];
  meshParts: GuidePartInfo[];
  instances: GuideInstance[];
  step: number;
  sel: string | null;
  onSelect: (iid: string | null) => void;
  onTransformed: (iid: string, m: number[][]) => void;
  t: (k: string, o?: Record<string, unknown>) => string;
}) {
  const gltfs = useLoader(GLTFLoader, urls);
  const scenes = useMemo(
    () =>
      Object.fromEntries(
        meshParts.map((p, i) => [p.id, gltfs[i]?.scene ?? new THREE.Group()]),
      ),
    [meshParts, gltfs],
  );
  const [selObj, setSelObj] = useState<THREE.Group | null>(null);
  const [mode, setMode] = useState<"translate" | "rotate">("translate");

  const visible = instances.filter((i) => (i.placed_in_step ?? 0) <= step);

  const commit = () => {
    if (!selObj || !sel) return;
    selObj.updateMatrix();
    const e = selObj.matrix.elements; // column-major → row-major 4x4
    onTransformed(sel, [
      [e[0], e[4], e[8], e[12]],
      [e[1], e[5], e[9], e[13]],
      [e[2], e[6], e[10], e[14]],
      [e[3], e[7], e[11], e[15]],
    ]);
  };

  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ position: [220, 180, 260], fov: 40 }}
        onPointerMissed={() => onSelect(null)}
      >
        <color attach="background" args={["#0b0d12"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[140, 220, 160]} intensity={1.1} />
        <directionalLight position={[-120, 80, -140]} intensity={0.35} />
        <Grid args={[800, 800]} cellSize={20} sectionSize={100} fadeDistance={900} />
        {visible.map((i) => (
          <Instance
            key={i.id}
            scene={scenes[i.part_id] ?? new THREE.Group()}
            inst={i}
            isNew={(i.placed_in_step ?? 0) === step}
            innerRef={i.id === sel ? setSelObj : undefined}
            onSelect={() => onSelect(i.id)}
          />
        ))}
        {selObj && (
          <TransformControls object={selObj} mode={mode} onMouseUp={commit} />
        )}
        <OrbitControls makeDefault />
      </Canvas>

      <div className="absolute right-2 top-2 flex gap-1">
        <IconBtn
          title={t("guide_move")}
          onClick={() => setMode("translate")}
          className={`bg-panel/80 backdrop-blur ${mode === "translate" ? "text-brand-400" : ""}`}
        >
          <Move3d className="h-4 w-4" />
        </IconBtn>
        <IconBtn
          title={t("guide_rot")}
          onClick={() => setMode("rotate")}
          className={`bg-panel/80 backdrop-blur ${mode === "rotate" ? "text-brand-400" : ""}`}
        >
          <Rotate3d className="h-4 w-4" />
        </IconBtn>
      </div>
    </div>
  );
}
