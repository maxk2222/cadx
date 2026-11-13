import { Grid, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { Boxes, Move3d, Rotate3d } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type Ref } from "react";
import { useTranslation } from "react-i18next";
import * as THREE from "three";
import { GLTFLoader } from "three-stdlib";
import type { GuideAnnotation, GuideInstance, GuidePartInfo } from "../../lib/types";
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

/** Annotation arrow for the CURRENT step: world-direction hint anchored at
 *  the annotated instance's origin (mirrors unity._annotation_arrows). */
function AnnArrow({
  axis,
  origin,
}: {
  axis: number[];
  origin: THREE.Vector3;
}) {
  const arrow = useMemo(() => {
    const dir = new THREE.Vector3(...(axis as [number, number, number]));
    const len = Math.max(dir.length(), 1);
    dir.normalize();
    return new THREE.ArrowHelper(
      dir,
      origin.clone().addScaledVector(dir, -30),
      Math.min(len, 80),
      0xffb020,
      12,
      6,
    );
  }, [axis, origin]);
  return <primitive object={arrow} />;
}

/** Frame the camera on the placed assembly's world bbox: without this the
 *  three.js defaults (near 0.1 / far 2000, camera aimed at the origin) clip
 *  millimetre assemblies on zoom and show an empty view for off-origin
 *  placements. Distance controls the clip planes, never hardcoded. */
export function CameraFit({ content, fitKey }: { content: Ref<THREE.Group>; fitKey?: unknown }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as { target?: THREE.Vector3; minDistance?: number; maxDistance?: number; update?: () => void } | null;
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    const grp = content && (content as React.RefObject<THREE.Group>).current;
    if (!grp) return;
    const box = new THREE.Box3().setFromObject(grp);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const diag = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
    const dist = diag * 1.35;
    const dir = new THREE.Vector3(0.55, 0.45, 0.7).normalize();
    camera.position.copy(center).addScaledVector(dir, dist);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = Math.max(diag / 500, 0.01);
      camera.far = Math.max(diag * 20, 1000);
      camera.updateProjectionMatrix();
    }
    if (controls) {
      controls.target?.copy(center);
      if ("minDistance" in controls) controls.minDistance = diag * 0.05;
      if ("maxDistance" in controls) controls.maxDistance = diag * 8;
      controls.update?.();
    }
    invalidate();
  }, [camera, controls, content, invalidate, fitKey]);
  return null;
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
  readonly,
  annotations,
}: {
  versionId: string;
  parts: GuidePartInfo[];
  instances: GuideInstance[];
  step: number;
  sel: string | null;
  onSelect: (iid: string | null) => void;
  onTransformed: (iid: string, m: number[][]) => void;
  readonly?: boolean;
  annotations?: GuideAnnotation[];
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
      readonly={readonly}
      annotations={annotations}
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
  readonly,
  annotations,
  t,
}: {
  urls: string[];
  meshParts: GuidePartInfo[];
  instances: GuideInstance[];
  step: number;
  sel: string | null;
  onSelect: (iid: string | null) => void;
  onTransformed: (iid: string, m: number[][]) => void;
  readonly?: boolean;
  annotations?: GuideAnnotation[];
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
  const [ctxLost, setCtxLost] = useState(0);
  const contentRef = useRef<THREE.Group>(null);

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
        key={ctxLost}
        frameloop="demand"
        dpr={[1, 2]}
        camera={{ position: [220, 180, 260], fov: 40 }}
        onPointerMissed={() => onSelect(null)}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener("webglcontextlost", (e) => {
            e.preventDefault();
            setCtxLost((n) => n + 1);
          });
        }}
      >
        <color attach="background" args={["#0b0d12"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[140, 220, 160]} intensity={1.1} />
        <directionalLight position={[-120, 80, -140]} intensity={0.35} />
        <Grid args={[800, 800]} cellSize={20} sectionSize={100} fadeDistance={900} />
        <group ref={contentRef}>
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
        </group>
        {(annotations ?? [])
          .filter((a) => a.kind === "arrow" && a.step === step)
          .map((a, ai) => {
            const inst = visible.find(
              (i) => i.id === a.part_id || i.part_id === a.part_id,
            );
            if (!inst?.transform?.length) return null;
            const origin = new THREE.Vector3(
              inst.transform[0][3],
              inst.transform[1][3],
              inst.transform[2][3],
            );
            return <AnnArrow key={ai} axis={a.approach_axis} origin={origin} />;
          })}
        <CameraFit content={contentRef} fitKey={step} />
        {selObj && !readonly && (
          <TransformControls object={selObj} mode={mode} onMouseUp={commit} />
        )}
        <OrbitControls makeDefault enableDamping dampingFactor={0.1} zoomToCursor />
      </Canvas>

      {!readonly && (
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
      )}
    </div>
  );
}
