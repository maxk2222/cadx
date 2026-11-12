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

/** A placed instance: cloned part GLB under its 4x4 world transform. The
 *  outer group's local matrix IS the instance transform, so the gizmo can
 *  edit it directly and commit reads it back. */
function Instance({
  scene,
  inst,
  innerRef,
  onSelect,
}: {
  scene: THREE.Group;
  inst: GuideInstance;
  innerRef?: Ref<THREE.Group>;
  onSelect: () => void;
}) {
  const obj = useMemo(() => scene.clone(true), [scene]);
  const { pos, quat, scl } = useMemo(() => {
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const m = new THREE.Matrix4();
    if (inst.transform?.length === 4)
      m.set(...(inst.transform.flat() as Parameters<THREE.Matrix4["set"]>));
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

/** Interactive guide editor (Kiara-style): per-part GLBs placed by instance
 *  transforms, a step scrubber that reveals the build order, click-to-select
 *  instances and a translate/rotate gizmo whose result persists via PATCH. */
export default function GuideScene({
  versionId,
  parts,
  instances,
  onTransformed,
}: {
  versionId: string;
  parts: GuidePartInfo[];
  instances: GuideInstance[];
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
      onTransformed={onTransformed}
      t={t}
    />
  );
}

function SceneInner({
  urls,
  meshParts,
  instances,
  onTransformed,
  t,
}: {
  urls: string[];
  meshParts: GuidePartInfo[];
  instances: GuideInstance[];
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
  const maxStep = useMemo(
    () => Math.max(0, ...instances.map((i) => i.placed_in_step ?? 0)),
    [instances],
  );
  const [step, setStep] = useState(maxStep);
  const [sel, setSel] = useState<string | null>(null);
  const [selObj, setSelObj] = useState<THREE.Group | null>(null);
  const [mode, setMode] = useState<"translate" | "rotate">("translate");
  const partName = (pid: string) => meshParts.find((p) => p.id === pid)?.name ?? pid;

  const visible = instances.filter((i) => (i.placed_in_step ?? 0) <= step);
  const selInst = instances.find((i) => i.id === sel);

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
        onPointerMissed={() => {
          setSel(null);
          setSelObj(null);
        }}
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
            innerRef={i.id === sel ? setSelObj : undefined}
            onSelect={() => setSel(i.id)}
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

      {selInst && (
        <div className="absolute left-2 top-2 rounded-md border border-edge bg-bg/90 px-2 py-1 text-[11px] text-fg backdrop-blur">
          <span className="font-medium">{selInst.id}</span>
          <span className="text-muted"> · {partName(selInst.part_id)}</span>
          <span className="text-faint">
            {" "}
            · {t("guide_step_n", { n: selInst.placed_in_step })}
          </span>
        </div>
      )}

      <div className="absolute inset-x-3 bottom-2 flex items-center gap-2 rounded-md border border-edge bg-bg/90 px-2.5 py-1.5 backdrop-blur">
        <span className="text-[10px] font-medium uppercase tracking-wider text-faint">
          {t("guide_step_n", { n: step })}
        </span>
        <input
          type="range"
          min={0}
          max={maxStep}
          step={1}
          value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          className="flex-1 accent-brand-500"
          aria-label={t("guide_step_scrub")}
        />
        <span className="text-[10px] text-muted">
          {visible.length}/{instances.length}
        </span>
      </div>
    </div>
  );
}
