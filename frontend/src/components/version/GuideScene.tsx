import { Grid, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { Boxes, Move3d, Rotate3d } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type Ref } from "react";
import { useTranslation } from "react-i18next";
import * as THREE from "three";
import { GLTFLoader } from "three-stdlib";
import type {
  GuideAnnotation,
  GuideInstance,
  GuidePartInfo,
} from "../../lib/types";
import WebGLGate from "../common/WebGLGate";
import { IconBtn } from "../ui";
import { Annotations } from "./guideAnnotations";
import { applyStepMaterials } from "./guidePalette";

const meshUrl = (vid: string, pid: string) =>
  `/api/versions/${vid}/guide/parts/${encodeURIComponent(pid)}/mesh`;

/* Step palette lives in ./guidePalette — shared with GuideEditor so every
 *  guide view paints parts identically. */

/** The solver world is Z-up mm; three.js renders Y-up. One -90degX
 *  wrapper presents a Z-up world upright — instance transforms stay in
 *  solver frame, so gizmo commits read back the same matrix they would
 *  without the wrapper. Blender-mode guides arrive already Y-up and get
 *  no wrapper. */
const ZUP_DISPLAY_ROT: [number, number, number] = [-Math.PI / 2, 0, 0];

/** Operator-view yaw: the solver's canonical front is +Y, which the tilt
 *  alone maps to -Z — facing AWAY from the default camera. An extra 180°
 *  turn about display-Y (outermost wrapper, applied after the tilt)
 *  brings the working face to +Z: the camera default sees the front, and
 *  operator-left (+X world) lands screen-left like the manual figure. */
const FRONT_YAW_ROT: [number, number, number] = [0, Math.PI, 0];

function cloneForStep(
  scene: THREE.Group,
  isNew: boolean,
  isSel: boolean,
): THREE.Group {
  const obj = scene.clone(true);
  applyStepMaterials(obj, {
    isNew,
    emissiveIntensity: isSel ? 0.45 : isNew ? 0.14 : 0.0,
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
  isSel,
  innerRef,
  onSelect,
}: {
  scene: THREE.Group;
  inst: GuideInstance;
  isNew: boolean;
  isSel: boolean;
  innerRef?: Ref<THREE.Group>;
  onSelect: () => void;
}) {
  const obj = useMemo(
    () => cloneForStep(scene, isNew, isSel),
    [scene, isNew, isSel],
  );
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

/** Annotation arrows + text placards render via the shared
 *  guideAnnotations components (same visuals as the editor). */

/** Ground plane that always stays subtly visible: sized and faded from
 *  the scene bbox instead of a fixed 800mm guess, parked at the
 *  assembly's underside, dark-quiet colors on the #0b0d12 background. */
function AdaptiveGrid({
  content,
  fitKey,
}: {
  content: Ref<THREE.Group>;
  fitKey?: unknown;
}) {
  const [cfg, setCfg] = useState<{
    size: number;
    cell: number;
    section: number;
    fade: number;
    y: number;
  } | null>(null);
  useEffect(() => {
    const grp = content && (content as React.RefObject<THREE.Group>).current;
    if (!grp) return;
    const box = new THREE.Box3().setFromObject(grp);
    if (box.isEmpty()) return;
    const diag = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
    const size = Math.ceil(diag * 3);
    setCfg({
      size,
      cell: Math.max(Math.round(size / 40), 1),
      section: Math.max(Math.round(size / 8), 10),
      fade: diag * 3.5,
      y: box.min.y - 0.5,
    });
  }, [content, fitKey]);
  if (!cfg) return null;
  return (
    <Grid
      position={[0, cfg.y, 0]}
      args={[cfg.size, cfg.size]}
      cellSize={cfg.cell}
      sectionSize={cfg.section}
      cellColor="#1d222b"
      sectionColor="#2b3140"
      fadeDistance={cfg.fade}
      fadeStrength={1.2}
    />
  );
}

/** Frame the camera on the placed assembly's world bbox: without this the
 *  three.js defaults (near 0.1 / far 2000, camera aimed at the origin) clip
 *  millimetre assemblies on zoom and show an empty view for off-origin
 *  placements. Distance controls the clip planes, never hardcoded. The
 *  +Z-octant start dir is deliberate: after the front-yaw wrapper the
 *  operator face points +Z, so the default view IS the manual's front. */
export function CameraFit({
  content,
  fitKey,
}: {
  content: Ref<THREE.Group>;
  fitKey?: unknown;
}) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as {
    target?: THREE.Vector3;
    minDistance?: number;
    maxDistance?: number;
    update?: () => void;
  } | null;
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
  worldFrame,
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
  worldFrame?: "z_up" | "y_up";
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
      worldFrame={worldFrame}
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
  worldFrame,
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
  worldFrame?: "z_up" | "y_up";
  t: (k: string, o?: Record<string, unknown>) => string;
}) {
  // manual loading instead of useLoader: a single 404/malformed mesh must
  // degrade to an empty group, not throw past the ErrorBoundary and kill
  // the whole page (observed: generated parts whose mesh path points
  // outside the guide dir)
  const [gltfs, setGltfs] = useState<(THREE.Group | null)[]>([]);
  useEffect(() => {
    let alive = true;
    const loader = new GLTFLoader();
    setGltfs(new Array(urls.length).fill(null));
    urls.forEach((u, i) => {
      loader
        .loadAsync(u)
        .then((g) => {
          if (alive)
            setGltfs((prev) => {
              const next = [...prev];
              next[i] = g.scene;
              return next;
            });
        })
        .catch(() => {
          // missing/corrupt mesh — leave the slot empty, other parts still show
        });
    });
    return () => {
      alive = false;
    };
  }, [urls]);
  const scenes = useMemo(
    () =>
      Object.fromEntries(
        meshParts.map((p, i) => [p.id, gltfs[i] ?? new THREE.Group()]),
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
      <WebGLGate>
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
          <AdaptiveGrid content={contentRef} fitKey={step} />
          {/* Z-up solver world -> Y-up display: rotate the whole world
            (parts + annotations) once, never each transform. The outer
            yaw puts the +Y working face toward the default camera —
            operator-left lands screen-left like the manual figure. */}
          <group rotation={worldFrame === "y_up" ? [0, 0, 0] : FRONT_YAW_ROT}>
            <group
              rotation={worldFrame === "y_up" ? [0, 0, 0] : ZUP_DISPLAY_ROT}
            >
              <group ref={contentRef}>
                {visible.map((i) => (
                  <Instance
                    key={i.id}
                    scene={scenes[i.part_id] ?? new THREE.Group()}
                    inst={i}
                    isNew={(i.placed_in_step ?? 0) === step}
                    isSel={i.id === sel}
                    innerRef={i.id === sel ? setSelObj : undefined}
                    onSelect={() => onSelect(i.id)}
                  />
                ))}
              </group>
              <Annotations
                anns={annotations ?? []}
                step={step}
                instances={visible}
              />
            </group>
          </group>
          <CameraFit content={contentRef} fitKey={step} />
          {selObj && !readonly && (
            <TransformControls object={selObj} mode={mode} onMouseUp={commit} />
          )}
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.1}
            zoomToCursor
          />
        </Canvas>
      </WebGLGate>

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
