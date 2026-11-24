import {
  ArrowDownToLine,
  ArrowUpFromDot,
  Camera,
  ChevronLeft,
  ChevronRight,
  CopyPlus,
  Crosshair,
  Eye,
  Maximize2,
  Move3d,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Plus,
  Redo2,
  Rotate3d,
  Square,
  Trash2,
  Type,
  Undo2,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Ref,
} from "react";
import { useTranslation } from "react-i18next";
import { createPortal } from "react-dom";
import * as THREE from "three";
import {
  GizmoHelper,
  GizmoViewport,
  Grid,
  Line,
  OrbitControls,
  TransformControls,
} from "@react-three/drei";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { GLTFLoader, type OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { api, errMsg } from "../../lib/api";
import type {
  Asset,
  GuideAnnotation,
  GuideDetail,
  GuideInstance,
  GuidePartInfo,
  GuideStepDetail,
} from "../../lib/types";
import SafeImg from "../SafeImg";
import WebGLGate from "../common/WebGLGate";
import { CameraFit } from "./GuideScene";
import { Annotations, axisVec } from "./guideAnnotations";
import { applyStepMaterials } from "./guidePalette";
import { FRONT_YAW_ROT, ZUP_DISPLAY_ROT, solverToDisplayVec } from "./guideFrame";
import { Btn, IconBtn, Spinner } from "../ui";

/**
 * Full KIARA guide-editor port, opened as a modal from GuideTab.
 * Layout mirrors editor/[id]: left sidebar (step navigator + inline step
 * text + asset library tabs), centre 3D canvas (per-step cumulative scene,
 * click/multi-select, pivot gizmo, context menu), right inspector (object
 * list + numeric transforms + quick rotations). Undo/redo over the whole
 * document; edits flush to the backend as diffs (PATCH steps/instances,
 * POST/DELETE instances) — debounced and on step switch/close.
 */

const meshUrl = (vid: string, pid: string) =>
  `/api/versions/${vid}/guide/parts/${encodeURIComponent(pid)}/mesh`;
const mediaUrl = (vid: string, name: string) =>
  `/api/versions/${vid}/guide/media/${encodeURIComponent(name)}`;

const PALETTE = ["#8c9bb3", "#e03131", "#2f9e44", "#1971c2", "#f08c00", "#e5e5e5"];

type Doc = {
  steps: GuideStepDetail[];
  instances: GuideInstance[];
  annotations: EdAnn[];
};
type Mode = "translate" | "rotate" | "scale";
/* solver→display wrappers come from ./guideFrame — same nested
 *  tilt+front-yaw as GuideScene so every view shows the model's front. */
/** Editable annotation: server dict + client bookkeeping. `k` is the
 *  identity key (never persisted), `idx` is the index in
 *  state.annotations — undefined means "not yet POSTed". */
type EdAnn = GuideAnnotation & { k: string; idx?: number };
const annPayload = (a: EdAnn): GuideAnnotation => {
  const rest = { ...a } as Record<string, unknown>;
  delete rest.k;
  delete rest.idx;
  return rest as unknown as GuideAnnotation;
};
const annEq = (a: EdAnn, b: EdAnn) =>
  JSON.stringify(annPayload(a)) === JSON.stringify(annPayload(b));

const DIR_PRESETS = ["up", "down", "left", "right", "front", "back"] as const;

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

function matFromRows(t: number[][] | undefined): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  if (t?.length === 4)
    m.set(...(t.flat() as Parameters<THREE.Matrix4["set"]>));
  return m;
}

function rowsFromMat(m: THREE.Matrix4): number[][] {
  const e = m.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];
}

function decompose(t: number[][]) {
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  matFromRows(t).decompose(p, q, s);
  return { pos: p, quat: q, scl: s, euler: new THREE.Euler().setFromQuaternion(q) };
}

function instEq(a: GuideInstance, b: GuideInstance): boolean {
  return (
    a.placed_in_step === b.placed_in_step &&
    (a.color ?? null) === (b.color ?? null) &&
    (a.opacity ?? null) === (b.opacity ?? null) &&
    JSON.stringify(a.transform) === JSON.stringify(b.transform)
  );
}

// ======================== 3D scene ========================

function PartObject({
  scene,
  inst,
  selected,
  ghost,
  isNew,
  innerRef,
  onSelect,
  onContext,
}: {
  scene: THREE.Group;
  inst: GuideInstance;
  selected: boolean;
  /** isolate-step mode: earlier steps fade to a ghost */
  ghost?: boolean;
  /** part was placed by the CURRENT step → amber highlight */
  isNew: boolean;
  innerRef?: Ref<THREE.Group>;
  onSelect: (e: { stopPropagation: () => void; ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) => void;
  onContext: (iid: string, x: number, y: number) => void;
}) {
  const obj = useMemo(() => {
    const o = scene.clone(true);
    // uniform flat step palette (shared with GuideScene) — cloning the GLB's
    // own material keeps glTF-default metallic=1 and renders parts BLACK on
    // the dark canvas without an env map.
    applyStepMaterials(o, {
      isNew,
      colorOverride: inst.color,
      opacity: ghost ? 0.15 : (inst.opacity ?? 1),
      emissive: selected ? new THREE.Color("#2b7cff") : undefined,
      emissiveIntensity: selected ? 0.3 : isNew ? 0.14 : 0.0,
    });
    return o;
  }, [scene, inst.color, inst.opacity, selected, ghost, isNew]);

  const { pos, quat, scl } = useMemo(() => decompose(inst.transform), [inst.transform]);

  return (
    <group
      ref={innerRef}
      position={pos}
      quaternion={quat}
      scale={scl}
      onClick={onSelect}
      onContextMenu={(e) => {
        e.stopPropagation();
        (e.nativeEvent as MouseEvent).preventDefault();
        const n = e.nativeEvent as MouseEvent;
        onContext(inst.id, n.clientX, n.clientY);
      }}
    >
      <primitive object={obj} />
    </group>
  );
}

function Scene({
  versionId,
  parts,
  instances,
  stepNo,
  sel,
  primary,
  mode,
  gizmoDrag,
  onSelect,
  onContext,
  onMiss,
  onTransformedMany,
  focus,
  anns,
  selAnn,
  onAnnSelect,
  onAnnTransformed,
  isolate,
  playing,
  glRef,
  worldFrame,
}: {
  versionId: string;
  parts: GuidePartInfo[];
  instances: GuideInstance[];
  stepNo: number;
  sel: Set<string>;
  primary: string | null;
  mode: Mode;
  gizmoDrag: (d: boolean) => void;
  onSelect: (iid: string | null, multi: boolean) => void;
  onContext: (iid: string, x: number, y: number) => void;
  onMiss: () => void;
  onTransformedMany: (t: Record<string, number[][]>) => void;
  focus: { iid: string; v: number } | null;
  anns: EdAnn[];
  selAnn: EdAnn | null;
  onAnnSelect: (k: string | null) => void;
  onAnnTransformed: (k: string, p: { axis?: number[]; position?: number[] }) => void;
  isolate: boolean;
  playing: { t0: number } | null;
  glRef: React.MutableRefObject<THREE.WebGLRenderer | null>;
  worldFrame?: "z_up" | "y_up";
}) {
  const meshParts = useMemo(() => parts.filter((p) => p.has_mesh), [parts]);
  const urls = useMemo(
    () => meshParts.map((p) => meshUrl(versionId, p.id)),
    [versionId, meshParts],
  );
  const key = urls.join("|");
  const [ctxLost, setCtxLost] = useState(0);
  const visible = useMemo(
    () => instances.filter((i) => (i.placed_in_step ?? 0) <= stepNo),
    [instances, stepNo],
  );
  return (
    <WebGLGate>
    <Canvas
      key={`${key}:${ctxLost}`}
      frameloop={playing ? "always" : "demand"}
      dpr={[1, 2]}
      camera={{ position: [220, 200, 260], fov: 40 }}
      onPointerMissed={onMiss}
      onCreated={({ gl }) => {
        glRef.current = gl;
        gl.domElement.addEventListener("webglcontextlost", (e) => {
          e.preventDefault();
          setCtxLost((n) => n + 1);
        });
      }}
      gl={{ antialias: true, powerPreference: "high-performance", alpha: false, stencil: false, preserveDrawingBuffer: true }}
    >
      <color attach="background" args={["#181818"]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[160, 240, 180]} intensity={1.1} />
      <directionalLight position={[-140, 90, -150]} intensity={0.35} />
      <Grid args={[1200, 1200]} cellSize={10} sectionSize={100} fadeDistance={1500} infiniteGrid />
      <Line points={[[0, 0.5, 0], [400, 0.5, 0]]} color="#ff4d5a" lineWidth={1.5} />
      <Line points={[[0, 0.5, 0], [0, 0.5, 400]]} color="#2b7cff" lineWidth={1.5} />
      <Line points={[[0, 0, 0], [0, 400, 0]]} color="#00d084" lineWidth={1.5} />
      <SceneInner
        urls={urls}
        meshParts={meshParts}
        instances={visible}
        sel={sel}
        primary={primary}
        mode={mode}
        gizmoDrag={gizmoDrag}
        onSelect={onSelect}
        onContext={onContext}
        onTransformedMany={onTransformedMany}
        focus={focus}
        anns={anns}
        stepNo={stepNo}
        selAnn={selAnn}
        onAnnSelect={onAnnSelect}
        onAnnTransformed={onAnnTransformed}
        isolate={isolate}
        playing={playing}
        worldFrame={worldFrame}
      />
      <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
        <GizmoViewport axisColors={["#ff4d5a", "#00d084", "#2b7cff"]} labelColor="#ccc" />
      </GizmoHelper>
    </Canvas>
    </WebGLGate>
  );
}

/** Per-step assembly animation: freshly placed instances slide into
 *  their final transform along the step's arrow axis (KIARA preview). */
function AnimatedIn({
  dir,
  delay,
  t0,
  children,
}: {
  dir: THREE.Vector3;
  delay: number;
  t0: number;
  children: React.ReactNode;
}) {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    const g = ref.current;
    if (!g) return;
    const el = (performance.now() - t0) / 1000 - delay;
    const p = Math.min(1, Math.max(0, el / 0.9));
    const ease = 1 - (1 - p) * (1 - p); // ease-out quad
    g.position.copy(dir).multiplyScalar((1 - ease) * 140);
  });
  return <group ref={ref}>{children}</group>;
}

function SceneInner({
  urls,
  meshParts,
  instances,
  sel,
  mode,
  gizmoDrag,
  onSelect,
  onContext,
  onTransformedMany,
  focus,
  anns,
  stepNo,
  selAnn,
  onAnnSelect,
  onAnnTransformed,
  isolate,
  playing,
  worldFrame,
}: {
  urls: string[];
  meshParts: GuidePartInfo[];
  instances: GuideInstance[];
  sel: Set<string>;
  primary: string | null;
  mode: Mode;
  gizmoDrag: (d: boolean) => void;
  onSelect: (iid: string | null, multi: boolean) => void;
  onContext: (iid: string, x: number, y: number) => void;
  onTransformedMany: (t: Record<string, number[][]>) => void;
  focus: { iid: string; v: number } | null;
  anns: EdAnn[];
  stepNo: number;
  selAnn: EdAnn | null;
  onAnnSelect: (k: string | null) => void;
  onAnnTransformed: (k: string, p: { axis?: number[]; position?: number[] }) => void;
  isolate: boolean;
  playing: { t0: number } | null;
  worldFrame?: "z_up" | "y_up";
}) {
  const gltfs = useLoader(GLTFLoader, urls);
  const scenes = useMemo(
    () =>
      Object.fromEntries(
        meshParts.map((p, i) => [p.id, gltfs[i]?.scene ?? new THREE.Group()]),
      ),
    [meshParts, gltfs],
  );

  const refs = useRef<Record<string, THREE.Group | null>>({});
  const annRefs = useRef<Record<string, THREE.Group | null>>({});
  const [annObj, setAnnObj] = useState<THREE.Group | null>(null);
  const pivotRef = useRef<THREE.Group>(null!);
  const [pivotObj, setPivotObj] = useState<THREE.Group | null>(null);
  const pivotStart = useRef(new THREE.Matrix4());
  const itemStart = useRef<Record<string, THREE.Matrix4>>({});
  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  const contentRef = useRef<THREE.Group>(null);
  const zUp = worldFrame !== "y_up";
  const visibleIds = useMemo(() => new Set(instances.map((i) => i.id)), [instances]);
  const liveSel = useMemo(
    () => new Set([...sel].filter((i) => visibleIds.has(i))),
    [sel, visibleIds],
  );

  // selection pivot — centroid of the live selection (KIARA multi-select pivot)
  const pivotPos = useMemo(() => {
    const c = new THREE.Vector3();
    let n = 0;
    for (const iid of liveSel) {
      const inst = instances.find((i) => i.id === iid);
      if (!inst) continue;
      c.add(decompose(inst.transform).pos);
      n += 1;
    }
    if (n) c.multiplyScalar(1 / n);
    return c;
  }, [liveSel, instances]);

  // camera focus request
  const focusSeen = useRef(0);
  useEffect(() => {
    if (!focus || focus.v === focusSeen.current) return;
    focusSeen.current = focus.v;
    const inst = instances.find((i) => i.id === focus.iid);
    const controls = controlsRef.current;
    if (!inst || !controls) return;
    const p = decompose(inst.transform).pos;
    solverToDisplayVec(p, zUp); // solver coords -> display coords
    const cam = controls.object as THREE.Camera;
    const dir = cam.position.clone().sub(controls.target).normalize();
    controls.target.copy(p);
    cam.position.copy(p.clone().add(dir.multiplyScalar(160)));
    controls.update();
  }, [focus, instances, zUp]);

  const applyDelta = useCallback(() => {
    const piv = pivotRef.current;
    if (!piv) return;
    const delta = piv.matrix.clone().multiply(pivotStart.current.clone().invert());
    for (const [iid, m0] of Object.entries(itemStart.current)) {
      const g = refs.current[iid];
      if (!g) continue;
      const m = delta.clone().multiply(m0);
      const p = new THREE.Vector3();
      const q = new THREE.Quaternion();
      const s = new THREE.Vector3();
      m.decompose(p, q, s);
      g.position.copy(p);
      g.quaternion.copy(q);
      g.scale.copy(s);
    }
  }, []);

  const commit = useCallback(() => {
    const out: Record<string, number[][]> = {};
    for (const iid of Object.keys(itemStart.current)) {
      const g = refs.current[iid];
      if (!g) continue;
      g.updateMatrix();
      out[iid] = rowsFromMat(g.matrix);
    }
    if (Object.keys(out).length) onTransformedMany(out);
  }, [onTransformedMany]);

  const showGizmo = liveSel.size > 0 && !playing;

  // animation direction per freshly-placed instance: its step's arrow
  // axis if annotated, else +Z (slide down)
  const dirFor = useMemo(() => {
    const m = new Map<string, THREE.Vector3>();
    for (const a of anns) {
      if (a.kind === "arrow" && a.step === stepNo && a.part_id)
        m.set(a.part_id, axisVec(a));
    }
    return (i: GuideInstance) =>
      m.get(i.id) ?? m.get(i.part_id) ?? new THREE.Vector3(0, 0, 1);
  }, [anns, stepNo]);

  // stagger order among this step's new instances
  const newOrder = useMemo(() => {
    const m = new Map<string, number>();
    let n = 0;
    for (const i of instances)
      if ((i.placed_in_step ?? 0) === stepNo) m.set(i.id, n++);
    return m;
  }, [instances, stepNo]);

  // annotation gizmo: rotate changes the arrow axis, translate moves the
  // text placard — commits back into the doc
  const commitAnn = useCallback(() => {
    if (!selAnn) return;
    const g = annRefs.current[selAnn.k];
    if (!g) return;
    if (selAnn.kind === "text") {
      onAnnTransformed(selAnn.k, {
        position: [g.position.x, g.position.y, g.position.z],
      });
    } else {
      const v = new THREE.Vector3(0, 1, 0).applyQuaternion(g.quaternion).normalize();
      onAnnTransformed(selAnn.k, { axis: [v.x, v.y, v.z] });
    }
  }, [selAnn, onAnnTransformed]);

  return (
    <>
      {/* Z-up solver world -> Y-up display + front-yaw: nested wrappers
          rotate parts, annotations AND the selection pivot together, so
          every local matrix the gizmo touches stays in solver frame. */}
      <group rotation={zUp ? FRONT_YAW_ROT : [0, 0, 0]}>
      <group rotation={zUp ? ZUP_DISPLAY_ROT : [0, 0, 0]}>
        <group ref={contentRef}>
          {instances.map((i) => {
            const isNew = (i.placed_in_step ?? 0) === stepNo;
            const node = (
              <PartObject
                key={i.id}
                scene={scenes[i.part_id] ?? new THREE.Group()}
                inst={i}
                selected={liveSel.has(i.id)}
                ghost={isolate && !isNew}
                isNew={isNew}
                innerRef={(g) => {
                  refs.current[i.id] = g;
                }}
                onSelect={(e) => {
                  e.stopPropagation();
                  onSelect(
                    i.id,
                    !!(e.ctrlKey || e.metaKey || e.shiftKey),
                  );
                }}
                onContext={onContext}
              />
            );
            if (playing && isNew) {
              return (
                <AnimatedIn key={i.id} dir={dirFor(i)} delay={(newOrder.get(i.id) ?? 0) * 0.15} t0={playing.t0}>
                  {node}
                </AnimatedIn>
              );
            }
            return node;
          })}
        </group>
        <Annotations
          anns={anns}
          step={stepNo}
          instances={instances}
          selKey={selAnn?.k ?? null}
          annKey={(a) => (a as EdAnn).k}
          onSelect={(a) => onAnnSelect((a as EdAnn).k)}
          annRef={(key, g) => {
            annRefs.current[key] = g;
            if (selAnn && key === selAnn.k) setAnnObj(g);
          }}
        />
        {showGizmo && (
          <group
            position={pivotPos}
            ref={(g) => {
              pivotRef.current = g as THREE.Group;
              setPivotObj(g);
            }}
          />
        )}
      </group>
      </group>
      <CameraFit content={contentRef} fitKey={`${stepNo}:${urls.length}`} />

      {showGizmo && pivotObj && (
        <TransformControls
          object={pivotObj}
          mode={mode}
          size={0.9}
          onMouseDown={() => {
            gizmoDrag(true);
            pivotStart.current.copy(pivotRef.current.matrix);
            itemStart.current = {};
            for (const iid of liveSel) {
              const g = refs.current[iid];
              if (g) {
                g.updateMatrix();
                itemStart.current[iid] = g.matrix.clone();
              }
            }
          }}
          onObjectChange={applyDelta}
          onMouseUp={() => {
            gizmoDrag(false);
            commit();
          }}
        />
      )}
      {selAnn && annObj && (
        <TransformControls
          object={annObj}
          mode={selAnn.kind === "text" ? "translate" : "rotate"}
          size={0.8}
          onMouseDown={() => gizmoDrag(true)}
          onMouseUp={() => {
            gizmoDrag(false);
            commitAnn();
          }}
        />
      )}
      <OrbitControls makeDefault ref={controlsRef} enableDamping dampingFactor={0.1} zoomToCursor />
    </>
  );
}

// ======================== editor shell ========================

export default function GuideEditor({
  versionId,
  name,
  onClose,
  projectId,
}: {
  versionId: string;
  name: string;
  onClose: () => void;
  /** owning project — enables the project-asset→media copy picker */
  projectId?: string;
}) {
  const { t } = useTranslation("version");
  const [guide, setGuide] = useState<GuideDetail | null>(null);
  const [doc, setDoc] = useState<Doc | null>(null);
  const [base, setBase] = useState<Doc | null>(null);
  const [history, setHistory] = useState<Doc[]>([]);
  const [redoStack, setRedoStack] = useState<Doc[]>([]);
  const [cur, setCur] = useState(0);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [primary, setPrimary] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("translate");
  const [assetTab, setAssetTab] = useState<"parts" | "media">("parts");
  const [ctx, setCtx] = useState<{ iid: string; x: number; y: number; sub: "main" | "color" | "alpha" } | null>(null);
  const [focus, setFocus] = useState<{ iid: string; v: number } | null>(null);
  // selected annotation (EdAnn.k) — mutually exclusive with instance sel
  const [annSel, setAnnSel] = useState<string | null>(null);
  // right rail: collapsible steps/objects tabs
  const [railOpen, setRailOpen] = useState(true);
  const [railTab, setRailTab] = useState<"steps" | "objects">("steps");
  const [dirPreset, setDirPreset] = useState<(typeof DIR_PRESETS)[number]>("down");
  const [annLabel, setAnnLabel] = useState("");
  const [isolateStep, setIsolateStep] = useState(false);
  const [playing, setPlaying] = useState<{ t0: number } | null>(null);
  const [mediaFor, setMediaFor] = useState<number | null>(null); // step no with open picker
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const tmpAnnSeq = useRef(0);
  const [saveState, setSaveState] = useState<"saved" | "dirty" | "saving" | "error">("saved");
  const [feedback, setFeedback] = useState<{ msg: string; err?: boolean } | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const tmpSeq = useRef(0);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const docRef = useRef<Doc | null>(null);
  const baseRef = useRef<Doc | null>(null);
  docRef.current = doc;
  baseRef.current = base;

  const reloadDoc = useCallback(() => {
    api
      .getGuide(versionId)
      .then((g) => {
        setGuide(g);
        const d: Doc = {
          steps: clone(g.steps),
          instances: clone(g.instance_list ?? []),
          annotations: (g.annotations ?? []).map((a, i) => ({
            ...(clone(a) as GuideAnnotation),
            k: `a${i}_${Date.now()}`,
            idx: i,
          })),
        };
        setDoc(d);
        setBase(clone(d));
      })
      .catch((e) => setLoadErr(errMsg(e)));
  }, [versionId]);

  useEffect(reloadDoc, [reloadDoc]);

  const flash = (msg: string, err = false) => {
    setFeedback({ msg, err });
    setTimeout(() => setFeedback(null), 2500);
  };

  // ---- diff-based persistence: doc is truth, base is last synced ----
  const flush = useCallback(async () => {
    const d = docRef.current;
    const b = baseRef.current;
    if (!d || !b || JSON.stringify(d) === JSON.stringify(b)) {
      setSaveState("saved");
      return;
    }
    setSaveState("saving");
    try {
      const bSteps = new Map(b.steps.map((s) => [s.number, s]));
      for (const s of d.steps) {
        const o = bSteps.get(s.number);
        if (
          !o ||
          o.title !== s.title ||
          o.text !== s.text ||
          o.display_text !== s.display_text ||
          o.safety_hint !== s.safety_hint ||
          (o.media_image ?? null) !== (s.media_image ?? null)
        ) {
          await api.patchGuideStep(versionId, s.number, {
            title: s.title,
            text: s.text,
            display_text: s.display_text,
            safety_hint: s.safety_hint,
            media_image: s.media_image ?? "",
          });
        }
      }
      const bInst = new Map(b.instances.map((i) => [i.id, i]));
      const next: Doc = clone(d);
      for (const inst of d.instances) {
        const o = bInst.get(inst.id);
        if (inst.id.startsWith("new_")) {
          const r = await api.addGuideInstance(versionId, inst.part_id, inst.placed_in_step);
          const j = next.instances.find((x) => x.id === inst.id);
          if (j) {
            j.id = r.id;
            j.transform = r.transform;
          }
          // arrows anchored to this fresh instance must point at the real
          // server id — remap in both the outgoing doc and the synced copy
          for (const a of d.annotations) if (a.part_id === inst.id) a.part_id = r.id;
          for (const a of next.annotations) if (a.part_id === inst.id) a.part_id = r.id;
          continue;
        }
        if (o && !instEq(inst, o)) {
          await api.patchGuideInstance(versionId, inst.id, {
            transform: inst.transform,
            placed_in_step: inst.placed_in_step,
            color: inst.color ?? "",
            ...(inst.opacity != null ? { opacity: inst.opacity } : {}),
          });
        }
      }
      const dIds = new Set(d.instances.map((i) => i.id));
      for (const o of b.instances) {
        if (!dIds.has(o.id)) await api.deleteGuideInstance(versionId, o.id);
      }
      // annotations: keyed diff — POST new (idx undefined), PATCH changed,
      // DELETE removed. Deletes must run descending by server index or the
      // indexes of later deletes drift.
      const bAnn = new Map(b.annotations.map((a) => [a.k, a]));
      for (const a of d.annotations) {
        const o = bAnn.get(a.k);
        if (a.idx === undefined) {
          const r = await api.addGuideAnnotation(versionId, annPayload(a));
          const j = next.annotations.find((x) => x.k === a.k);
          if (j) j.idx = r.index;
          continue;
        }
        if (o && !annEq(a, o)) {
          await api.patchGuideAnnotation(versionId, a.idx, {
            step: a.step,
            part_id: a.part_id,
            approach_axis: a.approach_axis,
            label: a.label,
            position: a.position,
            text: a.text,
            color: a.color ?? "",
            ...(a.size != null ? { size: a.size } : {}),
          });
        }
      }
      const dKeys = new Set(d.annotations.map((a) => a.k));
      const removals = b.annotations
        .filter((o) => !dKeys.has(o.k) && o.idx !== undefined)
        .map((o) => o.idx as number)
        .sort((x, y) => y - x); // descending — see above
      for (const idx of removals) {
        await api.deleteGuideAnnotation(versionId, idx);
      }
      // deletes renumber the server list — shift surviving idx or the
      // next flush would PATCH/DELETE the wrong annotations
      if (removals.length) {
        for (const a of next.annotations) {
          if (a.idx === undefined) continue;
          a.idx -= removals.filter((r) => r < (a.idx as number)).length;
        }
      }
      setDoc((cur) => (cur === d ? next : cur));
      setBase(next);
      setSaveState("saved");
    } catch (e) {
      setSaveState("error");
      flash(errMsg(e), true);
    }
  }, [versionId]);

  const scheduleFlush = useCallback(() => {
    setSaveState("dirty");
    if (flushTimer.current) clearTimeout(flushTimer.current);
    flushTimer.current = setTimeout(() => void flush(), 900);
  }, [flush]);

  const mutate = useCallback(
    (fn: (d: Doc) => void) => {
      setDoc((d) => {
        if (!d) return d;
        setHistory((h) => [...h.slice(-49), clone(d)]);
        setRedoStack([]);
        const n = clone(d);
        fn(n);
        return n;
      });
      scheduleFlush();
    },
    [scheduleFlush],
  );

  const undo = useCallback(() => {
    setHistory((h) => {
      if (!h.length) return h;
      setDoc((d) => {
        if (d) setRedoStack((r) => [...r, clone(d)]);
        return clone(h[h.length - 1]);
      });
      return h.slice(0, -1);
    });
    setSel(new Set());
    setPrimary(null);
    scheduleFlush();
    flash(t("guide_undo"));
  }, [scheduleFlush, t]);

  const redo = useCallback(() => {
    setRedoStack((r) => {
      if (!r.length) return r;
      setDoc((d) => {
        if (d) setHistory((h) => [...h, clone(d)]);
        return clone(r[r.length - 1]);
      });
      return r.slice(0, -1);
    });
    setSel(new Set());
    setPrimary(null);
    scheduleFlush();
    flash(t("guide_redo"));
  }, [scheduleFlush, t]);


  const steps = useMemo(
    () => [...(doc?.steps ?? [])].sort((a, b) => a.number - b.number),
    [doc],
  );
  const step = steps[Math.min(cur, Math.max(0, steps.length - 1))];
  const visible = useMemo(
    () => (doc?.instances ?? []).filter((i) => (i.placed_in_step ?? 0) <= (step?.number ?? 0)),
    [doc, step],
  );
  const selInst = primary ? doc?.instances.find((i) => i.id === primary) ?? null : null;

  const goto = useCallback(
    (idx: number) => {
      setCur(Math.max(0, Math.min(steps.length - 1, idx)));
      setSel(new Set());
      setPrimary(null);
      setAnnSel(null);
      setCtx(null);
      setPlaying(null);
      void flush(); // KIARA: autosave on step switch
    },
    [steps.length, flush],
  );

  const close = () => {
    if (flushTimer.current) clearTimeout(flushTimer.current);
    void flush().finally(onClose);
  };

  const select = (iid: string | null, multi: boolean) => {
    if (iid === null) {
      setSel(new Set());
      setPrimary(null);
      setAnnSel(null);
      return;
    }
    setAnnSel(null);
    if (multi) {
      setSel((s) => {
        const n = new Set(s);
        if (n.has(iid)) n.delete(iid);
        else n.add(iid);
        return n;
      });
      setPrimary(iid);
    } else {
      setSel(new Set([iid]));
      setPrimary(iid);
    }
  };

  const applyTransforms = (m: Record<string, number[][]>) =>
    mutate((d) => {
      for (const inst of d.instances) {
        if (m[inst.id]) inst.transform = m[inst.id];
      }
    });

  const updateInst = (iid: string, fn: (i: GuideInstance) => void) =>
    mutate((d) => {
      const i = d.instances.find((x) => x.id === iid);
      if (i) fn(i);
    });

  const updateSel = (fn: (i: GuideInstance) => void) =>
    mutate((d) => {
      for (const i of d.instances) if (sel.has(i.id)) fn(i);
    });

  const deleteInst = useCallback(
    (iid: string) => {
      mutate((d) => {
        d.instances = d.instances.filter((i) => i.id !== iid);
      });
      setSel((s) => {
        const n = new Set(s);
        n.delete(iid);
        return n;
      });
      if (primary === iid) setPrimary(null);
      setCtx(null);
    },
    [mutate, primary],
  );

  const insertPart = (pid: string) => {
    const iid = `new_${++tmpSeq.current}`;
    // spread estimate beside the visible union (server re-solves on flush)
    const vis = (doc?.instances ?? []).filter((i) => (i.placed_in_step ?? 0) <= (step?.number ?? 0));
    const mx = Math.max(0, ...vis.map((i) => decompose(i.transform).pos.x));
    const m = new THREE.Matrix4().makeTranslation(mx + 140, 0, 0);
    mutate((d) => {
      d.instances.push({
        id: iid,
        part_id: pid,
        placed_in_step: step?.number ?? 0,
        transform: rowsFromMat(m),
      });
    });
    setSel(new Set([iid]));
    setPrimary(iid);
  };

  // ---------------- annotations ----------------
  const selAnn = annSel
    ? (doc?.annotations.find((a) => a.k === annSel) ?? null)
    : null;

  const updateAnn = (k: string, fn: (a: EdAnn) => void) =>
    mutate((d) => {
      const a = d.annotations.find((x) => x.k === k);
      if (a) fn(a);
    });

  const deleteAnn = useCallback(
    (k: string) => {
      mutate((d) => {
        d.annotations = d.annotations.filter((a) => a.k !== k);
      });
      if (annSel === k) setAnnSel(null);
      setCtx(null);
    },
    [mutate, annSel],
  );

  const selectAnn = (k: string | null) => {
    setAnnSel(k);
    if (k) {
      setSel(new Set());
      setPrimary(null);
    }
  };

  const DIR_VEC: Record<(typeof DIR_PRESETS)[number], number[]> = {
    // mirrors annotate_tools._DIR_VECS — Z-up assembly frame
    up: [0, 0, 1], down: [0, 0, -1],
    left: [-1, 0, 0], right: [1, 0, 0],
    front: [0, -1, 0], back: [0, 1, 0],
  };

  const addArrow = (iid?: string) => {
    const target = iid ?? primary;
    if (!target) return;
    const k = `n_${++tmpAnnSeq.current}`;
    mutate((d) => {
      d.annotations.push({
        k,
        kind: "arrow",
        step: step?.number ?? 1,
        part_id: target,
        approach_axis: DIR_VEC[dirPreset],
        label: annLabel.trim() || undefined,
      });
    });
    selectAnn(k);
  };

  const addText = (iid?: string) => {
    const k = `n_${++tmpAnnSeq.current}`;
    const anchor = iid
      ? doc?.instances.find((i) => i.id === iid)
      : selInst;
    const p = anchor ? decompose(anchor.transform).pos : new THREE.Vector3();
    mutate((d) => {
      d.annotations.push({
        k,
        kind: "text",
        step: step?.number ?? 1,
        position: [p.x, p.y, p.z + 60],
        text: "",
      });
    });
    selectAnn(k);
  };

  /** gizmo commit: arrow → rotated approach_axis, text → moved position */
  const onAnnTransformed = (k: string, p: { axis?: number[]; position?: number[] }) =>
    updateAnn(k, (a) => {
      if (p.axis) a.approach_axis = p.axis;
      if (p.position) a.position = p.position;
    });

  const setStepMedia = (no: number, name: string | null) =>
    mutate((d) => {
      const s = d.steps.find((x) => x.number === no);
      if (s) s.media_image = name;
    });

  const refreshGuide = () =>
    api.getGuide(versionId).then((g) => setGuide(g)).catch(() => {});

  /** viewport snapshot → guide media → step.media_image */
  const snapshotStep = (no: number) => {
    const gl = glRef.current;
    if (!gl) return;
    gl.domElement.toBlob((b) => {
      if (!b) return;
      void api
        .uploadGuideMedia(versionId, b, `step-${no}-snapshot.png`)
        .then((r) => {
          setStepMedia(no, r.name);
          refreshGuide();
          flash(t("guide_snapshot_done"));
        })
        .catch((e) => flash(errMsg(e), true));
    }, "image/png");
  };

  /** server-side step ops renumber state — always settle the pending
   *  doc diff first or a debounced flush could PATCH shifted numbers */
  const runStepOp = useCallback(
    async (op: () => Promise<void>) => {
      if (flushTimer.current) clearTimeout(flushTimer.current);
      try {
        await flush();
        await op();
      } catch (e) {
        flash(errMsg(e), true);
      }
    },
    [flush],
  );

  const duplicateStep = () => {
    if (!step) return;
    const srcNo = step.number;
    const srcStep = step;
    const anns = doc?.annotations ?? [];
    void runStepOp(async () => {
      const r = await api.addGuideStep(versionId, { insert_at: srcNo + 1 });
      const newNo = r.step.number;
      await api.patchGuideStep(versionId, newNo, {
        title: srcStep.title,
        text: srcStep.text,
        display_text: srcStep.display_text,
        safety_hint: srcStep.safety_hint,
        media_image: srcStep.media_image ?? "",
      });
      for (const a of anns.filter((x) => x.step === srcNo)) {
        await api.addGuideAnnotation(versionId, {
          ...annPayload(a),
          step: newNo,
        });
      }
      reloadDoc();
      setCur(Math.min(srcNo, steps.length));
    });
  };

  const quickRotate = (axis: "x" | "y" | "z", deg: number) =>
    updateSel((i) => {
      const rot = new THREE.Matrix4().makeRotationAxis(
        new THREE.Vector3(axis === "x" ? 1 : 0, axis === "y" ? 1 : 0, axis === "z" ? 1 : 0),
        THREE.MathUtils.degToRad(deg),
      );
      i.transform = rowsFromMat(rot.multiply(matFromRows(i.transform)));
    });

  const alignHeight = () => {
    const z = selInst ? decompose(selInst.transform).pos.z : 0;
    updateSel((i) => {
      const d = decompose(i.transform);
      i.transform = rowsFromMat(
        new THREE.Matrix4().compose(
          new THREE.Vector3(d.pos.x, d.pos.y, z),
          d.quat,
          d.scl,
        ),
      );
    });
  };

  const flatten = () =>
    updateSel((i) => {
      const d = decompose(i.transform);
      const yawOnly = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, d.euler.y, 0));
      i.transform = rowsFromMat(new THREE.Matrix4().compose(d.pos, yawOnly, d.scl));
    });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
      if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) {
        e.preventDefault();
        redo();
      } else if (e.key === "Escape") {
        setCtx(null);
      } else if (e.key === "ArrowLeft") {
        goto(cur - 1);
      } else if (e.key === "ArrowRight") {
        goto(cur + 1);
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (primary) deleteInst(primary);
        else if (annSel) deleteAnn(annSel);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, goto, cur, primary, deleteInst, annSel, deleteAnn]);

  const scaleSel = (k: number) =>
    updateSel((i) => {
      const d = decompose(i.transform);
      i.transform = rowsFromMat(
        new THREE.Matrix4().compose(d.pos, d.quat, new THREE.Vector3(k, k, k)),
      );
    });

  if (loadErr)
    return (
      <Shell onClose={onClose}>
        <div className="flex h-full items-center justify-center text-sm text-red-400">
          {t("guide_load_failed", { err: loadErr })}
        </div>
      </Shell>
    );
  if (!doc || !guide)
    return (
      <Shell onClose={onClose}>
        <div className="flex h-full items-center justify-center">
          <Spinner />
        </div>
      </Shell>
    );

  return (
    <Shell onClose={close} wide>
      {/* ——— toolbar ——— */}
      <div className="flex items-center gap-2 border-b border-edge px-3 py-2">
        <IconBtn title={t("guide_undo")} disabled={!history.length} onClick={undo}>
          <Undo2 className="h-4 w-4" />
        </IconBtn>
        <IconBtn title={t("guide_redo")} disabled={!redoStack.length} onClick={redo}>
          <Redo2 className="h-4 w-4" />
        </IconBtn>
        <div className="mx-2 h-5 w-px bg-edge" />
        <div className="flex items-center gap-1">
          <Btn size="sm" disabled={cur === 0} onClick={() => goto(cur - 1)}>
            <ChevronLeft className="h-3.5 w-3.5" />
          </Btn>
          <span className="px-1 text-xs text-fg">
            {t("guide_step_of", { n: step?.number ?? 0, m: steps.length })}
          </span>
          <Btn size="sm" disabled={cur >= steps.length - 1} onClick={() => goto(cur + 1)}>
            <ChevronRight className="h-3.5 w-3.5" />
          </Btn>
          <IconBtn
            title={t("guide_step_add")}
            onClick={() => {
              const at = (step?.number ?? steps.length) + 1;
              void runStepOp(async () => {
                await api.addGuideStep(versionId, { insert_at: at });
                reloadDoc();
                setCur(Math.min(at - 1, steps.length));
              });
            }}
          >
            <Plus className="h-3.5 w-3.5" />
          </IconBtn>
          <IconBtn title={t("guide_step_dup")} disabled={!step} onClick={duplicateStep}>
            <CopyPlus className="h-3.5 w-3.5" />
          </IconBtn>
          <IconBtn
            title={t("guide_step_delete")}
            className="text-red-400"
            disabled={steps.length <= 1}
            onClick={() => {
              if (!step || steps.length <= 1) return;
              const no = step.number;
              void runStepOp(async () => {
                await api.deleteGuideStep(versionId, no);
                reloadDoc();
                setCur((c) => Math.max(0, c - 1));
              });
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </IconBtn>
        </div>
        <div className="mx-2 h-5 w-px bg-edge" />
        <span className="truncate text-sm font-medium text-fg">{name}</span>
        <span
          className={
            "rounded px-1.5 py-0.5 text-[10px] " +
            (saveState === "saved"
              ? "bg-emerald-500/15 text-emerald-400"
              : saveState === "saving"
                ? "bg-amber-500/15 text-amber-400"
                : saveState === "error"
                  ? "bg-red-500/15 text-red-400"
                  : "bg-panel2 text-muted")
          }
        >
          {t(`guide_save_${saveState}`)}
        </span>
        <div className="flex-1" />
        {feedback && (
          <span className={"text-[11px] " + (feedback.err ? "text-red-400" : "text-emerald-400")}>
            {feedback.msg}
          </span>
        )}
        <IconBtn title={t("close")} onClick={close}>
          <X className="h-4 w-4" />
        </IconBtn>
      </div>

      {/* ——— 3-pane body ——— */}
      <div className="flex min-h-0 flex-1" onClick={() => setCtx(null)}>
        {/* left sidebar — Schritte + Asset-Bibliothek */}
        <aside className="flex w-56 shrink-0 flex-col gap-3 overflow-y-auto border-r border-edge p-3">
          <section>
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {t("guide_annotations")}
            </h2>
            <div className="flex flex-col gap-1.5">
              <div className="flex gap-1">
                <select
                  className="input min-w-0 flex-1 py-1 text-[11px]"
                  value={dirPreset}
                  onChange={(e) => setDirPreset(e.target.value as (typeof DIR_PRESETS)[number])}
                  title={t("guide_ann_dir")}
                >
                  {DIR_PRESETS.map((d) => (
                    <option key={d} value={d}>{t(`guide_dir_${d}`)}</option>
                  ))}
                </select>
                <IconBtn
                  title={primary ? t("guide_arrow_add") : t("guide_arrow_add_hint")}
                  disabled={!primary}
                  onClick={() => addArrow()}
                >
                  <ArrowUpFromDot className="h-4 w-4" />
                </IconBtn>
                <IconBtn title={t("guide_text_add")} onClick={() => addText()}>
                  <Type className="h-4 w-4" />
                </IconBtn>
              </div>
              <input
                className="input py-1 text-[11px]"
                value={annLabel}
                placeholder={t("guide_ann_label_ph")}
                onChange={(e) => setAnnLabel(e.target.value)}
              />
              {selAnn && (
                <div className="flex flex-col gap-1 rounded border border-edge2 bg-panel2/50 p-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase text-faint">
                      {selAnn.kind === "arrow" ? t("guide_ann_arrow") : t("guide_ann_text")}
                    </span>
                    <button
                      className="text-faint hover:text-red-400"
                      title={t("guide_delete")}
                      onClick={() => deleteAnn(selAnn.k)}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                  {selAnn.kind === "arrow" ? (
                    <input
                      className="input py-0.5 text-[11px]"
                      value={selAnn.label ?? ""}
                      placeholder={t("guide_ann_label_ph")}
                      onChange={(e) => updateAnn(selAnn.k, (a) => { a.label = e.target.value; })}
                    />
                  ) : (
                    <textarea
                      className="input w-full text-[11px]"
                      rows={2}
                      value={selAnn.text ?? ""}
                      placeholder={t("guide_ann_text_ph")}
                      onChange={(e) => updateAnn(selAnn.k, (a) => { a.text = e.target.value; })}
                    />
                  )}
                  <div className="grid grid-cols-6 gap-1">
                    {PALETTE.map((c) => (
                      <button
                        key={c}
                        className={
                          "h-4 w-4 rounded-full border " +
                          (selAnn.color === c ? "border-fg ring-1 ring-fg" : "border-edge2")
                        }
                        style={{ backgroundColor: c }}
                        onClick={() => updateAnn(selAnn.k, (a) => { a.color = c; })}
                      />
                    ))}
                  </div>
                  <label className="flex items-center gap-1.5 text-[10px] text-faint">
                    {t("guide_ann_size")}
                    <input
                      type="range" min={0.5} max={3} step={0.1}
                      value={selAnn.size ?? 1}
                      className="min-w-0 flex-1"
                      onChange={(e) => updateAnn(selAnn.k, (a) => { a.size = Number(e.target.value); })}
                    />
                  </label>
                </div>
              )}
            </div>
          </section>

          <section className="min-h-0 flex-1">
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {t("guide_assets")}
            </h2>
            <div className="mb-2 flex gap-1">
              {(["parts", "media"] as const).map((tab) => (
                <button
                  key={tab}
                  className={
                    "flex-1 rounded px-2 py-1 text-[11px] " +
                    (assetTab === tab ? "bg-brand-500/20 text-brand-300" : "bg-panel2 text-muted hover:text-fg")
                  }
                  onClick={() => setAssetTab(tab)}
                >
                  {t(tab === "parts" ? "guide_assets_parts" : "guide_assets_media")}
                </button>
              ))}
            </div>
            {assetTab === "parts" ? (
              <div className="flex flex-col gap-1">
                {guide.parts.map((p) => (
                  <button
                    key={p.id}
                    className="flex items-center justify-between rounded border border-edge2 bg-panel2 px-2 py-1.5 text-left text-[11px] text-fg hover:border-brand-500"
                    onClick={() => insertPart(p.id)}
                    title={t("guide_insert_hint")}
                  >
                    <span className="truncate">{p.name}</span>
                    <Plus className="h-3 w-3 shrink-0 text-faint" />
                  </button>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-1.5">
                {guide.media.map((m) => (
                  <button
                    key={m}
                    className={
                      "overflow-hidden rounded border " +
                      (m === step?.media_image
                        ? "border-brand-500 ring-1 ring-brand-500"
                        : "border-edge2 hover:border-brand-500")
                    }
                    title={m}
                    onClick={() =>
                      mutate((d) => {
                        const s = d.steps.find((x) => x.number === step?.number);
                        if (s) s.media_image = s.media_image === m ? null : m;
                      })
                    }
                  >
                    <SafeImg
                      src={mediaUrl(versionId, m)}
                      alt={m}
                      className="h-14 w-full object-cover"
                      fallback={<div className="h-14 bg-panel2" />}
                    />
                  </button>
                ))}
              </div>
            )}
          </section>
        </aside>

        {/* centre canvas */}
        <div className="relative min-w-0 flex-1">
          <Scene
            versionId={versionId}
            parts={guide.parts}
            instances={doc.instances}
            stepNo={step?.number ?? 0}
            sel={sel}
            primary={primary}
            mode={mode}
            gizmoDrag={() => {}}
            onSelect={select}
            onContext={(iid, x, y) => setCtx({ iid, x, y, sub: "main" })}
            onMiss={() => select(null, false)}
            onTransformedMany={applyTransforms}
            focus={focus}
            anns={doc.annotations}
            selAnn={selAnn}
            onAnnSelect={selectAnn}
            onAnnTransformed={onAnnTransformed}
            isolate={isolateStep}
            playing={playing}
            glRef={glRef}
            worldFrame={guide.world_frame}
          />
          {/* canvas overlay — preview transport, isolate, snapshot */}
          <div className="absolute left-2 top-2 flex flex-col gap-1">
            <IconBtn
              title={playing ? t("guide_anim_stop") : t("guide_anim_play")}
              className="bg-panel/80 backdrop-blur"
              onClick={() => setPlaying(playing ? null : { t0: performance.now() })}
            >
              {playing ? <Square className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </IconBtn>
            <IconBtn
              title={t("guide_isolate_step")}
              className={
                "bg-panel/80 backdrop-blur " +
                (isolateStep ? "text-brand-400 ring-1 ring-brand-500" : "")
              }
              onClick={() => setIsolateStep((v) => !v)}
            >
              <Eye className="h-4 w-4" />
            </IconBtn>
            <IconBtn
              title={t("guide_snapshot")}
              className="bg-panel/80 backdrop-blur"
              onClick={() => step && snapshotStep(step.number)}
            >
              <Camera className="h-4 w-4" />
            </IconBtn>
          </div>
          {ctx && (
            <ContextMenu
              x={ctx.x}
              y={ctx.y}
              sub={ctx.sub}
              inst={doc.instances.find((i) => i.id === ctx.iid) ?? null}
              setSub={(s) => setCtx({ ...ctx, sub: s })}
              onColor={(c) => {
                updateInst(ctx.iid, (i) => {
                  i.color = c;
                });
                setCtx(null);
              }}
              onOpacity={(o) => updateInst(ctx.iid, (i) => {
                i.opacity = o;
              })}
              onFocus={() => {
                setFocus({ iid: ctx.iid, v: Date.now() });
                setCtx(null);
              }}
              onReset={() => {
                updateInst(ctx.iid, (i) => {
                  const d = decompose(i.transform);
                  i.transform = rowsFromMat(
                    new THREE.Matrix4().compose(
                      d.pos,
                      new THREE.Quaternion(),
                      new THREE.Vector3(1, 1, 1),
                    ),
                  );
                });
                setCtx(null);
              }}
              onArrow={() => {
                addArrow(ctx.iid);
                setCtx(null);
              }}
              onText={() => {
                addText(ctx.iid);
                setCtx(null);
              }}
              onDelete={() => deleteInst(ctx.iid)}
            />
          )}
        </div>

        {/* right rail — collapsible: Steps editor (all steps) | Objects */}
        <div className="flex w-9 shrink-0 flex-col items-center border-l border-edge py-1.5">
          <IconBtn
            title={railOpen ? t("guide_rail_close") : t("guide_rail_open")}
            onClick={() => setRailOpen((v) => !v)}
          >
            {railOpen ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
          </IconBtn>
        </div>
        {railOpen && (
          <aside className="flex w-80 shrink-0 flex-col border-l border-edge xl:w-96">
            <div className="flex gap-1 border-b border-edge p-1.5">
              {(["steps", "objects"] as const).map((tab) => (
                <button
                  key={tab}
                  className={
                    "flex-1 rounded px-2 py-1 text-[11px] " +
                    (railTab === tab
                      ? "bg-brand-500/20 text-brand-300"
                      : "bg-panel2 text-muted hover:text-fg")
                  }
                  onClick={() => setRailTab(tab)}
                >
                  {t(tab === "steps" ? "guide_steps" : "guide_objects")}
                </button>
              ))}
            </div>
            {railTab === "steps" ? (
              <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-2">
                {steps.map((s, idx) => (
                  <StepCard
                    key={s.number}
                    step={s}
                    active={s.number === step?.number}
                    annCount={doc.annotations.filter((a) => a.step === s.number)}
                    onFocus={() => goto(idx)}
                    onField={(k, v) =>
                      mutate((d) => {
                        const st = d.steps.find((x) => x.number === s.number);
                        if (st) (st as unknown as Record<string, unknown>)[k] = v;
                      })
                    }
                    versionId={versionId}
                    guideMedia={guide.media}
                    mediaOpen={mediaFor === s.number}
                    onToggleMedia={() =>
                      setMediaFor((m) => (m === s.number ? null : s.number))
                    }
                    onMedia={(name) => {
                      setStepMedia(s.number, name);
                      setMediaFor(null);
                    }}
                    onSnapshot={() => snapshotStep(s.number)}
                    onUpload={(blob, fn) =>
                      void api
                        .uploadGuideMedia(versionId, blob, fn)
                        .then((r) => {
                          setStepMedia(s.number, r.name);
                          setMediaFor(null);
                          refreshGuide();
                        })
                        .catch((e) => flash(errMsg(e), true))
                    }
                    onAsset={(aid) =>
                      void api
                        .guideMediaFromAsset(versionId, aid)
                        .then((r) => {
                          setStepMedia(s.number, r.name);
                          setMediaFor(null);
                          refreshGuide();
                        })
                        .catch((e) => flash(errMsg(e), true))
                    }
                    projectId={projectId}
                    canDelete={steps.length > 1}
                    onDelete={() =>
                      void runStepOp(async () => {
                        await api.deleteGuideStep(versionId, s.number);
                        reloadDoc();
                        setCur((c) => Math.max(0, c - (idx <= c ? 1 : 0)));
                      })
                    }
                  />
                ))}
              </div>
            ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
          <section>
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {t("guide_objects")}
            </h2>
            <div className="flex flex-col gap-0.5">
              {visible.map((i) => {
                const pname = guide.parts.find((p) => p.id === i.part_id)?.name ?? i.part_id;
                const ord =
                  visible.filter((x) => x.part_id === i.part_id).indexOf(i) + 1;
                return (
                  <div
                    key={i.id}
                    className={
                      "group flex cursor-pointer items-center justify-between rounded px-2 py-1 text-[11px] " +
                      (sel.has(i.id) ? "bg-brand-500/20 text-brand-200" : "text-muted hover:bg-panel2 hover:text-fg")
                    }
                    title={`${i.id} · ${pname}`}
                    onClick={(e) => select(i.id, e.ctrlKey || e.metaKey || e.shiftKey)}
                  >
                    <span className="truncate">
                      {pname}
                      <span className="text-faint"> · {ord}</span>
                    </span>
                    <button
                      className="hidden text-faint hover:text-red-400 group-hover:block"
                      title={t("guide_delete")}
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteInst(i.id);
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                );
              })}
              {!visible.length && (
                <p className="text-[11px] text-faint">{t("guide_no_objects")}</p>
              )}
            </div>
          </section>

          {sel.size > 0 && (
            <section>
              <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
                {sel.size > 1
                  ? t("guide_sel_edit_multi", { n: sel.size })
                  : t("guide_sel_edit")}
              </h2>
              <div className="mb-2 flex gap-1">
                {(
                  [
                    ["translate", Move3d, "guide_move"],
                    ["rotate", Rotate3d, "guide_rot"],
                    ["scale", Maximize2, "guide_scale"],
                  ] as const
                ).map(([m, Icon, k]) => (
                  <IconBtn
                    key={m}
                    title={t(k)}
                    onClick={() => setMode(m)}
                    className={mode === m ? "text-brand-400 ring-1 ring-brand-500" : ""}
                  >
                    <Icon className="h-4 w-4" />
                  </IconBtn>
                ))}
              </div>
              {sel.size === 1 && selInst && (
                <NumericTransform
                  inst={selInst}
                  onApply={(m) => updateInst(selInst.id, (i) => (i.transform = m))}
                />
              )}
              {sel.size === 1 && selInst && (
                <div className="mb-2">
                  <div className="mb-1 text-[10px] uppercase text-faint">
                    {t("guide_scale_uniform")}
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={4}
                    step={0.1}
                    defaultValue={decompose(selInst.transform).scl.x}
                    className="w-full"
                    onChange={(e) => scaleSel(Number(e.target.value))}
                  />
                </div>
              )}
              <div className="mb-1 text-[10px] uppercase text-faint">
                {t("guide_quick_rot")}
              </div>
              <div className="mb-2 grid grid-cols-3 gap-1">
                {(["x", "y", "z"] as const).flatMap((a) =>
                  [90, -90].map((d) => (
                    <Btn key={`${a}${d}`} size="sm" onClick={() => quickRotate(a, d)}>
                      {a.toUpperCase()}
                      {d > 0 ? "+" : "-"}
                    </Btn>
                  )),
                )}
              </div>
              <Btn size="sm" className="mb-1 w-full" onClick={alignHeight}>
                <ArrowDownToLine className="h-3 w-3" /> {t("guide_align_height")}
              </Btn>
              <Btn size="sm" className="w-full" onClick={flatten}>
                {t("guide_flatten")}
              </Btn>
            </section>
          )}

          <section>
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {t("guide_annotations")}
            </h2>
            <div className="flex flex-col gap-0.5">
              {doc.annotations
                .filter((a) => a.step === step?.number)
                .map((a) => (
                  <div
                    key={a.k}
                    className={
                      "group flex cursor-pointer items-center justify-between rounded px-2 py-1 text-[11px] " +
                      (annSel === a.k
                        ? "bg-brand-500/20 text-brand-200"
                        : "text-muted hover:bg-panel2 hover:text-fg")
                    }
                    onClick={() => selectAnn(annSel === a.k ? null : a.k)}
                  >
                    <span className="flex min-w-0 items-center gap-1.5">
                      {a.kind === "arrow" ? (
                        <ArrowUpFromDot className="h-3 w-3 shrink-0 text-amber-400" />
                      ) : (
                        <Type className="h-3 w-3 shrink-0 text-teal-300" />
                      )}
                      <span className="truncate">
                        {a.kind === "arrow"
                          ? a.label || a.part_id || t("guide_ann_arrow")
                          : a.text || t("guide_ann_text")}
                      </span>
                    </span>
                    <button
                      className="hidden text-faint hover:text-red-400 group-hover:block"
                      title={t("guide_delete")}
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteAnn(a.k);
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              {!doc.annotations.filter((a) => a.step === step?.number).length && (
                <p className="text-[11px] text-faint">{t("guide_ann_none")}</p>
              )}
            </div>
          </section>
          </div>
            )}
          </aside>
        )}
      </div>
    </Shell>
  );
}

// ======================== pieces ========================

/** One step inside the right rail's scrollable steps editor: title/text/
 *  display/safety fields, media image swap (guide media · project assets
 *  · upload · viewport snapshot), annotation counts, delete. */
function StepCard({
  step,
  active,
  annCount,
  onFocus,
  onField,
  versionId,
  guideMedia,
  mediaOpen,
  onToggleMedia,
  onMedia,
  onSnapshot,
  onUpload,
  onAsset,
  projectId,
  canDelete,
  onDelete,
}: {
  step: GuideStepDetail;
  active: boolean;
  annCount: EdAnn[];
  onFocus: () => void;
  onField: (k: keyof GuideStepDetail, v: string) => void;
  versionId: string;
  guideMedia: string[];
  mediaOpen: boolean;
  onToggleMedia: () => void;
  onMedia: (name: string | null) => void;
  onSnapshot: () => void;
  onUpload: (blob: Blob, filename: string) => void;
  onAsset: (assetId: string) => void;
  projectId?: string;
  canDelete: boolean;
  onDelete: () => void;
}) {
  const { t } = useTranslation("version");
  const ref = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest" });
  }, [active]);
  const arrows = annCount.filter((a) => a.kind === "arrow").length;
  const texts = annCount.filter((a) => a.kind === "text").length;
  return (
    <div
      ref={ref}
      className={
        "rounded-lg border p-2 " +
        (active ? "border-brand-600 bg-panel2/60 ring-1 ring-brand-700/50" : "border-edge bg-panel")
      }
    >
      <div className="flex items-center gap-1.5">
        <button
          className={
            "shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold " +
            (active ? "bg-brand-500/20 text-brand-300" : "bg-panel2 text-faint")
          }
          title={t("guide_step_focus")}
          onClick={onFocus}
        >
          S{step.number}
        </button>
        <input
          className="input min-w-0 flex-1 py-0.5 text-[11px]"
          value={step.title}
          placeholder={t("guide_title_ph")}
          onFocus={onFocus}
          onChange={(e) => onField("title", e.target.value)}
        />
        <button
          className="shrink-0 text-faint hover:text-red-400 disabled:opacity-30"
          title={t("guide_step_delete")}
          disabled={!canDelete}
          onClick={onDelete}
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
      <textarea
        className="input mt-1.5 w-full text-[11px]"
        rows={3}
        value={step.text}
        placeholder={t("guide_text_ph")}
        onFocus={onFocus}
        onChange={(e) => onField("text", e.target.value)}
      />
      <input
        className="input mt-1 w-full py-0.5 text-[11px]"
        value={step.display_text}
        placeholder={t("guide_display_ph")}
        onFocus={onFocus}
        onChange={(e) => onField("display_text", e.target.value)}
      />
      <input
        className="input mt-1 w-full py-0.5 text-[11px]"
        value={step.safety_hint}
        placeholder={t("guide_safety_ph")}
        onFocus={onFocus}
        onChange={(e) => onField("safety_hint", e.target.value)}
      />

      {/* step photo — swap from guide media / project assets / upload /
          viewport snapshot */}
      <div className="mt-1.5 flex items-center gap-1.5">
        <div className="h-11 w-14 shrink-0 overflow-hidden rounded border border-edge2 bg-panel2">
          {step.media_image ? (
            <SafeImg
              src={mediaUrl(versionId, step.media_image)}
              alt={step.media_image}
              className="h-full w-full object-cover"
              fallback={<div className="h-full w-full" />}
            />
          ) : null}
        </div>
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
          <Btn size="sm" variant="ghost" onClick={onToggleMedia}>
            {t("guide_media_swap")}
          </Btn>
          <IconBtn title={t("guide_snapshot")} onClick={onSnapshot}>
            <Camera className="h-3.5 w-3.5" />
          </IconBtn>
          {step.media_image && (
            <IconBtn title={t("guide_media_clear")} onClick={() => onMedia(null)}>
              <X className="h-3.5 w-3.5" />
            </IconBtn>
          )}
          {(arrows > 0 || texts > 0) && (
            <span className="ml-auto shrink-0 text-[10px] text-faint">
              {arrows > 0 && `➤ ${arrows}`}
              {arrows > 0 && texts > 0 && " · "}
              {texts > 0 && `▤ ${texts}`}
            </span>
          )}
        </div>
      </div>

      {mediaOpen && (
        <MediaPicker
          versionId={versionId}
          guideMedia={guideMedia}
          current={step.media_image}
          onPick={onMedia}
          onUpload={onUpload}
          onAsset={onAsset}
          projectId={projectId}
          fileRef={fileRef}
        />
      )}
    </div>
  );
}

/** Popover body for step-photo swap: guide-extracted media, project
 *  image assets (copied in), or a fresh upload. */
function MediaPicker({
  versionId,
  guideMedia,
  current,
  onPick,
  onUpload,
  onAsset,
  projectId,
  fileRef,
}: {
  versionId: string;
  guideMedia: string[];
  current: string | null | undefined;
  onPick: (name: string) => void;
  onUpload: (blob: Blob, filename: string) => void;
  onAsset: (assetId: string) => void;
  projectId?: string;
  fileRef: React.RefObject<HTMLInputElement>;
}) {
  const { t } = useTranslation("version");
  const [assets, setAssets] = useState<Asset[] | null>(null);
  useEffect(() => {
    if (!projectId || assets !== null) return;
    api
      .listAssets(projectId)
      .then((a) =>
        setAssets(a.filter((x) => x.kind === "image" || x.kind === "generated_image")),
      )
      .catch(() => setAssets([]));
  }, [projectId, assets]);
  return (
    <div className="mt-1.5 space-y-2 rounded border border-edge2 bg-panel2/40 p-1.5">
      {guideMedia.length > 0 && (
        <div className="grid grid-cols-4 gap-1">
          {guideMedia.map((m) => (
            <button
              key={m}
              className={
                "overflow-hidden rounded border " +
                (m === current ? "border-brand-500 ring-1 ring-brand-500" : "border-edge2 hover:border-brand-500")
              }
              title={m}
              onClick={() => onPick(m)}
            >
              <SafeImg
                src={mediaUrl(versionId, m)}
                alt={m}
                className="h-12 w-full object-cover"
                fallback={<div className="h-12 bg-panel2" />}
              />
            </button>
          ))}
        </div>
      )}
      {projectId && assets && assets.length > 0 && (
        <>
          <div className="text-[10px] uppercase tracking-wider text-faint">
            {t("guide_media_project")}
          </div>
          <div className="grid grid-cols-4 gap-1">
            {assets.map((a) => (
              <button
                key={a.id}
                className="overflow-hidden rounded border border-edge2 hover:border-brand-500"
                title={a.filename}
                onClick={() => onAsset(a.id)}
              >
                <SafeImg
                  src={`/api/assets/${a.id}/file`}
                  alt={a.filename}
                  className="h-12 w-full object-cover"
                  fallback={<div className="h-12 bg-panel2" />}
                />
              </button>
            ))}
          </div>
        </>
      )}
      <div>
        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onUpload(f, f.name);
            e.target.value = "";
          }}
        />
        <Btn size="sm" variant="ghost" className="w-full" onClick={() => fileRef.current?.click()}>
          {t("guide_media_upload")}
        </Btn>
      </div>
    </div>
  );
}

function Shell({
  children,
  onClose,
  wide,
}: {
  children: React.ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  // portal at document.body: `fixed` is viewport-relative only outside any
  // transformed ancestor — inside the page layout the content column offsets
  // the overlay and its left edge lands behind the nav bar
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={
          "flex flex-col overflow-hidden rounded-xl border border-edge bg-panel shadow-2xl " +
          (wide ? "h-[94vh] w-[97vw]" : "h-[40vh] w-[40vw] min-w-[360px]")
        }
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}

function ContextMenu({
  x,
  y,
  sub,
  inst,
  setSub,
  onColor,
  onOpacity,
  onFocus,
  onReset,
  onArrow,
  onText,
  onDelete,
}: {
  x: number;
  y: number;
  sub: "main" | "color" | "alpha";
  inst: GuideInstance | null;
  setSub: (s: "main" | "color" | "alpha") => void;
  onColor: (c: string) => void;
  onOpacity: (o: number) => void;
  onFocus: () => void;
  onReset: () => void;
  onArrow: () => void;
  onText: () => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation("version");
  if (!inst) return null;
  const item = "w-full rounded px-3 py-1.5 text-left text-[12px] text-fg hover:bg-panel2";
  return (
    <div
      className="fixed z-[60] w-52 rounded-lg border border-edge bg-panel py-1 shadow-xl"
      style={{ left: x, top: y }}
      onClick={(e) => e.stopPropagation()}
    >
      {sub === "main" && (
        <>
          <button className={item} onClick={() => setSub("color")}>
            {t("guide_color")}
          </button>
          <button className={item} onClick={() => setSub("alpha")}>
            {t("guide_opacity")}
          </button>
          <div className="my-1 border-t border-edge" />
          <button className={item} onClick={onFocus}>
            <Crosshair className="mr-1.5 inline h-3 w-3" />
            {t("guide_focus")}
          </button>
          <button className={item} onClick={onReset}>
            {t("guide_reset_transform")}
          </button>
          <div className="my-1 border-t border-edge" />
          <button className={item} onClick={onArrow}>
            <ArrowUpFromDot className="mr-1.5 inline h-3 w-3" />
            {t("guide_arrow_add")}
          </button>
          <button className={item} onClick={onText}>
            <Type className="mr-1.5 inline h-3 w-3" />
            {t("guide_text_add")}
          </button>
          <div className="my-1 border-t border-edge" />
          <button className={item + " text-red-400"} onClick={onDelete}>
            <Trash2 className="mr-1.5 inline h-3 w-3" />
            {t("guide_delete")}
          </button>
        </>
      )}
      {sub === "color" && (
        <div className="px-2 py-1">
          <button className={item} onClick={() => setSub("main")}>
            ‹ {t("back")}
          </button>
          <div className="grid grid-cols-6 gap-1.5 p-1">
            {PALETTE.map((c) => (
              <button
                key={c}
                className="h-6 w-6 rounded-full border border-edge2 hover:scale-110"
                style={{ backgroundColor: c }}
                onClick={() => onColor(c)}
              />
            ))}
          </div>
        </div>
      )}
      {sub === "alpha" && (
        <div className="px-3 py-2">
          <button className={item} onClick={() => setSub("main")}>
            ‹ {t("back")}
          </button>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            defaultValue={inst.opacity ?? 1}
            className="w-full"
            onChange={(e) => onOpacity(Number(e.target.value))}
          />
        </div>
      )}
    </div>
  );
}

function NumericTransform({
  inst,
  onApply,
}: {
  inst: GuideInstance;
  onApply: (m: number[][]) => void;
}) {
  const { t } = useTranslation("version");
  const d = useMemo(() => decompose(inst.transform), [inst.transform]);
  const [f, setF] = useState({
    px: +d.pos.x.toFixed(2),
    py: +d.pos.y.toFixed(2),
    pz: +d.pos.z.toFixed(2),
    rx: +THREE.MathUtils.radToDeg(d.euler.x).toFixed(1),
    ry: +THREE.MathUtils.radToDeg(d.euler.y).toFixed(1),
    rz: +THREE.MathUtils.radToDeg(d.euler.z).toFixed(1),
  });
  const lastId = useRef(inst.id);
  useEffect(() => {
    if (lastId.current === inst.id) return;
    lastId.current = inst.id;
    const dd = decompose(inst.transform);
    setF({
      px: +dd.pos.x.toFixed(2),
      py: +dd.pos.y.toFixed(2),
      pz: +dd.pos.z.toFixed(2),
      rx: +THREE.MathUtils.radToDeg(dd.euler.x).toFixed(1),
      ry: +THREE.MathUtils.radToDeg(dd.euler.y).toFixed(1),
      rz: +THREE.MathUtils.radToDeg(dd.euler.z).toFixed(1),
    });
  }, [inst.id, inst.transform]);

  const num = (k: keyof typeof f) => (
    <label key={k} className="flex items-center gap-1 text-[10px] text-faint">
      <span className="w-3">{k[1].toUpperCase()}</span>
      <input
        type="number"
        step={k.startsWith("r") ? 5 : 1}
        className="input w-full py-0.5 text-[11px]"
        value={f[k]}
        onChange={(e) => setF({ ...f, [k]: Number(e.target.value) })}
        onBlur={apply}
        onKeyDown={(e) => e.key === "Enter" && apply()}
      />
    </label>
  );

  const apply = () => {
    const m = new THREE.Matrix4().compose(
      new THREE.Vector3(f.px, f.py, f.pz),
      new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          THREE.MathUtils.degToRad(f.rx),
          THREE.MathUtils.degToRad(f.ry),
          THREE.MathUtils.degToRad(f.rz),
        ),
      ),
      decompose(inst.transform).scl,
    );
    onApply(rowsFromMat(m));
  };

  return (
    <div className="mb-2 flex flex-col gap-1">
      <div className="text-[10px] uppercase text-faint">{t("guide_pos")}</div>
      {num("px")}
      {num("py")}
      {num("pz")}
      <div className="mt-1 text-[10px] uppercase text-faint">{t("guide_rot_deg")}</div>
      {num("rx")}
      {num("ry")}
      {num("rz")}
      <div className="mt-1 flex items-center justify-between text-[10px] text-faint">
        <span>{t("guide_step_n", { n: inst.placed_in_step })}</span>
        <span className="truncate pl-2">{inst.id}</span>
      </div>
    </div>
  );
}




