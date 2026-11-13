import {
  ArrowDownToLine,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Maximize2,
  Move3d,
  Plus,
  Redo2,
  Rotate3d,
  Trash2,
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
import * as THREE from "three";
import {
  GizmoHelper,
  GizmoViewport,
  Grid,
  Line,
  OrbitControls,
  TransformControls,
} from "@react-three/drei";
import { Canvas, useLoader } from "@react-three/fiber";
import { GLTFLoader, type OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { api, errMsg } from "../../lib/api";
import type {
  GuideDetail,
  GuideInstance,
  GuidePartInfo,
  GuideStepDetail,
} from "../../lib/types";
import SafeImg from "../SafeImg";
import { CameraFit } from "./GuideScene";
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

type Doc = { steps: GuideStepDetail[]; instances: GuideInstance[] };
type Mode = "translate" | "rotate" | "scale";

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
  innerRef,
  onSelect,
  onContext,
}: {
  scene: THREE.Group;
  inst: GuideInstance;
  selected: boolean;
  innerRef?: Ref<THREE.Group>;
  onSelect: (e: { stopPropagation: () => void; ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) => void;
  onContext: (iid: string, x: number, y: number) => void;
}) {
  const obj = useMemo(() => {
    const o = scene.clone(true);
    o.traverse((c) => {
      const m = c as THREE.Mesh;
      if (!m.isMesh || !m.material) return;
      const apply = (src: THREE.Material) => {
        const mat = (src as THREE.MeshStandardMaterial).clone();
        if (inst.color && "color" in mat)
          (mat as THREE.MeshStandardMaterial).color = new THREE.Color(inst.color);
        const op = inst.opacity ?? 1;
        if (op < 1) {
          mat.transparent = true;
          mat.opacity = op;
        }
        if (selected && "emissive" in mat) {
          (mat as THREE.MeshStandardMaterial).emissive = new THREE.Color("#2b7cff");
          (mat as THREE.MeshStandardMaterial).emissiveIntensity = 0.3;
        }
        return mat;
      };
      m.material = Array.isArray(m.material)
        ? m.material.map(apply)
        : apply(m.material);
    });
    return o;
  }, [scene, inst.color, inst.opacity, selected]);

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
}) {
  const meshParts = useMemo(() => parts.filter((p) => p.has_mesh), [parts]);
  const urls = useMemo(
    () => meshParts.map((p) => meshUrl(versionId, p.id)),
    [versionId, meshParts],
  );
  const key = urls.join("|");
  const [ctxLost, setCtxLost] = useState(0);
  return (
    <Canvas
      key={`${key}:${ctxLost}`}
      frameloop="demand"
      dpr={[1, 2]}
      camera={{ position: [220, 200, 260], fov: 40 }}
      onPointerMissed={onMiss}
      onCreated={({ gl }) => {
        gl.domElement.addEventListener("webglcontextlost", (e) => {
          e.preventDefault();
          setCtxLost((n) => n + 1);
        });
      }}
      gl={{ antialias: true, powerPreference: "high-performance", alpha: false, stencil: false }}
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
        instances={instances.filter((i) => (i.placed_in_step ?? 0) <= stepNo)}
        sel={sel}
        primary={primary}
        mode={mode}
        gizmoDrag={gizmoDrag}
        onSelect={onSelect}
        onContext={onContext}
        onTransformedMany={onTransformedMany}
        focus={focus}
      />
      <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
        <GizmoViewport axisColors={["#ff4d5a", "#00d084", "#2b7cff"]} labelColor="#ccc" />
      </GizmoHelper>
    </Canvas>
  );
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
  const pivotRef = useRef<THREE.Group>(null!);
  const [pivotObj, setPivotObj] = useState<THREE.Group | null>(null);
  const pivotStart = useRef(new THREE.Matrix4());
  const itemStart = useRef<Record<string, THREE.Matrix4>>({});
  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  const contentRef = useRef<THREE.Group>(null);
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
    const cam = controls.object as THREE.Camera;
    const dir = cam.position.clone().sub(controls.target).normalize();
    controls.target.copy(p);
    cam.position.copy(p.clone().add(dir.multiplyScalar(160)));
    controls.update();
  }, [focus, instances]);

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

  const showGizmo = liveSel.size > 0;

  return (
    <>
      <group ref={contentRef}>
        {instances.map((i) => (
          <PartObject
            key={i.id}
            scene={scenes[i.part_id] ?? new THREE.Group()}
            inst={i}
            selected={liveSel.has(i.id)}
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
        ))}
      </group>
      <CameraFit content={contentRef} fitKey={urls.length} />

      {showGizmo && (
        <>
          <group
            position={pivotPos}
            ref={(g) => {
              pivotRef.current = g as THREE.Group;
              setPivotObj(g);
            }}
          />
          {pivotObj && (
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
        </>
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
}: {
  versionId: string;
  name: string;
  onClose: () => void;
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
  const [saveState, setSaveState] = useState<"saved" | "dirty" | "saving" | "error">("saved");
  const [feedback, setFeedback] = useState<string | null>(null);
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
        const d: Doc = { steps: clone(g.steps), instances: clone(g.instance_list ?? []) };
        setDoc(d);
        setBase(clone(d));
      })
      .catch((e) => setLoadErr(errMsg(e)));
  }, [versionId]);

  useEffect(reloadDoc, [reloadDoc]);

  const flash = (msg: string) => {
    setFeedback(msg);
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
      setDoc((cur) => (cur === d ? next : cur));
      setBase(next);
      setSaveState("saved");
    } catch (e) {
      setSaveState("error");
      flash(errMsg(e));
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
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo]);

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

  const goto = (idx: number) => {
    setCur(Math.max(0, Math.min(steps.length - 1, idx)));
    setSel(new Set());
    setPrimary(null);
    setCtx(null);
    void flush(); // KIARA: autosave on step switch
  };

  const close = () => {
    if (flushTimer.current) clearTimeout(flushTimer.current);
    void flush().finally(onClose);
  };

  const setStepField = (k: keyof GuideStepDetail, v: string) =>
    mutate((d) => {
      const s = d.steps.find((x) => x.number === step?.number);
      if (s) (s as unknown as Record<string, unknown>)[k] = v;
    });

  const select = (iid: string | null, multi: boolean) => {
    if (iid === null) {
      setSel(new Set());
      setPrimary(null);
      return;
    }
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

  const deleteInst = (iid: string) => {
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
  };

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
        {feedback && <span className="text-[11px] text-emerald-400">{feedback}</span>}
        <IconBtn title={t("close")} onClick={close}>
          <X className="h-4 w-4" />
        </IconBtn>
      </div>

      {/* ——— 3-pane body ——— */}
      <div className="flex min-h-0 flex-1" onClick={() => setCtx(null)}>
        {/* left sidebar — Schritte + Asset-Bibliothek */}
        <aside className="flex w-60 shrink-0 flex-col gap-3 overflow-y-auto border-r border-edge p-3">
          <section>
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {t("guide_steps")}
            </h2>
            <div className="mb-2 flex items-center justify-between">
              <Btn size="sm" disabled={cur === 0} onClick={() => goto(cur - 1)}>
                <ChevronLeft className="h-3.5 w-3.5" />
              </Btn>
              <span className="text-xs text-fg">
                {t("guide_step_of", { n: step?.number ?? 0, m: steps.length })}
              </span>
              <Btn size="sm" disabled={cur >= steps.length - 1} onClick={() => goto(cur + 1)}>
                <ChevronRight className="h-3.5 w-3.5" />
              </Btn>
            </div>
            <div className="mb-2 flex gap-1">
              <IconBtn
                title={t("guide_step_add")}
                className="flex-1"
                onClick={() => {
                  const at = (step?.number ?? steps.length) + 1;
                  api
                    .addGuideStep(versionId, { insert_at: at })
                    .then(() => {
                      reloadDoc();
                      setCur(Math.min(at - 1, steps.length));
                    })
                    .catch((e) => setFeedback(errMsg(e)));
                }}
              >
                <Plus className="h-3.5 w-3.5" />
              </IconBtn>
              <IconBtn
                title={t("guide_step_delete")}
                className="flex-1 text-red-400"
                onClick={() => {
                  if (!step || steps.length <= 1) return;
                  api
                    .deleteGuideStep(versionId, step.number)
                    .then(() => {
                      reloadDoc();
                      setCur(Math.max(0, cur - 1));
                    })
                    .catch((e) => setFeedback(errMsg(e)));
                }}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </IconBtn>
            </div>
            {step && (
              <div className="flex flex-col gap-1.5">
                <input
                  className="input py-1 text-xs"
                  value={step.title}
                  placeholder={t("guide_title_ph")}
                  onChange={(e) => setStepField("title", e.target.value)}
                />
                <textarea
                  className="input w-full text-[11px]"
                  rows={7}
                  value={step.text}
                  placeholder={t("guide_text_ph")}
                  onChange={(e) => setStepField("text", e.target.value)}
                />
                <input
                  className="input py-1 text-[11px]"
                  value={step.display_text}
                  placeholder={t("guide_display_ph")}
                  onChange={(e) => setStepField("display_text", e.target.value)}
                />
                <input
                  className="input py-1 text-[11px]"
                  value={step.safety_hint}
                  placeholder={t("guide_safety_ph")}
                  onChange={(e) => setStepField("safety_hint", e.target.value)}
                />
              </div>
            )}
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
          />
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
              onDelete={() => deleteInst(ctx.iid)}
            />
          )}
        </div>

        {/* right inspector — Objekte in Schritt + Auswahl bearbeiten */}
        <aside className="flex w-64 shrink-0 flex-col gap-3 overflow-y-auto border-l border-edge p-3">
          <section>
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {t("guide_objects")}
            </h2>
            <div className="flex flex-col gap-0.5">
              {visible.map((i) => {
                const pname = guide.parts.find((p) => p.id === i.part_id)?.name ?? i.part_id;
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
                    <span className="truncate">{i.id}</span>
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
        </aside>
      </div>
    </Shell>
  );
}

// ======================== pieces ========================

function Shell({
  children,
  onClose,
  wide,
}: {
  children: React.ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  return (
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
    </div>
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




