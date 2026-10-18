import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  Check,
  CircleDot,
  Copy,
  Eye,
  Layers,
  MousePointer2,
  Ruler,
  Trash2,
  Undo2,
  Wand2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import * as THREE from "three";
import GizmoControls from "../components/editor/GizmoControls";
import ModelCanvas, { type PickInfo, type ViewPreset } from "../components/editor/ModelCanvas";
import {
  Btn,
  Card,
  Chip,
  Crumbs,
  Field,
  IconBtn,
  NumInput,
  SectionTitle,
  Skeleton,
  inputCls,
  useToast,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import {
  readStoredAnchors,
  readUiEditor,
  recordedMeshOps,
  savedOps,
} from "../lib/editorState";
import {
  isRigidOp,
  meshAabbEvaluator,
  recordedPrefixLen,
  rigidOpsMatrix,
  type AabbEval,
} from "../lib/opsPreview";
import type { MeshOp, MeshPreviewResult } from "../lib/types";
import { useTitle } from "../lib/useTitle";

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

type Axis = "x" | "y" | "z";
const AXES: Axis[] = ["x", "y", "z"];
const UP_DIRS = ["+z", "-z", "+x", "-x", "+y", "-y"] as const;

type PickMode =
  | "measure"
  | "bool_center"
  | "slice_at"
  | "crop_min"
  | "crop_max"
  | "drill"
  | "clip_z"
  | null;

type Tool = "boolean" | "slice" | "crop" | "drill" | "clip" | "measure" | null;

const fmt = (n: number) => (Math.abs(n) < 0.01 ? "0" : n.toFixed(2));

function opLabel(o: MeshOp): string {
  const a = o.args as Record<string, unknown>;
  switch (o.op) {
    case "set_up":
      return `orient up=${a.up}`;
    case "rotate":
      return `rotate ${a.axis} ${a.deg}°`;
    case "mirror":
      return `mirror ${a.axis}`;
    case "scale":
      return a.factor != null
        ? `scale ×${a.factor}`
        : `scale (${fmt(Number(a.x ?? 1))}, ${fmt(Number(a.y ?? 1))}, ${fmt(Number(a.z ?? 1))})`;
    case "flatten_bottom":
      return `flatten bottom ${a.height_mm}mm`;
    case "slice":
      return `slice ${a.axis}@${fmt(Number(a.at_mm))} keep ${a.keep}`;
    case "crop":
      return "crop box";
    case "decimate":
      return `decimate → ${a.faces} faces`;
    case "smooth":
      return `smooth ${a.scheme ?? "laplacian"} ×${a.iterations ?? 5}`;
    case "smooth_preserve":
      return `smooth (preserve) ×${a.iterations ?? 10}`;
    case "clip_to_outline":
      return "clip to outline";
    case "keep_largest":
      return "keep largest body";
    case "drop_small":
      return `drop small bodies <${Number(a.min_volume_frac ?? 0.02) * 100}%`;
    case "repair":
      return "repair mesh";
    case "clip_height":
      return `clip height ≤ ${fmt(Number(a.z_max))}mm`;
    case "symmetrize":
      return `symmetrize ${a.axis} (${a.keep ?? "auto"})`;
    case "remesh":
      return `remesh voxel ${a.voxel_mm}mm`;
    case "drill_holes":
      return a.from_anchors
        ? `drill holes from anchors${(a.holes as unknown[] | undefined)?.length ? ` +${(a.holes as unknown[]).length}` : ""}`
        : `drill ${(a.holes as unknown[] | undefined)?.length ?? 0} hole(s)`;
    case "shell":
      return `shell ${a.thickness_mm}mm (${a.opening ?? "bottom"})`;
    case "boolean":
      return `${a.op === "union" ? "add" : "cut"} ${a.shape}`;
    case "organic":
      return `organic ${a.style ?? "gyroid"} (seed ${a.seed ?? 0})`;
    default:
      return o.op;
  }
}

/* ------------------------------------------------------------------ */
/* tool form primitives                                                */
/* ------------------------------------------------------------------ */

function ToolCard({
  title,
  hint,
  children,
  onAdd,
  addLabel = "+ add op",
  disabled,
  active,
  onToggle,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
  onAdd?: () => void;
  addLabel?: string;
  disabled?: boolean;
  /** overlay tool — highlighted while its gizmo/overlay is armed */
  active?: boolean;
  onToggle?: () => void;
}) {
  return (
    <Card className={`p-3 ${active ? "border-amber-700/60" : ""}`}>
      <div className="mb-1.5 flex items-center justify-between">
        <button
          className={`text-xs font-semibold ${active ? "text-amber-300" : "text-zinc-300"} ${onToggle ? "hover:text-zinc-100" : ""}`}
          onClick={onToggle}
          disabled={!onToggle}
          title={onToggle ? (active ? "deactivate tool overlay" : "activate tool overlay") : undefined}
        >
          {title}
        </button>
        {onAdd && (
          <Btn variant="ghost" className="px-2 py-0.5 text-xs" disabled={disabled} onClick={onAdd}>
            {addLabel}
          </Btn>
        )}
      </div>
      {hint && <div className="mb-2 text-[11px] text-zinc-500">{hint}</div>}
      {children}
    </Card>
  );
}

function AxisPicker({ value, onChange }: { value: Axis; onChange: (a: Axis) => void }) {
  return (
    <div className="flex gap-1">
      {AXES.map((a) => (
        <button
          key={a}
          onClick={() => onChange(a)}
          className={`flex-1 rounded border px-2 py-1 text-xs font-semibold uppercase ${
            value === a
              ? "border-brand-600 bg-brand-950/60 text-brand-300"
              : "border-zinc-700 text-zinc-500 hover:text-zinc-300"
          }`}
        >
          {a}
        </button>
      ))}
    </div>
  );
}

/** Generic per-op arg editor: numeric fields for scalars/vec3 args. */
function OpArgEditor({ op, onChange }: { op: MeshOp; onChange: (o: MeshOp) => void }) {
  const setArg = (k: string, v: unknown) => onChange({ ...op, args: { ...op.args, [k]: v } });
  const rows: React.ReactNode[] = [];
  Object.entries(op.args).forEach(([k, v], i) => {
    if (typeof v === "number") {
      rows.push(
        <label key={i} className="flex items-center gap-1">
          <span className="w-20 truncate text-[10px] text-zinc-500" title={k}>{k}</span>
          <NumInput value={v} onChange={(n) => setArg(k, n)} step={1} className="flex-1" />
        </label>
      );
    } else if (typeof v === "boolean") {
      rows.push(
        <label key={i} className="flex items-center gap-1 text-[11px] text-zinc-400">
          <input
            type="checkbox"
            checked={v}
            onChange={(e) => setArg(k, e.target.checked)}
            className="accent-brand-500"
          />
          {k}
        </label>
      );
    } else if (Array.isArray(v) && v.length === 3 && v.every((x) => typeof x === "number")) {
      rows.push(
        <div key={i} className="flex items-center gap-1">
          <span className="w-20 truncate text-[10px] text-zinc-500" title={k}>{k}</span>
          {v.map((x, j) => (
            <NumInput
              key={j}
              value={x}
              step={0.5}
              className="flex-1"
              onChange={(n) => setArg(k, (v as number[]).map((o2, jj) => (jj === j ? n : o2)))}
            />
          ))}
        </div>
      );
    } else if (typeof v === "string") {
      rows.push(
        <div key={i} className="flex items-center gap-1 text-[11px] text-zinc-500">
          <span className="w-20 truncate text-[10px]" title={k}>{k}</span>
          <span className="truncate">{v}</span>
        </div>
      );
    } else {
      rows.push(
        <div key={i} className="truncate text-[10px] text-zinc-500" title={JSON.stringify(v)}>
          {k}: {JSON.stringify(v).slice(0, 80)}
        </div>
      );
    }
  });
  return <div className="mt-1 flex flex-col gap-1 border-t border-zinc-800 pt-1">{rows}</div>;
}

/* ------------------------------------------------------------------ */
/* overlays (displayed-model space mm)                                 */
/* ------------------------------------------------------------------ */

function Marker({ p, color }: { p: [number, number, number]; color: string }) {
  return (
    <mesh position={p}>
      <sphereGeometry args={[0.8, 16, 12]} />
      <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
    </mesh>
  );
}

function BoolGhost({
  shape,
  op,
  sizeXYZ,
  radius,
  height,
  axis,
}: {
  shape: "box" | "cylinder" | "sphere";
  op: "union" | "difference";
  sizeXYZ: [number, number, number];
  radius: number;
  height: number;
  axis: Axis;
}) {
  const color = op === "union" ? "#34d399" : "#f87171";
  return (
    <mesh
      rotation={
        axis === "x"
          ? [0, Math.PI / 2, 0]
          : axis === "y"
            ? [-Math.PI / 2, 0, 0]
            : [0, 0, 0]
      }
    >
      {shape === "cylinder" ? (
        <cylinderGeometry args={[radius, radius, height, 32]} />
      ) : shape === "box" ? (
        <boxGeometry args={sizeXYZ} />
      ) : (
        <sphereGeometry args={[radius, 24, 16]} />
      )}
      <meshBasicMaterial color={color} transparent opacity={0.3} depthTest={false} />
    </mesh>
  );
}

interface DrillHoleIn {
  x: number;
  y: number;
  dia: number;
  depth_mm?: number | null;
  csk_dia?: number | null;
  csk_angle_deg?: number;
}

/* ------------------------------------------------------------------ */
/* page                                                                */
/* ------------------------------------------------------------------ */

export default function EditPage() {
  const { id = "" } = useParams<{ id: string }>();
  const nav = useNavigate();
  const toast = useToast();
  const qc = useQueryClient();

  const version = useQuery({ queryKey: ["version", id], queryFn: () => api.getVersion(id) });
  const v = version.data;
  useTitle(v ? `edit v${v.number}` : "Edit");

  // draft ops stack — mesh parents replay their recorded ops on the raw
  // generator mesh; every other version starts empty on the finished STL.
  const [ops, setOps] = useState<MeshOp[]>([]);
  const [initFor, setInitFor] = useState("");
  const [preview, setPreview] = useState<MeshPreviewResult | null>(null);
  const [previewOpsKey, setPreviewOpsKey] = useState("");
  const [saveFailed, setSaveFailed] = useState(false);
  const previewSeq = useRef(0);
  useEffect(() => {
    if (!v || initFor === v.id) return;
    const saved = readUiEditor<{ ops?: MeshOp[] }>(v);
    if (saved && "ops" in saved) {
      // persisted stack wins — even an empty one (user dropped every op)
      setOps(saved.ops ?? []);
    } else {
      setOps(recordedMeshOps(v));
    }
    // never keep the previous version's scratch preview — its GLB URL would
    // render as this version's model; also invalidates in-flight responses
    previewSeq.current += 1;
    setPreview(null);
    setPreviewOpsKey("");
    setSaveFailed(false);
    setInitFor(v.id);
  }, [v, initFor]);

  // tool form state
  const [rotAxis, setRotAxis] = useState<Axis>("z");
  const [rotDeg, setRotDeg] = useState(90);
  const [scaleMode, setScaleMode] = useState<"uniform" | "axes">("uniform");
  const [scaleF, setScaleF] = useState(1.1);
  const [scaleXYZ, setScaleXYZ] = useState<[number, number, number]>([1, 1, 1]);
  const [mirAxis, setMirAxis] = useState<Axis>("x");
  const [flatH, setFlatH] = useState(0.4);
  const [sliceAxis, setSliceAxis] = useState<Axis>("z");
  const [sliceAt, setSliceAt] = useState(0);
  const [sliceKeep, setSliceKeep] = useState<"low" | "high">("high");
  const [cropMin, setCropMin] = useState<[number, number, number]>([-50, -50, 0]);
  const [cropMax, setCropMax] = useState<[number, number, number]>([50, 50, 50]);
  const [smoothIter, setSmoothIter] = useState(5);
  const [smoothScheme, setSmoothScheme] = useState<"laplacian" | "taubin">("laplacian");
  const [smoothPIter, setSmoothPIter] = useState(10);
  const [decFaces, setDecFaces] = useState(50000);
  const [dropFrac, setDropFrac] = useState(0.02);
  const [remeshVoxel, setRemeshVoxel] = useState(0.5);
  const [clipZ, setClipZ] = useState(10);
  const [symAxis, setSymAxis] = useState<Axis>("y");
  const [symKeep, setSymKeep] = useState<"auto" | "pos" | "neg">("auto");
  const [shellTh, setShellTh] = useState(2);
  const [shellOpen, setShellOpen] = useState<"none" | "bottom" | "top">("bottom");
  const [boolOp, setBoolOp] = useState<"difference" | "union">("difference");
  const [boolShape, setBoolShape] = useState<"box" | "cylinder" | "sphere">("cylinder");
  const [boolR, setBoolR] = useState(2.5);
  const [boolH, setBoolH] = useState(20);
  const [boolSizeXYZ, setBoolSizeXYZ] = useState<[number, number, number]>([10, 10, 10]);
  const [boolAxis, setBoolAxis] = useState<Axis>("z");
  const [boolCenter, setBoolCenter] = useState<[number, number, number]>([0, 0, 5]);
  const [boolRotZ, setBoolRotZ] = useState(0);
  const [boolMode, setBoolMode] = useState<"translate" | "rotate">("translate");
  const [drillDia, setDrillDia] = useState(5);
  const [drillDepth, setDrillDepth] = useState(0); // 0 = through
  const [drillCsk, setDrillCsk] = useState(0); // 0 = no countersink
  const [drillHoles, setDrillHoles] = useState<DrillHoleIn[]>([]);
  const [drillAnchors, setDrillAnchors] = useState(false);

  // picking + tool focus
  const [pickMode, setPickMode] = useState<PickMode>(null);
  const [activeTool, setActiveTool] = useState<Tool>(null);
  const [measurePts, setMeasurePts] = useState<[number, number, number][]>([]);
  const [hover, setHover] = useState<PickInfo | null>(null);

  // model handle + instant rigid preview
  const [modelObj, setModelObj] = useState<THREE.Group | null>(null);
  const aabbEval = useMemo<AabbEval | null>(
    () => (modelObj ? meshAabbEvaluator(modelObj) : null),
    [modelObj]
  );

  // preview + apply
  const [busy, setBusy] = useState<"preview" | "apply" | null>(null);
  const [viewCmd] = useState<{ view: ViewPreset; n: number }>({ view: "iso", n: 0 });
  const [fitKey, setFitKey] = useState(0);
  const [grid, setGrid] = useState(true);
  const [autoPreview, setAutoPreview] = useState(true);
  const [editIdx, setEditIdx] = useState<number | null>(null);

  const glb = v?.artifacts.find((a) => a.kind === "glb");
  const baseUrl = glb?.url ?? null;

  /** ops baked into the displayed GLB (mesh parents only) — the instant
   *  rigid preview may only evaluate the suffix after this prefix. */
  const bakedOps = useMemo<MeshOp[]>(() => recordedMeshOps(v), [v]);

  const prefixLen = useMemo(() => recordedPrefixLen(ops, bakedOps), [ops, bakedOps]);
  const needsServerPreview = useMemo(
    () => prefixLen < bakedOps.length || ops.slice(prefixLen).some((o) => !isRigidOp(o)),
    [ops, prefixLen, bakedOps]
  );
  const instantMatrix = useMemo(() => {
    if (!aabbEval || needsServerPreview || !ops.length) return null;
    return rigidOpsMatrix(ops, prefixLen, aabbEval);
  }, [aabbEval, needsServerPreview, ops, prefixLen]);

  const displayUrl = needsServerPreview ? (preview?.glb_url ?? baseUrl) : baseUrl;
  const previewStale = !!preview && JSON.stringify(ops) !== previewOpsKey;
  const dirty = useMemo(
    () => JSON.stringify(ops) !== JSON.stringify(savedOps(v ?? null)),
    [ops, v]
  );

  const addOp = (op: MeshOp) => {
    setOps((o) => [...o, op]);
    setEditIdx(null);
  };

  const armTool = (t: Tool) => {
    setActiveTool((cur) => (cur === t ? null : t));
    setPickMode(null);
  };
  /** arm a click-pick — also activates the matching tool overlay so the
   *  plane/box/holes stay visible while picking */
  const TOOL_OF: Record<Exclude<PickMode, null>, Tool> = {
    measure: "measure",
    bool_center: "boolean",
    slice_at: "slice",
    crop_min: "crop",
    crop_max: "crop",
    drill: "drill",
    clip_z: "clip",
  };
  const armPick = (mode: Exclude<PickMode, null>) => {
    setPickMode((m) => (m === mode ? null : mode));
    setActiveTool(TOOL_OF[mode]);
  };

  const handlePick = (info: PickInfo) => {
    const p = info.point;
    switch (pickMode) {
      case "measure":
        setMeasurePts((pts) => (pts.length >= 2 ? [p] : [...pts, p]));
        return;
      case "bool_center":
        setBoolCenter([p[0], p[1], p[2]]);
        setPickMode(null);
        return;
      case "slice_at":
        setSliceAt(Number(p["xyz".indexOf(sliceAxis)].toFixed(2)));
        setPickMode(null);
        return;
      case "crop_min":
        setCropMin([p[0], p[1], p[2]]);
        setPickMode(null);
        return;
      case "crop_max":
        setCropMax([p[0], p[1], p[2]]);
        setPickMode(null);
        return;
      case "drill":
        setDrillHoles((hs) => [
          ...hs,
          {
            x: Number(p[0].toFixed(2)),
            y: Number(p[1].toFixed(2)),
            dia: drillDia,
            depth_mm: drillDepth > 0 ? drillDepth : null,
            csk_dia: drillCsk > 0 ? drillCsk : null,
            csk_angle_deg: 90,
          },
        ]);
        return;
      case "clip_z":
        setClipZ(Number(p[2].toFixed(2)));
        setPickMode(null);
        return;
      default:
        return;
    }
  };

  const measureDist =
    measurePts.length === 2
      ? Math.hypot(
          measurePts[1][0] - measurePts[0][0],
          measurePts[1][1] - measurePts[0][1],
          measurePts[1][2] - measurePts[0][2]
        )
      : null;

  const boolArgs = (): Record<string, unknown> => {
    const base: Record<string, unknown> = {
      op: boolOp,
      shape: boolShape,
      center: boolCenter,
      axis: boolAxis,
      rotate_z_deg: boolRotZ,
    };
    if (boolShape === "box") base.size_xyz = boolSizeXYZ;
    else if (boolShape === "cylinder") Object.assign(base, { radius: boolR, height: boolH });
    else base.radius = boolR;
    return base;
  };

  const drillArgs = () => ({
    holes: drillHoles.map((h) => ({
      x: h.x,
      y: h.y,
      dia: h.dia,
      ...(h.depth_mm ? { depth_mm: h.depth_mm } : {}),
      ...(h.csk_dia ? { csk_dia: h.csk_dia, csk_angle_deg: h.csk_angle_deg ?? 90 } : {}),
    })),
    from_anchors: drillAnchors,
  });

  const doPreview = async () => {
    if (!v || !ops.length) return;
    const seq = ++previewSeq.current;
    setBusy("preview");
    try {
      const r = await api.meshPreview(id, { ops });
      if (seq !== previewSeq.current) return; // stale response
      setPreview(r);
      setPreviewOpsKey(JSON.stringify(ops));
      if (!r.ok) toast(r.error ?? "preview failed", "error");
      else setFitKey((k) => k + 1);
    } catch (e) {
      if (seq === previewSeq.current) toast(errMsg(e), "error");
    } finally {
      if (seq === previewSeq.current) setBusy(null);
    }
  };

  // debounced auto-preview for stacks containing geometry ops
  useEffect(() => {
    if (!autoPreview || !needsServerPreview || !ops.length || busy === "apply") return;
    const t = setTimeout(doPreview, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ops, autoPreview, needsServerPreview]);

  // autosave the ops draft (merged into ui_editor server-side)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!v || initFor !== v.id) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      api.saveEditorState(id, { ops }).then(
        () => setSaveFailed(false),
        // a dropped draft must be visible — silent loss is a data bug
        () => setSaveFailed(true)
      );
    }, 1500);
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ops, initFor, v?.id]);

  const modelBox = useMemo(
    () => (aabbEval ? aabbEval(new THREE.Matrix4()) : null),
    [aabbEval]
  );

  const doApply = async () => {
    if (!v) return;
    setBusy("apply");
    try {
      await api.saveEditorState(id, { ops });
      const r = await api.meshEdit(id, { ops, note: `visual edit of v${v.number}` });
      nav(`/versions/${r.version_id}?job=${r.job_id}`);
    } catch (e) {
      toast(errMsg(e), "error");
      setBusy(null);
    }
  };

  if (version.isLoading)
    return (
      <div className="flex h-screen gap-4 p-4">
        <Skeleton className="flex-1" />
        <Skeleton className="w-96" />
      </div>
    );
  if (!v)
    return (
      <div className="p-6 text-sm text-zinc-400">Version not found.</div>
    );

  const editable = v.status === "succeeded" || v.status === "needs_review";
  const anchors = readStoredAnchors(v);
  const previewFaces = preview?.stats.faces;
  const baseFaces = (v.analysis as { triangle_count?: number } | null)?.triangle_count;
  const effFaces = previewFaces ?? baseFaces;
  const hasDecimateAhead = ops.some((o) => o.op === "decimate");
  const boolTooDense =
    (effFaces ?? 0) > 200_000 &&
    !(hasDecimateAhead && (previewFaces == null || (previewFaces ?? 0) <= 200_000));
  const failedOp = preview?.stats.failed_op;
  const opReport = preview?.stats.report;

  const cropValid = cropMax.every((mx, i) => mx > cropMin[i]);
  const pickCursor =
    pickMode && hover ? (
      <mesh position={hover.point}>
        <sphereGeometry args={[0.7, 16, 12]} />
        <meshBasicMaterial color="#f59e0b" transparent opacity={0.85} depthTest={false} />
      </mesh>
    ) : null;

  /* --- slice plane drag --- */
  const sliceDrag = (obj: THREE.Object3D) => {
    const val = [obj.position.x, obj.position.y, obj.position.z]["xyz".indexOf(sliceAxis)];
    setSliceAt(Number(val.toFixed(2)));
  };
  const clipDrag = (obj: THREE.Object3D) => setClipZ(Number(obj.position.z.toFixed(2)));

  return (
    <div className="flex h-screen flex-col">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
        <Link
          to={`/versions/${id}`}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <ArrowLeft className="h-4 w-4" /> v{v.number}
        </Link>
        <Crumbs
          items={[
            <Link key="p" to={`/projects/${v.project_id}`} className="hover:text-zinc-300">
              project
            </Link>,
            <span key="v" className="font-medium text-zinc-200">
              mesh editor
            </span>,
          ]}
        />
        <Chip tone="zinc" className="uppercase">
          {v.resolved_mode ?? v.kind}
        </Chip>
        {instantMatrix && <Chip tone="green">instant preview</Chip>}
        {needsServerPreview && preview?.glb_url && (
          <Chip tone={previewStale ? "amber" : "blue"}>
            {previewStale ? "stale preview" : "preview"}
          </Chip>
        )}
        <span className="flex-1" />
        {pickMode && (
          <Chip tone="amber">
            <MousePointer2 className="h-3 w-3" /> click on the model (right-drag orbits) —{" "}
            {pickMode === "measure"
              ? "measure"
              : pickMode === "bool_center"
                ? "place primitive"
                : pickMode === "drill"
                  ? "add drill hole"
                  : pickMode === "clip_z"
                    ? "set clip height"
                    : pickMode.startsWith("crop")
                      ? "set crop corner"
                      : "set slice position"}
          </Chip>
        )}
        <Btn variant="ghost" onClick={() => setFitKey((k) => k + 1)} title="re-fit camera">
          fit
        </Btn>
        <Btn variant="ghost" onClick={() => setGrid((g) => !g)} title="toggle grid">
          grid
        </Btn>
        <label className="flex items-center gap-1.5 text-xs text-zinc-500" title="debounced server preview after stack changes">
          <input
            type="checkbox"
            className="accent-brand-500"
            checked={autoPreview}
            onChange={(e) => setAutoPreview(e.target.checked)}
          />
          auto
        </label>
        <Btn
          variant="ghost"
          disabled={!ops.length || busy === "preview"}
          busy={busy === "preview"}
          onClick={doPreview}
        >
          <Eye className="h-4 w-4" /> Preview
        </Btn>
        <Btn
          variant="success"
          disabled={!editable || !ops.length}
          busy={busy === "apply"}
          onClick={doApply}
          title={editable ? "apply ops → new child version" : "version is still running"}
        >
          <Wand2 className="h-4 w-4" /> Apply as new version
        </Btn>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* canvas */}
        <div className="relative min-w-0 flex-1 bg-zinc-950">
          {displayUrl ? (
            <ModelCanvas
              url={displayUrl}
              grid={grid}
              viewCmd={viewCmd}
              fitKey={fitKey}
              pickEnabled={!!pickMode}
              modelMatrix={needsServerPreview ? null : instantMatrix}
              // click-only tools — ignore drag-move repeats
              onPick={(i, viaDrag) => {
                if (!viaDrag) handlePick(i);
              }}
              onHover={setHover}
              onModel={setModelObj}
            >
              {pickCursor}
              {measurePts.map((p, i) => (
                <Marker key={i} p={p} color="#f59e0b" />
              ))}
              {/* slice plane — draggable along its axis while the tool is armed */}
              {(activeTool === "slice" || pickMode === "slice_at") && (
                <GizmoControls
                  enabled={!pickMode}
                  axes={sliceAxis}
                  position={[
                    sliceAxis === "x" ? sliceAt : 0,
                    sliceAxis === "y" ? sliceAt : 0,
                    sliceAxis === "z" ? sliceAt : 0,
                  ]}
                  onChange={sliceDrag}
                >
                  <mesh
                    rotation={
                      sliceAxis === "x"
                        ? [0, Math.PI / 2, 0]
                        : sliceAxis === "y"
                          ? [Math.PI / 2, 0, 0]
                          : [0, 0, 0]
                    }
                  >
                    <planeGeometry args={[400, 400]} />
                    <meshBasicMaterial color="#f59e0b" transparent opacity={0.15} side={2} depthTest={false} />
                  </mesh>
                </GizmoControls>
              )}
              {/* clip-height plane */}
              {activeTool === "clip" && (
                <GizmoControls enabled={!pickMode} axes="z" position={[0, 0, clipZ]} onChange={clipDrag}>
                  <mesh>
                    <planeGeometry args={[400, 400]} />
                    <meshBasicMaterial color="#38bdf8" transparent opacity={0.15} side={2} depthTest={false} />
                  </mesh>
                </GizmoControls>
              )}
              {/* boolean primitive ghost — draggable/rotatable while armed */}
              {(activeTool === "boolean" || pickMode === "bool_center") && (
                <GizmoControls
                  enabled={!pickMode}
                  mode={boolMode}
                  axes={boolMode === "rotate" ? "z" : "xyz"}
                  position={boolCenter}
                  rotation={[0, 0, (boolRotZ * Math.PI) / 180]}
                  onChange={(obj) => {
                    if (boolMode === "rotate") {
                      setBoolRotZ(Number(((obj.rotation.z * 180) / Math.PI).toFixed(1)));
                    } else {
                      setBoolCenter([
                        Number(obj.position.x.toFixed(2)),
                        Number(obj.position.y.toFixed(2)),
                        Number(obj.position.z.toFixed(2)),
                      ]);
                    }
                  }}
                >
                  <BoolGhost
                    shape={boolShape}
                    op={boolOp}
                    sizeXYZ={boolSizeXYZ}
                    radius={boolR}
                    height={boolH}
                    axis={boolAxis}
                  />
                </GizmoControls>
              )}
              {/* crop box helper + corner handles */}
              {activeTool === "crop" && cropValid && (
                <>
                  <box3Helper
                    args={[
                      new THREE.Box3(
                        new THREE.Vector3(...cropMin),
                        new THREE.Vector3(...cropMax)
                      ),
                      new THREE.Color("#38bdf8"),
                    ]}
                  />
                  {(
                    [
                      [cropMin, setCropMin, "min"],
                      [cropMax, setCropMax, "max"],
                    ] as const
                  ).map(([corner, setCorner, label]) => (
                    <GizmoControls
                      key={label}
                      enabled={!pickMode}
                      position={corner}
                      onChange={(obj) =>
                        setCorner([
                          Number(obj.position.x.toFixed(2)),
                          Number(obj.position.y.toFixed(2)),
                          Number(obj.position.z.toFixed(2)),
                        ])
                      }
                    >
                      <mesh>
                        <boxGeometry args={[1.5, 1.5, 1.5]} />
                        <meshBasicMaterial color="#38bdf8" transparent opacity={0.8} depthTest={false} />
                      </mesh>
                    </GizmoControls>
                  ))}
                </>
              )}
              {/* drill holes */}
              {(activeTool === "drill" || pickMode === "drill") &&
                drillHoles.map((h, i) => (
                  <group key={i} position={[h.x, h.y, modelBox ? modelBox.min.z : 0]}>
                    <mesh position={[0, 0, ((modelBox?.max.z ?? 10) - (modelBox?.min.z ?? 0)) / 2]}>
                      <cylinderGeometry
                        args={[h.dia / 2, h.dia / 2, Math.max(0.1, (modelBox?.max.z ?? 10) - (modelBox?.min.z ?? 0)), 20]}
                      />
                      <meshBasicMaterial color="#f87171" transparent opacity={0.35} depthTest={false} />
                    </mesh>
                  </group>
                ))}
              {/* anchor markers while drill tool armed */}
              {activeTool === "drill" &&
                anchors?.map((a, i) => (
                  <Marker key={`a${i}`} p={[a.x, a.y, (modelBox?.max.z ?? 10) + 2]} color="#22d3ee" />
                ))}
            </ModelCanvas>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-zinc-500">
              no mesh artifact to edit
            </div>
          )}

          {/* measure overlay */}
          {measureDist != null && (
            <div className="absolute bottom-3 left-3 rounded-lg border border-amber-800/60 bg-zinc-950/90 px-3 py-1.5 text-sm text-amber-300 backdrop-blur">
              <Ruler className="mr-1.5 inline h-3.5 w-3.5" />
              {measureDist.toFixed(2)} mm
              <button
                className="ml-2 text-xs text-zinc-500 hover:text-zinc-300"
                onClick={() => setMeasurePts([])}
              >
                clear
              </button>
            </div>
          )}

          {/* preview stats */}
          {needsServerPreview && preview?.ok && (
            <div className="absolute bottom-3 right-3 flex flex-wrap gap-1.5">
              {preview.stats.faces != null && (
                <Chip>faces {preview.stats.faces.toLocaleString()}</Chip>
              )}
              {preview.stats.bbox_size && (
                <Chip>{preview.stats.bbox_size.map((n) => fmt(n)).join(" × ")} mm</Chip>
              )}
              {preview.stats.watertight != null && (
                <Chip tone={preview.stats.watertight ? "green" : "amber"}>
                  {preview.stats.watertight ? "watertight" : "not watertight"}
                </Chip>
              )}
              <button
                className="text-xs text-zinc-500 underline hover:text-zinc-300"
                onClick={() => setPreview(null)}
              >
                back to original
              </button>
            </div>
          )}
        </div>

        {/* tools panel */}
        <div className="flex w-[400px] shrink-0 flex-col overflow-y-auto border-l border-zinc-800 bg-zinc-950">
          <div className="flex flex-col gap-3 p-3">
            <SectionTitle>
              <span className="flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5" /> tools
              </span>
            </SectionTitle>

            {/* orient */}
            <ToolCard title="Orient" hint="re-ground the part — map a source axis to up (+Z)">
              <div className="grid grid-cols-3 gap-1">
                {UP_DIRS.map((u) => (
                  <button
                    key={u}
                    className="rounded border border-zinc-700 px-1 py-1 text-[11px] font-semibold uppercase text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                    onClick={() => addOp({ op: "set_up", args: { up: u } })}
                  >
                    {u} up
                  </button>
                ))}
              </div>
            </ToolCard>

            {/* rotate */}
            <ToolCard
              title="Rotate"
              onAdd={() => addOp({ op: "rotate", args: { axis: rotAxis, deg: rotDeg } })}
            >
              <div className="flex gap-2">
                <AxisPicker value={rotAxis} onChange={setRotAxis} />
                <NumInput value={rotDeg} onChange={setRotDeg} step={15} className="w-20" />
              </div>
            </ToolCard>

            {/* scale */}
            <ToolCard
              title="Scale"
              onAdd={() =>
                addOp({
                  op: "scale",
                  args:
                    scaleMode === "uniform"
                      ? { factor: scaleF }
                      : { x: scaleXYZ[0], y: scaleXYZ[1], z: scaleXYZ[2] },
                })
              }
              disabled={scaleF <= 0 || scaleXYZ.some((n) => n <= 0)}
            >
              <div className="mb-1.5 flex gap-1">
                {(["uniform", "axes"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setScaleMode(m)}
                    className={`flex-1 rounded border px-2 py-1 text-xs ${
                      scaleMode === m
                        ? "border-brand-600 bg-brand-950/60 text-brand-300"
                        : "border-zinc-700 text-zinc-500"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
              {scaleMode === "uniform" ? (
                <NumInput value={scaleF} onChange={setScaleF} step={0.1} />
              ) : (
                <div className="flex gap-1">
                  {scaleXYZ.map((n, i) => (
                    <NumInput
                      key={i}
                      value={n}
                      step={0.1}
                      onChange={(val) =>
                        setScaleXYZ((s) => s.map((x, j) => (j === i ? val : x)) as [number, number, number])
                      }
                    />
                  ))}
                </div>
              )}
            </ToolCard>

            {/* mirror */}
            <ToolCard
              title="Mirror"
              onAdd={() => addOp({ op: "mirror", args: { axis: mirAxis } })}
            >
              <AxisPicker value={mirAxis} onChange={setMirAxis} />
            </ToolCard>

            {/* flatten bottom */}
            <ToolCard
              title="Flatten bottom"
              hint="planes off the lowest mm — flat-back for FDM printing"
              onAdd={() => addOp({ op: "flatten_bottom", args: { height_mm: flatH } })}
              disabled={flatH <= 0 || flatH > 50}
            >
              <Field label="height (mm)">
                <NumInput value={flatH} onChange={setFlatH} step={0.1} />
              </Field>
            </ToolCard>

            {/* slice */}
            <ToolCard
              title="Slice"
              hint="cut with an axis plane and keep one side — drag the plane or pick a point"
              active={activeTool === "slice"}
              onToggle={() => armTool("slice")}
              onAdd={() =>
                addOp({ op: "slice", args: { axis: sliceAxis, at_mm: sliceAt, keep: sliceKeep } })
              }
            >
              <div className="flex flex-col gap-2">
                <AxisPicker value={sliceAxis} onChange={setSliceAxis} />
                <div className="flex items-center gap-1.5">
                  <NumInput value={sliceAt} onChange={setSliceAt} step={0.5} className="flex-1" />
                  <span className="text-xs text-zinc-500">mm</span>
                  <Btn
                    variant="ghost"
                    className={`px-2 py-0.5 text-xs ${pickMode === "slice_at" ? "border border-amber-600 text-amber-300" : ""}`}
                    onClick={() => armPick("slice_at")}
                  >
                    <CircleDot className="h-3 w-3" /> pick
                  </Btn>
                </div>
                <div className="flex gap-1">
                  {(["low", "high"] as const).map((k) => (
                    <button
                      key={k}
                      onClick={() => setSliceKeep(k)}
                      className={`flex-1 rounded border px-2 py-1 text-xs ${
                        sliceKeep === k
                          ? "border-brand-600 bg-brand-950/60 text-brand-300"
                          : "border-zinc-700 text-zinc-500"
                      }`}
                    >
                      keep {k}
                    </button>
                  ))}
                </div>
              </div>
            </ToolCard>

            {/* clip height */}
            <ToolCard
              title="Clip height"
              hint="hard ceiling — cut + cap everything above z_max (drag the plane)"
              active={activeTool === "clip"}
              onToggle={() => armTool("clip")}
              onAdd={() => addOp({ op: "clip_height", args: { z_max: clipZ } })}
              disabled={clipZ <= 0.1}
            >
              <div className="flex items-center gap-1.5">
                <NumInput value={clipZ} onChange={setClipZ} step={0.5} className="flex-1" />
                <span className="text-xs text-zinc-500">mm</span>
                <Btn
                  variant="ghost"
                  className={`px-2 py-0.5 text-xs ${pickMode === "clip_z" ? "border border-amber-600 text-amber-300" : ""}`}
                  onClick={() => armPick("clip_z")}
                >
                  <CircleDot className="h-3 w-3" /> pick
                </Btn>
              </div>
            </ToolCard>

            {/* crop */}
            <ToolCard
              title="Crop box"
              hint="keep only geometry inside the box — drag the corner handles or pick on the model"
              active={activeTool === "crop"}
              onToggle={() => armTool("crop")}
              onAdd={() => addOp({ op: "crop", args: { min: cropMin, max: cropMax } })}
              disabled={!cropValid}
            >
              <div className="flex flex-col gap-1.5">
                <Btn
                  variant="ghost"
                  className="px-2 py-0.5 text-xs"
                  disabled={!modelBox}
                  onClick={() => {
                    if (!modelBox) return;
                    setCropMin([modelBox.min.x, modelBox.min.y, modelBox.min.z]);
                    setCropMax([modelBox.max.x, modelBox.max.y, modelBox.max.z]);
                    setActiveTool("crop");
                  }}
                >
                  fit to part
                </Btn>
                {(["min", "max"] as const).map((k) => {
                  const val = k === "min" ? cropMin : cropMax;
                  const set = k === "min" ? setCropMin : setCropMax;
                  const mode: Exclude<PickMode, null> = k === "min" ? "crop_min" : "crop_max";
                  return (
                    <div key={k} className="flex items-center gap-1">
                      <span className="w-8 text-[10px] uppercase text-zinc-500">{k}</span>
                      {val.map((n, i) => (
                        <NumInput
                          key={i}
                          value={n}
                          step={1}
                          onChange={(x) =>
                            set(val.map((o, j) => (j === i ? x : o)) as [number, number, number])
                          }
                        />
                      ))}
                      <IconBtn title="pick corner on model" onClick={() => armPick(mode)}>
                        <CircleDot
                          className={`h-3.5 w-3.5 ${pickMode === mode ? "text-amber-300" : ""}`}
                        />
                      </IconBtn>
                    </div>
                  );
                })}
              </div>
            </ToolCard>

            {/* smooth */}
            <ToolCard
              title="Smooth"
              onAdd={() =>
                addOp({ op: "smooth", args: { iterations: smoothIter, scheme: smoothScheme } })
              }
            >
              <div className="flex gap-1.5">
                <NumInput value={smoothIter} onChange={setSmoothIter} className="w-16" />
                <select
                  className={`${inputCls} flex-1 px-2 py-1 text-xs`}
                  value={smoothScheme}
                  onChange={(e) => setSmoothScheme(e.target.value as "laplacian" | "taubin")}
                >
                  <option value="laplacian">laplacian</option>
                  <option value="taubin">taubin (shrink-free)</option>
                </select>
              </div>
            </ToolCard>

            <ToolCard
              title="Smooth (preserve features)"
              hint="edge-aware smoothing — keeps sharp features"
              onAdd={() => addOp({ op: "smooth_preserve", args: { iterations: smoothPIter } })}
            >
              <NumInput value={smoothPIter} onChange={setSmoothPIter} className="w-24" />
            </ToolCard>

            {/* decimate */}
            <ToolCard
              title="Decimate"
              hint="reduce triangle count — needed before booleans on dense meshes"
              onAdd={() => addOp({ op: "decimate", args: { faces: Math.round(decFaces) } })}
              disabled={decFaces < 100}
            >
              <Field label="target faces">
                <NumInput value={decFaces} onChange={setDecFaces} step={5000} />
              </Field>
            </ToolCard>

            {/* remesh */}
            <ToolCard
              title="Remesh"
              hint="uniform voxel remesh — watertight by construction, evens out topology"
              onAdd={() => addOp({ op: "remesh", args: { voxel_mm: remeshVoxel } })}
              disabled={remeshVoxel < 0.05 || remeshVoxel > 5}
            >
              <Field label="voxel (mm)">
                <NumInput value={remeshVoxel} onChange={setRemeshVoxel} step={0.1} />
              </Field>
            </ToolCard>

            {/* symmetrize */}
            <ToolCard
              title="Symmetrize"
              hint="mirror the better half across the bbox-centre plane"
              onAdd={() => addOp({ op: "symmetrize", args: { axis: symAxis, keep: symKeep } })}
            >
              <div className="flex flex-col gap-1.5">
                <AxisPicker value={symAxis} onChange={setSymAxis} />
                <div className="flex gap-1">
                  {(["auto", "pos", "neg"] as const).map((k) => (
                    <button
                      key={k}
                      onClick={() => setSymKeep(k)}
                      className={`flex-1 rounded border px-2 py-1 text-xs ${
                        symKeep === k
                          ? "border-brand-600 bg-brand-950/60 text-brand-300"
                          : "border-zinc-700 text-zinc-500"
                      }`}
                    >
                      {k === "auto" ? "auto" : `keep ${k}`}
                    </button>
                  ))}
                </div>
              </div>
            </ToolCard>

            {/* shell */}
            <ToolCard
              title="Shell"
              hint="hollow the part — optional opening on a face (FDM-friendly)"
              onAdd={() => addOp({ op: "shell", args: { thickness_mm: shellTh, opening: shellOpen } })}
              disabled={shellTh < 0.2 || shellTh > 20}
            >
              <div className="flex items-center gap-1.5">
                <NumInput value={shellTh} onChange={setShellTh} step={0.2} className="w-20" />
                <span className="text-xs text-zinc-500">mm</span>
                <select
                  className={`${inputCls} flex-1 px-2 py-1 text-xs`}
                  value={shellOpen}
                  onChange={(e) => setShellOpen(e.target.value as typeof shellOpen)}
                >
                  <option value="none">sealed</option>
                  <option value="bottom">open bottom</option>
                  <option value="top">open top</option>
                </select>
              </div>
            </ToolCard>

            {/* drill holes */}
            <ToolCard
              title="Drill holes"
              hint={`vertical holes at picked XY points${anchors?.length ? ` — ${anchors.length} measured anchor(s) available` : ""}`}
              active={activeTool === "drill"}
              onToggle={() => armTool("drill")}
              onAdd={() => {
                addOp({ op: "drill_holes", args: drillArgs() });
                setDrillHoles([]); // holes are baked into the op — clear the picker
              }}
              disabled={!drillAnchors && !drillHoles.length}
            >
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-14 text-[11px] text-zinc-500">dia</span>
                  <NumInput value={drillDia} onChange={setDrillDia} step={0.5} className="flex-1" />
                  <span className="w-14 text-[11px] text-zinc-500">depth</span>
                  <NumInput value={drillDepth} onChange={setDrillDepth} step={1} className="flex-1" />
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-14 text-[11px] text-zinc-500">csk dia</span>
                  <NumInput value={drillCsk} onChange={setDrillCsk} step={0.5} className="flex-1" />
                  <Btn
                    variant="ghost"
                    className={`px-2 py-0.5 text-xs ${pickMode === "drill" ? "border border-amber-600 text-amber-300" : ""}`}
                    onClick={() => armPick("drill")}
                  >
                    <CircleDot className="h-3 w-3" /> pick holes
                  </Btn>
                </div>
                <div className="text-[10px] text-zinc-500">depth/csk 0 = through / none</div>
                {!!drillHoles.length && (
                  <div className="flex flex-col gap-0.5">
                    {drillHoles.map((h, i) => (
                      <div key={i} className="flex items-center gap-1.5 rounded border border-zinc-800 px-1.5 py-0.5 text-[11px]">
                        <span className="flex-1 text-zinc-400">
                          Ø{h.dia} @ ({h.x}, {h.y})
                          {h.depth_mm ? ` · ${h.depth_mm}mm` : " · through"}
                          {h.csk_dia ? ` · csk Ø${h.csk_dia}` : ""}
                        </span>
                        <button
                          title="remove drill hole"
                          aria-label="remove drill hole"
                          className="text-zinc-500 hover:text-red-400"
                          onClick={() => setDrillHoles((hs) => hs.filter((_, j) => j !== i))}
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                {!!anchors?.length && (
                  <label className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                    <input
                      type="checkbox"
                      className="accent-brand-500"
                      checked={drillAnchors}
                      onChange={(e) => setDrillAnchors(e.target.checked)}
                    />
                    use {anchors.length} measured anchor hole(s)
                  </label>
                )}
              </div>
            </ToolCard>

            {/* boolean */}
            <ToolCard
              title="Boolean add / cut"
              hint={
                boolTooDense
                  ? `mesh has ${effFaces?.toLocaleString()} faces (> 200k) — decimate first`
                  : "place the primitive, then drag/rotate the ghost — 'place' re-arms picking"
              }
              active={activeTool === "boolean"}
              onToggle={() => armTool("boolean")}
              onAdd={() => addOp({ op: "boolean", args: boolArgs() })}
              disabled={boolTooDense}
            >
              <div className="flex flex-col gap-2">
                <div className="flex gap-1">
                  {(["difference", "union"] as const).map((o) => (
                    <button
                      key={o}
                      onClick={() => setBoolOp(o)}
                      className={`flex-1 rounded border px-2 py-1 text-xs ${
                        boolOp === o
                          ? "border-brand-600 bg-brand-950/60 text-brand-300"
                          : "border-zinc-700 text-zinc-500"
                      }`}
                    >
                      {o === "difference" ? "cut" : "add"}
                    </button>
                  ))}
                </div>
                <div className="flex gap-1">
                  {(["cylinder", "box", "sphere"] as const).map((s) => (
                    <button
                      key={s}
                      onClick={() => setBoolShape(s)}
                      className={`flex-1 rounded border px-2 py-1 text-xs ${
                        boolShape === s
                          ? "border-brand-600 bg-brand-950/60 text-brand-300"
                          : "border-zinc-700 text-zinc-500"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                {boolShape === "box" && (
                  <div className="flex items-center gap-1.5">
                    <span className="w-14 text-[11px] text-zinc-500">size xyz</span>
                    {boolSizeXYZ.map((n, i) => (
                      <NumInput
                        key={i}
                        value={n}
                        step={1}
                        className="flex-1"
                        onChange={(x) =>
                          setBoolSizeXYZ(boolSizeXYZ.map((o, j) => (j === i ? x : o)) as [number, number, number])
                        }
                      />
                    ))}
                  </div>
                )}
                {boolShape !== "box" && (
                  <div className="flex items-center gap-1.5">
                    <span className="w-14 text-[11px] text-zinc-500">radius</span>
                    <NumInput value={boolR} onChange={setBoolR} step={0.5} className="flex-1" />
                    <span className="text-xs text-zinc-500">mm</span>
                  </div>
                )}
                {boolShape === "cylinder" && (
                  <>
                    <div className="flex items-center gap-1.5">
                      <span className="w-14 text-[11px] text-zinc-500">height</span>
                      <NumInput value={boolH} onChange={setBoolH} step={1} className="flex-1" />
                      <span className="text-xs text-zinc-500">mm</span>
                    </div>
                    <AxisPicker value={boolAxis} onChange={setBoolAxis} />
                  </>
                )}
                <div className="flex items-center gap-1.5">
                  <span className="w-14 text-[11px] text-zinc-500">center</span>
                  {boolCenter.map((n, i) => (
                    <NumInput
                      key={i}
                      value={n}
                      step={0.5}
                      onChange={(x) =>
                        setBoolCenter(
                          boolCenter.map((o, j) => (j === i ? x : o)) as [number, number, number]
                        )
                      }
                    />
                  ))}
                  <Btn
                    variant="ghost"
                    className={`px-2 py-0.5 text-xs ${pickMode === "bool_center" ? "border border-amber-600 text-amber-300" : ""}`}
                    onClick={() => armPick("bool_center")}
                  >
                    place
                  </Btn>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-14 text-[11px] text-zinc-500">rotate z</span>
                  <NumInput value={boolRotZ} onChange={setBoolRotZ} step={15} className="flex-1" />
                  <span className="text-xs text-zinc-500">°</span>
                  <div className="flex gap-1">
                    {(["translate", "rotate"] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => setBoolMode(m)}
                        className={`rounded border px-2 py-0.5 text-[10px] ${
                          boolMode === m
                            ? "border-brand-600 text-brand-300"
                            : "border-zinc-700 text-zinc-500"
                        }`}
                      >
                        {m === "translate" ? "move" : "rot"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </ToolCard>

            {/* bodies + repair */}
            <ToolCard title="Bodies & repair">
              <div className="flex flex-wrap gap-1.5">
                <Btn
                  variant="ghost"
                  className="px-2 py-1 text-xs"
                  onClick={() => addOp({ op: "keep_largest", args: {} })}
                >
                  keep largest
                </Btn>
                <Btn
                  variant="ghost"
                  className="px-2 py-1 text-xs"
                  onClick={() => addOp({ op: "repair", args: {} })}
                >
                  repair
                </Btn>
                <div className="flex items-center gap-1">
                  <Btn
                    variant="ghost"
                    className="px-2 py-1 text-xs"
                    onClick={() =>
                      addOp({ op: "drop_small", args: { min_volume_frac: dropFrac } })
                    }
                  >
                    drop small
                  </Btn>
                  <NumInput
                    value={dropFrac}
                    onChange={setDropFrac}
                    step={0.01}
                    className="w-16"
                  />
                </div>
              </div>
            </ToolCard>

            {/* measure */}
            <ToolCard
              title="Measure"
              hint="click two surface points — distance in mm (not recorded as an op)"
            >
              <Btn
                variant="ghost"
                className={`px-2 py-1 text-xs ${pickMode === "measure" ? "border border-amber-600 text-amber-300" : ""}`}
                onClick={() => {
                  armPick("measure");
                  setMeasurePts([]);
                }}
              >
                <Ruler className="h-3.5 w-3.5" /> measure distance
              </Btn>
            </ToolCard>
          </div>

          {/* ops stack */}
          <div className="mt-auto border-t border-zinc-800 p-3">
            <SectionTitle
              right={
                <div className="flex gap-1">
                  <IconBtn
                    title="undo last op"
                    disabled={!ops.length}
                    onClick={() => setOps((o) => o.slice(0, -1))}
                  >
                    <Undo2 className="h-4 w-4" />
                  </IconBtn>
                  <IconBtn
                    title="clear all ops"
                    danger
                    disabled={!ops.length}
                    onClick={() => setOps([])}
                  >
                    <Trash2 className="h-4 w-4" />
                  </IconBtn>
                </div>
              }
            >
              op stack ({ops.length})
            </SectionTitle>
            {!ops.length && (
              <div className="text-xs text-zinc-500">
                No operations yet — build a stack with the tools, then Preview / Apply.
              </div>
            )}
            <div className="flex max-h-52 flex-col gap-1 overflow-y-auto">
              {ops.map((o, i) => (
                <div
                  key={i}
                  className={`rounded border px-2 py-1 text-xs ${
                    failedOp === i
                      ? "border-red-600 bg-red-950/30"
                      : "border-zinc-800 bg-zinc-900/60"
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="w-4 text-zinc-500">{i + 1}</span>
                    <button
                      className="min-w-0 flex-1 truncate text-left text-zinc-300 hover:text-zinc-100"
                      title={`${JSON.stringify(o.args)} — click to edit args`}
                      onClick={() => setEditIdx(editIdx === i ? null : i)}
                    >
                      {opLabel(o)}
                    </button>
                    {opReport?.[i] && (
                      <span
                        className={`shrink-0 text-[10px] ${opReport[i].watertight === false ? "text-amber-400" : "text-zinc-500"}`}
                        title={`faces ${opReport[i].faces_in?.toLocaleString()} → ${opReport[i].faces_out?.toLocaleString()}${opReport[i].watertight === false ? " · not watertight" : ""}`}
                      >
                        {opReport[i].watertight === false ? "!" : "·"}
                      </span>
                    )}
                    <IconBtn
                      title="duplicate op"
                      onClick={() =>
                        setOps((os) => [...os.slice(0, i + 1), o, ...os.slice(i + 1)])
                      }
                    >
                      <Copy className="h-3 w-3" />
                    </IconBtn>
                    <IconBtn
                      title="move up"
                      disabled={i === 0}
                      onClick={() =>
                        setOps((os) => {
                          const c = [...os];
                          [c[i - 1], c[i]] = [c[i], c[i - 1]];
                          return c;
                        })
                      }
                    >
                      <ArrowUp className="h-3 w-3" />
                    </IconBtn>
                    <IconBtn
                      title="move down"
                      disabled={i === ops.length - 1}
                      onClick={() =>
                        setOps((os) => {
                          const c = [...os];
                          [c[i], c[i + 1]] = [c[i + 1], c[i]];
                          return c;
                        })
                      }
                    >
                      <ArrowDown className="h-3 w-3" />
                    </IconBtn>
                    <IconBtn
                      title="remove op"
                      danger
                      onClick={() => setOps((os) => os.filter((_, j) => j !== i))}
                    >
                      <Trash2 className="h-3 w-3" />
                    </IconBtn>
                  </div>
                  {editIdx === i && (
                    <OpArgEditor
                      op={o}
                      onChange={(no) => setOps((os) => os.map((x, j) => (j === i ? no : x)))}
                    />
                  )}
                </div>
              ))}
            </div>
            {dirty && (
              <div className="mt-2 flex items-center gap-1.5 text-[11px] text-zinc-500">
                {saveFailed ? (
                  <span className="text-amber-400" title="draft changes could not be saved — saving retries automatically">
                    draft not saved
                  </span>
                ) : (
                  <>
                    <Check className="h-3 w-3 text-emerald-500" /> draft autosaved
                  </>
                )}
                <button
                  className="underline hover:text-zinc-300"
                  onClick={() =>
                    api.saveEditorState(id, { ops }).then(
                      () => {
                        setSaveFailed(false);
                        toast("draft saved", "success");
                        qc.invalidateQueries({ queryKey: ["version", id] });
                      },
                      (e) => {
                        setSaveFailed(true);
                        toast(errMsg(e), "error");
                      }
                    )
                  }
                >
                  save now
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
