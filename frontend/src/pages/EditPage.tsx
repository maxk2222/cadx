import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  CircleDot,
  Eye,
  Layers,
  MousePointer2,
  Ruler,
  Trash2,
  Undo2,
  Wand2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import ModelCanvas, { type PickInfo, type ViewPreset } from "../components/editor/ModelCanvas";
import {
  Btn,
  Card,
  Chip,
  Crumbs,
  Field,
  IconBtn,
  SectionTitle,
  Skeleton,
  inputCls,
  useToast,
} from "../components/ui";
import { api } from "../lib/api";
import type { MeshOp, MeshPreviewResult } from "../lib/types";
import { useTitle } from "../lib/useTitle";

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

type Axis = "x" | "y" | "z";
const AXES: Axis[] = ["x", "y", "z"];
const UP_DIRS = ["+z", "-z", "+x", "-x", "+y", "-y"] as const;

type PickMode = "measure" | "bool_center" | "slice_at" | "crop_min" | "crop_max" | null;

/** The ops stack this version was opened with (persisted draft or recorded
 *  mesh ops — used to detect unsaved draft changes). */
function savedOps(ver: {
  kind?: string;
  resolved_mode?: string | null;
  params?: unknown;
  options?: unknown;
} | null): MeshOp[] {
  if (!ver) return [];
  const saved = (ver.options as Record<string, unknown> | undefined)?.ui_editor as
    | { ops?: MeshOp[] }
    | undefined;
  if (saved?.ops?.length) return saved.ops;
  if (ver.resolved_mode === "mesh" && ver.kind !== "mesh_edit") {
    const recorded = (ver.params as { ops?: MeshOp[] } | null)?.ops;
    if (Array.isArray(recorded)) return recorded as MeshOp[];
  }
  return [];
}

const num = (s: string, fallback = 0) => {
  const n = Number(s);
  return Number.isFinite(n) ? n : fallback;
};

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
    case "boolean":
      return `${a.op === "union" ? "add" : "cut"} ${a.shape}`;
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
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
  onAdd?: () => void;
  addLabel?: string;
  disabled?: boolean;
}) {
  return (
    <Card className="p-3">
      <div className="mb-1.5 flex items-center justify-between">
        <div className="text-xs font-semibold text-zinc-300">{title}</div>
        {onAdd && (
          <Btn variant="ghost" className="px-2 py-0.5 text-xs" disabled={disabled} onClick={onAdd}>
            {addLabel}
          </Btn>
        )}
      </div>
      {hint && <div className="mb-2 text-[11px] text-zinc-600">{hint}</div>}
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

function NumInput({
  value,
  onChange,
  step = 1,
  className = "",
}: {
  value: number;
  onChange: (n: number) => void;
  step?: number;
  className?: string;
}) {
  return (
    <input
      type="number"
      step={step}
      className={`${inputCls} px-2 py-1 text-xs ${className}`}
      value={value}
      onChange={(e) => onChange(num(e.target.value))}
    />
  );
}

/* ------------------------------------------------------------------ */
/* overlays (model-space mm)                                           */
/* ------------------------------------------------------------------ */

function Marker({ p, color }: { p: [number, number, number]; color: string }) {
  return (
    <mesh position={p}>
      <sphereGeometry args={[0.8, 16, 12]} />
      <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
    </mesh>
  );
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
  useEffect(() => {
    if (!v || initFor === v.id) return;
    const saved = (v.options as Record<string, unknown> | undefined)?.ui_editor as
      | { ops?: MeshOp[] }
      | undefined;
    if (saved?.ops?.length) {
      setOps(saved.ops);
    } else if (v.resolved_mode === "mesh" && v.kind !== "mesh_edit") {
      const recorded = (v.params as { ops?: MeshOp[] } | null)?.ops;
      if (Array.isArray(recorded)) setOps(recorded as MeshOp[]);
    }
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
  const [boolOp, setBoolOp] = useState<"difference" | "union">("difference");
  const [boolShape, setBoolShape] = useState<"box" | "cylinder" | "sphere">("cylinder");
  const [boolR, setBoolR] = useState(2.5);
  const [boolH, setBoolH] = useState(20);
  const [boolSize, setBoolSize] = useState(10);
  const [boolAxis, setBoolAxis] = useState<Axis>("z");
  const [boolCenter, setBoolCenter] = useState<[number, number, number]>([0, 0, 5]);
  const [boolRotZ, setBoolRotZ] = useState(0);

  // picking
  const [pickMode, setPickMode] = useState<PickMode>(null);
  const [measurePts, setMeasurePts] = useState<[number, number, number][]>([]);

  // preview + apply
  const [preview, setPreview] = useState<MeshPreviewResult | null>(null);
  const [busy, setBusy] = useState<"preview" | "apply" | null>(null);
  const [viewCmd, setViewCmd] = useState<{ view: ViewPreset; n: number }>({ view: "iso", n: 0 });
  const [fitKey, setFitKey] = useState(0);
  const [grid, setGrid] = useState(true);

  const glb = v?.artifacts.find((a) => a.kind === "glb");
  const baseUrl = glb?.url ?? null;
  const displayUrl = preview?.glb_url ?? baseUrl;
  const dirty = useMemo(() => JSON.stringify(ops) !== JSON.stringify(savedOps(v ?? null)), [ops, v]);

  const addOp = (op: MeshOp) => {
    setOps((o) => [...o, op]);
    setPreview(null);
  };

  const handlePick = (info: PickInfo) => {
    const p = info.point;
    if (pickMode === "measure") {
      setMeasurePts((pts) => (pts.length >= 2 ? [p] : [...pts, p]));
      return;
    }
    if (pickMode === "bool_center") {
      setBoolCenter([p[0], p[1], p[2]]);
      setPickMode(null);
      return;
    }
    if (pickMode === "slice_at") {
      setSliceAt(Number(p["xyz".indexOf(sliceAxis)].toFixed(2)));
      setPickMode(null);
      return;
    }
    if (pickMode === "crop_min") {
      setCropMin([p[0], p[1], p[2]]);
      setPickMode(null);
      return;
    }
    if (pickMode === "crop_max") {
      setCropMax([p[0], p[1], p[2]]);
      setPickMode(null);
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
    if (boolShape === "box") base.size = boolSize;
    else if (boolShape === "cylinder") Object.assign(base, { radius: boolR, height: boolH });
    else base.radius = boolR;
    return base;
  };

  const doPreview = async () => {
    if (!v) return;
    setBusy("preview");
    try {
      const r = await api.meshPreview(id, { ops });
      setPreview(r);
      if (!r.ok) toast(r.error ?? "preview failed", "error");
      else setFitKey((k) => k + 1);
    } catch (e) {
      toast(String(e), "error");
    } finally {
      setBusy(null);
    }
  };

  const doApply = async () => {
    if (!v) return;
    setBusy("apply");
    try {
      await api.saveEditorState(id, { ops });
      const r = await api.meshEdit(id, { ops, note: `visual edit of v${v.number}` });
      nav(`/versions/${r.version_id}?job=${r.job_id}`);
    } catch (e) {
      toast(String(e), "error");
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
  const faces = (v.analysis as { triangle_count?: number } | null)?.triangle_count;
  const boolTooDense = (faces ?? 0) > 200_000;

  const armPick = (mode: Exclude<PickMode, null>) =>
    setPickMode((m) => (m === mode ? null : mode));

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
        {preview?.glb_url && <Chip tone="blue">preview</Chip>}
        <span className="flex-1" />
        {pickMode && (
          <Chip tone="amber">
            <MousePointer2 className="h-3 w-3" /> click on the model —{" "}
            {pickMode === "measure"
              ? "measure"
              : pickMode === "bool_center"
                ? "place primitive"
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
              onPick={handlePick}
            >
              {measurePts.map((p, i) => (
                <Marker key={i} p={p} color="#f59e0b" />
              ))}
              {/* slice plane indicator */}
              <SlicePlane axis={sliceAxis} at={sliceAt} show={pickMode === "slice_at"} />
              {/* boolean primitive ghost while placing */}
              {pickMode === "bool_center" && (
                <group position={boolCenter} rotation={[0, 0, (boolRotZ * Math.PI) / 180]}>
                  <mesh
                    rotation={
                      boolAxis === "x"
                        ? [0, Math.PI / 2, 0]
                        : boolAxis === "y"
                          ? [-Math.PI / 2, 0, 0]
                          : [0, 0, 0]
                    }
                  >
                    {boolShape === "cylinder" ? (
                      <cylinderGeometry args={[boolR, boolR, boolH, 32]} />
                    ) : boolShape === "box" ? (
                      <boxGeometry args={[boolSize, boolSize, boolSize]} />
                    ) : (
                      <sphereGeometry args={[boolR, 24, 16]} />
                    )}
                    <meshBasicMaterial
                      color={boolOp === "union" ? "#34d399" : "#f87171"}
                      transparent
                      opacity={0.3}
                      depthTest={false}
                    />
                  </mesh>
                </group>
              )}
            </ModelCanvas>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-zinc-600">
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
          {preview?.ok && (
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
        <div className="flex w-[380px] shrink-0 flex-col overflow-y-auto border-l border-zinc-800 bg-zinc-950">
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
                      onChange={(v) =>
                        setScaleXYZ((s) => s.map((x, j) => (j === i ? v : x)) as [number, number, number])
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
              hint="cut with an axis plane and keep one side — click 'pick' then the model"
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

            {/* crop */}
            <ToolCard
              title="Crop box"
              hint="keep only geometry inside the box — pick corners on the model or type mm"
              onAdd={() => addOp({ op: "crop", args: { min: cropMin, max: cropMax } })}
              disabled={cropMax.some((mx, i) => mx <= cropMin[i])}
            >
              <div className="flex flex-col gap-1.5">
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

            {/* boolean */}
            <ToolCard
              title="Boolean add / cut"
              hint={
                boolTooDense
                  ? `mesh has ${faces?.toLocaleString()} faces (> 200k) — decimate first`
                  : "place a primitive on the surface — click 'place' then the model"
              }
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
                    <span className="w-14 text-[11px] text-zinc-500">size</span>
                    <NumInput value={boolSize} onChange={setBoolSize} step={1} className="flex-1" />
                    <span className="text-xs text-zinc-600">mm</span>
                  </div>
                )}
                {boolShape !== "box" && (
                  <div className="flex items-center gap-1.5">
                    <span className="w-14 text-[11px] text-zinc-500">radius</span>
                    <NumInput value={boolR} onChange={setBoolR} step={0.5} className="flex-1" />
                    <span className="text-xs text-zinc-600">mm</span>
                  </div>
                )}
                {boolShape === "cylinder" && (
                  <>
                    <div className="flex items-center gap-1.5">
                      <span className="w-14 text-[11px] text-zinc-500">height</span>
                      <NumInput value={boolH} onChange={setBoolH} step={1} className="flex-1" />
                      <span className="text-xs text-zinc-600">mm</span>
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
                {boolShape === "box" && (
                  <div className="flex items-center gap-1.5">
                    <span className="w-14 text-[11px] text-zinc-500">rotate z</span>
                    <NumInput value={boolRotZ} onChange={setBoolRotZ} step={15} className="flex-1" />
                    <span className="text-xs text-zinc-600">°</span>
                  </div>
                )}
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
                    onClick={() => {
                      setOps((o) => o.slice(0, -1));
                      setPreview(null);
                    }}
                  >
                    <Undo2 className="h-4 w-4" />
                  </IconBtn>
                  <IconBtn
                    title="clear all ops"
                    danger
                    disabled={!ops.length}
                    onClick={() => {
                      setOps([]);
                      setPreview(null);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </IconBtn>
                </div>
              }
            >
              op stack ({ops.length})
            </SectionTitle>
            {!ops.length && (
              <div className="text-xs text-zinc-600">
                No operations yet — build a stack with the tools, then Preview / Apply.
              </div>
            )}
            <div className="flex max-h-44 flex-col gap-1 overflow-y-auto">
              {ops.map((o, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1 text-xs"
                >
                  <span className="w-4 text-zinc-600">{i + 1}</span>
                  <span className="min-w-0 flex-1 truncate text-zinc-300" title={JSON.stringify(o.args)}>
                    {opLabel(o)}
                  </span>
                  <IconBtn
                    title="remove op"
                    danger
                    onClick={() => {
                      setOps((os) => os.filter((_, j) => j !== i));
                      setPreview(null);
                    }}
                  >
                    <Trash2 className="h-3 w-3" />
                  </IconBtn>
                </div>
              ))}
            </div>
            {dirty && ops.length > 0 && (
              <button
                className="mt-2 text-[11px] text-zinc-500 underline hover:text-zinc-300"
                onClick={() => {
                  api.saveEditorState(id, { ops }).then(
                    () => {
                      toast("draft saved", "success");
                      qc.invalidateQueries({ queryKey: ["version", id] });
                    },
                    (e) => toast(String(e), "error")
                  );
                }}
              >
                save draft to version
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* overlay helpers */
function SlicePlane({ axis, at, show }: { axis: Axis; at: number; show: boolean }) {
  if (!show) return null;
  const pos: [number, number, number] = [0, 0, 0];
  pos["xyz".indexOf(axis)] = at;
  const rot: [number, number, number] =
    axis === "x" ? [0, Math.PI / 2, 0] : axis === "y" ? [Math.PI / 2, 0, 0] : [0, 0, 0];
  return (
    <mesh position={pos} rotation={rot}>
      <planeGeometry args={[400, 400]} />
      <meshBasicMaterial color="#f59e0b" transparent opacity={0.15} side={2} depthTest={false} />
    </mesh>
  );
}
