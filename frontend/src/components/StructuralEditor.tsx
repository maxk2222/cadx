import {
  Brush,
  Crosshair,
  Eraser,
  Loader2,
  Plus,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { api } from "../lib/api";
import type {
  FemPreviewResult,
  LoadCase,
  RegionSpec,
  VersionDetail,
} from "../lib/types";
import ModelCanvas, { type PickInfo } from "./editor/ModelCanvas";
import RegionFill, { type RegionFillEntry } from "./editor/RegionFill";
import { MATERIAL_OPTIONS } from "./OptionsPanel";
import { Btn, Chip, Field, IconBtn, inputCls, useToast } from "./ui";

/* ------------------------------------------------------------------ */
/* model                                                               */
/* ------------------------------------------------------------------ */

type RegionColor = "fixed" | "load" | "keep";
const COLORS: Record<RegionColor, string> = {
  fixed: "#3b82f6",
  load: "#ef4444",
  keep: "#22c55e",
};

/** Where the next brush stroke (or center-pick) appends. */
type PaintTarget =
  | { kind: "fixed"; ci: number; ri: number }
  | { kind: "load"; ci: number; li: number }
  | { kind: "keep"; ri: number }
  | null;

type Mode = "paint" | "erase";

const num = (s: string, fb = 0) => {
  const n = Number(s);
  return Number.isFinite(n) ? n : fb;
};

const FACES = ["-x", "+x", "-y", "+y", "-z", "+z"] as const;
const FACE_LABELS: Record<string, string> = {
  "-x": "left (−X)",
  "+x": "right (+X)",
  "-y": "back (−Y)",
  "+y": "front (+Y)",
  "-z": "bottom (−Z)",
  "+z": "top (+Z)",
};
const CYL_AXES = ["x", "y", "z"] as const;
const REGION_KINDS: { id: NonNullable<RegionSpec["kind"]>; label: string }[] = [
  { id: "face", label: "face" },
  { id: "surface", label: "surface" },
  { id: "area", label: "area" },
  { id: "cylinder", label: "cylinder" },
  { id: "box", label: "box" },
  { id: "hole", label: "hole" },
  { id: "all", label: "all" },
];

function emptyPointsRegion(radius: number): RegionSpec {
  return { kind: "points", points: [], radius_mm: radius };
}

function newRegion(kind: NonNullable<RegionSpec["kind"]>): RegionSpec {
  switch (kind) {
    case "face":
      return { kind: "face", axis: "-x" };
    case "surface":
      return { kind: "surface", axis: "+y" };
    case "area":
      return { kind: "area", center: [0, 0], diameter_mm: 20 };
    case "cylinder":
      return { kind: "cylinder", axis: "z", center: [0, 0, 0], diameter_mm: 10 };
    case "box":
      return { kind: "box", center: [0, 0, 5], diameter_mm: 10 };
    case "hole":
      return { kind: "hole", hole_index: 1 };
    case "all":
      return { kind: "all" };
    default:
      return { kind: "face", axis: "-x" };
  }
}

function regionLabel(r: RegionSpec): string {
  switch (r.kind) {
    case "points":
      return `painted · ${r.points?.length ?? 0} pts`;
    case "face":
      return `face ${r.axis ?? "?"}`;
    case "surface":
      return `surface ${r.axis ?? "?"}`;
    case "area":
      return `area Ø${r.diameter_mm ?? "?"}mm`;
    case "cylinder":
      return `cylinder ${r.axis ?? "z"} Ø${r.diameter_mm ?? "?"}mm`;
    case "box":
      return `box Ø${r.diameter_mm ?? "?"}mm`;
    case "hole":
      return `hole #${r.hole_index ?? "?"}`;
    case "all":
      return "whole part";
    default:
      return `${r.kind ?? "region"}${r.description ? ` · ${r.description}` : ""}`;
  }
}

/* ------------------------------------------------------------------ */
/* number input (commits on blur/enter — no mid-typing coercion)       */
/* ------------------------------------------------------------------ */

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
  const [txt, setTxt] = useState(String(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setTxt(String(value));
  }, [value, editing]);
  const commit = () => {
    setEditing(false);
    const n = Number(txt);
    if (Number.isFinite(n)) onChange(n);
    else setTxt(String(value));
  };
  return (
    <input
      type="number"
      step={step}
      className={`${inputCls} px-1.5 py-0.5 text-[11px] ${className}`}
      value={editing ? txt : String(value)}
      onFocus={() => {
        setEditing(true);
        setTxt(String(value));
      }}
      onChange={(e) => setTxt(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

/* ------------------------------------------------------------------ */
/* region row — label + kind-specific fields + expandable details      */
/* ------------------------------------------------------------------ */

function RegionRow({
  r,
  color,
  painting,
  hasCenterPick,
  anchors,
  onPaint,
  onPickCenter,
  onChange,
  onDelete,
}: {
  r: RegionSpec;
  color: string;
  painting: boolean;
  /** whether this region kind supports click-to-place center */
  hasCenterPick?: boolean;
  anchors?: { x: number; y: number; dia?: number }[];
  onPaint?: () => void;
  onPickCenter?: () => void;
  onChange: (r: RegionSpec) => void;
  onDelete: () => void;
}) {
  const [open, setOpen] = useState(false);
  const kind = r.kind ?? "face";
  const setA = (patch: Partial<RegionSpec>) => onChange({ ...r, ...patch });

  return (
    <div
      className={`mb-1 rounded border px-1.5 py-1 text-xs ${
        painting ? "border-amber-600 bg-amber-950/30" : "border-zinc-800 bg-zinc-900/60"
      }`}
    >
      <div className="flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
        {kind === "face" || kind === "surface" ? (
          <select
            className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
            value={r.axis ?? "+x"}
            onChange={(e) => setA({ axis: e.target.value })}
          >
            {FACES.map((f) => (
              <option key={f} value={f}>
                {kind === "face" ? "face" : "surf"} {FACE_LABELS[f]}
              </option>
            ))}
          </select>
        ) : kind === "cylinder" ? (
          <select
            className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
            value={r.axis ?? "z"}
            onChange={(e) => setA({ axis: e.target.value })}
          >
            {CYL_AXES.map((a) => (
              <option key={a} value={a}>
                cylinder ∥ {a.toUpperCase()}
              </option>
            ))}
          </select>
        ) : kind === "hole" && anchors?.length ? (
          <select
            className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
            value={r.hole_index ?? 1}
            onChange={(e) => setA({ hole_index: Number(e.target.value) })}
          >
            {anchors.map((a, i) => (
              <option key={i} value={i + 1}>
                hole #{i + 1} Ø{a.dia ? a.dia.toFixed(1) : "?"} @ ({a.x.toFixed(1)}, {a.y.toFixed(1)})
              </option>
            ))}
          </select>
        ) : (
          <button
            className="min-w-0 flex-1 truncate text-left text-zinc-300 hover:text-zinc-100"
            title={`${regionLabel(r)} — click for details`}
            onClick={() => setOpen((o) => !o)}
          >
            {r.description || regionLabel(r)}
          </button>
        )}
        {hasCenterPick && onPickCenter && (
          <button
            title="pick region center on the model"
            onClick={onPickCenter}
            className="rounded p-0.5 text-zinc-500 hover:text-zinc-200"
          >
            <Crosshair className="h-3.5 w-3.5" />
          </button>
        )}
        {kind === "points" && onPaint && (
          <button
            title={painting ? "stop painting" : "paint on the model"}
            onClick={onPaint}
            className={`rounded p-0.5 ${
              painting ? "text-amber-300" : "text-zinc-500 hover:text-zinc-200"
            }`}
          >
            <Brush className="h-3.5 w-3.5" />
          </button>
        )}
        <button title="remove region" onClick={onDelete} className="text-zinc-600 hover:text-red-400">
          <Trash2 className="h-3 w-3" />
        </button>
      </div>

      {/* quick row for painted regions: radius slider */}
      {kind === "points" && (
        <div className="mt-1 flex items-center gap-1.5 pl-4">
          <input
            type="range"
            min={1}
            max={30}
            step={0.5}
            title={`selection radius ${r.radius_mm ?? 3} mm`}
            className="w-20 accent-brand-500"
            value={r.radius_mm ?? 3}
            onChange={(e) => setA({ radius_mm: num(e.target.value, 3) })}
          />
          <span className="w-10 text-[10px] text-zinc-500">{r.radius_mm ?? 3}mm</span>
          <button
            className="ml-auto text-[10px] text-zinc-600 underline hover:text-zinc-300"
            onClick={() => setOpen((o) => !o)}
          >
            details
          </button>
        </div>
      )}

      {/* expanded details */}
      {open && (
        <div className="mt-1 flex flex-col gap-1 border-t border-zinc-800 pl-4 pt-1">
          {(kind === "cylinder" || kind === "box" || kind === "area") && (
            <div className="flex items-center gap-1">
              <span className="w-14 text-[10px] text-zinc-500">center</span>
              {(r.center ?? [0, 0, 0]).map((c, i) => (
                <NumInput
                  key={i}
                  value={c}
                  step={0.5}
                  className="flex-1"
                  onChange={(n) =>
                    setA({ center: (r.center ?? [0, 0, 0]).map((x, j) => (j === i ? n : x)) })
                  }
                />
              ))}
            </div>
          )}
          {(kind === "cylinder" || kind === "box" || kind === "area") && (
            <div className="flex items-center gap-1">
              <span className="w-14 text-[10px] text-zinc-500">
                {kind === "box" ? "edge" : "dia"}
              </span>
              <NumInput
                value={r.diameter_mm ?? 10}
                step={1}
                className="flex-1"
                onChange={(n) => setA({ diameter_mm: n })}
              />
              <span className="text-[10px] text-zinc-600">mm</span>
            </div>
          )}
          <div className="flex items-center gap-1">
            <span className="w-14 text-[10px] text-zinc-500">margin</span>
            <NumInput
              value={r.margin_mm ?? 2}
              step={0.5}
              className="flex-1"
              onChange={(n) => setA({ margin_mm: n })}
            />
            <span className="text-[10px] text-zinc-600">mm</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-14 text-[10px] text-zinc-500">label</span>
            <input
              className={`${inputCls} flex-1 px-1.5 py-0.5 text-[11px]`}
              placeholder="e.g. mounting holes"
              value={r.description ?? ""}
              onChange={(e) => setA({ description: e.target.value })}
            />
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* add-region button row                                               */
/* ------------------------------------------------------------------ */

function AddRegionButtons({
  onPaint,
  onAdd,
  hasAnchors,
  paintLabel,
}: {
  onPaint: () => void;
  onAdd: (r: RegionSpec) => void;
  hasAnchors: boolean;
  paintLabel: string;
}) {
  return (
    <div className="mb-2 flex flex-wrap gap-1">
      <button
        className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
        onClick={onPaint}
      >
        <Brush className="h-3 w-3" /> {paintLabel}
      </button>
      {REGION_KINDS.filter((k) => k.id !== "hole" || hasAnchors).map((k) => (
        <button
          key={k.id}
          className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
          onClick={() => onAdd(newRegion(k.id))}
        >
          <Plus className="h-3 w-3" /> {k.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* overlay: point cloud (model-space mm)                               */
/* ------------------------------------------------------------------ */

function PointCloud({
  pts,
  color,
  size,
  opacity = 0.95,
  depthTest = false,
}: {
  pts: number[][];
  color: string;
  size: number;
  opacity?: number;
  depthTest?: boolean;
}) {
  const arr = useMemo(() => new Float32Array(pts.flat()), [pts]);
  if (!pts.length) return null;
  return (
    <points key={pts.length}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[arr, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color={color}
        size={size}
        sizeAttenuation
        transparent
        opacity={opacity}
        depthTest={depthTest}
      />
    </points>
  );
}

/* ------------------------------------------------------------------ */
/* main component                                                      */
/* ------------------------------------------------------------------ */

export default function StructuralEditor({
  open,
  onClose,
  version,
  canTopopt,
  onJobStarted,
}: {
  open: boolean;
  onClose: () => void;
  version: VersionDetail;
  canTopopt: boolean;
  onJobStarted: (r: { version_id: string; job_id: string }) => void;
}) {
  const toast = useToast();
  const [cases, setCases] = useState<LoadCase[]>([]);
  const [keep, setKeep] = useState<RegionSpec[]>([]);
  const [volfrac, setVolfrac] = useState(0.4);
  const [material, setMaterial] = useState("");
  const [brushR, setBrushR] = useState(4);
  const [mode, setMode] = useState<Mode>("paint");
  const [target, setTarget] = useState<PaintTarget>(null);
  const [centerPick, setCenterPick] = useState<Exclude<PaintTarget, null> | null>(null);
  const [hover, setHover] = useState<PickInfo | null>(null);
  const [model, setModel] = useState<THREE.Group | null>(null);
  const [preview, setPreview] = useState<FemPreviewResult | null>(null);
  const [showCells, setShowCells] = useState(true);
  const [busy, setBusy] = useState<"preview" | "fem" | "opt" | null>(null);
  const lastPt = useRef<[number, number, number] | null>(null);
  const lastPickT = useRef(0);
  const undoStack = useRef<{ t: Exclude<PaintTarget, null>; prev: number[][] }[]>([]);

  // init once per open/version — persisted GUI state wins over spec defaults
  useEffect(() => {
    if (!open) return;
    const saved = (version.options as Record<string, unknown> | undefined)?.ui_editor as
      | {
          load_cases?: LoadCase[];
          keep_regions?: RegionSpec[];
          material?: string;
          volfrac?: number;
        }
      | undefined;
    if (saved?.load_cases?.length) {
      setCases(saved.load_cases);
      setKeep(saved.keep_regions ?? []);
      setMaterial(saved.material ?? "");
      setVolfrac(saved.volfrac ?? 0.4);
    } else if (version.spec?.load_cases?.length) {
      setCases(version.spec.load_cases);
      setKeep(version.spec.optimize?.keep_regions ?? []);
      setMaterial(version.options?.material ?? "");
    } else {
      setCases([{ name: "case1", fixed: [], loads: [] }]);
      setKeep([]);
    }
    setPreview(null);
    setTarget(null);
    setCenterPick(null);
    undoStack.current = [];
    lastPt.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, version.id]);

  /* --- region list mutation helpers --- */

  const setCase = (ci: number, patch: Partial<LoadCase>) =>
    setCases((cs) => cs.map((c, i) => (i === ci ? { ...c, ...patch } : c)));

  const getRegion = (t: Exclude<PaintTarget, null>): RegionSpec | null => {
    if (t.kind === "fixed") return cases[t.ci]?.fixed[t.ri] ?? null;
    if (t.kind === "load") return cases[t.ci]?.loads[t.li]?.region ?? null;
    return keep[t.ri] ?? null;
  };

  const setRegion = (t: Exclude<PaintTarget, null>, r: RegionSpec) => {
    if (t.kind === "fixed") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { fixed: c.fixed.map((x, i) => (i === t.ri ? r : x)) });
    } else if (t.kind === "load") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { loads: c.loads.map((x, i) => (i === t.li ? { ...x, region: r } : x)) });
    } else {
      setKeep((k) => k.map((x, i) => (i === t.ri ? r : x)));
    }
  };

  const delRegion = (t: Exclude<PaintTarget, null>) => {
    if (t.kind === "fixed") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { fixed: c.fixed.filter((_, i) => i !== t.ri) });
    } else if (t.kind === "load") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { loads: c.loads.filter((_, i) => i !== t.li) });
    } else {
      setKeep((k) => k.filter((_, i) => i !== t.ri));
    }
    if (target && JSON.stringify(target) === JSON.stringify(t)) setTarget(null);
  };

  const pushUndo = (t: Exclude<PaintTarget, null>, prev: number[][]) => {
    undoStack.current.push({ t, prev });
    if (undoStack.current.length > 60) undoStack.current.shift();
  };

  const undo = () => {
    const e = undoStack.current.pop();
    if (!e) return;
    const r = getRegion(e.t);
    if (r?.kind === "points") setRegion(e.t, { ...r, points: e.prev });
  };

  /** Append a fresh painted region and arm the brush on it. */
  const startPaint = (t: Exclude<PaintTarget, null>, append: (r: RegionSpec) => void) => {
    append(emptyPointsRegion(brushR));
    setTarget(t);
    setMode("paint");
    setCenterPick(null);
    lastPt.current = null;
  };

  /* --- painting / erasing / center-pick --- */

  const onPick = (info: PickInfo) => {
    // center placement for geometric regions
    if (centerPick) {
      const r = getRegion(centerPick);
      const p = info.point;
      if (r) {
        const k = r.kind;
        const center =
          k === "area" ? [p[0], p[1]] : [p[0], p[1], p[2]];
        setRegion(centerPick, { ...r, center });
      }
      setCenterPick(null);
      return;
    }
    if (!target) return;
    const r = getRegion(target);
    if (!r || r.kind !== "points") return;
    const p = info.point;
    const rad = r.radius_mm ?? brushR;
    const now = Date.now();
    const strokeStart = now - lastPickT.current > 350;
    lastPickT.current = now;

    // a pause >350ms counts as a new stroke — snapshot for undo
    if (strokeStart) {
      pushUndo(target, r.points ?? []);
      lastPt.current = null;
    }

    if (mode === "erase") {
      const keepPts = (r.points ?? []).filter(
        (q) => Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) > rad
      );
      setRegion(target, { ...r, points: keepPts });
      return;
    }

    if (lastPt.current) {
      const d = Math.hypot(
        p[0] - lastPt.current[0],
        p[1] - lastPt.current[1],
        p[2] - lastPt.current[2]
      );
      if (d < rad * 0.4) return;
    }
    lastPt.current = p;
    const pts = [...(r.points ?? []), p];
    // running average of face normals — used by FEM for pressure direction
    const prev = (r.normal?.length === 3 ? r.normal : info.normal) as number[];
    const n = [
      prev[0] * 0.8 + info.normal[0] * 0.2,
      prev[1] * 0.8 + info.normal[1] * 0.2,
      prev[2] * 0.8 + info.normal[2] * 0.2,
    ];
    setRegion(target, { ...r, points: pts.slice(-4000), normal: n });
  };

  /* --- payload --- */

  const valid = cases.length > 0 && cases.every(
    (c) =>
      c.fixed.length > 0 &&
      c.loads.length > 0 &&
      c.loads.every(
        (l) =>
          (l.force_n?.some((f) => Math.abs(f) > 1e-9) ?? false) ||
          (l.pressure_mpa ?? 0) > 0
      )
  );

  const persist = () =>
    api.saveEditorState(version.id, {
      load_cases: cases,
      keep_regions: keep,
      material,
      volfrac,
    });

  // autosave draft state (merged into ui_editor server-side)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!open) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      persist().catch(() => {});
    }, 1500);
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, keep, material, volfrac, open]);

  const checkRegions = async () => {
    setBusy("preview");
    setPreview(null);
    try {
      await persist();
      const r = await api.femPreview(version.id, {
        load_cases: cases,
        keep_regions: keep,
        render_pngs: false,
      });
      setPreview(r);
      setShowCells(true);
      if (!r.ok) toast(r.error ?? "region check failed", "error");
    } catch (e) {
      toast(String(e), "error");
    } finally {
      setBusy(null);
    }
  };

  const runFem = async () => {
    setBusy("fem");
    try {
      await persist();
      const r = await api.runFem(version.id, {
        load_cases: cases,
        keep_regions: keep,
        material: material || undefined,
      });
      onJobStarted(r);
    } catch (e) {
      toast(String(e), "error");
      setBusy(null);
    }
  };

  const runTopopt = async () => {
    setBusy("opt");
    try {
      await persist();
      const r = await api.optimize(version.id, {
        load_cases: cases,
        optimize: {
          enabled: true,
          target_volume_fraction: volfrac,
          keep_regions: keep,
        },
        material: material || undefined,
      });
      onJobStarted(r);
    } catch (e) {
      toast(String(e), "error");
      setBusy(null);
    }
  };

  /* --- overlays --- */

  /** painted points → vertex-coloured fills on the model surface */
  const fills = useMemo<RegionFillEntry[]>(() => {
    const out: RegionFillEntry[] = [];
    const add = (r: RegionSpec, color: string) => {
      if (r.kind === "points" && r.points?.length)
        out.push({ points: r.points, radius: r.radius_mm ?? 3, color });
    };
    cases.forEach((c) => {
      c.fixed.forEach((r) => add(r, COLORS.fixed));
      c.loads.forEach((l) => add(l.region, COLORS.load));
    });
    keep.forEach((r) => add(r, COLORS.keep));
    return out;
  }, [cases, keep]);

  /** non-painted geometric regions → center markers */
  const markers = useMemo(() => {
    const out: { p: [number, number, number]; color: string }[] = [];
    const add = (r: RegionSpec, color: string) => {
      const k = r.kind;
      if ((k === "area" || k === "cylinder" || k === "box") && r.center?.length)
        out.push({
          p: [r.center[0] ?? 0, r.center[1] ?? 0, k === "area" ? 0 : (r.center[2] ?? 0)],
          color,
        });
    };
    cases.forEach((c) => {
      c.fixed.forEach((r) => add(r, COLORS.fixed));
      c.loads.forEach((l) => add(l.region, COLORS.load));
    });
    keep.forEach((r) => add(r, COLORS.keep));
    return out;
  }, [cases, keep]);

  const anchors = (
    version.analysis as { anchors?: { holes?: { x: number; y: number; dia?: number }[] } } | null
  )?.anchors?.holes;
  const glb = version.artifacts.find((a) => a.kind === "glb");

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;

  const sameTarget = (t: Exclude<PaintTarget, null>) =>
    target != null && JSON.stringify(target) === JSON.stringify(t);
  const sameCenterPick = (t: Exclude<PaintTarget, null>) =>
    centerPick != null && JSON.stringify(centerPick) === JSON.stringify(t);

  const activeRegion = target ? getRegion(target) : null;
  const cursorR = mode === "erase" ? brushR : (activeRegion?.radius_mm ?? brushR);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-zinc-950">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
        <h2 className="text-sm font-semibold">Structural editor — v{version.number}</h2>
        <span className="text-xs text-zinc-500">
          paint support / load / keep regions directly on the surface
        </span>
        <span className="flex-1" />
        {(target || centerPick) && (
          <Chip tone="amber">
            <Brush className="h-3 w-3" />{" "}
            {centerPick
              ? "click to place region center"
              : `${mode === "erase" ? "erasing" : "painting"} ${
                  target!.kind === "fixed" ? "support" : target!.kind === "load" ? "load" : "keep"
                } — drag on the model`}
          </Chip>
        )}
        {target && (
          <>
            <div className="flex overflow-hidden rounded border border-zinc-700 text-[11px]">
              {(["paint", "erase"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`flex items-center gap-1 px-2 py-1 ${
                    mode === m ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {m === "paint" ? <Brush className="h-3 w-3" /> : <Eraser className="h-3 w-3" />}
                  {m}
                </button>
              ))}
            </div>
            <IconBtn title="undo last stroke" onClick={undo} disabled={!undoStack.current.length}>
              <Undo2 className="h-3.5 w-3.5" />
            </IconBtn>
          </>
        )}
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          brush
          <input
            type="range"
            min={1}
            max={30}
            step={0.5}
            className="w-24 accent-brand-500"
            value={brushR}
            onChange={(e) => setBrushR(num(e.target.value, 4))}
          />
          <span className="w-10 tabular-nums">{brushR} mm</span>
        </label>
        <IconBtn title="close (Esc)" onClick={onClose}>
          <X className="h-4 w-4" />
        </IconBtn>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* canvas */}
        <div className="relative min-w-0 flex-1">
          {glb ? (
            <ModelCanvas
              url={glb.url}
              pickEnabled={!!target || !!centerPick}
              onPick={onPick}
              onHover={setHover}
              onModel={setModel}
            >
              <RegionFill model={model} entries={fills} />
              {markers.map((m, i) => (
                <mesh key={i} position={m.p}>
                  <sphereGeometry args={[1, 16, 12]} />
                  <meshBasicMaterial color={m.color} depthTest={false} transparent opacity={0.9} />
                </mesh>
              ))}
              {/* resolved cells from the last region check (voxel-domain truth) */}
              {showCells &&
                preview?.ok &&
                preview.cells?.map((c, i) => (
                  <group key={i}>
                    <PointCloud pts={c.fixed} color={COLORS.fixed} size={1.2} opacity={0.7} />
                    <PointCloud pts={c.load} color={COLORS.load} size={1.2} opacity={0.7} />
                    <PointCloud pts={c.keep} color={COLORS.keep} size={1.2} opacity={0.7} />
                  </group>
                ))}
              {(target || centerPick) && hover && (
                <mesh position={hover.point}>
                  <sphereGeometry args={[cursorR, 24, 16]} />
                  <meshBasicMaterial
                    color={
                      centerPick
                        ? "#f59e0b"
                        : mode === "erase"
                          ? "#a1a1aa"
                          : COLORS[target!.kind]
                    }
                    transparent
                    opacity={0.25}
                    depthTest={false}
                  />
                </mesh>
              )}
            </ModelCanvas>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-zinc-600">
              no mesh artifact
            </div>
          )}
          {/* legend */}
          <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/90 px-3 py-1.5 text-[11px] backdrop-blur">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: COLORS.fixed }} />
              support
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: COLORS.load }} />
              load
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: COLORS.keep }} />
              keep
            </span>
            {preview?.ok && !!preview.cells?.length && (
              <label className="ml-2 flex items-center gap-1 text-zinc-500">
                <input
                  type="checkbox"
                  className="accent-brand-500"
                  checked={showCells}
                  onChange={(e) => setShowCells(e.target.checked)}
                />
                resolved cells
              </label>
            )}
          </div>
        </div>

        {/* right panel */}
        <div className="flex w-[420px] shrink-0 flex-col overflow-y-auto border-l border-zinc-800">
          <div className="flex-1 space-y-3 p-3">
            {cases.map((c, ci) => (
              <div key={ci} className="rounded-lg border border-zinc-800 p-2.5">
                <div className="mb-2 flex items-center gap-2">
                  <input
                    className={`${inputCls} w-32 px-2 py-1 text-xs`}
                    value={c.name}
                    onChange={(e) => setCase(ci, { name: e.target.value })}
                  />
                  <span className="flex-1" />
                  {cases.length > 1 && (
                    <button
                      title="remove case"
                      className="text-zinc-600 hover:text-red-400"
                      onClick={() => setCases((cs) => cs.filter((_, i) => i !== ci))}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>

                {/* fixed regions */}
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  support regions
                </div>
                {c.fixed.map((r, ri) => (
                  <RegionRow
                    key={ri}
                    r={r}
                    color={COLORS.fixed}
                    painting={sameTarget({ kind: "fixed", ci, ri })}
                    hasCenterPick={r.kind === "area" || r.kind === "cylinder" || r.kind === "box"}
                    anchors={anchors}
                    onPaint={() =>
                      setTarget(sameTarget({ kind: "fixed", ci, ri }) ? null : { kind: "fixed", ci, ri })
                    }
                    onPickCenter={() =>
                      setCenterPick(sameCenterPick({ kind: "fixed", ci, ri }) ? null : { kind: "fixed", ci, ri })
                    }
                    onChange={(nr) => setRegion({ kind: "fixed", ci, ri }, nr)}
                    onDelete={() => delRegion({ kind: "fixed", ci, ri })}
                  />
                ))}
                <AddRegionButtons
                  paintLabel="paint support"
                  hasAnchors={!!anchors?.length}
                  onPaint={() =>
                    startPaint({ kind: "fixed", ci, ri: c.fixed.length }, (r) =>
                      setCase(ci, { fixed: [...c.fixed, r] })
                    )
                  }
                  onAdd={(r) => {
                    setCase(ci, { fixed: [...c.fixed, r] });
                    const t = { kind: "fixed" as const, ci, ri: c.fixed.length };
                    if (r.kind === "area" || r.kind === "cylinder" || r.kind === "box")
                      setCenterPick(t);
                  }}
                />

                {/* loads */}
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  loads
                </div>
                {c.loads.map((l, li) => {
                  const force = l.force_n ?? [0, 0, 0];
                  const isPressure = (l.pressure_mpa ?? 0) > 0 && !force.some((f) => Math.abs(f) > 1e-9);
                  const setLoad = (patch: Partial<typeof l>) =>
                    setCase(ci, {
                      loads: c.loads.map((x, i) => (i === li ? { ...x, ...patch } : x)),
                    });
                  return (
                    <div key={li} className="mb-1.5">
                      <RegionRow
                        r={l.region}
                        color={COLORS.load}
                        painting={sameTarget({ kind: "load", ci, li })}
                        hasCenterPick={
                          l.region.kind === "area" ||
                          l.region.kind === "cylinder" ||
                          l.region.kind === "box"
                        }
                        anchors={anchors}
                        onPaint={() =>
                          setTarget(sameTarget({ kind: "load", ci, li }) ? null : { kind: "load", ci, li })
                        }
                        onPickCenter={() =>
                          setCenterPick(
                            sameCenterPick({ kind: "load", ci, li }) ? null : { kind: "load", ci, li }
                          )
                        }
                        onChange={(nr) => setRegion({ kind: "load", ci, li }, nr)}
                        onDelete={() => delRegion({ kind: "load", ci, li })}
                      />
                      <div className="mt-1 flex flex-col gap-1 pl-4">
                        {/* force vs pressure toggle */}
                        <div className="flex gap-1">
                          {(["force", "pressure"] as const).map((m) => (
                            <button
                              key={m}
                              onClick={() =>
                                setLoad(
                                  m === "force"
                                    ? { force_n: [0, 0, -50], pressure_mpa: null }
                                    : { force_n: [], pressure_mpa: 2 }
                                )
                              }
                              className={`flex-1 rounded border px-2 py-0.5 text-[10px] ${
                                (m === "pressure") === isPressure
                                  ? "border-brand-600 bg-brand-950/60 text-brand-300"
                                  : "border-zinc-700 text-zinc-500"
                              }`}
                            >
                              {m === "force" ? "force vector" : "pressure"}
                            </button>
                          ))}
                        </div>
                        {isPressure ? (
                          <div className="flex items-center gap-1.5">
                            <NumInput
                              value={l.pressure_mpa ?? 0}
                              step={0.5}
                              className="w-24"
                              onChange={(n) => setLoad({ pressure_mpa: n })}
                            />
                            <span className="text-[10px] text-zinc-500">
                              MPa — direction follows the region's surface normal
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-1">
                            {(["fx", "fy", "fz"] as const).map((ax, i) => (
                              <label key={ax} className="flex flex-1 items-center gap-0.5">
                                <span className="text-[9px] uppercase text-zinc-600">{ax}</span>
                                <NumInput
                                  value={force[i] ?? 0}
                                  step={5}
                                  className="w-full"
                                  onChange={(n) =>
                                    setLoad({ force_n: force.map((f, j) => (j === i ? n : f)) })
                                  }
                                />
                              </label>
                            ))}
                            <span className="text-[10px] text-zinc-500">N</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
                <AddRegionButtons
                  paintLabel="paint load"
                  hasAnchors={!!anchors?.length}
                  onPaint={() =>
                    startPaint({ kind: "load", ci, li: c.loads.length }, (r) =>
                      setCase(ci, {
                        loads: [...c.loads, { region: r, force_n: [0, 0, -50] }],
                      })
                    )
                  }
                  onAdd={(r) => {
                    setCase(ci, {
                      loads: [...c.loads, { region: r, force_n: [0, 0, -50] }],
                    });
                    const t = { kind: "load" as const, ci, li: c.loads.length };
                    if (r.kind === "area" || r.kind === "cylinder" || r.kind === "box")
                      setCenterPick(t);
                  }}
                />
              </div>
            ))}

            <button
              className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200"
              onClick={() =>
                setCases((cs) => [...cs, { name: `case${cs.length + 1}`, fixed: [], loads: [] }])
              }
            >
              <Plus className="h-3.5 w-3.5" /> add load case
            </button>

            {/* keep regions */}
            {canTopopt && (
              <div className="rounded-lg border border-zinc-800 p-2.5">
                <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  keep regions (TopOpt design-preserve)
                </div>
                {keep.map((r, ri) => (
                  <RegionRow
                    key={ri}
                    r={r}
                    color={COLORS.keep}
                    painting={sameTarget({ kind: "keep", ri })}
                    hasCenterPick={r.kind === "area" || r.kind === "cylinder" || r.kind === "box"}
                    anchors={anchors}
                    onPaint={() =>
                      setTarget(sameTarget({ kind: "keep", ri }) ? null : { kind: "keep", ri })
                    }
                    onPickCenter={() =>
                      setCenterPick(sameCenterPick({ kind: "keep", ri }) ? null : { kind: "keep", ri })
                    }
                    onChange={(nr) => setRegion({ kind: "keep", ri }, nr)}
                    onDelete={() => delRegion({ kind: "keep", ri })}
                  />
                ))}
                <AddRegionButtons
                  paintLabel="paint keep region"
                  hasAnchors={!!anchors?.length}
                  onPaint={() =>
                    startPaint({ kind: "keep", ri: keep.length }, (r) => setKeep((k) => [...k, r]))
                  }
                  onAdd={(r) => {
                    setKeep((k) => [...k, r]);
                    if (r.kind === "area" || r.kind === "cylinder" || r.kind === "box")
                      setCenterPick({ kind: "keep", ri: keep.length });
                  }}
                />
              </div>
            )}

            {/* settings */}
            <div className="grid grid-cols-2 gap-2">
              <Field label="Material" hint="empty = project default">
                <select
                  className={`${inputCls} px-2 py-1 text-xs`}
                  value={material}
                  onChange={(e) => setMaterial(e.target.value)}
                >
                  <option value="">— default —</option>
                  {MATERIAL_OPTIONS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </Field>
              {canTopopt && (
                <Field label={`Target volume ${Math.round(volfrac * 100)}%`}>
                  <input
                    type="range"
                    min={10}
                    max={90}
                    step={5}
                    className="mt-2 w-full accent-brand-500"
                    value={Math.round(volfrac * 100)}
                    onChange={(e) => setVolfrac(num(e.target.value, 40) / 100)}
                  />
                </Field>
              )}
            </div>

            {/* region check result */}
            {preview && (
              <div className="rounded-lg border border-zinc-800 p-2.5 text-xs">
                <div className="mb-1 font-semibold text-zinc-300">
                  region check{" "}
                  {preview.ok
                    ? `— voxel ${preview.dims.join("×")} @ ${preview.pitch_mm?.toFixed(2)}mm · ${preview.solid_elements} elements`
                    : "failed"}
                </div>
                {preview.cases.map((c, i) => (
                  <div key={i} className="mb-1 text-zinc-400">
                    <b className="text-zinc-300">{c.case}</b>: {c.fixed_nodes} fixed nodes,{" "}
                    {c.loaded_nodes} loaded nodes, {c.total_force_n.toFixed(1)} N
                    {c.warnings.map((w, j) => (
                      <div key={j} className="text-amber-400">
                        ⚠ {w}
                      </div>
                    ))}
                  </div>
                ))}
                {!!preview.pngs?.length && (
                  <div className="mt-2 grid grid-cols-2 gap-1.5">
                    {preview.pngs.map((u) => (
                      <img key={u} src={u} className="rounded border border-zinc-800" alt="" />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* actions */}
          <div className="flex items-center gap-2 border-t border-zinc-800 p-3">
            <Btn
              variant="ghost"
              disabled={!valid || !!busy}
              busy={busy === "preview"}
              onClick={checkRegions}
              title="resolve regions on the voxel domain — shows hit counts, warnings and a 3D overlay"
            >
              {busy === "preview" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Check regions
            </Btn>
            <span className="flex-1" />
            <Btn
              variant="ghost"
              className="border border-purple-800/60 text-purple-300 hover:bg-purple-950/40"
              disabled={!valid || !!busy}
              busy={busy === "fem"}
              onClick={runFem}
              title="run a FEM strength check on this version (no optimization)"
            >
              FEM check
            </Btn>
            {canTopopt && (
              <Btn
                variant="success"
                disabled={!valid || !!busy}
                busy={busy === "opt"}
                onClick={runTopopt}
                title="start SIMP topology optimization → new child version"
              >
                Optimize
              </Btn>
            )}
          </div>
          {!valid && (
            <div className="border-t border-zinc-800 bg-amber-950/30 px-3 py-1.5 text-[11px] text-amber-300">
              every case needs at least one support region and one load with force &gt; 0
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
