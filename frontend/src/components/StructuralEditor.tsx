import {
  Brush,
  Check,
  CircleDot,
  Crosshair,
  Eraser,
  Layers,
  Loader2,
  Plus,
  Spline,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { api, errMsg } from "../lib/api";
import { num, readStoredAnchors, readUiEditor } from "../lib/editorState";
import {
  buildFacets,
  facetAtHit,
  facetGeometry,
  facetLoop,
  facetPickPoints,
  facetRadius,
  resampleLoop,
  type Facet,
  type FacetSet,
} from "../lib/facets";
import type {
  FeatureAnchor,
  FemPreviewResult,
  LoadCase,
  LoadSpec,
  RegionSpec,
  VersionDetail,
} from "../lib/types";
import {
  AnchorMarker,
  anchorMouth,
  anchorPos,
  axisVec,
  dirQuat,
} from "./editor/anchors";
import ModelCanvas, { type PickInfo } from "./editor/ModelCanvas";
import RegionFill, { type RegionFillEntry } from "./editor/RegionFill";
import { MATERIAL_OPTIONS } from "./OptionsPanel";
import { Btn, Chip, Field, IconBtn, inputCls, NumInput, useToast } from "./ui";

/* ------------------------------------------------------------------ */
/* model                                                               */
/* ------------------------------------------------------------------ */

type RegionColor = "fixed" | "load" | "keep" | "void" | "anchor";
const COLORS: Record<RegionColor, string> = {
  fixed: "#3b82f6",
  load: "#ef4444",
  keep: "#22c55e",
  void: "#a855f7",
  anchor: "#22d3ee",
};
const GIZMO_COLORS = { force: "#f59e0b", moment: "#f472b6", preload: "#facc15" };

/** Where the next brush stroke (or center-pick) appends. */
type PaintTarget =
  | { kind: "fixed"; ci: number; ri: number }
  | { kind: "load"; ci: number; li: number }
  | { kind: "keep"; ri: number }
  | { kind: "void"; ri: number }
  | null;

type Mode = "paint" | "erase" | "face" | "edge";

/** Which region list a feature pick (hole/thread/bore) appends to. */
type FeatureTarget =
  | { kind: "fixed"; ci: number }
  | { kind: "load"; ci: number }
  | { kind: "keep" }
  | { kind: "void" }
  | null;

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
const ANCHOR_KINDS = new Set(["hole", "anchor", "bore"]);
const REGION_KINDS: { id: NonNullable<RegionSpec["kind"]>; label: string }[] = [
  { id: "face", label: "face" },
  { id: "surface", label: "surface" },
  { id: "area", label: "area" },
  { id: "cylinder", label: "cylinder" },
  { id: "box", label: "box" },
  { id: "anchor", label: "anchor" },
  { id: "bore", label: "bore" },
  { id: "hole", label: "hole" },
  { id: "all", label: "all" },
];

/** nearest feature anchor to a surface pick: the point must lie inside the
 *  anchor's bore corridor (perpendicular distance < dia/2 + snap) and within
 *  the part span along the axis. Returns null when nothing is close — a
 *  plain surface click should never silently snap to a far-away hole. */
function nearestAnchorAt(
  p: [number, number, number],
  anchors: FeatureAnchor[],
  box: THREE.Box3 | null,
  snapMm = 2.5
): FeatureAnchor | null {
  let best: FeatureAnchor | null = null;
  let bd = Infinity;
  for (const a of anchors) {
    const ap = anchorPos(a, box);
    if (!ap) continue;
    const av = axisVec(a.axis);
    const d = [p[0] - ap[0], p[1] - ap[1], p[2] - ap[2]];
    const t = d[0] * av[0] + d[1] * av[1] + d[2] * av[2];
    const perp = Math.hypot(d[0] - t * av[0], d[1] - t * av[1], d[2] - t * av[2]);
    const tol = (a.dia || 0) / 2 + snapMm;
    if (perp > tol) continue;
    // axial bound: the pick must lie within the part span along the axis
    if (box) {
      const i = av.findIndex((v) => v !== 0);
      if (i >= 0) {
        const lo = box.min.getComponent(i) - snapMm;
        const hi = box.max.getComponent(i) + snapMm;
        if (ap[i] + t < lo || ap[i] + t > hi) continue;
      }
    }
    // prefer corridor hits — penalize picks far along the axis
    const score = perp + Math.abs(t) * 0.05;
    if (score < bd) {
      bd = score;
      best = a;
    }
  }
  return best;
}

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
    case "anchor":
      return { kind: "anchor", hole_index: 1, margin_mm: 2 };
    case "bore":
      return { kind: "bore", hole_index: 1 };
    case "all":
      return { kind: "all" };
    default:
      return { kind: "face", axis: "-x" };
  }
}

function regionLabel(r: RegionSpec, anchors?: FeatureAnchor[]): string {
  switch (r.kind) {
    case "points":
      return r.description || `painted · ${r.points?.length ?? 0} pts`;
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
    case "anchor":
    case "bore": {
      const a = anchors?.find((x) => x.index === r.hole_index);
      const det = a ? ` ${a.kind} ${a.axis} Ø${a.dia.toFixed(1)}` : "";
      return `${r.kind} #${r.hole_index ?? "?"}${det}`;
    }
    case "all":
      return "whole part";
    default:
      return `${r.kind ?? "region"}${r.description ? ` · ${r.description}` : ""}`;
  }
}

/* ------------------------------------------------------------------ */
/* load kinds — which fields of a LoadSpec are active                  */
/* ------------------------------------------------------------------ */

type LoadKind =
  | "force"
  | "pressure"
  | "bearing"
  | "moment"
  | "preload"
  | "remote"
  | "gravity";

const LOAD_KINDS: { id: LoadKind; label: string; hint: string }[] = [
  { id: "force", label: "force", hint: "distributed [N] over the region" },
  { id: "pressure", label: "pressure", hint: "MPa, follows the region normal" },
  { id: "bearing", label: "bearing", hint: "cosine bolt/bearing load on a bore wall" },
  { id: "moment", label: "moment", hint: "pure couple [N·m]" },
  { id: "preload", label: "preload", hint: "bolt pretension along the bore axis" },
  { id: "remote", label: "remote", hint: "force applied at a remote point (adds moment)" },
  { id: "gravity", label: "g-load", hint: "body acceleration in g units" },
];

function loadKind(l: LoadSpec): LoadKind {
  if ((l.bearing_n ?? 0) > 0) return "bearing";
  if (l.moment_nm?.some((v) => Math.abs(v) > 1e-9)) return "moment";
  if ((l.preload_n ?? 0) > 0) return "preload";
  if (l.remote_point?.length === 3) return "remote";
  if (l.g_load?.some((v) => Math.abs(v) > 1e-9)) return "gravity";
  if ((l.pressure_mpa ?? 0) > 0 && !(l.force_n?.some((f) => Math.abs(f) > 1e-9))) return "pressure";
  return "force";
}

function blankLoad(kind: LoadKind): Partial<LoadSpec> {
  const base: Partial<LoadSpec> = {
    force_n: undefined,
    pressure_mpa: null,
    bearing_n: null,
    bearing_dir: undefined,
    moment_nm: undefined,
    preload_n: null,
    remote_point: undefined,
    g_load: undefined,
    direction_cone_deg: null,
  };
  switch (kind) {
    case "force":
      return { ...base, force_n: [0, 0, -50] };
    case "pressure":
      return { ...base, pressure_mpa: 2 };
    case "bearing":
      return { ...base, bearing_n: 200, bearing_dir: [0, 0, -1] };
    case "moment":
      return { ...base, moment_nm: [0, 0, 1] };
    case "preload":
      return { ...base, preload_n: 500 };
    case "remote":
      return { ...base, force_n: [0, 0, -50], remote_point: [0, 0, 0] };
    case "gravity":
      return { ...base, g_load: [0, 0, -1] };
  }
}

/** does this load actually apply something? (for the run-button guard) */
function loadActive(l: LoadSpec): boolean {
  return (
    (l.force_n?.some((f) => Math.abs(f) > 1e-9) ?? false) ||
    (l.pressure_mpa ?? 0) > 0 ||
    (l.bearing_n ?? 0) > 0 ||
    (l.moment_nm?.some((v) => Math.abs(v) > 1e-9) ?? false) ||
    (l.preload_n ?? 0) > 0 ||
    (l.g_load?.some((v) => Math.abs(v) > 1e-9) ?? false)
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
  showFixMode,
  showMaterial,
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
  anchors?: FeatureAnchor[];
  /** fixed/support regions: offer the encastre-vs-roller switch */
  showFixMode?: boolean;
  /** keep regions: offer a material override (insert zones) */
  showMaterial?: boolean;
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
        ) : ANCHOR_KINDS.has(kind) && anchors?.length ? (
          <>
            <select
              className="rounded bg-zinc-800 px-1 py-0.5 text-[10px] text-zinc-400"
              value={kind}
              title="ring = material around the mouth, wall = bore wall cells"
              onChange={(e) => setA({ kind: e.target.value as RegionSpec["kind"] })}
            >
              <option value="hole">ring</option>
              <option value="anchor">ring+</option>
              <option value="bore">wall</option>
            </select>
            <select
              className={`${inputCls} min-w-0 flex-1 px-1 py-0.5 text-[11px]`}
              value={r.hole_index ?? 1}
              onChange={(e) => setA({ hole_index: Number(e.target.value) })}
            >
              {anchors.map((a) => (
                <option key={a.index} value={a.index}>
                  #{a.index} {a.kind} {a.axis} Ø{a.dia ? a.dia.toFixed(1) : "?"}
                  {a.bolt_size ? ` ${a.bolt_size}` : ""} @ ({a.x.toFixed(1)}, {a.y.toFixed(1)}
                  {a.z != null ? `, ${a.z.toFixed(1)}` : ""})
                </option>
              ))}
            </select>
          </>
        ) : (
          <button
            className="min-w-0 flex-1 truncate text-left text-zinc-300 hover:text-zinc-100"
            title={`${regionLabel(r, anchors)} — click for details`}
            onClick={() => setOpen((o) => !o)}
          >
            {r.description || regionLabel(r, anchors)}
          </button>
        )}
        {hasCenterPick && onPickCenter && (
          <button
            title="pick region center on the model"
            aria-label="pick region center on the model"
            onClick={onPickCenter}
            className="rounded p-0.5 text-zinc-500 hover:text-zinc-200"
          >
            <Crosshair className="h-3.5 w-3.5" />
          </button>
        )}
        {kind === "points" && onPaint && (
          <button
            title={painting ? "stop painting" : "paint on the model"}
            aria-label={painting ? "stop painting" : "paint on the model"}
            onClick={onPaint}
            className={`rounded p-0.5 ${
              painting ? "text-amber-300" : "text-zinc-500 hover:text-zinc-200"
            }`}
          >
            <Brush className="h-3.5 w-3.5" />
          </button>
        )}
        <button
          title="remove region"
          aria-label="remove region"
          onClick={onDelete}
          className="text-zinc-500 hover:text-red-400"
        >
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
            className="ml-auto text-[10px] text-zinc-500 underline hover:text-zinc-300"
            onClick={() => setOpen((o) => !o)}
          >
            details
          </button>
        </div>
      )}

      {/* expanded details */}
      {open && (
        <div className="mt-1 flex flex-col gap-1 border-t border-zinc-800 pl-4 pt-1">
          {showFixMode && (
            <div className="flex items-center gap-1">
              <span className="w-14 text-[10px] text-zinc-500">support</span>
              <select
                className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
                value={r.fix_mode ?? "all"}
                onChange={(e) => setA({ fix_mode: e.target.value as "all" | "normal" })}
              >
                <option value="all">encastre (all DOFs)</option>
                <option value="normal">roller / resting (normal DOF only)</option>
              </select>
            </div>
          )}
          {showMaterial && (
            <div className="flex items-center gap-1">
              <span className="w-14 text-[10px] text-zinc-500">material</span>
              <select
                className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
                value={r.material ?? ""}
                onChange={(e) => setA({ material: e.target.value || undefined })}
              >
                <option value="">part material</option>
                {MATERIAL_OPTIONS.filter((m) => m).map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <span className="text-[10px] text-zinc-500">insert</span>
            </div>
          )}
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
              <span className="text-[10px] text-zinc-500">mm</span>
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
            <span className="text-[10px] text-zinc-500">mm</span>
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
  onPickFace,
  onPickEdge,
  onPickFeature,
  onAdd,
  hasAnchors,
  paintLabel,
}: {
  onPaint: () => void;
  onPickFace: () => void;
  onPickEdge: () => void;
  onPickFeature: () => void;
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
      <button
        className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
        title="arm face pick — click surface patches on the model"
        onClick={onPickFace}
      >
        <Layers className="h-3 w-3" /> face
      </button>
      <button
        className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
        title="arm edge pick — click a face to take its boundary loop"
        onClick={onPickEdge}
      >
        <Spline className="h-3 w-3" /> edge
      </button>
      <button
        className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 disabled:opacity-40 hover:border-zinc-500 hover:text-zinc-200"
        title="arm feature pick — click a hole / thread / bore on the model"
        onClick={onPickFeature}
        disabled={!hasAnchors}
      >
        <CircleDot className="h-3 w-3" /> hole
      </button>
      {REGION_KINDS.filter((k) => !ANCHOR_KINDS.has(k.id ?? "") || hasAnchors).map((k) => (
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
/* anchor markers + load gizmos                                        */
/* ------------------------------------------------------------------ */

/** one arrow: shaft + head, centred so `pos` is the tail */
function Arrow({
  pos,
  dir,
  len,
  color,
  r = 0.6,
}: {
  pos: [number, number, number];
  dir: [number, number, number];
  len: number;
  color: string;
  r?: number;
}) {
  const q = dirQuat(dir);
  const head = Math.min(len * 0.35, 4);
  return (
    <group position={pos} quaternion={q}>
      <mesh position={[0, (len - head) / 2, 0]}>
        <cylinderGeometry args={[r, r, Math.max(len - head, 0.1), 10]} />
        <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
      </mesh>
      <mesh position={[0, len - head / 2, 0]}>
        <coneGeometry args={[r * 2.4, head, 12]} />
        <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
      </mesh>
    </group>
  );
}

/** translucent bore sleeve + mouth ring shown when a feature pick hovers
 *  an anchor — the cylinder follows the anchor axis into the part. */
function AnchorSleeve({
  a,
  box,
  color,
}: {
  a: FeatureAnchor;
  box: THREE.Box3;
  color: string;
}) {
  const pos = anchorMouth(a, box);
  if (!pos) return null;
  const av = axisVec(a.axis);
  const idx = av.findIndex((v) => v !== 0);
  const span = idx >= 0 ? box.max.getComponent(idx) - box.min.getComponent(idx) : 10;
  const len = Math.min(Math.max((a.depth_mm ?? 0) * 2, span * 0.3), span);
  const r = Math.max((a.dia || 4) / 2, 1);
  return (
    <group position={pos} quaternion={dirQuat(av)}>
      <mesh position={[0, -len / 2, 0]}>
        <cylinderGeometry args={[r, r, len, 24, 1, true]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.28}
          side={THREE.DoubleSide}
          depthTest={false}
        />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[r + 0.5, Math.max(r * 0.08, 0.3), 8, 40]} />
        <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
      </mesh>
    </group>
  );
}

/** where does a load's gizmo sit? — region centre estimate in model space */
function regionCenter3(
  r: RegionSpec,
  anchors: FeatureAnchor[] | undefined,
  box: THREE.Box3 | null
): [number, number, number] | null {
  const mid = box ? box.getCenter(new THREE.Vector3()).toArray() : [0, 0, 0];
  const k = r.kind ?? "face";
  if (ANCHOR_KINDS.has(k)) {
    const a = anchors?.find((x) => x.index === (r.hole_index ?? 1));
    return a ? anchorPos(a, box) : null;
  }
  if (k === "area") return [r.center?.[0] ?? mid[0]!, r.center?.[1] ?? mid[1]!, mid[2]!];
  if (k === "cylinder" || k === "box")
    return [
      r.center?.[0] ?? mid[0]!,
      r.center?.[1] ?? mid[1]!,
      r.center?.[2] ?? mid[2]!,
    ];
  if (k === "points" && r.points?.length) {
    const n = r.points.length;
    return [
      r.points.reduce((s, p) => s + p[0], 0) / n,
      r.points.reduce((s, p) => s + p[1], 0) / n,
      r.points.reduce((s, p) => s + p[2], 0) / n,
    ];
  }
  if ((k === "face" || k === "surface" || k === "all") && box) {
    const c = mid.slice() as [number, number, number];
    if (k !== "all" && r.axis) {
      const i = { x: 0, y: 1, z: 2 }[r.axis.replace(/^[+-]/, "")] ?? 2;
      c[i] = r.axis.startsWith("-") ? box.min.toArray()[i]! : box.max.toArray()[i]!;
    }
    return c;
  }
  return null;
}

/** gizmos for one load: force arrow / moment ring / preload double-arrow */
function LoadGizmo({
  l,
  anchors,
  box,
  gsize,
}: {
  l: LoadSpec;
  anchors: FeatureAnchor[] | undefined;
  box: THREE.Box3 | null;
  gsize: number;
}) {
  const kind = loadKind(l);
  const pos = regionCenter3(
    kind === "remote" ? { ...l.region } : l.region,
    anchors,
    box
  );
  if (!pos) return null;
  const len = gsize;
  if (kind === "moment" && l.moment_nm) {
    const q = dirQuat(axisVecFromVec(l.moment_nm));
    const rad = len * 0.45;
    return (
      <group position={pos} quaternion={q}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[rad, 0.5, 8, 40, Math.PI * 1.6]} />
          <meshBasicMaterial color={GIZMO_COLORS.moment} depthTest={false} transparent opacity={0.9} />
        </mesh>
        <mesh position={[rad * Math.cos(Math.PI * 1.6), 0, rad * Math.sin(Math.PI * 1.6)]}>
          <coneGeometry args={[1.6, 4, 10]} />
          <meshBasicMaterial color={GIZMO_COLORS.moment} depthTest={false} transparent opacity={0.9} />
        </mesh>
      </group>
    );
  }
  if (kind === "preload") {
    const avec = axisVec(
      l.region.kind && ANCHOR_KINDS.has(l.region.kind)
        ? anchors?.find((x) => x.index === (l.region.hole_index ?? 1))?.axis
        : l.region.axis
    );
    return (
      <group>
        <Arrow
          pos={[pos[0] - avec[0] * len, pos[1] - avec[1] * len, pos[2] - avec[2] * len]}
          dir={avec}
          len={len * 0.8}
          color={GIZMO_COLORS.preload}
        />
        <Arrow
          pos={[pos[0] + avec[0] * len, pos[1] + avec[1] * len, pos[2] + avec[2] * len]}
          dir={[-avec[0], -avec[1], -avec[2]]}
          len={len * 0.8}
          color={GIZMO_COLORS.preload}
        />
      </group>
    );
  }
  if (kind === "gravity" && l.g_load) {
    return <Arrow pos={pos} dir={l.g_load as [number, number, number]} len={len} color={GIZMO_COLORS.force} />;
  }
  const dir =
    kind === "bearing"
      ? (l.bearing_dir as [number, number, number]) || (l.force_n as [number, number, number])
      : (l.force_n as [number, number, number]);
  const at = kind === "remote" && l.remote_point ? (l.remote_point as [number, number, number]) : pos;
  if (!dir?.length) return null;
  return <Arrow pos={at} dir={dir} len={len} color={GIZMO_COLORS.force} />;
}

/** moment vector → its rotation axis as a dir (for the torus orientation) */
function axisVecFromVec(v: number[]): [number, number, number] {
  return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 1];
}

/* ------------------------------------------------------------------ */
/* main component                                                      */
/* ------------------------------------------------------------------ */

/** persisted ui_editor keys owned by this editor — the blob itself is
 *  shared: EditPage writes `ops` into the same object, so key PRESENCE
 *  (not truthiness) marks whether this editor has a saved draft. A type
 *  alias (not interface) is required for the implicit index signature. */
type StructuralDraft = {
  load_cases?: LoadCase[];
  keep_regions?: RegionSpec[];
  void_regions?: RegionSpec[];
  material?: string;
  volfrac?: number;
  keep_mode?: "manual" | "auto";
  min_member_mm?: number;
  density_threshold?: number;
  auto_refine?: boolean;
  self_supporting?: boolean;
  print_direction?: "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
  multires?: boolean;
  solver?: "oc" | "mma";
  case_objective?: "weighted" | "minmax";
  max_displacement_mm?: number;
  max_stress_mpa?: number;
  min_gap_mm?: number;
  penal_init?: number;
};

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
  const qc = useQueryClient();
  const [saveFailed, setSaveFailed] = useState(false);
  const [cases, setCases] = useState<LoadCase[]>([]);
  const [keep, setKeep] = useState<RegionSpec[]>([]);
  const [voids, setVoids] = useState<RegionSpec[]>([]);
  const [volfrac, setVolfrac] = useState(0.4);
  const [keepMode, setKeepMode] = useState<"manual" | "auto">("manual");
  const [minMember, setMinMember] = useState(0);
  const [densityThreshold, setDensityThreshold] = useState(0.5);
  const [autoRefine, setAutoRefine] = useState(false);
  const [selfSupport, setSelfSupport] = useState(false);
  const [printDir, setPrintDir] = useState<"+x" | "-x" | "+y" | "-y" | "+z" | "-z">("+z");
  const [multires, setMultires] = useState(false);
  const [solver, setSolver] = useState<"oc" | "mma">("oc");
  const [minmax, setMinmax] = useState(false);
  const [dispMax, setDispMax] = useState(0);
  const [stressMax, setStressMax] = useState(0);
  const [minGap, setMinGap] = useState(0);
  const [penalCont, setPenalCont] = useState(false);
  const [material, setMaterial] = useState("");
  const [anchors, setAnchors] = useState<FeatureAnchor[]>([]);
  const [brushR, setBrushR] = useState(4);
  const [mode, setMode] = useState<Mode>("paint");
  const [target, setTarget] = useState<PaintTarget>(null);
  const [centerPick, setCenterPick] = useState<Exclude<PaintTarget, null> | null>(null);
  const [hover, setHover] = useState<PickInfo | null>(null);
  const [model, setModel] = useState<THREE.Group | null>(null);
  const [fillsCapped, setFillsCapped] = useState(false);
  const [preview, setPreview] = useState<FemPreviewResult | null>(null);
  const [showCells, setShowCells] = useState(true);
  const [busy, setBusy] = useState<"preview" | "fem" | "opt" | null>(null);
  const lastPt = useRef<[number, number, number] | null>(null);
  const lastPickT = useRef(0);
  const undoStack = useRef<{ t: Exclude<PaintTarget, null>; prev: number[][] }[]>([]);

  // surface/feature picking state
  const [featureTarget, setFeatureTarget] = useState<FeatureTarget>(null);
  const [creaseDeg, setCreaseDeg] = useState(30);
  const [facetSets, setFacetSets] = useState<Map<THREE.BufferGeometry, FacetSet> | null>(null);
  const [facetBusy, setFacetBusy] = useState(false);
  const [hoverFacet, setHoverFacet] = useState<{
    fs: FacetSet;
    facet: Facet;
    key: string;
  } | null>(null);
  const [hoverAnchor, setHoverAnchor] = useState<FeatureAnchor | null>(null);
  const [hoverLoop, setHoverLoop] = useState<{
    pts: [number, number, number][];
    closed: boolean;
  } | null>(null);
  const facetGeoms = useRef(new Map<string, THREE.BufferGeometry>());
  const pickedFacets = useRef(new Set<string>());

  // on-demand feature scan — same query key as AnchorsPanel, so both panels
  // share one fetch and one cache entry
  const anchorsQ = useQuery({
    queryKey: ["anchors", version.id],
    queryFn: () => api.anchors(version.id),
    enabled: open,
    staleTime: 60_000,
  });
  useEffect(() => {
    // the scan supersedes the stored analysis table once it arrives — `open`
    // in deps so a warm cache re-applies on re-open after init restored the
    // stored table
    const r = anchorsQ.data;
    if (open && r?.ok && r.anchors.length) setAnchors(r.anchors);
  }, [anchorsQ.data, open]);

  // init once per open/version — persisted GUI state wins over spec defaults
  useEffect(() => {
    if (!open) return;
    const saved = readUiEditor<StructuralDraft>(version);
    const specOpt = version.spec?.optimize;
    if (saved && "load_cases" in saved) {
      // a persisted [] is the user's "deleted every case" — honour it
      setCases(saved.load_cases ?? []);
      setKeep(saved.keep_regions ?? []);
      setVoids(saved.void_regions ?? []);
      setMaterial(saved.material ?? "");
      setVolfrac(saved.volfrac ?? 0.4);
      setKeepMode(saved.keep_mode ?? "manual");
      setMinMember(saved.min_member_mm ?? specOpt?.min_member_mm ?? 0);
      setDensityThreshold(saved.density_threshold ?? specOpt?.density_threshold ?? 0.5);
      setAutoRefine(saved.auto_refine ?? specOpt?.auto_refine_features ?? false);
      setSelfSupport(saved.self_supporting ?? specOpt?.self_supporting ?? false);
      setPrintDir(saved.print_direction ?? specOpt?.print_direction ?? "+z");
      setMultires(saved.multires ?? specOpt?.multires ?? false);
      setSolver(saved.solver ?? specOpt?.solver ?? "oc");
      setMinmax(
        (saved.case_objective ?? specOpt?.case_objective ?? "weighted") === "minmax"
      );
      setDispMax(saved.max_displacement_mm ?? specOpt?.max_displacement_mm ?? 0);
      setStressMax(saved.max_stress_mpa ?? specOpt?.max_stress_mpa ?? 0);
      setMinGap(saved.min_gap_mm ?? specOpt?.min_gap_mm ?? 0);
      setPenalCont(
        (saved.penal_init ?? specOpt?.penal_init) != null &&
          (saved.penal_init ?? specOpt?.penal_init ?? 0) > 0
      );
    } else if (version.spec?.load_cases?.length) {
      setCases(version.spec.load_cases);
      setKeep(specOpt?.keep_regions ?? []);
      setVoids(specOpt?.void_regions ?? []);
      setMaterial(version.options?.material ?? "");
      setKeepMode(specOpt?.keep_mode ?? "manual");
      setMinMember(specOpt?.min_member_mm ?? 0);
      setDensityThreshold(specOpt?.density_threshold ?? 0.5);
      setAutoRefine(specOpt?.auto_refine_features ?? false);
      setSelfSupport(specOpt?.self_supporting ?? false);
      setPrintDir(specOpt?.print_direction ?? "+z");
      setMultires(specOpt?.multires ?? false);
      setSolver(specOpt?.solver ?? "oc");
      setMinmax(specOpt?.case_objective === "minmax");
      setDispMax(specOpt?.max_displacement_mm ?? 0);
      setStressMax(specOpt?.max_stress_mpa ?? 0);
      setMinGap(specOpt?.min_gap_mm ?? 0);
      setPenalCont(specOpt?.penal_init != null && specOpt.penal_init > 0);
    } else {
      setCases([{ name: "case1", fixed: [], loads: [] }]);
      setKeep([]);
      setVoids([]);
    }
    // typed anchors: stored analysis table first, the on-demand scan query
    // (above) replaces it when it resolves
    setAnchors(readStoredAnchors(version));
    setPreview(null);
    setTarget(null);
    setCenterPick(null);
    setFeatureTarget(null);
    setHoverFacet(null);
    setHoverAnchor(null);
    setHoverLoop(null);
    pickedFacets.current.clear();
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
    if (t.kind === "void") return voids[t.ri] ?? null;
    return keep[t.ri] ?? null;
  };

  const setRegion = (t: Exclude<PaintTarget, null>, r: RegionSpec) => {
    if (t.kind === "fixed") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { fixed: c.fixed.map((x, i) => (i === t.ri ? r : x)) });
    } else if (t.kind === "load") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { loads: c.loads.map((x, i) => (i === t.li ? { ...x, region: r } : x)) });
    } else if (t.kind === "void") {
      setVoids((k) => k.map((x, i) => (i === t.ri ? r : x)));
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
    } else if (t.kind === "void") {
      setVoids((k) => k.filter((_, i) => i !== t.ri));
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
    // re-picking must be possible after undo — the key set only prevents
    // double-adds within the current selection
    pickedFacets.current.clear();
  };

  /** clear hover-highlight state (tool switches happen without a pointer move) */
  const clearHoverFx = () => {
    setHoverFacet(null);
    setHoverAnchor(null);
    setHoverLoop(null);
  };

  /** remove the armed region when it is a still-empty points region */
  const dropEmptyTarget = () => {
    if (!target) return;
    const r = getRegion(target);
    if (r?.kind === "points" && !r.points?.length) delRegion(target);
  };

  /** drop any armed tool; an empty just-armed paint region goes with it */
  const disarm = () => {
    dropEmptyTarget();
    setTarget(null);
    setCenterPick(null);
    setFeatureTarget(null);
    clearHoverFx();
    pickedFacets.current.clear();
  };

  const armPaint = (t: Exclude<PaintTarget, null>) => {
    if (sameTarget(t) && (mode === "paint" || mode === "erase")) {
      disarm();
      return;
    }
    dropEmptyTarget();
    setTarget(t);
    setMode("paint");
    setCenterPick(null);
    setFeatureTarget(null);
    clearHoverFx();
    pickedFacets.current.clear();
  };

  const armCenter = (t: Exclude<PaintTarget, null>) => {
    if (sameCenterPick(t)) {
      setCenterPick(null);
      return;
    }
    dropEmptyTarget();
    setCenterPick(t);
    setTarget(null);
    setFeatureTarget(null);
    clearHoverFx();
  };

  /** Append a fresh painted region and arm the brush on it. */
  const startPaint = (t: Exclude<PaintTarget, null>, append: (r: RegionSpec) => void) => {
    dropEmptyTarget();
    append(emptyPointsRegion(brushR));
    setTarget(t);
    setMode("paint");
    setCenterPick(null);
    setFeatureTarget(null);
    clearHoverFx();
    lastPt.current = null;
    pickedFacets.current.clear();
  };

  /** Append a fresh points region and arm a surface-pick tool on it. */
  const startToolPick = (
    t: Exclude<PaintTarget, null>,
    m: "face" | "edge",
    append: (r: RegionSpec) => void
  ) => {
    dropEmptyTarget();
    append(emptyPointsRegion(m === "edge" ? 2 : brushR));
    setTarget(t);
    setMode(m);
    setCenterPick(null);
    setFeatureTarget(null);
    clearHoverFx();
    lastPt.current = null;
    pickedFacets.current.clear();
  };

  const armTool = (
    t: Exclude<PaintTarget, null>,
    m: "face" | "edge",
    append: (r: RegionSpec) => void
  ) => {
    if (sameTarget(t) && mode === m) disarm();
    else startToolPick(t, m, append);
  };

  const toggleFeaturePick = (ft: Exclude<FeatureTarget, null>) => {
    dropEmptyTarget();
    setFeatureTarget((cur) =>
      cur && JSON.stringify(cur) === JSON.stringify(ft) ? null : ft
    );
    setTarget(null);
    setCenterPick(null);
    clearHoverFx();
  };

  /* --- painting / erasing / picking --- */

  /** facet containing the picked triangle (needs facetSets + faceIndex). */
  const facetAt = (info: PickInfo) => {
    const g = (info.object as THREE.Mesh | undefined)?.geometry as
      | THREE.BufferGeometry
      | undefined;
    const fs = g ? facetSets?.get(g) : null;
    const facet = fs ? facetAtHit(fs, info.faceIndex) : null;
    return fs && facet ? { fs, facet, key: `${g!.uuid}:${facet.id}` } : null;
  };

  /** append a hole/bore/anchor region bound to a scanned feature */
  const addAnchorRegion = (t: Exclude<FeatureTarget, null>, a: FeatureAnchor) => {
    const r: RegionSpec = {
      kind: t.kind === "load" ? "bore" : "hole",
      hole_index: a.index,
      margin_mm: 2,
      description: `${a.kind} #${a.index} Ø${a.dia.toFixed(1)}`,
    };
    if (t.kind === "fixed") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { fixed: [...c.fixed, r] });
    } else if (t.kind === "load") {
      const c = cases[t.ci];
      if (c) setCase(t.ci, { loads: [...c.loads, { region: r, force_n: [0, 0, -50] }] });
    } else if (t.kind === "void") {
      setVoids((k) => [...k, r]);
    } else {
      setKeep((k) => [...k, r]);
    }
  };

  /** hover routing — drives facet/edge/anchor highlight state */
  const handleHover = (info: PickInfo | null) => {
    setHover(info);
    if (!info) {
      setHoverFacet(null);
      setHoverAnchor(null);
      setHoverLoop(null);
      return;
    }
    if (featureTarget) {
      setHoverAnchor(nearestAnchorAt(info.point, anchors, modelBox));
      setHoverFacet(null);
      setHoverLoop(null);
      return;
    }
    if (target && (mode === "face" || mode === "edge")) {
      const hit = facetAt(info);
      setHoverFacet(hit);
      setHoverLoop(
        hit && mode === "edge" ? facetLoop(hit.fs, hit.facet, info.point) : null
      );
      setHoverAnchor(null);
      return;
    }
    setHoverFacet(null);
    setHoverAnchor(null);
    setHoverLoop(null);
  };

  const onPick = (info: PickInfo, viaDrag = false) => {
    // feature pick: click inside/near a bore corridor → hole/bore/anchor region
    if (featureTarget) {
      if (viaDrag) return;
      const a = nearestAnchorAt(info.point, anchors, modelBox);
      if (a) addAnchorRegion(featureTarget, a);
      return;
    }
    // center placement for geometric regions — for anchor kinds it snaps to
    // the nearest measured anchor (feature picking)
    if (centerPick) {
      if (viaDrag) return;
      const r = getRegion(centerPick);
      const p = info.point;
      if (r) {
        const k = r.kind;
        if (k && ANCHOR_KINDS.has(k) && anchors.length) {
          const a = nearestAnchorAt(info.point, anchors, modelBox);
          if (a) {
            setRegion(centerPick, { ...r, hole_index: a.index });
            setCenterPick(null);
            return;
          }
        }
        const center = k === "area" ? [p[0], p[1]] : [p[0], p[1], p[2]];
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

    // face pick: each click adds the whole facet's triangles as pick points
    if (mode === "face") {
      if (viaDrag) return;
      const hit = facetAt(info);
      if (!hit || pickedFacets.current.has(hit.key)) return;
      pickedFacets.current.add(hit.key);
      pushUndo(target, r.points ?? []);
      // stride-sample into the remaining 4000-point budget so later facet
      // clicks don't evict earlier ones
      const budget = Math.max(300, 4000 - (r.points?.length ?? 0));
      const merged = [...(r.points ?? []), ...facetPickPoints(hit.fs, hit.facet, budget)].slice(
        -4000
      );
      setRegion(target, {
        ...r,
        points: merged,
        radius_mm: r.points?.length ? r.radius_mm : facetRadius(hit.facet),
        normal: hit.facet.normal,
        description: r.description || `face · ${Math.round(hit.facet.area)} mm²`,
      });
      return;
    }
    // edge pick: click a face → its boundary loop nearest the click becomes
    // the region (resampled — raw vertices would gap along long edges)
    if (mode === "edge") {
      if (viaDrag) return;
      const hit = facetAt(info);
      if (!hit) return;
      const loop = facetLoop(hit.fs, hit.facet, info.point);
      if (!loop) return;
      pushUndo(target, r.points ?? []);
      const pts = resampleLoop(loop.pts, Math.min(rad, 2), 3000, loop.closed);
      const merged = [...(r.points ?? []), ...pts].slice(-4000);
      setRegion(target, {
        ...r,
        points: merged,
        normal: r.normal ?? hit.facet.normal,
        description: r.description || `edge loop · ${loop.pts.length} verts`,
      });
      return;
    }

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

  const modelBox = useMemo(() => {
    if (!model) return null;
    try {
      const b = new THREE.Box3().setFromObject(model);
      return b.isEmpty() ? null : b;
    } catch {
      return null;
    }
  }, [model]);

  // facet segmentation — deferred so arming a tool doesn't block the frame
  const surfacePickArmed = !!target && (mode === "face" || mode === "edge");
  useEffect(() => {
    if (!model || !surfacePickArmed) {
      setFacetSets(null);
      setFacetBusy(false);
      return;
    }
    setFacetBusy(true);
    const t = setTimeout(() => {
      const m = new Map<THREE.BufferGeometry, FacetSet>();
      model.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.isMesh && mesh.geometry) {
          const fs = buildFacets(mesh, model, creaseDeg);
          if (fs) m.set(mesh.geometry as THREE.BufferGeometry, fs);
        }
      });
      setFacetSets(m);
      setFacetBusy(false);
    }, 120);
    return () => clearTimeout(t);
  }, [model, surfacePickArmed, creaseDeg]);

  // drop cached facet overlays + pick bookkeeping when facets change, and
  // dispose the superseded welded geometry (highlight geoms share its
  // position attribute — they're dropped at the same time)
  useEffect(() => {
    facetGeoms.current.forEach((g) => g.dispose());
    facetGeoms.current.clear();
    pickedFacets.current.clear();
    setHoverFacet(null);
    setHoverLoop(null);
    return () => facetSets?.forEach((fs) => fs.welded.dispose());
  }, [facetSets]);

  /** highlight geometry for the hovered facet (built once, cached) */
  const facetGeom = useMemo(() => {
    if (!hoverFacet) return null;
    let g = facetGeoms.current.get(hoverFacet.key);
    if (!g) {
      if (facetGeoms.current.size > 96) {
        facetGeoms.current.forEach((x) => x.dispose());
        facetGeoms.current.clear();
      }
      g = facetGeometry(hoverFacet.fs, hoverFacet.facet);
      facetGeoms.current.set(hoverFacet.key, g);
    }
    return g;
  }, [hoverFacet]);

  const valid = cases.length > 0 && cases.every(
    (c) => c.fixed.length > 0 && c.loads.length > 0 && c.loads.every(loadActive)
  );

  const persist = () =>
    api
      .saveEditorState(version.id, {
        load_cases: cases,
        keep_regions: keep,
        void_regions: voids,
        material,
        volfrac,
        keep_mode: keepMode,
        min_member_mm: minMember || undefined,
        density_threshold: densityThreshold,
        auto_refine: autoRefine,
        self_supporting: selfSupport,
        print_direction: printDir,
        multires,
        solver,
        case_objective: solver === "mma" && minmax ? "minmax" : "weighted",
        max_displacement_mm: dispMax || undefined,
        max_stress_mpa: stressMax || undefined,
        min_gap_mm: minGap || undefined,
        penal_init: penalCont ? 1.5 : undefined,
      })
      .then(
        (r) => {
          setSaveFailed(false);
          return r;
        },
        (e) => {
          // a dropped draft must be visible — silent loss is a data bug
          setSaveFailed(true);
          throw e;
        }
      );

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
  }, [cases, keep, voids, material, volfrac, keepMode, minMember, densityThreshold, autoRefine, selfSupport, printDir, multires, solver, minmax, dispMax, stressMax, minGap, penalCont, open]);

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
      if (r.ok && r.anchors?.length) {
        setAnchors(r.anchors);
        // the preview's scan is fresher than the query — publish it so the
        // AnchorsPanel key sees the same table
        qc.setQueryData(["anchors", version.id], { ok: true, anchors: r.anchors });
      }
      if (!r.ok) toast(r.error ?? "region check failed", "error");
    } catch (e) {
      toast(errMsg(e), "error");
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
      toast(errMsg(e), "error");
      setBusy(null);
    }
  };

  const runTopopt = async () => {
    setBusy("opt");
    try {
      await persist();
      const r = await api.optimize(version.id, {
        load_cases: cases,
        // allow_reopt is a top-level field of the optimize request — nested
        // inside optimize:{} the backend silently drops it
        allow_reopt: version.kind === "optimize" || undefined,
        optimize: {
          enabled: true,
          target_volume_fraction: volfrac,
          keep_regions: keep,
          void_regions: voids.length ? voids : undefined,
          keep_mode: keepMode,
          min_member_mm: minMember || undefined,
          density_threshold: densityThreshold,
          auto_refine_features: autoRefine || undefined,
          self_supporting: selfSupport || undefined,
          print_direction: selfSupport ? printDir : undefined,
          multires: multires || undefined,
          solver,
          case_objective:
            solver === "mma" && minmax ? "minmax" : undefined,
          max_displacement_mm: solver === "mma" && dispMax ? dispMax : undefined,
          max_stress_mpa: solver === "mma" && stressMax ? stressMax : undefined,
          min_gap_mm: minGap || undefined,
          penal_init: penalCont ? 1.5 : undefined,
        },
        material: material || undefined,
      });
      onJobStarted(r);
    } catch (e) {
      toast(errMsg(e), "error");
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
    voids.forEach((r) => add(r, COLORS.void));
    return out;
  }, [cases, keep, voids]);

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
    voids.forEach((r) => add(r, COLORS.void));
    return out;
  }, [cases, keep, voids]);

  /** paint trail — world-sized dots at every pick point so coverage reads on
   *  any tessellation (RegionFill can look sparse on coarse meshes) */
  const trails = useMemo(() => {
    const out: { pts: number[][]; color: string; size: number }[] = [];
    const add = (r: RegionSpec, color: string) => {
      if (r.kind === "points" && r.points?.length)
        out.push({ pts: r.points, color, size: (r.radius_mm ?? 3) * 1.4 });
    };
    cases.forEach((c) => {
      c.fixed.forEach((r) => add(r, COLORS.fixed));
      c.loads.forEach((l) => add(l.region, COLORS.load));
    });
    keep.forEach((r) => add(r, COLORS.keep));
    voids.forEach((r) => add(r, COLORS.void));
    return out;
  }, [cases, keep, voids]);

  /** anchor mouth positions in model space (for markers) */
  const anchorMarkers = useMemo(
    () =>
      anchors
        .map((a) => ({ a, p: anchorMouth(a, modelBox) }))
        .filter((x): x is { a: FeatureAnchor; p: [number, number, number] } => !!x.p),
    [anchors, modelBox]
  );

  /** which anchor indices are referenced by anchor-kind regions? */
  const usedAnchors = useMemo(() => {
    const s = new Set<number>();
    const scan = (r: RegionSpec) => {
      if (r.kind && ANCHOR_KINDS.has(r.kind) && r.hole_index) s.add(r.hole_index);
    };
    cases.forEach((c) => {
      c.fixed.forEach(scan);
      c.loads.forEach((l) => scan(l.region));
    });
    keep.forEach(scan);
    voids.forEach(scan);
    return s;
  }, [cases, keep, voids]);

  const glb = version.artifacts.find((a) => a.kind === "glb");

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // don't yank the tool while typing in a field
      if ((e.target as HTMLElement)?.matches?.("input,select,textarea")) return;
      if (target || centerPick || featureTarget) disarm();
      else onClose();
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, onClose, target, centerPick, featureTarget]);
  if (!open) return null;

  const sameTarget = (t: Exclude<PaintTarget, null>) =>
    target != null && JSON.stringify(target) === JSON.stringify(t);
  const sameCenterPick = (t: Exclude<PaintTarget, null>) =>
    centerPick != null && JSON.stringify(centerPick) === JSON.stringify(t);

  const activeRegion = target ? getRegion(target) : null;
  const cursorR = mode === "erase" ? brushR : (activeRegion?.radius_mm ?? brushR);
  const dotR = modelBox
    ? Math.max(0.5, modelBox.getSize(new THREE.Vector3()).length() * 0.004)
    : 1;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-zinc-950">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
        <h2 className="text-sm font-semibold">Structural editor — v{version.number}</h2>
        <span className="text-xs text-zinc-500">
          paint or click faces / edges / holes — right-drag orbits while a tool is armed
        </span>
        <span className="flex-1" />
        {(target || centerPick || featureTarget) && (
          <Chip tone="amber">
            <Brush className="h-3 w-3" />{" "}
            {featureTarget
              ? "feature pick — click a hole / thread / bore on the model"
              : centerPick
                ? "click to place region center"
                : mode === "face"
                  ? "face pick — click faces to add them to the region"
                  : mode === "edge"
                    ? "edge pick — click near a face edge to take its loop"
                    : `${mode === "erase" ? "erasing" : "painting"} ${
                        target!.kind === "fixed"
                          ? "support"
                          : target!.kind === "load"
                            ? "load"
                            : target!.kind
                      } — drag on the model`}
          </Chip>
        )}
        {target && (mode === "paint" || mode === "erase") && (
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
        )}
        {target && (mode === "face" || mode === "edge") && (
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            crease
            <input
              type="range"
              min={10}
              max={60}
              step={5}
              className="w-20 accent-brand-500"
              value={creaseDeg}
              onChange={(e) => setCreaseDeg(num(e.target.value, 30))}
            />
            <span className="w-8 tabular-nums">{creaseDeg}°</span>
            {facetBusy && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
            {!facetBusy && facetSets && facetSets.size === 0 && (
              <span className="text-amber-400">mesh too large for picking</span>
            )}
          </label>
        )}
        {target && (
          <IconBtn title="undo last stroke" onClick={undo} disabled={!undoStack.current.length}>
            <Undo2 className="h-3.5 w-3.5" />
          </IconBtn>
        )}
        {(target || centerPick || featureTarget) && (
          <IconBtn title="done (Esc)" onClick={disarm}>
            <Check className="h-3.5 w-3.5" />
          </IconBtn>
        )}
        {saveFailed && (
          <Chip tone="amber" title="draft changes could not be saved — saving retries automatically">
            draft not saved
          </Chip>
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
              pickEnabled={!!target || !!centerPick || !!featureTarget}
              onPick={onPick}
              onHover={handleHover}
              onModel={setModel}
            >
              <RegionFill model={model} entries={fills} onTooDense={setFillsCapped} />
              {/* paint trails — visible coverage regardless of tessellation */}
              {trails.map((t, i) => (
                <PointCloud key={i} pts={t.pts} color={t.color} size={t.size} opacity={0.8} />
              ))}
              {markers.map((m, i) => (
                <mesh key={i} position={m.p}>
                  <sphereGeometry args={[1, 16, 12]} />
                  <meshBasicMaterial color={m.color} depthTest={false} transparent opacity={0.9} />
                </mesh>
              ))}
              {/* hovered facet */}
              {facetGeom && (
                <mesh geometry={facetGeom} renderOrder={2}>
                  <meshBasicMaterial
                    color={target ? COLORS[target.kind] : "#f59e0b"}
                    transparent
                    opacity={mode === "edge" ? 0.15 : 0.32}
                    side={THREE.DoubleSide}
                    polygonOffset
                    polygonOffsetFactor={-2}
                    depthWrite={false}
                  />
                </mesh>
              )}
              {/* hovered boundary loop */}
              {hoverLoop && hoverLoop.pts.length >= 2 && (
                <Line
                  points={
                    hoverLoop.closed ? [...hoverLoop.pts, hoverLoop.pts[0]] : hoverLoop.pts
                  }
                  color={target ? COLORS[target.kind] : "#f59e0b"}
                  lineWidth={3}
                  depthTest={false}
                  renderOrder={3}
                />
              )}
              {/* hovered feature anchor (bore sleeve + mouth ring) */}
              {featureTarget && hoverAnchor && modelBox && (
                <AnchorSleeve a={hoverAnchor} box={modelBox} color={COLORS.anchor} />
              )}
              {/* measured feature anchors — markers sit at the hole mouth and
                  are clickable while the feature pick tool is armed */}
              {anchorMarkers.map(({ a, p }) => (
                <AnchorMarker
                  key={a.index}
                  a={a}
                  pos={p}
                  selected={usedAnchors.has(a.index)}
                  onSelect={
                    featureTarget ? () => addAnchorRegion(featureTarget, a) : undefined
                  }
                />
              ))}
              {/* load gizmos: arrows / moment rings at the region centres */}
              {modelBox &&
                cases.map((c, ci) =>
                  c.loads.map((l, li) => (
                    <LoadGizmo
                      key={`${ci}-${li}`}
                      l={l}
                      anchors={anchors}
                      box={modelBox}
                      gsize={modelBox.getSize(new THREE.Vector3()).length() * 0.14}
                    />
                  ))
                )}
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
              {(target || centerPick) && hover && (mode === "paint" || mode === "erase") && (
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
              {/* small dot cursor for pick modes */}
              {(centerPick || featureTarget || (target && (mode === "face" || mode === "edge"))) &&
                hover && (
                  <mesh position={hover.point}>
                    <sphereGeometry args={[dotR, 12, 8]} />
                    <meshBasicMaterial color="#f59e0b" depthTest={false} transparent opacity={0.9} />
                  </mesh>
                )}
            </ModelCanvas>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-zinc-500">
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
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: COLORS.void }} />
              void
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: COLORS.anchor }} />
              anchors {anchors.length ? `(${anchors.length})` : ""}
            </span>
            {fillsCapped && (
              <span className="text-amber-400" title="paint fills are skipped above 500k vertices">
                fills hidden — mesh too dense
              </span>
            )}
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
                      aria-label="remove case"
                      className="text-zinc-500 hover:text-red-400"
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
                    hasCenterPick={
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    }
                    showFixMode
                    anchors={anchors}
                    onPaint={() => armPaint({ kind: "fixed", ci, ri })}
                    onPickCenter={() => armCenter({ kind: "fixed", ci, ri })}
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
                  onPickFace={() =>
                    armTool({ kind: "fixed", ci, ri: c.fixed.length }, "face", (r) =>
                      setCase(ci, { fixed: [...c.fixed, r] })
                    )
                  }
                  onPickEdge={() =>
                    armTool({ kind: "fixed", ci, ri: c.fixed.length }, "edge", (r) =>
                      setCase(ci, { fixed: [...c.fixed, r] })
                    )
                  }
                  onPickFeature={() => toggleFeaturePick({ kind: "fixed", ci })}
                  onAdd={(r) => {
                    setCase(ci, { fixed: [...c.fixed, r] });
                    const t = { kind: "fixed" as const, ci, ri: c.fixed.length };
                    if (
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    )
                      armCenter(t);
                  }}
                />

                {/* loads */}
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  loads
                </div>
                {c.loads.map((l, li) => {
                  const force = l.force_n ?? [0, 0, 0];
                  const lk = loadKind(l);
                  const setLoad = (patch: Partial<typeof l>) =>
                    setCase(ci, {
                      loads: c.loads.map((x, i) => (i === li ? { ...x, ...patch } : x)),
                    });
                  const vecRow = (
                    vec: number[] | undefined,
                    labels: string[],
                    unit: string,
                    step: number,
                    set: (v: number[]) => void
                  ) => (
                    <div className="flex items-center gap-1">
                      {labels.map((ax, i) => (
                        <label key={ax} className="flex flex-1 items-center gap-0.5">
                          <span className="text-[9px] uppercase text-zinc-500">{ax}</span>
                          <NumInput
                            value={vec?.[i] ?? 0}
                            step={step}
                            className="w-full"
                            onChange={(n) =>
                              set((vec ?? [0, 0, 0]).map((f, j) => (j === i ? n : f)))
                            }
                          />
                        </label>
                      ))}
                      <span className="text-[10px] text-zinc-500">{unit}</span>
                    </div>
                  );
                  return (
                    <div key={li} className="mb-1.5">
                      <RegionRow
                        r={l.region}
                        color={COLORS.load}
                        painting={sameTarget({ kind: "load", ci, li })}
                        hasCenterPick={
                          l.region.kind === "area" ||
                          l.region.kind === "cylinder" ||
                          l.region.kind === "box" ||
                          ANCHOR_KINDS.has(l.region.kind ?? "")
                        }
                        anchors={anchors}
                        onPaint={() => armPaint({ kind: "load", ci, li })}
                        onPickCenter={() => armCenter({ kind: "load", ci, li })}
                        onChange={(nr) => setRegion({ kind: "load", ci, li }, nr)}
                        onDelete={() => delRegion({ kind: "load", ci, li })}
                      />
                      <div className="mt-1 flex flex-col gap-1 pl-4">
                        {/* load type selector */}
                        <select
                          className={`${inputCls} w-full px-1 py-0.5 text-[11px]`}
                          value={lk}
                          title={LOAD_KINDS.find((k) => k.id === lk)?.hint}
                          onChange={(e) =>
                            setLoad(blankLoad(e.target.value as LoadKind))
                          }
                        >
                          {LOAD_KINDS.map((k) => (
                            <option key={k.id} value={k.id} title={k.hint}>
                              {k.label} — {k.hint}
                            </option>
                          ))}
                        </select>
                        {(lk === "force" || lk === "remote") &&
                          vecRow(force, ["fx", "fy", "fz"], "N", 5, (v) => setLoad({ force_n: v }))}
                        {lk === "pressure" && (
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
                        )}
                        {lk === "bearing" && (
                          <>
                            <div className="flex items-center gap-1.5">
                              <NumInput
                                value={l.bearing_n ?? 0}
                                step={25}
                                className="w-24"
                                onChange={(n) => setLoad({ bearing_n: n })}
                              />
                              <span className="text-[10px] text-zinc-500">
                                N — cosine-distributed on the bore wall (needs a bore/anchor region)
                              </span>
                            </div>
                            {vecRow(
                              l.bearing_dir ?? [0, 0, -1],
                              ["dx", "dy", "dz"],
                              "dir",
                              0.1,
                              (v) => setLoad({ bearing_dir: v })
                            )}
                          </>
                        )}
                        {lk === "moment" &&
                          vecRow(
                            l.moment_nm ?? [0, 0, 0],
                            ["mx", "my", "mz"],
                            "N·m",
                            0.5,
                            (v) => setLoad({ moment_nm: v })
                          )}
                        {lk === "preload" && (
                          <div className="flex items-center gap-1.5">
                            <NumInput
                              value={l.preload_n ?? 0}
                              step={50}
                              className="w-24"
                              onChange={(n) => setLoad({ preload_n: n })}
                            />
                            <span className="text-[10px] text-zinc-500">
                              N pretension along the bore axis (needs a bore/anchor region)
                            </span>
                          </div>
                        )}
                        {lk === "remote" &&
                          vecRow(
                            l.remote_point ?? [0, 0, 0],
                            ["px", "py", "pz"],
                            "mm",
                            5,
                            (v) => setLoad({ remote_point: v })
                          )}
                        {lk === "gravity" &&
                          vecRow(
                            l.g_load ?? [0, 0, -1],
                            ["gx", "gy", "gz"],
                            "g",
                            0.5,
                            (v) => setLoad({ g_load: v })
                          )}
                        {(lk === "force" || lk === "bearing") && (
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] text-zinc-500">dir. cone</span>
                            <NumInput
                              value={l.direction_cone_deg ?? 0}
                              step={5}
                              className="w-16"
                              onChange={(n) =>
                                setLoad({ direction_cone_deg: n > 0 ? Math.min(n, 90) : null })
                              }
                            />
                            <span className="text-[10px] text-zinc-500">
                              ° robustness (TopOpt samples directions)
                            </span>
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
                  onPickFace={() =>
                    armTool({ kind: "load", ci, li: c.loads.length }, "face", (r) =>
                      setCase(ci, {
                        loads: [...c.loads, { region: r, force_n: [0, 0, -50] }],
                      })
                    )
                  }
                  onPickEdge={() =>
                    armTool({ kind: "load", ci, li: c.loads.length }, "edge", (r) =>
                      setCase(ci, {
                        loads: [...c.loads, { region: r, force_n: [0, 0, -50] }],
                      })
                    )
                  }
                  onPickFeature={() => toggleFeaturePick({ kind: "load", ci })}
                  onAdd={(r) => {
                    setCase(ci, {
                      loads: [...c.loads, { region: r, force_n: [0, 0, -50] }],
                    });
                    const t = { kind: "load" as const, ci, li: c.loads.length };
                    if (
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    )
                      armCenter(t);
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
                    hasCenterPick={
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    }
                    showMaterial
                    anchors={anchors}
                    onPaint={() => armPaint({ kind: "keep", ri })}
                    onPickCenter={() => armCenter({ kind: "keep", ri })}
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
                  onPickFace={() =>
                    armTool({ kind: "keep", ri: keep.length }, "face", (r) =>
                      setKeep((k) => [...k, r])
                    )
                  }
                  onPickEdge={() =>
                    armTool({ kind: "keep", ri: keep.length }, "edge", (r) =>
                      setKeep((k) => [...k, r])
                    )
                  }
                  onPickFeature={() => toggleFeaturePick({ kind: "keep" })}
                  onAdd={(r) => {
                    setKeep((k) => [...k, r]);
                    if (
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    )
                      armCenter({ kind: "keep", ri: keep.length });
                  }}
                />
              </div>
            )}

            {/* void regions — material forbidden zones */}
            {canTopopt && (
              <div className="rounded-lg border border-zinc-800 p-2.5">
                <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  void regions (TopOpt must keep empty)
                </div>
                {voids.map((r, ri) => (
                  <RegionRow
                    key={ri}
                    r={r}
                    color={COLORS.void}
                    painting={sameTarget({ kind: "void", ri })}
                    hasCenterPick={
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    }
                    anchors={anchors}
                    onPaint={() => armPaint({ kind: "void", ri })}
                    onPickCenter={() => armCenter({ kind: "void", ri })}
                    onChange={(nr) => setRegion({ kind: "void", ri }, nr)}
                    onDelete={() => delRegion({ kind: "void", ri })}
                  />
                ))}
                <AddRegionButtons
                  paintLabel="paint void region"
                  hasAnchors={!!anchors?.length}
                  onPaint={() =>
                    startPaint({ kind: "void", ri: voids.length }, (r) => setVoids((k) => [...k, r]))
                  }
                  onPickFace={() =>
                    armTool({ kind: "void", ri: voids.length }, "face", (r) =>
                      setVoids((k) => [...k, r])
                    )
                  }
                  onPickEdge={() =>
                    armTool({ kind: "void", ri: voids.length }, "edge", (r) =>
                      setVoids((k) => [...k, r])
                    )
                  }
                  onPickFeature={() => toggleFeaturePick({ kind: "void" })}
                  onAdd={(r) => {
                    setVoids((k) => [...k, r]);
                    if (
                      r.kind === "area" ||
                      r.kind === "cylinder" ||
                      r.kind === "box" ||
                      ANCHOR_KINDS.has(r.kind ?? "")
                    )
                      armCenter({ kind: "void", ri: voids.length });
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

            {canTopopt && (
              <div className="grid grid-cols-3 gap-2">
                <Field label="Anchors" hint="auto-keep ring">
                  <label className="flex h-7 items-center gap-1.5 text-[11px] text-zinc-400">
                    <input
                      type="checkbox"
                      className="accent-brand-500"
                      checked={keepMode === "auto"}
                      onChange={(e) => setKeepMode(e.target.checked ? "auto" : "manual")}
                    />
                    keep all
                  </label>
                </Field>
                <Field label="Min member" hint="mm — printable floor">
                  <NumInput
                    value={minMember}
                    step={0.2}
                    className="w-full"
                    onChange={(n) => setMinMember(Math.max(0, n))}
                  />
                </Field>
                <Field label="Threshold" hint="density iso (resample)">
                  <NumInput
                    value={densityThreshold}
                    step={0.05}
                    className="w-full"
                    onChange={(n) => setDensityThreshold(Math.min(0.9, Math.max(0.2, n)))}
                  />
                </Field>
                <Field label="Finishing" hint="re-cut features after">
                  <label className="flex h-7 items-center gap-1.5 text-[11px] text-zinc-400">
                    <input
                      type="checkbox"
                      className="accent-brand-500"
                      checked={autoRefine}
                      onChange={(e) => setAutoRefine(e.target.checked)}
                    />
                    auto-refine
                  </label>
                </Field>
                <Field label="45° support" hint="no overhangs (FDM)">
                  <label className="flex h-7 items-center gap-1.5 text-[11px] text-zinc-400">
                    <input
                      type="checkbox"
                      className="accent-brand-500"
                      checked={selfSupport}
                      onChange={(e) => setSelfSupport(e.target.checked)}
                    />
                    self-supporting
                  </label>
                </Field>
                <Field label="Print dir" hint="build direction">
                  <select
                    className={`${inputCls} px-2 py-1 text-xs`}
                    value={printDir}
                    disabled={!selfSupport}
                    onChange={(e) => setPrintDir(e.target.value as typeof printDir)}
                  >
                    {["+z", "-z", "+x", "-x", "+y", "-y"].map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Multires" hint="coarse warm-start">
                  <label className="flex h-7 items-center gap-1.5 text-[11px] text-zinc-400">
                    <input
                      type="checkbox"
                      className="accent-brand-500"
                      checked={multires}
                      onChange={(e) => setMultires(e.target.checked)}
                    />
                    fast start
                  </label>
                </Field>
                <Field label="Solver" hint="oc = fast, mma = constraints">
                  <select
                    className={`${inputCls} px-2 py-1 text-xs`}
                    value={solver}
                    onChange={(e) => setSolver(e.target.value as typeof solver)}
                  >
                    <option value="oc">oc</option>
                    <option value="mma">mma</option>
                  </select>
                </Field>
                {solver === "mma" && (
                  <>
                    <Field label="MinMax" hint="worst-case over load cases">
                      <label className="flex h-7 items-center gap-1.5 text-[11px] text-zinc-400">
                        <input
                          type="checkbox"
                          className="accent-brand-500"
                          checked={minmax}
                          onChange={(e) => setMinmax(e.target.checked)}
                        />
                        robust
                      </label>
                    </Field>
                    <Field label="Disp cap" hint="max displacement mm (0=off)">
                      <input
                        type="number"
                        min={0}
                        step={0.1}
                        className={`${inputCls} w-16 px-2 py-1 text-xs`}
                        value={dispMax || ""}
                        placeholder="off"
                        onChange={(e) => setDispMax(parseFloat(e.target.value) || 0)}
                      />
                    </Field>
                    <Field label="σ cap" hint="max von Mises MPa (0=off)">
                      <input
                        type="number"
                        min={0}
                        step={1}
                        className={`${inputCls} w-16 px-2 py-1 text-xs`}
                        value={stressMax || ""}
                        placeholder="off"
                        onChange={(e) => setStressMax(parseFloat(e.target.value) || 0)}
                      />
                    </Field>
                  </>
                )}
                <Field label="Min gap" hint="warn on thin gaps mm (0=off)">
                  <input
                    type="number"
                    min={0}
                    step={0.5}
                    className={`${inputCls} w-16 px-2 py-1 text-xs`}
                    value={minGap || ""}
                    placeholder="off"
                    onChange={(e) => setMinGap(parseFloat(e.target.value) || 0)}
                  />
                </Field>
                <Field label="Penal ramp" hint="1.0→target, less stalling">
                  <label className="flex h-7 items-center gap-1.5 text-[11px] text-zinc-400">
                    <input
                      type="checkbox"
                      className="accent-brand-500"
                      checked={penalCont}
                      onChange={(e) => setPenalCont(e.target.checked)}
                    />
                    continue
                  </label>
                </Field>
              </div>
            )}

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
              every case needs at least one support region and one active load
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
