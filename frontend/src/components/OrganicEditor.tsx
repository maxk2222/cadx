import {
  Brush,
  Dices,
  Loader2,
  Paintbrush,
  Sprout,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import type {
  MeshPreviewResult,
  OrganicParams,
  OrganicStyle,
  RegionSpec,
  VersionDetail,
} from "../lib/types";
import ModelCanvas, { type PickInfo } from "./editor/ModelCanvas";
import RegionFill, { type RegionFillEntry } from "./editor/RegionFill";
import { Btn, Chip, Field, IconBtn, inputCls, useToast } from "./ui";

/* ------------------------------------------------------------------ */
/* Pseudo-topology "organic" restyle — parameter editor with a live    */
/* meshPreview of the exact `organic` mesh op that the commit runs.    */
/* ------------------------------------------------------------------ */

const KEEP_COLOR = "#22c55e";

const STYLES: { id: OrganicStyle; label: string; hint: string }[] = [
  { id: "gyroid", label: "gyroid", hint: "smooth TPMS sheet — flowing minimal surface" },
  { id: "trabecular", label: "trabecular", hint: "bone-like spongy core" },
  { id: "voronoi", label: "voronoi", hint: "organic cell walls" },
  { id: "branches", label: "branches", hint: "floor-rooted struts + canopy" },
];

const QUALITY: { id: string; label: string; voxel_mm: number; max_elements: number }[] = [
  { id: "draft", label: "draft", voxel_mm: 3.0, max_elements: 120_000 },
  { id: "normal", label: "normal", voxel_mm: 1.5, max_elements: 150_000 },
  { id: "fine", label: "fine", voxel_mm: 1.0, max_elements: 400_000 },
];

const num = (s: string, fb = 0) => {
  const n = Number(s);
  return Number.isFinite(n) ? n : fb;
};

function Slider({
  label,
  value,
  min,
  max,
  step = 0.5,
  unit = "mm",
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="mb-3 block">
      <span className="flex items-baseline justify-between text-sm text-zinc-400">
        {label}
        <span className="tabular-nums text-zinc-500">
          {format ? format(value) : `${value} ${unit}`}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(num(e.target.value, value))}
        className="mt-1 w-full accent-brand-500"
      />
    </label>
  );
}

export default function OrganicEditor({
  open,
  onClose,
  version,
  onJobStarted,
}: {
  open: boolean;
  onClose: () => void;
  version: VersionDetail;
  onJobStarted: (r: { version_id: string; job_id: string }) => void;
}) {
  const toast = useToast();

  /* --- parameters ------------------------------------------------- */
  const [style, setStyle] = useState<OrganicStyle>("gyroid");
  const [volfrac, setVolfrac] = useState(0.35);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 9999));
  const [period, setPeriod] = useState(12.0);
  const [shellMm, setShellMm] = useState(0.0);
  const [floorMm, setFloorMm] = useState(2.0);
  const [minMember, setMinMember] = useState(0.0);
  const [autoKeep, setAutoKeep] = useState(true);
  const [quality, setQuality] = useState("normal");
  const [keeps, setKeeps] = useState<RegionSpec[]>([]);

  /* --- painting --------------------------------------------------- */
  const [painting, setPainting] = useState(false);
  const [brushR, setBrushR] = useState(4.0);
  const [model, setModel] = useState<import("three").Group | null>(null);
  const [hover, setHover] = useState<PickInfo | null>(null);
  const lastPt = useRef<[number, number, number] | null>(null);

  /* --- preview ---------------------------------------------------- */
  const [preview, setPreview] = useState<MeshPreviewResult | null>(null);
  const [busy, setBusy] = useState<"preview" | "apply" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const previewSeq = useRef(0);

  const q = QUALITY.find((x) => x.id === quality) ?? QUALITY[1];

  /** args identical for preview (mesh op) and commit (POST /organic) — the
   *  two code paths share organic_transform, so what you see is what you get. */
  const args = useMemo<OrganicParams>(
    () => ({
      style,
      volfrac,
      seed,
      period_mm: period,
      shell_mm: shellMm,
      floor_mm: floorMm,
      min_member_mm: minMember || undefined,
      auto_keep: autoKeep,
      keep_regions: keeps.length ? keeps : undefined,
      voxel_mm: q.voxel_mm,
      max_elements: q.max_elements,
    }),
    [style, volfrac, seed, period, shellMm, floorMm, minMember, autoKeep, keeps, q]
  );

  const glb = version.artifacts.find((a) => a.kind === "glb");
  const baseUrl = glb?.url ?? null;
  // while painting keeps the ORIGINAL surface is the pick target — show it,
  // not the organic preview (painted points must land on the real surface)
  const displayUrl = painting ? baseUrl : (preview?.glb_url ?? baseUrl);

  const doPreview = async () => {
    const seq = ++previewSeq.current;
    setBusy("preview");
    try {
      const r = await api.meshPreview(version.id, {
        ops: [{ op: "organic", args: args as Record<string, unknown> }],
      });
      if (seq !== previewSeq.current) return; // stale response
      setPreview(r);
      setErr(r.ok ? null : (r.error ?? "preview failed"));
    } catch (e) {
      if (seq === previewSeq.current) setErr(String(e));
    } finally {
      if (seq === previewSeq.current) setBusy(null);
    }
  };

  // debounced live preview — 700 ms after the last parameter change
  useEffect(() => {
    if (!open || !displayUrl) return;
    const t = setTimeout(doPreview, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [args, open]);

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);

  /* --- keep painting ----------------------------------------------- */

  const onPick = (info: PickInfo) => {
    if (!painting) return;
    const p = info.point;
    setKeeps((ks) => {
      // append to the active (last) keep region — start one on first pick
      const idx = ks.length ? ks.length - 1 : -1;
      if (idx >= 0 && lastPt.current) {
        const d = Math.hypot(
          p[0] - lastPt.current[0],
          p[1] - lastPt.current[1],
          p[2] - lastPt.current[2]
        );
        if (d < brushR * 0.4) return ks;
      }
      lastPt.current = p;
      const pt = [p[0], p[1], p[2]];
      if (idx < 0 || ks[idx].kind !== "points") {
        return [...ks, { kind: "points", points: [pt], radius_mm: brushR }];
      }
      const pts = [...(ks[idx].points ?? []), pt].slice(-4000);
      return ks.map((r, i) => (i === idx ? { ...r, points: pts } : r));
    });
  };

  const startPainting = () => {
    setKeeps((ks) => [...ks, { kind: "points", points: [], radius_mm: brushR }]);
    lastPt.current = null;
    setPainting(true);
  };

  const stopPainting = () => {
    setPainting(false);
    lastPt.current = null;
    // drop the armed-but-empty region so it doesn't produce a warning
    setKeeps((ks) =>
      ks.filter((r) => r.kind !== "points" || (r.points?.length ?? 0) > 0)
    );
  };

  /* --- overlays ---------------------------------------------------- */

  const fills = useMemo<RegionFillEntry[]>(
    () =>
      keeps
        .filter((r) => r.kind === "points" && r.points?.length)
        .map((r) => ({
          points: r.points!,
          radius: r.radius_mm ?? brushR,
          color: KEEP_COLOR,
        })),
    [keeps, brushR]
  );

  const apply = async () => {
    setBusy("apply");
    try {
      const r = await api.organic(version.id, args);
      onJobStarted(r);
    } catch (e) {
      toast(String(e), "error");
      setBusy(null);
    }
  };

  if (!open) return null;

  const stats = preview?.stats;
  const nPts = keeps.reduce((n, r) => n + (r.points?.length ?? 0), 0);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-zinc-950">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Sprout className="h-4 w-4 text-emerald-400" />
          Organic structure — v{version.number}
        </h2>
        <span className="text-xs text-zinc-500">
          pseudo-topology restyle — pick a style, tune, then apply
        </span>
        <span className="flex-1" />
        {painting && (
          <Chip tone="green">
            <Brush className="h-3 w-3" /> painting keep regions — drag on the model
          </Chip>
        )}
        {busy === "preview" && (
          <Chip tone="blue">
            <Loader2 className="h-3 w-3 animate-spin" /> preview…
          </Chip>
        )}
        <IconBtn title="close (Esc)" onClick={onClose}>
          <X className="h-4 w-4" />
        </IconBtn>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* canvas */}
        <div className="relative min-w-0 flex-1">
          {displayUrl ? (
            <ModelCanvas
              url={displayUrl}
              pickEnabled={painting}
              onPick={onPick}
              onHover={setHover}
              onModel={setModel}
            >
              <RegionFill model={model} entries={fills} />
              {painting && hover && (
                <mesh position={hover.point}>
                  <sphereGeometry args={[brushR, 24, 16]} />
                  <meshBasicMaterial
                    color={KEEP_COLOR}
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

          {/* legend / status */}
          <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/90 px-3 py-1.5 text-[11px] backdrop-blur">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: KEEP_COLOR }} />
              keep{nPts ? ` (${nPts} pts)` : ""}
            </span>
            <span className="text-zinc-600">·</span>
            {painting ? (
              <span className="text-emerald-300">painting on the source mesh</span>
            ) : preview?.ok ? (
              <span className="text-zinc-400">live preview of the organic structure</span>
            ) : (
              <span className="text-zinc-500">source mesh</span>
            )}
            {stats?.faces != null && preview?.ok && (
              <span className="tabular-nums text-zinc-500">
                · {stats.faces.toLocaleString()} tris
              </span>
            )}
            {stats?.watertight != null && preview?.ok && (
              <span className={stats.watertight ? "text-emerald-400" : "text-amber-400"}>
                · {stats.watertight ? "watertight" : "not watertight"}
              </span>
            )}
          </div>
        </div>

        {/* parameter panel */}
        <div className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-zinc-800 p-4">
          <Field label="Style">
            <div className="grid grid-cols-2 gap-1">
              {STYLES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setStyle(s.id)}
                  title={s.hint}
                  className={`rounded-md border px-2 py-1.5 text-left text-xs transition-colors ${
                    style === s.id
                      ? "border-emerald-600 bg-emerald-950/60 text-emerald-200"
                      : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <span className="mt-1 block text-xs text-zinc-600">
              {STYLES.find((s) => s.id === style)?.hint}
            </span>
          </Field>

          <Slider
            label="Density"
            value={volfrac}
            min={0.1}
            max={0.8}
            step={0.05}
            format={(v) => `${Math.round(v * 100)} %`}
            onChange={setVolfrac}
          />
          <Slider
            label="Feature size"
            value={period}
            min={4}
            max={40}
            step={0.5}
            format={(v) => `${v.toFixed(1)} mm`}
            onChange={setPeriod}
          />
          <Field label="Seed">
            <div className="flex gap-1.5">
              <input
                type="number"
                className={inputCls}
                value={seed}
                min={0}
                onChange={(e) => setSeed(Math.max(0, Math.round(num(e.target.value, 0))))}
              />
              <IconBtn
                title="random seed"
                onClick={() => setSeed(Math.floor(Math.random() * 9999))}
              >
                <Dices className="h-4 w-4" />
              </IconBtn>
            </div>
          </Field>

          <Slider
            label="Shell"
            value={shellMm}
            min={0}
            max={8}
            step={0.5}
            format={(v) => (v ? `${v.toFixed(1)} mm` : "off")}
            onChange={setShellMm}
          />
          <Slider
            label="Floor (solid stand band)"
            value={floorMm}
            min={0}
            max={15}
            step={0.5}
            format={(v) => (v ? `${v.toFixed(1)} mm` : "off")}
            onChange={setFloorMm}
          />
          <Slider
            label="Min member"
            value={minMember}
            min={0}
            max={8}
            step={0.25}
            format={(v) => (v ? `${v.toFixed(2)} mm` : "auto")}
            onChange={setMinMember}
          />

          <label className="mb-3 flex items-center gap-2 text-sm text-zinc-400">
            <input
              type="checkbox"
              checked={autoKeep}
              onChange={(e) => setAutoKeep(e.target.checked)}
              className="accent-brand-500"
            />
            protect detected anchors (holes / inserts)
          </label>

          <Field label="Quality">
            <div className="flex overflow-hidden rounded border border-zinc-700 text-[11px]">
              {QUALITY.map((x) => (
                <button
                  key={x.id}
                  onClick={() => setQuality(x.id)}
                  title={`voxel ${x.voxel_mm} mm · ≤${x.max_elements.toLocaleString()} cells`}
                  className={`flex-1 px-2 py-1 ${
                    quality === x.id
                      ? "bg-zinc-700 text-zinc-100"
                      : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {x.label}
                </button>
              ))}
            </div>
          </Field>

          {/* keep regions */}
          <div className="mb-3">
            <div className="mb-1 flex items-center justify-between">
              <span className="text-sm text-zinc-400">Keep regions</span>
              <div className="flex gap-1">
                {painting ? (
                  <Btn variant="ghost" className="!px-2 !py-1 text-xs" onClick={stopPainting}>
                    done
                  </Btn>
                ) : (
                  <button
                    onClick={startPainting}
                    disabled={!baseUrl}
                    className="flex items-center gap-1 rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-400 hover:border-emerald-700 hover:text-emerald-300 disabled:opacity-40"
                    title="paint zones that must stay solid (mounts, mating faces…)"
                  >
                    <Paintbrush className="h-3 w-3" /> paint
                  </button>
                )}
                {keeps.length > 0 && (
                  <IconBtn title="clear all keep regions" danger onClick={() => setKeeps([])}>
                    <Trash2 className="h-3.5 w-3.5" />
                  </IconBtn>
                )}
              </div>
            </div>
            {painting && (
              <label className="mb-2 flex items-center gap-2 text-xs text-zinc-500">
                brush
                <input
                  type="range"
                  min={1}
                  max={30}
                  step={0.5}
                  value={brushR}
                  onChange={(e) => setBrushR(num(e.target.value, 4))}
                  className="w-24 accent-emerald-500"
                />
                <span className="w-10 tabular-nums">{brushR} mm</span>
              </label>
            )}
            {keeps.length === 0 && !painting && (
              <p className="text-xs text-zinc-600">
                none — painted zones stay solid (mounts, mating faces)
              </p>
            )}
            {keeps.map((r, i) => (
              <div
                key={i}
                className="mb-1 flex items-center justify-between rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1 text-xs text-zinc-400"
              >
                <span>
                  keep #{i + 1} — {r.points?.length ?? 0} pts · r={r.radius_mm ?? brushR} mm
                </span>
                <IconBtn
                  title="delete region"
                  danger
                  onClick={() => setKeeps((k) => k.filter((_, j) => j !== i))}
                >
                  <Trash2 className="h-3 w-3" />
                </IconBtn>
              </div>
            ))}
          </div>

          {/* disclaimer */}
          <div className="mb-3 flex gap-2 rounded-md border border-amber-900/60 bg-amber-950/30 p-2 text-xs text-amber-200/90">
            <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              Aesthetic styling only — this is <b>not</b> a structural optimization. No FEM
              verification runs; check strength separately if the part is load-bearing.
            </span>
          </div>

          {err && (
            <div className="mb-3 rounded border border-red-900/60 bg-red-950/40 p-2 text-xs text-red-300">
              {err}
            </div>
          )}

          <div className="mt-auto flex gap-2 pt-2">
            <Btn variant="ghost" className="flex-1" onClick={onClose}>
              cancel
            </Btn>
            <Btn
              variant="success"
              className="flex-1"
              onClick={apply}
              busy={busy === "apply"}
              disabled={!!err && !preview?.ok}
              title="create a child version with this structure"
            >
              <Sprout className="h-4 w-4" /> apply
            </Btn>
          </div>
        </div>
      </div>
    </div>
  );
}
