import { Brush, Loader2, Plus, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { api } from "../lib/api";
import type {
  FemPreviewResult,
  LoadCase,
  RegionSpec,
  VersionDetail,
} from "../lib/types";
import ModelCanvas, { type PickInfo } from "./editor/ModelCanvas";
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

/** Where the next brush stroke appends its points. */
type PaintTarget =
  | { kind: "fixed"; ci: number; ri: number }
  | { kind: "load"; ci: number; li: number }
  | { kind: "keep"; ri: number }
  | null;

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
const FORCE_DIRS = FACES;

function emptyPointsRegion(radius: number): RegionSpec {
  return { kind: "points", points: [], radius_mm: radius };
}

function regionLabel(r: RegionSpec): string {
  if (r.kind === "points") return `painted · ${r.points?.length ?? 0} pts`;
  if (r.kind === "face") return `face ${r.axis ?? "?"}`;
  return `${r.kind ?? "region"}${r.description ? ` · ${r.description}` : ""}`;
}

/* ------------------------------------------------------------------ */
/* region row                                                          */
/* ------------------------------------------------------------------ */

function RegionRow({
  r,
  color,
  painting,
  onPaint,
  onChange,
  onDelete,
}: {
  r: RegionSpec;
  color: string;
  painting: boolean;
  onPaint: () => void;
  onChange: (r: RegionSpec) => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={`mb-1 flex items-center gap-1.5 rounded border px-1.5 py-1 text-xs ${
        painting ? "border-amber-600 bg-amber-950/30" : "border-zinc-800 bg-zinc-900/60"
      }`}
    >
      <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
      {r.kind === "face" ? (
        <select
          className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
          value={r.axis ?? "+x"}
          onChange={(e) => onChange({ ...r, axis: e.target.value })}
        >
          {FACES.map((f) => (
            <option key={f} value={f}>
              {FACE_LABELS[f]}
            </option>
          ))}
        </select>
      ) : (
        <span className="min-w-0 flex-1 truncate text-zinc-300" title={regionLabel(r)}>
          {regionLabel(r)}
        </span>
      )}
      {r.kind === "points" && (
        <>
          <input
            type="range"
            min={1}
            max={30}
            step={0.5}
            title={`brush radius ${r.radius_mm ?? 3} mm`}
            className="w-14 accent-brand-500"
            value={r.radius_mm ?? 3}
            onChange={(e) => onChange({ ...r, radius_mm: num(e.target.value, 3) })}
          />
          <span className="w-8 text-[10px] text-zinc-500">{r.radius_mm ?? 3}mm</span>
          <button
            title={painting ? "stop painting" : "paint on the model"}
            onClick={onPaint}
            className={`rounded p-0.5 ${
              painting ? "text-amber-300" : "text-zinc-500 hover:text-zinc-200"
            }`}
          >
            <Brush className="h-3.5 w-3.5" />
          </button>
        </>
      )}
      <button title="remove region" onClick={onDelete} className="text-zinc-600 hover:text-red-400">
        <Trash2 className="h-3 w-3" />
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* overlay: all painted points as point clouds (model-space mm)        */
/* ------------------------------------------------------------------ */

function PointCloud({ pts, color, size }: { pts: [number, number, number][]; color: string; size: number }) {
  const arr = useMemo(() => new Float32Array(pts.flat()), [pts]);
  if (!pts.length) return null;
  return (
    <points key={pts.length}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[arr, 3]} />
      </bufferGeometry>
      <pointsMaterial color={color} size={size} sizeAttenuation transparent opacity={0.95} depthTest={false} />
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
  const [target, setTarget] = useState<PaintTarget>(null);
  const [hover, setHover] = useState<PickInfo | null>(null);
  const [preview, setPreview] = useState<FemPreviewResult | null>(null);
  const [busy, setBusy] = useState<"preview" | "fem" | "opt" | null>(null);
  const lastPt = useState<{ p: [number, number, number] | null }>(() => ({ p: null }))[0];

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
      setCase(t.ci, { fixed: c.fixed.map((x, i) => (i === t.ri ? r : x)) });
    } else if (t.kind === "load") {
      const c = cases[t.ci];
      setCase(t.ci, {
        loads: c.loads.map((x, i) => (i === t.li ? { ...x, region: r } : x)),
      });
    } else {
      setKeep((k) => k.map((x, i) => (i === t.ri ? r : x)));
    }
  };

  const delRegion = (t: Exclude<PaintTarget, null>) => {
    if (t.kind === "fixed") {
      const c = cases[t.ci];
      setCase(t.ci, { fixed: c.fixed.filter((_, i) => i !== t.ri) });
    } else if (t.kind === "load") {
      const c = cases[t.ci];
      setCase(t.ci, { loads: c.loads.filter((_, i) => i !== t.li) });
    } else {
      setKeep((k) => k.filter((_, i) => i !== t.ri));
    }
    if (target && JSON.stringify(target) === JSON.stringify(t)) setTarget(null);
  };

  /** Append a fresh painted region and arm the brush on it. */
  const startPaint = (t: Exclude<PaintTarget, null>, append: (r: RegionSpec) => void) => {
    append(emptyPointsRegion(brushR));
    setTarget(t);
    lastPt.p = null;
  };

  /* --- painting --- */

  const onPick = (info: PickInfo) => {
    if (!target) return;
    const r = getRegion(target);
    if (!r || r.kind !== "points") return;
    const p = info.point;
    if (lastPt.p) {
      const d = Math.hypot(p[0] - lastPt.p[0], p[1] - lastPt.p[1], p[2] - lastPt.p[2]);
      if (d < brushR * 0.4) return;
    }
    lastPt.p = p;
    const pts = [...(r.points ?? []), p];
    // running average of face normals — used by FEM for pressure direction
    const prev = (r.normal ?? info.normal) as number[];
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

  const checkRegions = async () => {
    setBusy("preview");
    setPreview(null);
    try {
      await persist();
      const r = await api.femPreview(version.id, { load_cases: cases, keep_regions: keep });
      setPreview(r);
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

  /* --- overlay point groups --- */

  const clouds = useMemo(() => {
    const fixed: [number, number, number][] = [];
    const load: [number, number, number][] = [];
    cases.forEach((c) => {
      c.fixed.forEach((r) => r.points?.forEach((p) => fixed.push(p as [number, number, number])));
      c.loads.forEach((l) =>
        l.region.points?.forEach((p) => load.push(p as [number, number, number]))
      );
    });
    const keepPts = keep.flatMap(
      (r) => (r.points ?? []) as [number, number, number][]
    );
    return { fixed, load, keepPts };
  }, [cases, keep]);

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

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-zinc-950">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
        <h2 className="text-sm font-semibold">
          Structural editor — v{version.number}
        </h2>
        <span className="text-xs text-zinc-500">
          paint support / load / keep regions directly on the surface
        </span>
        <span className="flex-1" />
        {target && (
          <Chip tone="amber">
            <Brush className="h-3 w-3" /> painting{" "}
            {target.kind === "fixed" ? "support" : target.kind === "load" ? "load" : "keep"} —
            drag on the model
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
              pickEnabled={!!target}
              onPick={onPick}
              onHover={setHover}
            >
              <PointCloud pts={clouds.fixed} color={COLORS.fixed} size={brushR * 0.5} />
              <PointCloud pts={clouds.load} color={COLORS.load} size={brushR * 0.5} />
              <PointCloud pts={clouds.keepPts} color={COLORS.keep} size={brushR * 0.5} />
              {target && hover && (
                <mesh position={hover.point}>
                  <sphereGeometry args={[brushR, 24, 16]} />
                  <meshBasicMaterial
                    color={COLORS[target.kind]}
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
          <div className="absolute bottom-3 left-3 flex gap-2 rounded-lg border border-zinc-800 bg-zinc-950/90 px-3 py-1.5 text-[11px] backdrop-blur">
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
          </div>
        </div>

        {/* right panel */}
        <div className="flex w-[400px] shrink-0 flex-col overflow-y-auto border-l border-zinc-800">
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
                    onPaint={() =>
                      setTarget(sameTarget({ kind: "fixed", ci, ri }) ? null : { kind: "fixed", ci, ri })
                    }
                    onChange={(nr) => setRegion({ kind: "fixed", ci, ri }, nr)}
                    onDelete={() => delRegion({ kind: "fixed", ci, ri })}
                  />
                ))}
                <div className="mb-2 flex gap-1">
                  <button
                    className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                    onClick={() =>
                      startPaint({ kind: "fixed", ci, ri: c.fixed.length }, (r) =>
                        setCase(ci, { fixed: [...c.fixed, r] })
                      )
                    }
                  >
                    <Brush className="h-3 w-3" /> paint support
                  </button>
                  <button
                    className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                    onClick={() =>
                      setCase(ci, { fixed: [...c.fixed, { kind: "face", axis: "-x" }] })
                    }
                  >
                    <Plus className="h-3 w-3" /> face
                  </button>
                </div>

                {/* loads */}
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  loads
                </div>
                {c.loads.map((l, li) => {
                  const force = l.force_n ?? [0, 0, 0];
                  let fi = 0;
                  force.forEach((f, j) => {
                    if (Math.abs(f) > Math.abs(force[fi])) fi = j;
                  });
                  const mag = Math.abs(force[fi]) || 50;
                  const dir = `${force[fi] < 0 ? "-" : "+"}${"xyz"[fi]}`;
                  return (
                    <div key={li} className="mb-1.5">
                      <RegionRow
                        r={l.region}
                        color={COLORS.load}
                        painting={sameTarget({ kind: "load", ci, li })}
                        onPaint={() =>
                          setTarget(sameTarget({ kind: "load", ci, li }) ? null : { kind: "load", ci, li })
                        }
                        onChange={(nr) => setRegion({ kind: "load", ci, li }, nr)}
                        onDelete={() => delRegion({ kind: "load", ci, li })}
                      />
                      <div className="flex items-center gap-1.5 pl-4">
                        <input
                          type="number"
                          className={`${inputCls} w-20 px-2 py-0.5 text-[11px]`}
                          value={mag}
                          min={0.1}
                          step={5}
                          title="force magnitude (N) — set pressure via description for now"
                          onChange={(e) => {
                            const m = num(e.target.value, 0);
                            const f = [0, 0, 0];
                            const ax = "xyz".indexOf(dir[1]);
                            f[ax] = dir.startsWith("-") ? -m : m;
                            setCase(ci, {
                              loads: c.loads.map((x, i) =>
                                i === li ? { ...x, force_n: f } : x
                              ),
                            });
                          }}
                        />
                        <span className="text-[10px] text-zinc-500">N</span>
                        <select
                          className={`${inputCls} flex-1 px-1 py-0.5 text-[11px]`}
                          value={dir}
                          onChange={(e) => {
                            const f = [0, 0, 0];
                            const ax = "xyz".indexOf(e.target.value[1]);
                            f[ax] = e.target.value.startsWith("-") ? -mag : mag;
                            setCase(ci, {
                              loads: c.loads.map((x, i) =>
                                i === li ? { ...x, force_n: f } : x
                              ),
                            });
                          }}
                        >
                          {FORCE_DIRS.map((d) => (
                            <option key={d} value={d}>
                              {d.toUpperCase()}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  );
                })}
                <div className="flex gap-1">
                  <button
                    className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                    onClick={() =>
                      startPaint({ kind: "load", ci, li: c.loads.length }, (r) =>
                        setCase(ci, {
                          loads: [...c.loads, { region: r, force_n: [0, 0, -50] }],
                        })
                      )
                    }
                  >
                    <Brush className="h-3 w-3" /> paint load
                  </button>
                  <button
                    className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                    onClick={() =>
                      setCase(ci, {
                        loads: [
                          ...c.loads,
                          { region: { kind: "face", axis: "+x" }, force_n: [0, 0, -50] },
                        ],
                      })
                    }
                  >
                    <Plus className="h-3 w-3" /> face
                  </button>
                </div>
              </div>
            ))}

            <button
              className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200"
              onClick={() =>
                setCases((cs) => [
                  ...cs,
                  { name: `case${cs.length + 1}`, fixed: [], loads: [] },
                ])
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
                    onPaint={() =>
                      setTarget(sameTarget({ kind: "keep", ri }) ? null : { kind: "keep", ri })
                    }
                    onChange={(nr) => setRegion({ kind: "keep", ri }, nr)}
                    onDelete={() => delRegion({ kind: "keep", ri })}
                  />
                ))}
                <button
                  className="flex items-center gap-1 rounded border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                  onClick={() =>
                    startPaint({ kind: "keep", ri: keep.length }, (r) =>
                      setKeep((k) => [...k, r])
                    )
                  }
                >
                  <Brush className="h-3 w-3" /> paint keep region
                </button>
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
                {!!preview.pngs.length && (
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
              title="resolve regions on the voxel domain — shows hit counts + warnings"
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
