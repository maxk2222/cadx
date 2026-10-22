import {
  Brush,
  Dices,
  Eye,
  Loader2,
  Paintbrush,
  ShieldCheck,
  Sprout,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { api, errMsg } from "../lib/api";
import { num, readStoredAnchors } from "../lib/editorState";
import type {
  FeatureAnchor,
  MeshPreviewResult,
  OrganicParams,
  OrganicStyle,
  RegionSpec,
  VersionDetail,
} from "../lib/types";
import { AnchorMarker, anchorMouth } from "./editor/anchors";
import ModelCanvas, { type PickInfo } from "./editor/ModelCanvas";
import RegionFill, { type RegionFillEntry } from "./editor/RegionFill";
import { Btn, Chip, Field, IconBtn, inputCls, OpacitySlider, useToast } from "./ui";

/* ------------------------------------------------------------------ */
/* Pseudo-topology "organic" restyle — parameter editor with a live    */
/* meshPreview of the exact `organic` mesh op that the commit runs.    */
/* ------------------------------------------------------------------ */

const KEEP_COLOR = "#22c55e";
const CARVE_COLOR = "#a855f7";

const STYLES: OrganicStyle[] = ["gyroid", "trabecular", "voronoi", "branches"];

const QUALITY: { id: string; voxel_mm: number; max_elements: number }[] = [
  { id: "draft", voxel_mm: 3.0, max_elements: 120_000 },
  { id: "normal", voxel_mm: 1.5, max_elements: 150_000 },
  { id: "fine", voxel_mm: 1.0, max_elements: 400_000 },
];

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
      <span className="flex items-baseline justify-between text-sm text-muted">
        {label}
        <span className="tabular-nums text-faint">
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

type BrushMode = "keep" | "carve";

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
  const { t } = useTranslation("editor");
  const toast = useToast();

  /* --- core parameters --------------------------------------------- */
  const [style, setStyle] = useState<OrganicStyle>("gyroid");
  const [preservePct, setPreservePct] = useState(65); // 100 - surface_open
  const [volfrac, setVolfrac] = useState(0.35);
  const [period, setPeriod] = useState(12.0);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 9999));
  const [autoKeep, setAutoKeep] = useState(true);

  /* --- advanced ----------------------------------------------------- */
  const [shellMm, setShellMm] = useState(0.0); // 0 = auto
  const [floorMm, setFloorMm] = useState(2.0);
  const [minMember, setMinMember] = useState(0.0);
  const [threshold, setThreshold] = useState(0.0); // 0 = derive from density
  const [quality, setQuality] = useState("normal");

  /* --- regions ------------------------------------------------------ */
  const [keeps, setKeeps] = useState<RegionSpec[]>([]);
  const [voids, setVoids] = useState<RegionSpec[]>([]);
  const [anchors, setAnchors] = useState<FeatureAnchor[]>([]);

  /* --- painting ----------------------------------------------------- */
  const [brushMode, setBrushMode] = useState<BrushMode | null>(null);
  const [brushR, setBrushR] = useState(4.0);
  const [opacity, setOpacity] = useState(1);
  const [model, setModel] = useState<import("three").Group | null>(null);
  const [fillsCapped, setFillsCapped] = useState(false);
  const [hover, setHover] = useState<PickInfo | null>(null);
  const lastPt = useRef<[number, number, number] | null>(null);

  /* --- preview ------------------------------------------------------ */
  const [preview, setPreview] = useState<MeshPreviewResult | null>(null);
  const [busy, setBusy] = useState<"preview" | "apply" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);
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
      surface_open: 1 - preservePct / 100,
      shell_mm: shellMm || undefined,
      floor_mm: floorMm,
      min_member_mm: minMember || undefined,
      density_threshold: threshold || undefined,
      auto_keep: autoKeep,
      keep_regions: keeps.length ? keeps : undefined,
      void_regions: voids.length ? voids : undefined,
      voxel_mm: q.voxel_mm,
      max_elements: q.max_elements,
    }),
    [
      style,
      volfrac,
      seed,
      period,
      preservePct,
      shellMm,
      floorMm,
      minMember,
      threshold,
      autoKeep,
      keeps,
      voids,
      q,
    ]
  );

  const glb = version.artifacts.find((a) => a.kind === "glb");
  const baseUrl = glb?.url ?? null;
  // while painting keeps the ORIGINAL surface is the pick target — show it,
  // not the organic preview (painted points must land on the real surface)
  const displayUrl = brushMode || showOriginal ? baseUrl : (preview?.glb_url ?? baseUrl);

  /* --- anchor fetch on open: scans + persists the merged table, so the
   *  preview's auto-keep set equals what the commit will protect. The query
   *  key is shared with AnchorsPanel — one fetch serves both. */
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

  useEffect(() => {
    if (!open) return;
    setAnchors(readStoredAnchors(version));
    setPreview(null);
    setErr(null);
    lastPt.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, version.id]);

  const doPreview = async () => {
    const seq = ++previewSeq.current;
    setBusy("preview");
    try {
      const r = await api.meshPreview(version.id, {
        ops: [{ op: "organic", args: args as Record<string, unknown> }],
      });
      if (seq !== previewSeq.current) return; // stale response
      setPreview(r);
      setErr(r.ok ? null : (r.error ?? t("org_preview_failed")));
    } catch (e) {
      if (seq === previewSeq.current) setErr(errMsg(e));
    } finally {
      if (seq === previewSeq.current) setBusy(null);
    }
  };

  // debounced live preview — 700 ms after the last parameter change; skipped
  // while a brush is armed (the preview is hidden then anyway — saves the
  // server a voxelization per painted point)
  useEffect(() => {
    if (!open || !displayUrl || brushMode) return;
    const t = setTimeout(doPreview, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [args, open, brushMode]);

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);

  /* --- region painting ---------------------------------------------- */

  const setActiveList = brushMode === "carve" ? setVoids : setKeeps;

  const onPick = (info: PickInfo) => {
    if (!brushMode) return;
    const p = info.point;
    // dedupe outside the updater — mutating lastPt inside would break under
    // StrictMode's double-invoked updaters (first point of each stroke lost)
    if (lastPt.current) {
      const d = Math.hypot(
        p[0] - lastPt.current[0],
        p[1] - lastPt.current[1],
        p[2] - lastPt.current[2]
      );
      if (d < brushR * 0.4) return;
    }
    lastPt.current = p;
    const pt = [p[0], p[1], p[2]];
    setActiveList((rs) => {
      // append to the active (last) region — start one on first pick
      const idx = rs.length ? rs.length - 1 : -1;
      if (idx < 0 || rs[idx].kind !== "points") {
        return [...rs, { kind: "points", points: [pt], radius_mm: brushR }];
      }
      const pts = [...(rs[idx].points ?? []), pt].slice(-4000);
      return rs.map((r, i) => (i === idx ? { ...r, points: pts } : r));
    });
  };

  const startPainting = (mode: BrushMode) => {
    // pick the list by the requested mode — brushMode is still null here, so
    // the render-derived setActiveList would arm the region in keeps
    const list = mode === "carve" ? voids : keeps;
    const set = mode === "carve" ? setVoids : setKeeps;
    if (list.length >= 10) {
      toast(t("org_region_limit"), "error");
      return;
    }
    set((rs) => [...rs, { kind: "points", points: [], radius_mm: brushR }]);
    lastPt.current = null;
    setBrushMode(mode);
  };

  const stopPainting = () => {
    setBrushMode(null);
    lastPt.current = null;
    // drop the armed-but-empty region so it doesn't produce a warning
    const dropEmpty = (rs: RegionSpec[]) =>
      rs.filter((r) => r.kind !== "points" || (r.points?.length ?? 0) > 0);
    setKeeps(dropEmpty);
    setVoids(dropEmpty);
  };

  // brush-radius changes must reach the armed region — its radius_mm is what
  // the backend resolves, so a stale value would lie about the cursor size
  useEffect(() => {
    if (!brushMode) return;
    const set = brushMode === "carve" ? setVoids : setKeeps;
    set((rs) =>
      rs.map((r, i) =>
        i === rs.length - 1 && r.kind === "points" ? { ...r, radius_mm: brushR } : r
      )
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brushR]);

  /* --- overlays ------------------------------------------------------ */

  const fills = useMemo<RegionFillEntry[]>(
    () => [
      ...keeps
        .filter((r) => r.kind === "points" && r.points?.length)
        .map((r) => ({
          points: r.points!,
          radius: r.radius_mm ?? brushR,
          color: KEEP_COLOR,
        })),
      ...voids
        .filter((r) => r.kind === "points" && r.points?.length)
        .map((r) => ({
          points: r.points!,
          radius: r.radius_mm ?? brushR,
          color: CARVE_COLOR,
        })),
    ],
    [keeps, voids, brushR]
  );

  const modelBox = useMemo(
    () => (model ? new THREE.Box3().setFromObject(model) : null),
    [model]
  );
  const anchorMarkers = useMemo(
    () =>
      anchors
        .map((a) => ({ a, p: anchorMouth(a, modelBox) }))
        .filter((x): x is { a: FeatureAnchor; p: [number, number, number] } => !!x.p),
    [anchors, modelBox]
  );

  const apply = async () => {
    setBusy("apply");
    try {
      // drop armed-but-empty painted regions — a zero-point brush only
      // produces a "matched no cells" warning on the backend
      const nonEmpty = (rs: RegionSpec[]) =>
        rs.filter((r) => r.kind !== "points" || (r.points?.length ?? 0) > 0);
      const cleanKeeps = nonEmpty(keeps);
      const cleanVoids = nonEmpty(voids);
      const r = await api.organic(version.id, {
        ...args,
        keep_regions: cleanKeeps.length ? cleanKeeps : undefined,
        void_regions: cleanVoids.length ? cleanVoids : undefined,
      });
      onJobStarted(r);
    } catch (e) {
      toast(errMsg(e), "error");
      setBusy(null);
    }
  };

  if (!open) return null;

  const stats = preview?.stats;
  const orgMetrics = stats?.report?.find((r) => r.op === "organic")?.organic;
  const coverage = orgMetrics?.surface_coverage;
  const nKeepPts = keeps.reduce((n, r) => n + (r.points?.length ?? 0), 0);
  const nCarvePts = voids.reduce((n, r) => n + (r.points?.length ?? 0), 0);
  const regions: { type: BrushMode; r: RegionSpec; i: number }[] = [
    ...keeps.map((r, i) => ({ type: "keep" as const, r, i })),
    ...voids.map((r, i) => ({ type: "carve" as const, r, i })),
  ];

  const preserveHint =
    preservePct >= 95
      ? t("org_preserve_95")
      : preservePct >= 50
        ? t("org_preserve_50")
        : preservePct > 0
          ? t("org_preserve_1")
          : t("org_preserve_0");

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-bg">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-edge px-4 py-2.5">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Sprout className="h-4 w-4 text-emerald-400" />
          {t("org_title", { n: version.number })}
        </h2>
        <span className="text-xs text-faint">
          {t("org_subtitle")}
        </span>
        <span className="flex-1" />
        {brushMode && (
          <Chip tone={brushMode === "keep" ? "green" : "purple"}>
            <Brush className="h-3 w-3" /> {t("org_painting", { mode: t(`org_type_${brushMode}`) })}
          </Chip>
        )}
        {busy === "preview" && (
          <Chip tone="blue">
            <Loader2 className="h-3 w-3 animate-spin" /> {t("org_preview")}
          </Chip>
        )}
        <OpacitySlider value={opacity} onChange={setOpacity} />
        <IconBtn title={t("common:action.close_esc")} onClick={onClose}>
          <X className="h-4 w-4" />
        </IconBtn>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* canvas */}
        <div className="relative min-w-0 flex-1">
          {displayUrl ? (
            <ModelCanvas
              url={displayUrl}
              opacity={opacity}
              pickEnabled={!!brushMode}
              onPick={onPick}
              onHover={setHover}
              onModel={setModel}
            >
              <RegionFill model={model} entries={fills} onTooDense={setFillsCapped} />
              {anchorMarkers.map(({ a, p }) => (
                <AnchorMarker key={a.index} a={a} pos={p} selected={autoKeep} />
              ))}
              {brushMode && hover && (
                <mesh position={hover.point}>
                  <sphereGeometry args={[brushR, 24, 16]} />
                  <meshBasicMaterial
                    color={brushMode === "keep" ? KEEP_COLOR : CARVE_COLOR}
                    transparent
                    opacity={0.25}
                    depthTest={false}
                  />
                </mesh>
              )}
            </ModelCanvas>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-faint">
              {t("org_no_mesh")}
            </div>
          )}

          {/* legend / status */}
          <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-lg border border-edge bg-bg/90 px-3 py-1.5 text-[11px] backdrop-blur">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: KEEP_COLOR }} />
              {t("org_keep")}{nKeepPts ? ` (${nKeepPts} pts)` : ""}
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: CARVE_COLOR }} />
              {t("org_carve")}{nCarvePts ? ` (${nCarvePts} pts)` : ""}
            </span>
            {!!anchorMarkers.length && (
              <span className="flex items-center gap-1 text-cyan-300">
                <ShieldCheck className="h-3 w-3" /> {t("org_anchors_n", { count: anchorMarkers.length })}
              </span>
            )}
            {fillsCapped && (
              <span className="text-amber-400" title={t("org_fills_hint")}>
                {t("org_fills_hidden")}
              </span>
            )}
            <span className="text-faint">·</span>
            {brushMode ? (
              <span
                className={brushMode === "keep" ? "text-emerald-300" : "text-purple-300"}
              >
                {t("org_painting_src")}
              </span>
            ) : showOriginal ? (
              <span className="text-muted">{t("org_original")}</span>
            ) : preview?.ok ? (
              <span className="text-muted">{t("org_preview_live")}</span>
            ) : (
              <span className="text-faint">{t("org_source_mesh")}</span>
            )}
            {stats?.faces != null && preview?.ok && !showOriginal && (
              <span className="tabular-nums text-faint">
                · {t("org_tris", { n: stats.faces.toLocaleString() })}
              </span>
            )}
            {stats?.watertight != null && preview?.ok && !showOriginal && (
              <span className={stats.watertight ? "text-emerald-400" : "text-amber-400"}>
                · {stats.watertight ? t("org_watertight") : t("org_not_watertight")}
              </span>
            )}
            {coverage != null && preview?.ok && !showOriginal && (
              <span
                className={coverage >= 0.4 ? "text-muted" : "text-amber-400"}
                title={t("org_surface_hint")}
              >
                · {t("org_surface", { pct: Math.round(coverage * 100) })}
              </span>
            )}
          </div>

          {/* compare-original toggle */}
          {preview?.ok && preview.glb_url && !brushMode && (
            <button
              onClick={() => setShowOriginal((s) => !s)}
              className={`absolute right-3 top-3 flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs backdrop-blur transition-colors ${
                showOriginal
                  ? "border-emerald-700 bg-emerald-950/70 text-emerald-200"
                  : "border-edge2 bg-bg/80 text-muted hover:text-fg"
              }`}
              title={t("org_compare_hint")}
            >
              <Eye className="h-3.5 w-3.5" />
              {showOriginal ? t("org_original_btn") : t("org_compare_btn")}
            </button>
          )}
        </div>

        {/* parameter panel */}
        <div className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-edge p-4">
          <Field label={t("org_style")}>
            <div className="grid grid-cols-2 gap-1">
              {STYLES.map((sid) => (
                <button
                  key={sid}
                  onClick={() => setStyle(sid)}
                  title={t(`org_style_${sid}_hint`)}
                  className={`rounded-md border px-2 py-1.5 text-left text-xs transition-colors ${
                    style === sid
                      ? "border-emerald-600 bg-emerald-950/60 text-emerald-200"
                      : "border-edge2 bg-panel text-muted hover:border-edge2 hover:text-fg"
                  }`}
                >
                  {t(`org_style_${sid}`)}
                </button>
              ))}
            </div>
            <span className="mt-1 block text-xs text-faint">
              {t(`org_style_${style}_hint`)}
            </span>
          </Field>

          <Slider
            label={t("org_preserve")}
            value={preservePct}
            min={0}
            max={100}
            step={5}
            format={(v) => `${v} %`}
            onChange={setPreservePct}
          />
          <p className="-mt-2 mb-3 text-[11px] text-faint">{preserveHint}</p>

          <Slider
            label={t("org_density")}
            value={volfrac}
            min={0.1}
            max={0.8}
            step={0.05}
            format={(v) => `${Math.round(v * 100)} %`}
            onChange={setVolfrac}
          />
          <Slider
            label={t("org_cell")}
            value={period}
            min={4}
            max={40}
            step={0.5}
            format={(v) => `${v.toFixed(1)} mm`}
            onChange={setPeriod}
          />
          <Field label={t("org_seed")}>
            <div className="flex gap-1.5">
              <input
                type="number"
                className={inputCls}
                value={seed}
                min={0}
                onChange={(e) => setSeed(Math.max(0, Math.round(num(e.target.value, 0))))}
              />
              <IconBtn
                title={t("org_seed_random")}
                onClick={() => setSeed(Math.floor(Math.random() * 9999))}
              >
                <Dices className="h-4 w-4" />
              </IconBtn>
            </div>
          </Field>

          <label className="mb-3 flex items-center gap-2 text-sm text-muted">
            <input
              type="checkbox"
              checked={autoKeep}
              onChange={(e) => setAutoKeep(e.target.checked)}
              className="accent-brand-500"
            />
            {t("org_protect")}
            {anchors.length > 0 && (
              <span className="text-xs text-cyan-400">({t("org_found", { n: anchors.length })})</span>
            )}
          </label>

          {/* keep + carve regions */}
          <div className="mb-3">
            <div className="mb-1 flex items-center justify-between">
              <span className="text-sm text-muted">{t("org_regions")}</span>
              <div className="flex gap-1">
                {brushMode ? (
                  <Btn variant="ghost" className="!px-2 !py-1 text-xs" onClick={stopPainting}>
                    done
                  </Btn>
                ) : (
                  <>
                    <button
                      onClick={() => startPainting("keep")}
                      disabled={!baseUrl}
                      className="flex items-center gap-1 rounded border border-edge2 px-2 py-1 text-xs text-muted hover:border-emerald-700 hover:text-emerald-300 disabled:opacity-40"
                      title={t("org_keep_hint")}
                    >
                      <Paintbrush className="h-3 w-3" /> {t("org_keep")}
                    </button>
                    <button
                      onClick={() => startPainting("carve")}
                      disabled={!baseUrl}
                      className="flex items-center gap-1 rounded border border-edge2 px-2 py-1 text-xs text-muted hover:border-purple-700 hover:text-purple-300 disabled:opacity-40"
                      title={t("org_carve_hint")}
                    >
                      <Paintbrush className="h-3 w-3" /> {t("org_carve")}
                    </button>
                  </>
                )}
                {regions.length > 0 && (
                  <IconBtn
                    title={t("org_clear_regions")}
                    danger
                    onClick={() => {
                      setKeeps([]);
                      setVoids([]);
                    }}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </IconBtn>
                )}
              </div>
            </div>
            {brushMode && (
              <label className="mb-2 flex items-center gap-2 text-xs text-faint">
                {t("org_brush")}
                <input
                  type="range"
                  min={1}
                  max={30}
                  step={0.5}
                  value={brushR}
                  onChange={(e) => setBrushR(num(e.target.value, 4))}
                  className={`w-24 ${brushMode === "keep" ? "accent-emerald-500" : "accent-purple-500"}`}
                />
                <span className="w-10 tabular-nums">{brushR} mm</span>
              </label>
            )}
            {regions.length === 0 && !brushMode && (
              <p className="text-xs text-faint">
                {t("org_regions_none")}
              </p>
            )}
            {regions.map(({ type, r, i }) => (
              <div
                key={`${type}-${i}`}
                className="mb-1 flex items-center justify-between rounded border border-edge bg-panel/60 px-2 py-1 text-xs text-muted"
              >
                <span className="flex items-center gap-1.5">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ background: type === "keep" ? KEEP_COLOR : CARVE_COLOR }}
                  />
                  {t(`org_type_${type}`)} #{i + 1} — {r.points?.length ?? 0} {t("org_pts")} · r=
                  {r.radius_mm ?? brushR} mm
                </span>
                <IconBtn
                  title={t("org_del_region")}
                  danger
                  onClick={() =>
                    (type === "keep" ? setKeeps : setVoids)((k) =>
                      k.filter((_, j) => j !== i)
                    )
                  }
                >
                  <Trash2 className="h-3 w-3" />
                </IconBtn>
              </div>
            ))}
          </div>

          {/* advanced */}
          <details className="mb-3 rounded border border-edge bg-panel/40 px-3 py-2">
            <summary className="cursor-pointer text-xs font-medium text-muted select-none hover:text-fg">
              {t("org_advanced")}
            </summary>
            <div className="mt-3">
              <Slider
                label={t("org_skin")}
                value={shellMm}
                min={0}
                max={8}
                step={0.5}
                format={(v) => (v ? `${v.toFixed(1)} mm` : t("org_auto"))}
                onChange={setShellMm}
              />
              <Slider
                label={t("org_floor")}
                value={floorMm}
                min={0}
                max={15}
                step={0.5}
                format={(v) => (v ? `${v.toFixed(1)} mm` : t("org_off"))}
                onChange={setFloorMm}
              />
              <Slider
                label={t("org_min_member")}
                value={minMember}
                min={0}
                max={8}
                step={0.25}
                format={(v) => (v ? `${v.toFixed(2)} mm` : t("org_auto"))}
                onChange={setMinMember}
              />
              <Slider
                label={t("org_iso")}
                value={threshold}
                min={0}
                max={0.9}
                step={0.05}
                format={(v) => (v ? v.toFixed(2) : t("org_auto"))}
                onChange={(v) => setThreshold(v < 0.2 ? 0 : v)}
              />
              <Field label={t("org_quality")}>
                <div className="flex overflow-hidden rounded border border-edge2 text-[11px]">
                  {QUALITY.map((x) => (
                    <button
                      key={x.id}
                      onClick={() => setQuality(x.id)}
                      title={t("org_quality_hint", { v: x.voxel_mm, n: x.max_elements.toLocaleString() })}
                      className={`flex-1 px-2 py-1 ${
                        quality === x.id
                          ? "bg-edge2 text-fg"
                          : "text-faint hover:text-fg"
                      }`}
                    >
                      {t(`org_q_${x.id}`)}
                    </button>
                  ))}
                </div>
              </Field>
            </div>
          </details>

          {/* disclaimer */}
          <div className="mb-3 flex gap-2 rounded-md border border-amber-900/60 bg-amber-950/30 p-2 text-xs text-amber-200/90">
            <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              {t("org_disclaimer_a")} <b>{t("org_disclaimer_b")}</b> {t("org_disclaimer_c")}
            </span>
          </div>

          {err && (
            <div className="mb-3 rounded border border-red-900/60 bg-red-950/40 p-2 text-xs text-red-300">
              {err}
            </div>
          )}

          <div className="mt-auto flex gap-2 pt-2">
            <Btn variant="ghost" className="flex-1" onClick={onClose}>
              {t("common:action.cancel")}
            </Btn>
            <Btn
              variant="success"
              className="flex-1"
              onClick={apply}
              busy={busy === "apply"}
              disabled={!!err && !preview?.ok}
              title={t("org_apply_hint")}
            >
              <Sprout className="h-4 w-4" /> {t("org_apply")}
            </Btn>
          </div>
        </div>
      </div>
    </div>
  );
}
