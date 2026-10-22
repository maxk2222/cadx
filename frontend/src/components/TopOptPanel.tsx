import { Loader2, Shrink, Wrench } from "lucide-react";
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, errMsg } from "../lib/api";
import type { RefineInfo, TopOptInfo } from "../lib/types";
import { Btn, Card, SectionTitle, useToast } from "./ui";

export type { RefineInfo, TopOptInfo };

function sfTone(sf: number | null | undefined): string {
  if (sf == null) return "text-zinc-400";
  if (sf < 1) return "text-red-400 font-semibold";
  if (sf < 1.5) return "text-amber-400";
  return "text-emerald-400";
}

export default function TopOptPanel({
  topopt,
  refine,
  versionId,
  onJobStarted,
}: {
  topopt: TopOptInfo;
  refine?: RefineInfo;
  versionId?: string;
  onJobStarted?: (r: { version_id: string; job_id: string }) => void;
}) {
  const toast = useToast();
  const [threshold, setThreshold] = useState(topopt.density_threshold ?? 0.5);
  const [minMember, setMinMember] = useState(0);
  const [busy, setBusy] = useState(false);
  const [restorePlanes, setRestorePlanes] = useState(true);
  const [helicalThreads, setHelicalThreads] = useState(false);
  const [autoPlane, setAutoPlane] = useState(false);
  // the panel is rendered unkeyed — reset draft state when the version changes
  useEffect(() => {
    setThreshold(topopt.density_threshold ?? 0.5);
    setMinMember(0);
    setBusy(false);
    setRestorePlanes(true);
    setHelicalThreads(false);
    setAutoPlane(false);
  }, [versionId, topopt.density_threshold]);
  const resamplable = !!versionId && !!topopt.density_relpath;
  const refinable = !!versionId && !!onJobStarted;

  const resample = async () => {
    if (!versionId || !onJobStarted) return;
    setBusy(true);
    try {
      const r = await api.topoptResample(versionId, {
        density_threshold: threshold,
        min_member_mm: minMember || undefined,
      });
      onJobStarted(r);
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy(false);
    }
  };

  const runRefine = async () => {
    if (!versionId || !onJobStarted) return;
    setBusy(true);
    try {
      const r = await api.refine(versionId, {
        restore_planes: restorePlanes,
        helical_threads: helicalThreads,
        auto_plane_restore: autoPlane,
      });
      onJobStarted(r);
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy(false);
    }
  };

  const runs = topopt.agent_runs ?? [];
  // in-job refine reports live on the chosen agent run; standalone refine
  // children carry the same shape at mesh.refine
  const chosenRun = runs.find((r) => r.chosen);
  const refineEff = refine ?? chosenRun?.refine;
  return (
    <Card className="p-4">
      <SectionTitle>
        <span className="inline-flex items-center gap-1">
          <Shrink className="h-3 w-3" /> Topology optimization
        </span>
      </SectionTitle>
      <div className="mb-2 text-[11px] text-zinc-500">
        {topopt.volfrac_achieved != null &&
          `volume fraction ${(topopt.volfrac_achieved * 100).toFixed(0)}%`}
        {topopt.iterations != null && ` · ${topopt.iterations} SIMP iterations`}
        {topopt.converged != null && ` · ${topopt.converged ? "converged" : "not converged"}`}
      </div>
      {(topopt.warnings ?? []).map((w, i) => (
        <div key={i} className="mb-1 text-[10px] text-amber-400/90">
          {w}
        </div>
      ))}
      {(topopt.sf_verified != null ||
        topopt.sf_homog != null ||
        topopt.sf_micro != null ||
        topopt.bottleneck_mm2 != null ||
        topopt.load_path_ok != null ||
        topopt.strut_align_deg != null ||
        topopt.sf_mesh_sensitive) && (
        <div
          className="mb-2 text-[10px] text-zinc-500"
          data-testid="topopt-verification"
          title="hardened export verification — sf_verified is the worst case across the nominal/eroded/dilated printed-mask realizations; the load-path certificate bounds the thinnest section the load must pass"
        >
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
            verification:
          </span>
          {topopt.sf_verified != null && (
            <span className={sfTone(topopt.sf_verified)}>
              sf_verified {topopt.sf_verified.toFixed(2)}
            </span>
          )}
          {topopt.sf_realizations &&
            (topopt.sf_realizations.nominal != null ||
              topopt.sf_realizations.eroded != null ||
              topopt.sf_realizations.dilated != null) && (
              <span>
                {` (nom ${topopt.sf_realizations.nominal?.toFixed(2) ?? "—"} / ero ${topopt.sf_realizations.eroded?.toFixed(2) ?? "—"} / dil ${topopt.sf_realizations.dilated?.toFixed(2) ?? "—"})`}
              </span>
            )}
          {topopt.sf_homog != null && (
            <span className={sfTone(topopt.sf_homog)}>
              {" "}
              · sf_homog {topopt.sf_homog.toFixed(2)}
            </span>
          )}
          {topopt.sf_micro != null && (
            <span className={sfTone(topopt.sf_micro)}>
              {" "}
              · sf_micro {topopt.sf_micro.toFixed(2)}
            </span>
          )}
          {topopt.bottleneck_mm2 != null && (
            <span> · bottleneck {topopt.bottleneck_mm2.toFixed(1)} mm²</span>
          )}
          {topopt.sf_floor != null && <span> · floor {topopt.sf_floor.toFixed(2)}</span>}
          {topopt.load_path_ok != null && (
            <span
              className={
                topopt.load_path_ok
                  ? "text-emerald-400"
                  : "font-semibold text-red-400"
              }
              title="min-cut load-path certificate: every load case reaches its fixtures through the export solid"
            >
              {" "}
              · load path {topopt.load_path_ok ? "ok" : "BROKEN"}
            </span>
          )}
          {topopt.strut_align_deg != null && (
            <span> · strut align {topopt.strut_align_deg.toFixed(0)}°</span>
          )}
          {topopt.sf_mesh_sensitive && (
            <span
              className="text-amber-400"
              title="a half-pitch grid shift changes the export SF materially — the result is voxel-grid sensitive"
            >
              {" "}
              · mesh-sensitive!
            </span>
          )}
        </div>
      )}
      {(topopt.multires || topopt.input_normalized || topopt.base_mesh_relpath) && (
        <div className="mb-2 text-[10px] text-zinc-500">
          {topopt.multires &&
            `multires coarse pass: ${topopt.multires.coarse_cells ?? "?"} cells @ ${topopt.multires.coarse_pitch_mm ?? "?"}mm, ${topopt.multires.coarse_iterations ?? "?"} it, ${topopt.multires.seconds ?? "?"}s`}
          {topopt.input_normalized &&
            ` · input clipped to spec envelope${
              topopt.input_bbox ? ` (${topopt.input_bbox.map((n) => n.toFixed(1)).join("×")} → ${topopt.spec_bbox?.map((n) => n.toFixed(1)).join("×") ?? "spec"})` : ""
            }`}
          {topopt.base_mesh_relpath && ` · base: ${topopt.base_mesh_relpath}`}
        </div>
      )}
      {(topopt.history?.length ?? 0) > 2 && (
        <div className="mb-2 h-28">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={topopt.history} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="it" tick={{ fontSize: 10, fill: "#71717a" }} tickLine={false} axisLine={false} minTickGap={24} />
              <YAxis tick={{ fontSize: 10, fill: "#71717a" }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "#a1a1aa" }}
                formatter={(v: unknown, name?: unknown) => [
                  typeof v === "number" ? v.toPrecision(4) : String(v),
                  name === "compliance" ? "compliance (mJ)" : String(name),
                ]}
              />
              <Line type="monotone" dataKey="compliance" stroke="#7c9cf5" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {runs.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-[10px] uppercase tracking-wider text-zinc-500">
              <th className="pb-1 font-medium">run</th>
              <th className="pb-1 font-medium">volfrac</th>
              <th className="pb-1 font-medium">worst SF</th>
              <th className="pb-1 font-medium">iters</th>
              <th className="pb-1 font-medium">min member</th>
              <th className="pb-1 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r, i) => (
              <tr
                key={r.it ?? i}
                className={`border-t border-zinc-800/60 ${
                  r.chosen ? "text-emerald-300" : "text-zinc-400"
                }`}
              >
                <td className="py-1 pr-2">#{r.it ?? i + 1}</td>
                <td className="py-1 pr-2">
                  {r.achieved != null
                    ? `${(r.achieved * 100).toFixed(0)}%`
                    : r.volfrac != null
                      ? `${(r.volfrac * 100).toFixed(0)}%`
                      : "—"}
                </td>
                <td className={`py-1 pr-2 ${sfTone(r.worst_sf)}`}>
                  {r.worst_sf != null ? r.worst_sf.toFixed(2) : "—"}
                </td>
                <td className="py-1 pr-2">{r.iterations ?? "—"}</td>
                <td className="py-1 pr-2">
                  {r.min_member_mm != null ? `${r.min_member_mm.toFixed(1)} mm` : "—"}
                </td>
                <td className="py-1 text-right">{r.chosen ? "✓ kept" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {runs.some((r) => r.reasoning) && (
        <div className="mt-2 text-[10px] italic text-zinc-500">
          {runs.find((r) => r.chosen)?.reasoning ?? runs[runs.length - 1]?.reasoning}
        </div>
      )}
      {resamplable && (
        <div className="mt-3 border-t border-zinc-800 pt-2.5">
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500"
            title="re-slice the saved density field at a new iso threshold — instant, no re-optimization"
          >
            re-extract stored density (no new SIMP run)
          </div>
          <div className="flex items-center gap-3">
            <label className="flex flex-1 items-center gap-2 text-[11px] text-zinc-400">
              iso
              <input
                type="range"
                min={20}
                max={90}
                step={5}
                className="flex-1 accent-brand-500"
                value={Math.round(threshold * 100)}
                onChange={(e) => setThreshold(Number(e.target.value) / 100)}
              />
              <span className="w-8 tabular-nums">{threshold.toFixed(2)}</span>
            </label>
            <label className="flex items-center gap-1.5 text-[11px] text-zinc-400">
              min
              <input
                type="number"
                min={0}
                step={0.2}
                className="w-14 rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5 text-[11px]"
                value={minMember}
                onChange={(e) => setMinMember(Math.max(0, Number(e.target.value) || 0))}
              />
              mm
            </label>
            <Btn
              variant="ghost"
              className="border border-zinc-700"
              disabled={busy}
              busy={busy}
              onClick={resample}
              title="re-extract the mesh at this density threshold — creates a child version"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
              Resample
            </Btn>
          </div>
        </div>
      )}
      {(refinable || refineEff) && (
        <div className="mt-3 border-t border-zinc-800 pt-2.5">
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
            feature finishing
          </div>
          {refineEff ? (
            <div className="mb-2 text-[11px] text-zinc-400">
              {refineEff.applied?.length
                ? refineEff.applied.map((f, i) => (
                    <div key={i}>
                      ✓ {f.kind ?? "feature"}
                      {f.name ? ` ${f.name}` : ""}
                      {f.dia ? ` ⌀${f.dia}mm` : ""}
                      {f.axis ? ` ${f.axis}` : ""}
                      {f.depth ? ` ×${f.depth}mm` : ""}
                      {typeof f.dia_deviation_mm === "number" ? (
                        <span
                          className={
                            Math.abs(f.dia_deviation_mm) > 0.15
                              ? "text-amber-400/90"
                              : "text-zinc-500"
                          }
                        >
                          {" "}→ ⌀{f.measured_dia_mm}mm (Δ
                          {f.dia_deviation_mm}mm)
                        </span>
                      ) : null}
                    </div>
                  ))
                : "no features re-cut"}
              {!!refineEff.restored_regions &&
                ` · ${refineEff.restored_regions} keep surface(s) restored`}
              {!!refineEff.step_features?.length && (
                <div className="text-emerald-400/90">
                  STEP: {refineEff.step_features.length} analytic feature cut(s) (
                  {refineEff.step_features
                    .map((f) => `${f.kind ?? "?"}${f.axis ? ` ${f.axis}` : ""}`)
                    .join(", ")}
                  )
                </div>
              )}
              {(refineEff.probes ?? []).map((p, i) => (
                <div
                  key={`p${i}`}
                  className={p.open ? "text-emerald-400/90" : "text-red-400/90"}
                  title="raycast check that a non-through feature is still open after re-cutting"
                >
                  blind probe {p.name ?? i}: {p.open ? "open" : p.found ? "blocked" : "not found"}
                  {p.measured_depth_mm != null && ` · depth ${p.measured_depth_mm.toFixed(1)}mm`}
                  {p.measured_dia_mm != null && ` · ⌀${p.measured_dia_mm.toFixed(1)}mm`}
                </div>
              ))}
              {(refineEff.skipped ?? []).map((s, i) => (
                <div key={`s${i}`} className="text-amber-400/90">
                  skipped: {s.reason ?? "unmatched"}
                </div>
              ))}
              {(refineEff.warnings ?? []).map((w, i) => (
                <div key={`w${i}`} className="text-amber-400/90">
                  {w}
                </div>
              ))}
            </div>
          ) : (
            <div className="mb-2 text-[10px] text-zinc-500">
              re-cuts exact bores / printed threads / inserts at measured anchors and
              restores keep-region surfaces — FEM re-verified, child version
            </div>
          )}
          {refinable && (
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                <input
                  type="checkbox"
                  className="accent-brand-500"
                  checked={restorePlanes}
                  onChange={(e) => setRestorePlanes(e.target.checked)}
                />
                restore keep surfaces
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                <input
                  type="checkbox"
                  className="accent-brand-500"
                  checked={autoPlane}
                  onChange={(e) => setAutoPlane(e.target.checked)}
                />
                auto planar restore
              </label>
              <label
                className="flex items-center gap-1.5 text-[11px] text-zinc-400"
                title="cut real helical thread grooves instead of smooth bores (experimental)"
              >
                <input
                  type="checkbox"
                  className="accent-brand-500"
                  checked={helicalThreads}
                  onChange={(e) => setHelicalThreads(e.target.checked)}
                />
                helical threads
              </label>
              <span className="flex-1" />
              <Btn
                variant="ghost"
                className="border border-zinc-700"
                disabled={busy}
                busy={busy}
                onClick={runRefine}
                title="re-cut exact features on this mesh — creates a refine child version"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wrench className="h-3.5 w-3.5" />}
                Finish features
              </Btn>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
