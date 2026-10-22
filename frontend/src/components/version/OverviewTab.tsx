import { FileBox } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { UseQueryResult } from "@tanstack/react-query";
import { versionLabel } from "../ComparePanel";
import ComparePanel from "../ComparePanel";
import ExportPanel from "../ExportPanel";
import LivePreview from "../LivePreview";
import RenderGallery from "../RenderGallery";
import Viewer3D from "../Viewer3D";
import { Card, Chip, Spinner } from "../ui";
import type {
  Asset,
  Job,
  LibraryPart,
  VersionDetail,
  VersionSummary,
} from "../../lib/types";

/** Model tab: viewer(s) + consolidated model info + renders + export. */
export default function OverviewTab({
  v,
  jobLive,
  liveJob,
  streamConnected,
  glbUrl,
  compareOpen,
  others,
  otherId,
  setOtherId,
  otherSel,
  otherVersion,
  otherGlb,
  assetById,
  libById,
  versions,
  onRefresh,
}: {
  v: VersionDetail;
  jobLive: boolean;
  liveJob: Job | null;
  streamConnected: boolean;
  glbUrl: string | null;
  compareOpen: boolean;
  others: VersionSummary[];
  otherId: string | null;
  setOtherId: (id: string | null) => void;
  otherSel: VersionSummary | null;
  otherVersion: UseQueryResult<VersionDetail>;
  otherGlb: string | null;
  assetById: Map<string, Asset>;
  libById: Map<string, LibraryPart>;
  versions: VersionSummary[];
  onRefresh: () => void;
}) {
  const { t } = useTranslation("version");
  const a = v.analysis;
  const m = v.mesh;

  return (
    <>
      {/* source context: reference assets + library parts this version ran with */}
      {(v.asset_ids.length > 0 || v.library_part_ids.length > 0) && (
        <div className="mb-3 flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] font-medium uppercase tracking-wider text-faint">
            {t("sources")}
          </span>
          {v.asset_ids.map((aid) => {
            const as = assetById.get(aid);
            if (!as) return null;
            return as.kind === "image" ||
              as.kind === "generated_image" ||
              as.mime.startsWith("image/") ? (
              <img
                key={aid}
                src={as.url}
                alt={as.filename}
                title={as.filename}
                className="h-10 w-10 rounded border border-edge2 object-cover"
              />
            ) : (
              <span
                key={aid}
                title={as.filename}
                className="inline-flex max-w-40 items-center gap-1 rounded border border-edge2 bg-panel2 px-1.5 py-0.5 text-[10px] text-muted"
              >
                <FileBox className="h-3 w-3 shrink-0" />
                <span className="truncate">{as.filename}</span>
              </span>
            );
          })}
          {v.library_part_ids.map((pid) => {
            const p = libById.get(pid);
            return (
              <span
                key={pid}
                title={
                  p ? `${p.kind}: ${p.caption || p.filename || p.name}` : t("library_part")
                }
                className="inline-flex items-center gap-1 rounded-full border border-teal-800/60 bg-teal-950/40 px-2 py-0.5 text-[10px] text-teal-300"
              >
                <FileBox className="h-3 w-3" />
                {p?.name ?? t("library_part")}
              </span>
            );
          })}
        </div>
      )}

      {compareOpen && (
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("compare_with", { n: v.number })}
          </span>
          <select
            className="input max-w-md py-1 text-xs"
            value={otherId ?? ""}
            aria-label={t("compare_aria")}
            onChange={(e) => setOtherId(e.target.value || null)}
          >
            {!otherId && <option value="">{t("compare_pick")}</option>}
            {others.map((o) => (
              <option key={o.id} value={o.id}>
                {versionLabel(o)}
              </option>
            ))}
          </select>
          {otherSel && (
            <Chip tone="zinc" className="uppercase">
              {otherSel.kind}
            </Chip>
          )}
        </div>
      )}

      <div
        className={compareOpen ? "mb-4 grid grid-cols-1 gap-3 lg:grid-cols-2" : "mb-4"}
      >
        <div className="relative h-[460px] overflow-hidden rounded-lg border border-edge bg-zinc-950">
          {glbUrl || !jobLive ? (
            <Viewer3D url={glbUrl} />
          ) : (
            <span className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-xs text-muted">
              {t("waiting_render")}
              {liveJob && !streamConnected && (
                <span className="text-[10px] text-faint">{t("reconnecting")}</span>
              )}
            </span>
          )}
          {/* overlay while any job on this version is live — also on top
              of an existing model (fem/resample jobs produce renders on
              versions that already carry a glb) */}
          {jobLive && <LivePreview versionId={v.id} />}
          {compareOpen && (
            <span className="absolute bottom-2 left-2 rounded border border-edge2 bg-panel/80 px-1.5 py-0.5 text-[10px] font-medium text-zinc-300 backdrop-blur">
              a · v{v.number}
            </span>
          )}
        </div>
        {compareOpen && (
          <div className="relative h-[460px] overflow-hidden rounded-lg border border-brand-900/70 bg-zinc-950">
            {otherVersion.isLoading ? (
              <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-muted">
                <Spinner /> {t("loading_preview")}
              </span>
            ) : (
              <Viewer3D url={otherGlb} />
            )}
            <span className="absolute bottom-2 left-2 rounded border border-brand-800 bg-panel/80 px-1.5 py-0.5 text-[10px] font-medium text-brand-300 backdrop-blur">
              b · {otherSel ? `v${otherSel.number}` : "…"}
            </span>
          </div>
        )}
      </div>

      {compareOpen && (
        <ComparePanel base={v} otherId={otherId} versions={versions} />
      )}

      {/* consolidated model info — one definition list instead of two cards */}
      {(m || a) && (
        <Card className="mb-4 p-3">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-3">
            {m?.generation?.model && (
              <InfoItem k={t("mi_model")} title={m.generation.backend ?? ""}>
                {m.generation.model}
              </InfoItem>
            )}
            {m?.generation?.seed != null && (
              <InfoItem k={t("mi_seed")}>{m.generation.seed}</InfoItem>
            )}
            {(m?.triangles ?? m?.generation?.triangles) != null && (
              <InfoItem k={t("mi_triangles")}>
                {m!.triangles ?? m!.generation!.triangles}
              </InfoItem>
            )}
            {m?.orientation && (
              <InfoItem k={t("mi_orientation")} title={m.orientation_source ?? ""}>
                {m.orientation.up ?? "?"} / {m.orientation.front ?? "?"}
              </InfoItem>
            )}
            {m?.scale_factor != null && (
              <InfoItem k={t("mi_scale")}>×{m.scale_factor}</InfoItem>
            )}
            {m?.generation?.duration_s != null && (
              <InfoItem k={t("mi_gen_dur")}>
                {m.generation.duration_s.toFixed(0)}s
              </InfoItem>
            )}
            {a?.bbox_size && (
              <InfoItem k={t("mi_size")}>
                {a.bbox_size.map((n) => n.toFixed(1)).join(" × ")} mm
              </InfoItem>
            )}
            {a?.volume != null && (
              <InfoItem k={t("mi_volume")}>
                {(a.volume / 1000).toFixed(1)} cm³
              </InfoItem>
            )}
            {a?.triangle_count != null && (
              <InfoItem k={t("mi_triangles")}>{a.triangle_count}</InfoItem>
            )}
            {a && <InfoItem k={t("mi_valid")}>{String(a.is_valid)}</InfoItem>}
            {a && (
              <InfoItem k={t("mi_watertight")}>{String(a.watertight)}</InfoItem>
            )}
            {m?.step_tessellated && (
              <div className="col-span-full">
                <span
                  className="text-amber-400"
                  title={t("step_tessellated_hint")}
                >
                  {t("step_tessellated")}
                </span>
              </div>
            )}
          </dl>
        </Card>
      )}

      <RenderGallery artifacts={v.artifacts} />

      <div className="mt-4">
        <ExportPanel versionId={v.id} artifacts={v.artifacts} onRefresh={onRefresh} />
      </div>
    </>
  );
}

function InfoItem({
  k,
  title,
  children,
}: {
  k: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2" title={title}>
      <dt className="shrink-0 text-xs text-faint">{k}</dt>
      <dd className="truncate font-mono text-xs text-zinc-300">{children}</dd>
    </div>
  );
}
