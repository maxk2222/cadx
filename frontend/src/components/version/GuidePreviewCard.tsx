import { useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpenCheck, ChevronRight, PencilRuler } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { api, errMsg } from "../../lib/api";
import type { VersionSummary } from "../../lib/types";
import StatusBadge from "../StatusBadge";
import { Btn, Card, Chip, Spinner, inputCls } from "../ui";
import GuideEditor from "./GuideEditor";
import GuideScene from "./GuideScene";

const LIVE_STATUSES = new Set(["queued", "running", "waiting_for_input"]);

/** which guide to feature on the card: finished first, then a live one,
 *  then a dead one — newest wins inside each group */
const rank = (v: VersionSummary) =>
  v.status === "succeeded" || v.status === "needs_review"
    ? 0
    : LIVE_STATUSES.has(v.status) || v.active_job
      ? 1
      : 2;

/** Persistent guide preview on the project page: whenever the project holds
 *  guide versions, their cumulative 3D assembly is shown here with a button
 *  opening the full guide editor. */
export default function GuidePreviewCard({
  projectId,
  versions,
  guideNums,
}: {
  projectId: string;
  versions: VersionSummary[];
  guideNums: Map<string, number>;
}) {
  const { t } = useTranslation("projects");
  const qc = useQueryClient();
  const [picked, setPicked] = useState("");
  const [editorOpen, setEditorOpen] = useState(false);

  const guides = useMemo(
    () =>
      versions
        .filter((v) => v.kind === "guide")
        .sort((a, b) => rank(a) - rank(b) || b.number - a.number),
    [versions]
  );
  const gv = guides.find((v) => v.id === picked) ?? guides[0] ?? null;
  const live = !!gv && (LIVE_STATUSES.has(gv.status) || !!gv.active_job);

  const detail = useQuery({
    queryKey: ["guide", gv?.id],
    queryFn: () => api.getGuide(gv!.id),
    enabled: !!gv,
    retry: false,
    // a running guide job keeps growing state.json — follow along
    refetchInterval: live ? 4000 : false,
  });
  const guide = detail.data;
  if (!gv) return null;

  const gNum = guideNums.get(gv.part_id);
  const lastStep = guide?.steps?.length
    ? Math.max(...guide.steps.map((s) => s.number))
    : 0;

  return (
    <>
      <Card className="anim-fade-in-up overflow-hidden p-0">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 py-2">
          <BookOpenCheck className="h-4 w-4 shrink-0 text-purple-300" />
          <span className="shrink-0 text-[11px] font-semibold uppercase tracking-wider text-faint">
            {t("guide_card_title")}
          </span>
          {gNum != null && <Chip tone="purple">G{gNum}</Chip>}
          <span className="min-w-0 truncate text-xs text-muted">
            {gv.name || gv.prompt || `v${gv.number}`}
          </span>
          <StatusBadge status={gv.status} />
          <span className="ml-auto flex shrink-0 items-center gap-1.5">
            {guides.length > 1 && (
              <select
                className={`${inputCls} w-40 py-0.5 text-[11px]`}
                value={gv.id}
                onChange={(e) => setPicked(e.target.value)}
              >
                {guides.map((v) => (
                  <option key={v.id} value={v.id}>
                    G{guideNums.get(v.part_id) ?? "?"} · v{v.number}
                    {v.name ? ` · ${v.name}` : ""}
                  </option>
                ))}
              </select>
            )}
            <Btn
              size="sm"
              variant="ghost"
              disabled={live}
              title={live ? t("guide_building") : t("version:guide_open_editor")}
              onClick={() => setEditorOpen(true)}
            >
              <PencilRuler className="h-3.5 w-3.5" />
              {t("version:guide_open_editor")}
            </Btn>
            <Link
              to={`/versions/${gv.id}`}
              title={t("wizard:prog_open_version")}
              className="rounded p-1 text-faint transition-colors hover:bg-panel2 hover:text-fg"
            >
              <ChevronRight className="h-4 w-4" />
            </Link>
          </span>
        </div>
        <div className="h-80 border-t border-edge bg-bg">
          {detail.isLoading ? (
            <div className="flex h-full items-center justify-center">
              <Spinner />
            </div>
          ) : !guide ? (
            <div className="flex h-full items-center justify-center px-4 text-center text-xs text-faint">
              {/* a live job 404s until state.json lands — say "building",
                  not the raw error */}
              {live ? t("guide_building") : detail.error ? errMsg(detail.error) : t("guide_building")}
            </div>
          ) : (
            <GuideScene
              key={gv.id}
              versionId={gv.id}
              parts={guide.parts}
              instances={guide.instance_list ?? []}
              step={lastStep}
              sel={null}
              onSelect={() => {}}
              onTransformed={() => {}}
              readonly
              annotations={guide.annotations ?? []}
              worldFrame={guide.world_frame}
            />
          )}
        </div>
      </Card>
      {editorOpen && (
        <GuideEditor
          versionId={gv.id}
          name={gv.name || `G${gNum ?? ""} · v${gv.number}`}
          projectId={projectId}
          onClose={() => {
            setEditorOpen(false);
            detail.refetch();
            qc.invalidateQueries({ queryKey: ["versions", projectId] });
          }}
        />
      )}
    </>
  );
}
