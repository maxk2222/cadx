import { BadgeCheck, ChevronDown, ChevronRight, CircleDot } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { timeAgo } from "../lib/format";
import { stepLabel } from "../lib/steps";
import type { VersionSummary } from "../lib/types";
import { buildVersionRows } from "../lib/versionTree";
import StatusBadge from "./StatusBadge";

export default function VersionTree({
  versions,
  activeId,
}: {
  versions: VersionSummary[];
  activeId?: string;
}) {
  const { t } = useTranslation("version");
  // ids whose subtrees are hidden — deep refine chains get long
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set());
  const toggleCollapse = (id: string) =>
    setCollapsed((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const rows = buildVersionRows(versions, collapsed);
  // rebuild the same sibling map the row builder used, for the kids count
  const byParent = new Map<string | null, number>();
  for (const v of versions) {
    const key = v.parent_id ?? null;
    byParent.set(key, (byParent.get(key) ?? 0) + 1);
  }

  return (
    <div className="flex flex-col gap-1">
      {rows.map(({ v, depth }) => {
        const kids = byParent.get(v.id) ?? 0;
        const isCollapsed = collapsed.has(v.id);
        return (
        <Link
          key={v.id}
          to={
            v.active_job
              ? `/versions/${v.id}?job=${v.active_job.id}`
              : `/versions/${v.id}`
          }
          className={`flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-panel2 ${
            depth > 0 ? "border-l-2 border-edge" : ""
          } ${v.id === activeId ? "bg-panel2 ring-1 ring-faint" : ""}`}
          style={{ marginLeft: depth * 16 }}
        >
          {kids > 0 ? (
            <button
              type="button"
              aria-label={isCollapsed ? t("tree_expand_aria", { n: v.number }) : t("tree_collapse_aria", { n: v.number })}
              title={isCollapsed ? t("tree_expand_n", { n: kids }) : t("tree_collapse")}
              onClick={(e) => {
                e.preventDefault(); // don't follow the row link
                e.stopPropagation();
                toggleCollapse(v.id);
              }}
              className="-ml-1 flex h-4 w-4 shrink-0 items-center justify-center rounded text-faint hover:bg-edge2 hover:text-fg"
            >
              {isCollapsed ? (
                <ChevronRight className="h-3.5 w-3.5" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5" />
              )}
            </button>
          ) : (
            <span className="-ml-1 w-4 shrink-0" aria-hidden="true" />
          )}
          {v.thumbnail_url ? (
            <img src={v.thumbnail_url} className="h-8 w-8 rounded object-cover" alt="" />
          ) : (
            <div className="h-8 w-8 rounded bg-edge2" />
          )}
          <div className="min-w-0 flex-1">
            <div className="truncate">
              <span className="font-medium">v{v.number}</span>
              {v.accepted_at && (
                <span title={t("tree_accepted")}>
                  <BadgeCheck
                    className="ml-1 inline h-3.5 w-3.5 text-emerald-400"
                    role="img"
                    aria-label={t("tree_accepted")}
                  />
                </span>
              )}
              {v.name && (
                <span className="ml-1 font-medium text-fg">{v.name}</span>
              )}
              <span className="ml-1 text-muted">{v.kind}</span>
              {v.resolved_mode && (
                <span
                  className={`ml-1 rounded px-1 text-[9px] uppercase ${
                    v.resolved_mode === "cad"
                      ? "bg-edge2 text-fg"
                      : v.resolved_mode === "mesh"
                        ? "bg-purple-800 text-purple-200"
                        : v.resolved_mode === "topopt"
                          ? "bg-amber-800 text-amber-200"
                          : "bg-teal-800 text-teal-200"
                  }`}
                  title={t("tree_requested", { mode: v.gen_mode })}
                >
                  {v.resolved_mode}
                </span>
              )}
              {v.score != null && <span className="ml-1 text-faint">· {v.score}</span>}
            </div>
            <div className="truncate text-xs text-faint">
              {v.prompt}
              <span className="ml-1 text-faint">· {timeAgo(v.created_at)}</span>
              {v.active_job && (
                <span
                  className="ml-1 inline-flex items-center gap-1 text-brand-400"
                  title={`${v.active_job.type} job ${v.active_job.status}`}
                >
                  <CircleDot className="h-2.5 w-2.5 animate-pulse" />
                  {stepLabel(v.active_job.step) || v.active_job.type}
                  {v.active_job.max_iterations
                    ? ` ${Math.min(v.active_job.iteration, v.active_job.max_iterations)}/${v.active_job.max_iterations}`
                    : v.active_job.iteration
                      ? ` #${v.active_job.iteration}`
                      : ""}
                </span>
              )}
            </div>
          </div>
          <StatusBadge
            status={v.status}
            title={v.status === "failed" ? (v.error ?? undefined) : undefined}
          />
        </Link>
        );
      })}
      {rows.length === 0 && <div className="px-2 py-4 text-sm text-muted">{t("tree_empty")}</div>}
    </div>
  );
}
