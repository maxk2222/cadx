import { ChevronDown, ChevronRight, CircleDot } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { timeAgo } from "../lib/format";
import { stepLabel } from "../lib/steps";
import type { VersionSummary } from "../lib/types";
import StatusBadge from "./StatusBadge";

export default function VersionTree({
  versions,
  activeId,
}: {
  versions: VersionSummary[];
  activeId?: string;
}) {
  // ids whose subtrees are hidden — deep refine chains get long
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set());
  const toggleCollapse = (id: string) =>
    setCollapsed((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const byParent = new Map<string | null, VersionSummary[]>();
  for (const v of versions) {
    const key = v.parent_id ?? null;
    byParent.set(key, [...(byParent.get(key) ?? []), v]);
  }
  const rows: { v: VersionSummary; depth: number }[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const v of byParent.get(parent) ?? []) {
      rows.push({ v, depth });
      if (!collapsed.has(v.id)) walk(v.id, depth + 1);
    }
  };
  walk(null, 0);
  // orphan safety: versions never reached by the walk. Descendants hidden
  // behind a collapsed ancestor are NOT orphans — walk the parent chain to
  // tell them apart (also covers missing parents and parent cycles).
  const seen = new Set(rows.map((r) => r.v.id));
  const byId = new Map(versions.map((v) => [v.id, v]));
  const hiddenByCollapse = (v: VersionSummary) => {
    const guard = new Set<string>();
    let p = v.parent_id;
    while (p) {
      if (collapsed.has(p)) return true;
      if (guard.has(p)) return false; // parent cycle — treat as orphan
      guard.add(p);
      p = byId.get(p)?.parent_id ?? null; // absent parent ends the chain
    }
    return false;
  };
  for (const v of versions)
    if (!seen.has(v.id) && !hiddenByCollapse(v)) rows.push({ v, depth: 0 });

  return (
    <div className="flex flex-col gap-1">
      {rows.map(({ v, depth }) => {
        const kids = byParent.get(v.id)?.length ?? 0;
        const isCollapsed = collapsed.has(v.id);
        return (
        <Link
          key={v.id}
          to={
            v.active_job
              ? `/versions/${v.id}?job=${v.active_job.id}`
              : `/versions/${v.id}`
          }
          className={`flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-zinc-800 ${
            depth > 0 ? "border-l-2 border-zinc-800" : ""
          } ${v.id === activeId ? "bg-zinc-800 ring-1 ring-zinc-600" : ""}`}
          style={{ marginLeft: depth * 16 }}
        >
          {kids > 0 ? (
            <button
              type="button"
              aria-label={isCollapsed ? `expand subtree of v${v.number}` : `collapse subtree of v${v.number}`}
              title={isCollapsed ? `expand ${kids} descendant${kids > 1 ? "s" : ""}` : "collapse subtree"}
              onClick={(e) => {
                e.preventDefault(); // don't follow the row link
                e.stopPropagation();
                toggleCollapse(v.id);
              }}
              className="-ml-1 flex h-4 w-4 shrink-0 items-center justify-center rounded text-zinc-500 hover:bg-zinc-700 hover:text-zinc-200"
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
            <div className="h-8 w-8 rounded bg-zinc-700" />
          )}
          <div className="min-w-0 flex-1">
            <div className="truncate">
              <span className="font-medium">v{v.number}</span>
              <span className="ml-1 text-zinc-400">{v.kind}</span>
              {v.resolved_mode && (
                <span
                  className={`ml-1 rounded px-1 text-[9px] uppercase ${
                    v.resolved_mode === "cad"
                      ? "bg-zinc-700 text-zinc-300"
                      : v.resolved_mode === "mesh"
                        ? "bg-purple-800 text-purple-200"
                        : v.resolved_mode === "topopt"
                          ? "bg-amber-800 text-amber-200"
                          : "bg-teal-800 text-teal-200"
                  }`}
                  title={`requested: ${v.gen_mode}`}
                >
                  {v.resolved_mode}
                </span>
              )}
              {v.score != null && <span className="ml-1 text-zinc-500">· {v.score}</span>}
            </div>
            <div className="truncate text-xs text-zinc-500">
              {v.prompt}
              <span className="ml-1 text-zinc-600">· {timeAgo(v.created_at)}</span>
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
      {rows.length === 0 && <div className="px-2 py-4 text-sm text-zinc-500">no versions yet</div>}
    </div>
  );
}
