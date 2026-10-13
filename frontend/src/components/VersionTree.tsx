import { CircleDot } from "lucide-react";
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
  const byParent = new Map<string | null, VersionSummary[]>();
  for (const v of versions) {
    const key = v.parent_id ?? null;
    byParent.set(key, [...(byParent.get(key) ?? []), v]);
  }
  const rows: { v: VersionSummary; depth: number }[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const v of byParent.get(parent) ?? []) {
      rows.push({ v, depth });
      walk(v.id, depth + 1);
    }
  };
  walk(null, 0);
  // orphan safety: versions whose parent isn't in the list
  const seen = new Set(rows.map((r) => r.v.id));
  for (const v of versions) if (!seen.has(v.id)) rows.push({ v, depth: 0 });

  return (
    <div className="flex flex-col gap-1">
      {rows.map(({ v, depth }) => (
        <Link
          key={v.id}
          to={
            v.active_job
              ? `/versions/${v.id}?job=${v.active_job.id}`
              : `/versions/${v.id}`
          }
          className={`flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-zinc-800 ${
            v.id === activeId ? "bg-zinc-800 ring-1 ring-zinc-600" : ""
          }`}
          style={{ marginLeft: depth * 16 }}
        >
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
          <StatusBadge status={v.status} />
        </Link>
      ))}
      {rows.length === 0 && <div className="px-2 py-4 text-sm text-zinc-500">no versions yet</div>}
    </div>
  );
}
