import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleDot, Flag, Inbox, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import StatusBadge from "../components/StatusBadge";
import { EmptyState, ProgressBar, useToast } from "../components/ui";
import { api } from "../lib/api";
import { fmtDuration, fmtSecs, timeAgo } from "../lib/format";
import { stepLabel } from "../lib/steps";
import type { JobListItem } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const ACTIVE = new Set(["queued", "running", "waiting_for_input"]);

function JobRow({
  job,
  onChanged,
}: {
  job: JobListItem;
  onChanged: () => void;
}) {
  const nav = useNavigate();
  const toast = useToast();
  const live = job.live ?? {};
  const active = ACTIVE.has(job.status);
  const running = job.status === "running";
  const scores = live.scores ?? [];
  const best = scores.length ? Math.max(...scores.map((s) => s.score)) : null;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => nav(`/versions/${job.version_id}?job=${job.id}`)}
      onKeyDown={(e) => e.key === "Enter" && nav(`/versions/${job.version_id}?job=${job.id}`)}
      className="cursor-pointer rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 transition-colors hover:border-zinc-600"
    >
      <div className="flex flex-wrap items-center gap-2">
        {active ? (
          <CircleDot className="h-3.5 w-3.5 animate-pulse text-brand-400" />
        ) : (
          <StatusBadge status={job.status} />
        )}
        <span className="text-xs font-medium uppercase text-zinc-400">{job.type}</span>
        <Link
          to={`/projects/${job.project_id}`}
          onClick={(e) => e.stopPropagation()}
          className="text-sm font-medium text-zinc-200 hover:text-brand-300"
        >
          {job.project_name}
        </Link>
        <span className="text-sm text-zinc-500">· v{job.version_number}</span>
        <StatusBadge status={job.version_status} />
        <span className="flex-1" />
        {best != null && (
          <span
            className={`font-mono text-xs ${best >= 80 ? "text-emerald-400" : best >= 50 ? "text-amber-400" : "text-red-400"}`}
            title="best critique score"
          >
            best {best}
          </span>
        )}
        <span className="font-mono text-xs text-zinc-500">
          {active && job.started_at
            ? fmtDuration(job.started_at, null)
            : job.finished_at
              ? fmtDuration(job.started_at, job.finished_at)
              : timeAgo(job.created_at)}
        </span>
      </div>

      {job.prompt && (
        <div className="mt-1 truncate text-xs text-zinc-500">{job.prompt}</div>
      )}

      {active && (
        <div className="mt-2">
          <div className="mb-1 flex items-center gap-2 text-[11px] text-zinc-400">
            <span className="text-brand-300">
              {job.status === "queued"
                ? "queued"
                : job.status === "waiting_for_input"
                  ? "needs answers"
                  : stepLabel(job.step) || "starting"}
            </span>
            {live.max_iterations ? (
              <span className="font-mono text-zinc-300">
                it {Math.min(job.iteration, live.max_iterations)}/{live.max_iterations}
              </span>
            ) : job.iteration ? (
              <span className="font-mono">it {job.iteration}</span>
            ) : null}
            {live.sub_label && (
              <span className="truncate font-mono text-zinc-600">
                ↳ {live.sub_label}
                {live.sub_progress != null
                  ? ` ${Math.round(live.sub_progress * 100)}%`
                  : ""}
              </span>
            )}
            <span className="flex-1" />
            {job.cancel_requested && <span className="text-red-400">cancelling…</span>}
            {job.finish_requested && <span className="text-amber-400">finishing…</span>}
            {job.status === "running" && !job.finish_requested && (
              <button
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-amber-400 hover:bg-amber-950/50"
                title="stop after this iteration and keep the best result so far"
                onClick={(e) => {
                  e.stopPropagation();
                  api
                    .finishJob(job.id)
                    .then(onChanged)
                    .catch((err) => toast(String(err), "error"));
                }}
              >
                <Flag className="h-3 w-3" /> finish
              </button>
            )}
            {(job.status === "running" || job.status === "queued") && (
              <button
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-red-400 hover:bg-red-950/50"
                title="abort the job — discards the result"
                onClick={(e) => {
                  e.stopPropagation();
                  api
                    .cancelJob(job.id)
                    .then(onChanged)
                    .catch((err) => toast(String(err), "error"));
                }}
              >
                <Square className="h-3 w-3" /> cancel
              </button>
            )}
          </div>
          <ProgressBar
            value={job.progress}
            tone={job.status === "waiting_for_input" ? "amber" : "blue"}
          />
        </div>
      )}
      {!active && job.status === "failed" && (
        <div className="mt-1 truncate font-mono text-[10px] text-red-400/80">
          {(job.log ?? "").trim().split("\n").filter(Boolean).slice(-1)[0] ?? ""}
        </div>
      )}
    </div>
  );
}

export default function JobsPage() {
  useTitle("Jobs");
  const qc = useQueryClient();
  const jobs = useQuery({
    queryKey: ["jobs-all"],
    queryFn: () => api.listJobsAll(),
    refetchInterval: (q) =>
      (q.state.data ?? []).some((j) => ACTIVE.has(j.status)) ? 3000 : 15000,
  });
  // 1s tick so elapsed times count up while anything is active
  const anyActive = (jobs.data ?? []).some((j) => ACTIVE.has(j.status));
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!anyActive) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [anyActive]);

  const refresh = () => qc.invalidateQueries({ queryKey: ["jobs-all"] });
  const items = jobs.data ?? [];
  const active = items.filter((j) => ACTIVE.has(j.status));
  const recent = items.filter((j) => !ACTIVE.has(j.status)).slice(0, 40);

  return (
    <div className="mx-auto max-w-4xl p-6">
      <div className="mb-5 flex items-center gap-3">
        <h1 className="text-2xl font-bold">Jobs</h1>
        <span className="text-sm text-zinc-500">
          {active.length ? `${active.length} active · ` : ""}
          {items.length} total
        </span>
      </div>

      {jobs.isLoading ? null : !items.length ? (
        <EmptyState
          icon={<Inbox className="h-10 w-10" />}
          title="No jobs yet"
          hint="Jobs appear here as soon as a generation, edit or export is queued."
        />
      ) : (
        <div className="flex flex-col gap-2">
          {active.length > 0 && (
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Active
            </h2>
          )}
          {active.map((j) => (
            <JobRow key={j.id} job={j} onChanged={refresh} />
          ))}
          {active.length > 0 && recent.length > 0 && (
            <h2 className="mt-4 text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Recent
            </h2>
          )}
          {recent.map((j) => (
            <JobRow key={j.id} job={j} onChanged={refresh} />
          ))}
        </div>
      )}
    </div>
  );
}
