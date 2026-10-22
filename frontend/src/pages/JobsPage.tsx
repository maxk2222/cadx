import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleDot, Flag, Inbox, RotateCcw, Square } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router-dom";
import StatusBadge from "../components/StatusBadge";
import {
  Btn,
  EmptyState,
  PageShell,
  ProgressBar,
  inputCls,
  useToast,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import { fmtDuration, timeAgo } from "../lib/format";
import { stepLabel } from "../lib/steps";
import type { JobListItem } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const ACTIVE = new Set(["queued", "running", "waiting_for_input"]);

const STATUS_FILTERS = [
  { id: "all" },
  { id: "active" },
  { id: "waiting_for_input" },
  { id: "failed" },
  { id: "succeeded" },
] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number]["id"];

/** 1s self-ticking elapsed counter — keeps the re-render inside this leaf
 *  instead of invalidating the whole job list every second. */
function Elapsed({ since, live }: { since: string | null; live: boolean }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [live]);
  return <>{fmtDuration(since, null)}</>;
}

const JobRow = memo(function JobRow({
  job,
  onChanged,
}: {
  job: JobListItem;
  onChanged: () => void;
}) {
  const { t } = useTranslation("jobs");
  const nav = useNavigate();
  const toast = useToast();
  const live = job.live ?? {};
  const active = ACTIVE.has(job.status);
  const scores = live.scores ?? [];
  const best = scores.length ? Math.max(...scores.map((s) => s.score)) : null;
  const firstQuestion = job.questions?.[0]?.question;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => nav(`/versions/${job.version_id}?job=${job.id}`)}
      onKeyDown={(e) => e.key === "Enter" && nav(`/versions/${job.version_id}?job=${job.id}`)}
      className="panel panel-hover cursor-pointer p-3"
    >
      <div className="flex flex-wrap items-center gap-2">
        {active ? (
          <CircleDot className="h-3.5 w-3.5 animate-pulse text-brand-400" />
        ) : (
          <StatusBadge status={job.status} />
        )}
        <span className="text-xs font-medium uppercase text-muted">{job.type}</span>
        <Link
          to={`/projects/${job.project_id}`}
          onClick={(e) => e.stopPropagation()}
          className="text-sm font-medium text-fg transition-colors hover:text-brand-300"
        >
          {job.project_name}
        </Link>
        <span className="text-sm text-faint">· v{job.version_number}</span>
        {job.part_name && (
          <Link
            to={`/projects/${job.project_id}${job.part_id ? `?part=${job.part_id}` : ""}`}
            onClick={(e) => e.stopPropagation()}
            className="text-xs text-faint transition-colors hover:text-brand-300"
          >
            · {job.part_name}
          </Link>
        )}
        <StatusBadge status={job.version_status} />
        <span className="flex-1" />
        {best != null && (
          <span
            className={`font-mono text-xs ${best >= 80 ? "text-emerald-400" : best >= 50 ? "text-amber-400" : "text-red-400"}`}
            title={t("best_score_title")}
          >
            {t("best_score", { n: best })}
          </span>
        )}
        <span className="font-mono text-xs text-faint tabular-nums">
          {active && job.started_at ? (
            <Elapsed since={job.started_at} live />
          ) : job.finished_at ? (
            fmtDuration(job.started_at, job.finished_at)
          ) : (
            timeAgo(job.created_at)
          )}
        </span>
      </div>

      {job.prompt && (
        <div className="mt-1 truncate text-xs text-faint">{job.prompt}</div>
      )}

      {active && (
        <div className="mt-2">
          <div className="mb-1 flex items-center gap-2 text-[11px] text-muted">
            <span className="text-brand-300">
              {job.status === "queued"
                ? t("queued")
                : job.status === "waiting_for_input"
                  ? t("needs_answers")
                  : stepLabel(job.step) || t("starting")}
            </span>
            {live.max_iterations ? (
              <span className="font-mono text-fg">
                it {Math.min(job.iteration, live.max_iterations)}/{live.max_iterations}
              </span>
            ) : job.iteration ? (
              <span className="font-mono">it {job.iteration}</span>
            ) : null}
            {live.sub_label && (
              <span className="truncate font-mono text-faint">
                ↳ {live.sub_label}
                {live.sub_progress != null
                  ? ` ${Math.round(live.sub_progress * 100)}%`
                  : ""}
              </span>
            )}
            <span className="flex-1" />
            {job.cancel_requested && (
              <span className="text-red-400">{t("cancelling")}</span>
            )}
            {job.finish_requested && (
              <span className="text-amber-400">{t("finishing")}</span>
            )}
            {job.status === "running" && !job.finish_requested && (
              <button
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-amber-400 transition-colors hover:bg-amber-950/50"
                title={t("finish_title")}
                onClick={(e) => {
                  e.stopPropagation();
                  api
                    .finishJob(job.id)
                    .then(onChanged)
                    .catch((err) => toast(errMsg(err), "error"));
                }}
              >
                <Flag className="h-3 w-3" /> {t("finish")}
              </button>
            )}
            {(job.status === "running" || job.status === "queued") && (
              <button
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-red-400 transition-colors hover:bg-red-950/50"
                title={t("cancel_title")}
                onClick={(e) => {
                  e.stopPropagation();
                  api
                    .cancelJob(job.id)
                    .then(onChanged)
                    .catch((err) => toast(errMsg(err), "error"));
                }}
              >
                <Square className="h-3 w-3" /> {t("cancel")}
              </button>
            )}
          </div>
          <ProgressBar
            value={job.progress}
            tone={job.status === "waiting_for_input" ? "amber" : "blue"}
          />
          {job.status === "waiting_for_input" && firstQuestion && (
            <div
              className="mt-1.5 truncate text-[11px] text-amber-300/90"
              title={firstQuestion}
            >
              “{firstQuestion}”
              {(job.questions?.length ?? 0) > 1
                ? ` ${t("more_questions", { count: job.questions!.length - 1 })}`
                : ""}
            </div>
          )}
        </div>
      )}
      {!active && job.status === "failed" && (
        <div className="mt-1 flex items-center gap-2">
          <div className="min-w-0 flex-1 truncate font-mono text-[10px] text-red-400/80">
            {(job.log ?? "").trim().split("\n").filter(Boolean).slice(-1)[0] ?? ""}
          </div>
          <Btn
            variant="subtle"
            size="sm"
            className="shrink-0"
            title={t("retry_title")}
            onClick={(e) => {
              e.stopPropagation();
              api
                .restartJob(job.id)
                .then((r) => nav(`/versions/${r.version_id}?job=${r.job_id}`))
                .catch((err) => toast(errMsg(err), "error"));
            }}
          >
            <RotateCcw className="h-3 w-3" /> {t("retry")}
          </Btn>
        </div>
      )}
    </div>
  );
});

const RECENT_PAGE = 40;
const FETCH_PAGE = 200; // matches the server's default limit

export default function JobsPage() {
  const { t } = useTranslation("jobs");
  useTitle(t("title"));
  const qc = useQueryClient();
  const [fetchLimit, setFetchLimit] = useState(FETCH_PAGE);
  const [shown, setShown] = useState(RECENT_PAGE);
  const jobs = useQuery({
    queryKey: ["jobs", "all", fetchLimit],
    queryFn: () => api.listJobsAll({ limit: fetchLimit }),
    // keep the last page visible while a bigger limit refetches
    placeholderData: keepPreviousData,
    refetchInterval: (q) =>
      // no idle poll — the lifecycle stream's invalidation kicks activity in
      (q.state.data ?? []).some((j) => ACTIVE.has(j.status)) ? 3000 : false,
  });
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [typeFilter, setTypeFilter] = useState("all");

  const refresh = useCallback(
    () => qc.invalidateQueries({ queryKey: ["jobs", "all"] }),
    [qc]
  );
  const items = useMemo(() => jobs.data ?? [], [jobs.data]);
  const types = useMemo(
    () => [...new Set(items.map((j) => j.type))].sort(),
    [items]
  );
  const matches = (j: JobListItem) => {
    if (typeFilter !== "all" && j.type !== typeFilter) return false;
    if (statusFilter === "all") return true;
    if (statusFilter === "active")
      return j.status === "running" || j.status === "queued";
    return j.status === statusFilter;
  };
  const filtered = items.filter(matches);
  const active = filtered.filter((j) => ACTIVE.has(j.status));
  const recentAll = filtered.filter((j) => !ACTIVE.has(j.status));
  const recent = recentAll.slice(0, shown);
  // a full page means the server may still hold older jobs we can't see yet
  const serverMayHaveMore = items.length >= fetchLimit;

  return (
    <PageShell
      maxW="list"
      title={t("title")}
      subtitle={`${active.length ? `${t("active_n", { count: active.length })} · ` : ""}${t("total_n", { count: items.length })}`}
    >
      <div className="mb-4 flex flex-wrap items-center gap-1.5">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setStatusFilter(f.id)}
            className={`rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors ${
              statusFilter === f.id
                ? "border-brand-600 bg-brand-950/60 text-brand-200"
                : "border-edge2 bg-panel2 text-muted hover:border-faint/50 hover:text-fg"
            }`}
          >
            {t(`filter_${f.id}`)}
          </button>
        ))}
        {types.length > 1 && (
          <select
            className={`${inputCls} ml-auto w-auto px-2 py-1 text-[11px]`}
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            aria-label={t("type_filter_aria")}
          >
            <option value="all">{t("all_types")}</option>
            {types.map((ty) => (
              <option key={ty} value={ty}>
                {ty}
              </option>
            ))}
          </select>
        )}
      </div>

      {jobs.isLoading ? null : !filtered.length ? (
        <EmptyState
          icon={<Inbox className="h-10 w-10" />}
          title={items.length ? t("empty_filtered") : t("empty")}
          hint={items.length ? undefined : t("empty_hint")}
        />
      ) : (
        <div className="flex flex-col gap-2">
          {active.length > 0 && (
            <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">
              {t("section_active")}
            </h2>
          )}
          {active.map((j) => (
            <JobRow key={j.id} job={j} onChanged={refresh} />
          ))}
          {active.length > 0 && recent.length > 0 && (
            <h2 className="mt-4 text-xs font-semibold uppercase tracking-wider text-faint">
              {t("section_recent")}
            </h2>
          )}
          {recent.map((j) => (
            <JobRow key={j.id} job={j} onChanged={refresh} />
          ))}
          {(recentAll.length > recent.length || serverMayHaveMore) && (
            <div className="flex items-center justify-center gap-2 text-[11px] text-faint">
              <span>
                {t("showing", {
                  shown: recent.length,
                  total: recentAll.length,
                  more: serverMayHaveMore ? "+" : "",
                })}
              </span>
              <button
                type="button"
                aria-label={t("show_more_aria", { n: RECENT_PAGE })}
                className="rounded-md border border-edge2 px-2 py-0.5 text-brand-400 transition-colors hover:border-faint/60 hover:text-brand-300"
                onClick={() => {
                  setShown((s) => s + RECENT_PAGE);
                  // everything fetched is already visible — pull the next
                  // page of history from the server too
                  if (serverMayHaveMore) setFetchLimit((l) => l + FETCH_PAGE);
                }}
              >
                {t("show_more", { n: RECENT_PAGE })}
              </button>
            </div>
          )}
        </div>
      )}
    </PageShell>
  );
}
