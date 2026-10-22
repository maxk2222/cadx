import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, BarChart3, RefreshCw } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { StatArea, StatBars, StatComposed, StatDonut } from "../components/charts";
import {
  Card,
  EmptyState,
  PageShell,
  ProgressBar,
  SectionTitle,
  Skeleton,
  Spinner,
  Tabs,
} from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtSecs, fmtTokens, timeAgo } from "../lib/format";
import {
  bucketActivity,
  fmtBucketRange,
  fmtBucketTick,
  pickBucketMode,
  type ActivityBucket,
} from "../lib/statsBuckets";
import { stepLabel } from "../lib/steps";
import {
  meshCacheRate,
  passRateTone,
  scoreBucketColor,
  scoreTone,
  statsEmpty,
  statusColor,
  tokensPerSuccess,
} from "../lib/statsView";
import type { NamedCount, Stats } from "../lib/types";
import { useTitle } from "../lib/useTitle";

type Range = "7" | "30" | "90" | "0";
type View = "overview" | "quality" | "usage" | "data";

const SCORE_TONES = { red: "text-red-300", amber: "text-amber-300", green: "text-emerald-300" };

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <Card className="p-3 transition-colors hover:border-edge2">
      <div className="text-[11px] font-medium uppercase tracking-wider text-faint">
        {label}
      </div>
      <div className="mt-1 text-xl font-semibold text-fg tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-faint">{sub}</div>}
    </Card>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="anim-fade-in-up">
      <SectionTitle>{title}</SectionTitle>
      {children}
    </div>
  );
}

const TH =
  "px-3 py-1.5 text-left text-[11px] font-medium uppercase tracking-wider text-faint";
const TD = "px-3 py-1.5 text-sm text-fg";
const TD_MONO = `${TD} font-mono tabular-nums`;

interface Col<T> {
  label: string;
  align?: "left" | "right";
  render: (row: T) => ReactNode;
}

/** Generic dark table — dedupes the repeated table markup on this page. */
function StatTable<T>({
  cols,
  rows,
  rowKey,
}: {
  cols: Col<T>[];
  rows: T[];
  rowKey: (row: T) => string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-edge">
            {cols.map((c) => (
              <th key={c.label} className={`${TH} ${c.align === "right" ? "text-right" : ""}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={rowKey(r)} className="border-b border-edge/50 last:border-0">
              {cols.map((c) => (
                <td
                  key={c.label}
                  className={`${c.align === "right" ? TD_MONO : TD} ${
                    c.align === "right" ? "text-right" : ""
                  }`}
                >
                  {c.render(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function namedTable(data: NamedCount[], nameLabel: string, countLabel: string) {
  return (
    <StatTable
      cols={[
        { label: nameLabel, render: (r: NamedCount) => r.name },
        { label: countLabel, align: "right", render: (r) => r.count },
      ]}
      rows={data}
      rowKey={(r) => r.name}
    />
  );
}

export default function StatsPage() {
  const { t, i18n } = useTranslation("stats");
  useTitle(t("title"));
  const [range, setRange] = useState<Range>("30");
  const [view, setView] = useState<View>("overview");
  const days = Number(range);
  const stats = useQuery({
    queryKey: ["stats", days],
    queryFn: () => api.getStats(days),
    refetchInterval: 15000,
  });
  const s: Stats | undefined = stats.data;
  const lang = i18n.language;

  // fold the gapless daily timeline into week/month buckets for long ranges
  const buckets: ActivityBucket[] = useMemo(() => {
    const activity = s?.activity ?? [];
    const mode = pickBucketMode(activity.length);
    return bucketActivity(activity, mode).map((b) => b);
  }, [s?.activity]);
  const bucketMode = pickBucketMode((s?.activity ?? []).length);
  const chartData = useMemo(
    () =>
      buckets.map((b) => ({
        ...b,
        tick: fmtBucketTick(b, bucketMode, lang),
        rangeLabel: fmtBucketRange(b, bucketMode, lang),
      })),
    [buckets, bucketMode, lang]
  );

  return (
    <PageShell
      title={t("title")}
      subtitle={
        s ? (
          <span className="text-xs text-faint">
            {t("updated", { at: timeAgo(s.generated_at) })}
          </span>
        ) : undefined
      }
      actions={
        <>
          {stats.isFetching && <Spinner className="h-4 w-4 text-faint" />}
          <Tabs<Range>
            tabs={[
              { id: "7", label: t("range_7") },
              { id: "30", label: t("range_30") },
              { id: "90", label: t("range_90") },
              { id: "0", label: t("range_all") },
            ]}
            active={range}
            onChange={setRange}
          />
        </>
      }
    >
      {stats.isLoading ? (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
          <Skeleton className="h-56" />
        </div>
      ) : stats.isError || !s ? (
        <EmptyState
          icon={<AlertTriangle className="h-10 w-10" />}
          title={t("load_error")}
          hint={t("load_error_hint")}
        />
      ) : statsEmpty(s) ? (
        <EmptyState
          icon={<BarChart3 className="h-10 w-10" />}
          title={t("empty")}
          hint={t("empty_hint")}
        />
      ) : (
        <>
          <Tabs<View>
            className="mb-5 w-fit"
            active={view}
            onChange={setView}
            tabs={[
              { id: "overview", label: t("view_overview") },
              { id: "quality", label: t("view_quality") },
              { id: "usage", label: t("view_usage") },
              { id: "data", label: t("view_data") },
            ]}
          />

          {view === "overview" && (
            <div className="flex flex-col gap-6">
              {/* core KPI row */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                <StatCard
                  label={t("kpi_versions")}
                  value={s.totals.versions}
                  sub={t("kpi_versions_sub", {
                    projects: s.totals.projects,
                    jobs: s.totals.jobs,
                  })}
                />
                <StatCard
                  label={t("kpi_success")}
                  value={
                    s.score.pass_rate != null
                      ? `${Math.round(s.score.pass_rate * 100)}%`
                      : "—"
                  }
                  sub={t("kpi_success_sub")}
                />
                <StatCard
                  label={t("kpi_score")}
                  value={
                    <span className={SCORE_TONES[scoreTone(s.score.avg)]}>
                      {s.score.avg ?? "—"}
                    </span>
                  }
                  sub={s.score.best != null ? t("kpi_score_best", { n: s.score.best }) : undefined}
                />
                <StatCard
                  label={t("kpi_tokens")}
                  value={fmtTokens(s.totals.prompt_tokens + s.totals.completion_tokens)}
                  sub={t("kpi_tokens_sub", { count: s.totals.llm_calls })}
                />
                <StatCard
                  label={t("kpi_tok_success")}
                  value={(() => {
                    const n = tokensPerSuccess(s);
                    return n != null ? fmtTokens(n) : "—";
                  })()}
                  sub={t("kpi_tok_success_sub")}
                />
                <StatCard
                  label={t("kpi_compute")}
                  value={fmtSecs(s.totals.compute_seconds)}
                  sub={t("kpi_compute_sub", { n: s.totals.artifacts })}
                />
              </div>

              {/* secondary strip */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard
                  label={t("kpi_mesh")}
                  value={s.mesh.generations || "—"}
                  sub={(() => {
                    const rate = meshCacheRate(s);
                    if (!s.mesh.generations) return undefined;
                    return t("kpi_mesh_sub", {
                      cached: s.mesh.cached,
                      pct: Math.round((rate ?? 0) * 100),
                      dur: fmtSecs(s.mesh.avg_duration_s),
                    });
                  })()}
                />
                <StatCard
                  label={t("kpi_fem")}
                  value={s.fem?.versions || "—"}
                  sub={
                    s.fem?.avg_worst_sf != null
                      ? t("kpi_fem_sub", { sf: s.fem.avg_worst_sf })
                      : undefined
                  }
                />
                <StatCard label={t("kpi_iterations")} value={s.iterations_avg ?? "—"} />
                <StatCard
                  label={t("kpi_artifacts")}
                  value={s.totals.artifacts}
                  sub={fmtBytes(s.totals.artifact_bytes)}
                />
              </div>

              {/* trends — disjoint grouped bars (jobs ⊋ versions: never stack) */}
              <div className="grid gap-6 lg:grid-cols-2">
                <Section title={t("chart_activity")}>
                  <Card className="p-4">
                    <StatBars
                      data={chartData}
                      xKey="tick"
                      bars={[
                        { key: "versions", name: t("series_versions") },
                        { key: "jobs", name: t("series_jobs"), color: "#34d399" },
                      ]}
                    />
                  </Card>
                </Section>
                <Section title={t("chart_outcomes")}>
                  <Card className="p-4">
                    <StatComposed
                      data={chartData}
                      xKey="tick"
                      bars={[
                        { key: "succeeded", name: t("series_succeeded"), color: "#34d399" },
                        { key: "failed", name: t("series_failed"), color: "#f87171" },
                        { key: "cancelled", name: t("series_cancelled"), color: "#71717a" },
                      ]}
                      line={{ key: "score_avg", name: t("series_score"), color: "#fbbf24" }}
                    />
                  </Card>
                </Section>
              </div>

              {s.totals.llm_calls > 0 && (
                <Section title={t("chart_tokens")}>
                  <Card className="p-4">
                    <StatArea
                      data={chartData}
                      xKey="tick"
                      areas={[{ key: "tokens", name: t("series_tokens") }]}
                    />
                  </Card>
                </Section>
              )}

              {/* distributions */}
              <div className="grid gap-6 md:grid-cols-3">
                <Section title={t("chart_version_status")}>
                  <Card className="p-2">
                    <StatDonut
                      data={s.versions_by_status.map((d) => ({
                        name: t(`common:status.${d.name}`),
                        count: d.count,
                        color: statusColor(d.name),
                      }))}
                    />
                  </Card>
                </Section>
                <Section title={t("chart_mode")}>
                  <Card className="p-2">
                    <StatDonut
                      data={
                        s.versions_by_mode_resolved.length
                          ? s.versions_by_mode_resolved
                          : s.versions_by_mode_requested
                      }
                    />
                  </Card>
                </Section>
                <Section title={t("chart_job_status")}>
                  <Card className="p-2">
                    <StatDonut
                      data={s.jobs_by_status.map((d) => ({
                        name: t(`common:status.${d.name}`),
                        count: d.count,
                        color: statusColor(d.name),
                      }))}
                    />
                  </Card>
                </Section>
              </div>
            </div>
          )}

          {view === "quality" && (
            <div className="flex flex-col gap-6">
              {(s.mode_stats ?? []).length > 0 && (
                <Section title={t("tbl_mode_perf")}>
                  <Card>
                    <StatTable
                      cols={[
                        { label: t("col_mode"), render: (m) => m.mode },
                        {
                          label: t("col_versions"),
                          align: "right",
                          render: (m) => m.versions,
                        },
                        {
                          label: t("col_pass_rate"),
                          align: "right",
                          render: (m) =>
                            m.pass_rate != null ? (
                              <span className="inline-flex items-center gap-2">
                                <ProgressBar
                                  value={m.pass_rate}
                                  tone={passRateTone(m.pass_rate)}
                                  className="w-16"
                                />
                                {Math.round(m.pass_rate * 100)}%
                              </span>
                            ) : (
                              "—"
                            ),
                        },
                        {
                          label: t("col_avg_score"),
                          align: "right",
                          render: (m) =>
                            m.avg_score != null ? (
                              <span className={SCORE_TONES[scoreTone(m.avg_score)]}>
                                {m.avg_score}
                              </span>
                            ) : (
                              "—"
                            ),
                        },
                        {
                          label: t("col_avg_iters"),
                          align: "right",
                          render: (m) => m.avg_iterations ?? "—",
                        },
                        {
                          label: t("col_tokens"),
                          align: "right",
                          render: (m) => fmtTokens(m.tokens),
                        },
                      ]}
                      rows={s.mode_stats ?? []}
                      rowKey={(m) => m.mode}
                    />
                  </Card>
                </Section>
              )}

              <div className="grid gap-6 lg:grid-cols-2">
                <Section title={t("chart_score_dist")}>
                  <Card className="p-4">
                    <StatBars
                      data={s.score.buckets}
                      xKey="name"
                      bars={[{ key: "count", name: t("series_versions") }]}
                      height={180}
                      colorFor={scoreBucketColor}
                    />
                  </Card>
                </Section>
                <Section title={t("tbl_issue_sources")}>
                  <Card>
                    {(s.issue_sources ?? []).length === 0 ? (
                      <div className="p-3 text-sm text-faint">{t("no_issues")}</div>
                    ) : (
                      <StatTable
                        cols={[
                          { label: t("col_source"), render: (i) => i.source },
                          {
                            label: t("col_major"),
                            align: "right",
                            render: (i) =>
                              i.major ? (
                                <span className="text-red-400">{i.major}</span>
                              ) : (
                                "—"
                              ),
                          },
                          {
                            label: t("col_minor"),
                            align: "right",
                            render: (i) => i.minor || "—",
                          },
                          { label: t("col_total"), align: "right", render: (i) => i.total },
                        ]}
                        rows={s.issue_sources ?? []}
                        rowKey={(i) => i.source}
                      />
                    )}
                  </Card>
                </Section>
              </div>
            </div>
          )}

          {view === "usage" && (
            <div className="flex flex-col gap-6">
              {s.models.length > 0 && (
                <div className="grid gap-6 lg:grid-cols-2">
                  <Section title={t("tbl_by_model")}>
                    <Card>
                      <StatTable
                        cols={[
                          {
                            label: t("col_model"),
                            render: (m) => (
                              <span className="block max-w-40 truncate" title={m.model}>
                                {m.model}
                              </span>
                            ),
                          },
                          { label: t("col_calls"), align: "right", render: (m) => m.calls },
                          {
                            label: t("col_prompt"),
                            align: "right",
                            render: (m) => fmtTokens(m.prompt_tokens),
                          },
                          {
                            label: t("col_completion"),
                            align: "right",
                            render: (m) => fmtTokens(m.completion_tokens),
                          },
                          {
                            label: t("col_latency"),
                            align: "right",
                            render: (m) => fmtSecs(m.avg_latency_ms / 1000),
                          },
                        ]}
                        rows={s.models}
                        rowKey={(m) => m.model}
                      />
                    </Card>
                  </Section>
                  <Section title={t("tbl_by_step")}>
                    <Card>
                      <StatTable
                        cols={[
                          {
                            label: t("col_step"),
                            render: (st) => stepLabel(st.step) || st.step,
                          },
                          {
                            label: t("col_calls"),
                            align: "right",
                            render: (st) => st.calls,
                          },
                          {
                            label: t("col_tokens"),
                            align: "right",
                            render: (st) => fmtTokens(st.tokens),
                          },
                          {
                            label: t("col_latency"),
                            align: "right",
                            render: (st) =>
                              st.avg_latency_ms != null
                                ? fmtSecs(st.avg_latency_ms / 1000)
                                : "—",
                          },
                          {
                            label: t("col_wall"),
                            align: "right",
                            render: (st) =>
                              st.wall_seconds ? fmtSecs(st.wall_seconds) : "—",
                          },
                        ]}
                        rows={s.steps}
                        rowKey={(st) => st.step}
                      />
                    </Card>
                  </Section>
                </div>
              )}

              <div className="grid gap-6 lg:grid-cols-2">
                <Section title={t("tbl_job_duration")}>
                  <Card>
                    <StatTable
                      cols={[
                        { label: t("col_type"), render: (t2) => t2.type },
                        { label: t("col_count"), align: "right", render: (t2) => t2.count },
                        {
                          label: t("col_failed"),
                          align: "right",
                          render: (t2) =>
                            t2.failed ? (
                              <span className="text-red-400">{t2.failed}</span>
                            ) : (
                              "—"
                            ),
                        },
                        {
                          label: t("col_avg_dur"),
                          align: "right",
                          render: (t2) => fmtSecs(t2.avg_duration_s),
                        },
                        {
                          label: t("col_p50"),
                          align: "right",
                          render: (t2) => fmtSecs(t2.p50_duration_s),
                        },
                        {
                          label: t("col_max"),
                          align: "right",
                          render: (t2) => fmtSecs(t2.max_duration_s),
                        },
                      ]}
                      rows={s.job_types}
                      rowKey={(t2) => t2.type}
                    />
                  </Card>
                </Section>
                <Section title={t("tbl_failures")}>
                  <Card>
                    {s.top_errors.length === 0 && s.failed_steps.length === 0 ? (
                      <div className="p-3 text-sm text-faint">{t("no_failures")}</div>
                    ) : (
                      <div className="divide-y divide-edge/50">
                        {s.failed_steps.length > 0 && (
                          <div className="p-3">
                            <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-faint">
                              {t("last_step_failure")}
                            </div>
                            {s.failed_steps.map((f) => (
                              <div key={f.name} className="flex justify-between text-sm">
                                <span className="text-fg">
                                  {stepLabel(f.name) || f.name}
                                </span>
                                <span className="font-mono text-faint">{f.count}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {s.top_errors.length > 0 && (
                          <div className="p-3">
                            <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-faint">
                              {t("top_errors")}
                            </div>
                            {s.top_errors.map((e) => (
                              <div
                                key={e.name}
                                className="flex justify-between gap-2 text-sm"
                              >
                                <span
                                  className="min-w-0 flex-1 truncate font-mono text-xs text-red-300/90"
                                  title={e.name}
                                >
                                  {e.name}
                                </span>
                                <span className="shrink-0 font-mono text-faint">
                                  {e.count}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </Card>
                </Section>
              </div>
            </div>
          )}

          {view === "data" && (
            <div className="flex flex-col gap-6">
              <div className="grid gap-6 lg:grid-cols-3">
                {s.versions_by_kind.length > 0 && (
                  <Section title={t("tbl_version_kinds")}>
                    <Card>{namedTable(s.versions_by_kind, t("col_kind"), t("col_count"))}</Card>
                  </Section>
                )}
                {(s.assets_by_kind ?? []).length > 0 && (
                  <Section title={t("tbl_assets")}>
                    <Card>{namedTable(s.assets_by_kind ?? [], t("col_kind"), t("col_count"))}</Card>
                  </Section>
                )}
                {s.artifacts_by_kind.length > 0 && (
                  <Section title={t("tbl_artifacts")}>
                    <Card>{namedTable(s.artifacts_by_kind, t("col_kind"), t("col_count"))}</Card>
                  </Section>
                )}
                {s.mesh.models.length > 0 && (
                  <Section title={t("tbl_mesh_models")}>
                    <Card>{namedTable(s.mesh.models, t("col_model"), t("col_count"))}</Card>
                  </Section>
                )}
              </div>

              {s.projects.length > 0 && (
                <Section title={t("tbl_projects")}>
                  <Card>
                    <StatTable
                      cols={[
                        {
                          label: t("col_project"),
                          render: (p) => (
                            <Link
                              to={`/projects/${p.id}`}
                              className="text-fg transition-colors hover:text-brand-300"
                            >
                              {p.name}
                            </Link>
                          ),
                        },
                        {
                          label: t("col_versions"),
                          align: "right",
                          render: (p) => p.versions,
                        },
                        {
                          label: t("col_succeeded"),
                          align: "right",
                          render: (p) =>
                            `${p.succeeded} (${Math.round((p.succeeded / p.versions) * 100)}%)`,
                        },
                        {
                          label: t("col_avg_score"),
                          align: "right",
                          render: (p) =>
                            p.avg_score != null ? (
                              <span className={SCORE_TONES[scoreTone(p.avg_score)]}>
                                {p.avg_score}
                              </span>
                            ) : (
                              "—"
                            ),
                        },
                        {
                          label: t("col_tokens"),
                          align: "right",
                          render: (p) => fmtTokens(p.tokens),
                        },
                        {
                          label: t("col_last_activity"),
                          align: "right",
                          render: (p) => (
                            <span className="text-faint">{timeAgo(p.last_activity)}</span>
                          ),
                        },
                      ]}
                      rows={s.projects}
                      rowKey={(p) => p.id}
                    />
                  </Card>
                </Section>
              )}
            </div>
          )}

          <div className="mt-6 flex items-center gap-2 text-xs text-faint/70">
            <RefreshCw className="h-3 w-3" />
            {t("auto_refresh", { at: timeAgo(s.generated_at) })}
          </div>
        </>
      )}
    </PageShell>
  );
}
