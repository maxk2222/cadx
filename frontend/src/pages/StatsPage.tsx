import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, BarChart3, RefreshCw } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { StatArea, StatBars, StatComposed, StatDonut } from "../components/charts";
import { Card, EmptyState, ProgressBar, SectionTitle, Spinner, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtSecs, fmtTokens, timeAgo } from "../lib/format";
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
    <Card className="p-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {label}
      </div>
      <div className="mt-1 text-xl font-semibold text-zinc-100">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-zinc-500">{sub}</div>}
    </Card>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <SectionTitle>{title}</SectionTitle>
      {children}
    </div>
  );
}

const TH =
  "px-3 py-1.5 text-left text-[11px] font-medium uppercase tracking-wider text-zinc-500";
const TD = "px-3 py-1.5 text-sm text-zinc-300";
const TD_MONO = `${TD} font-mono`;

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
          <tr className="border-b border-zinc-800">
            {cols.map((c) => (
              <th key={c.label} className={`${TH} ${c.align === "right" ? "text-right" : ""}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={rowKey(r)} className="border-b border-zinc-800/50 last:border-0">
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

function namedTable(data: NamedCount[], nameLabel: string) {
  return (
    <StatTable
      cols={[
        { label: nameLabel, render: (r: NamedCount) => r.name },
        { label: "count", align: "right", render: (r) => r.count },
      ]}
      rows={data}
      rowKey={(r) => r.name}
    />
  );
}

function KpiSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Card key={i} className="h-20 animate-pulse p-3" children={null} />
        ))}
      </div>
      <Card className="h-56 animate-pulse" children={null} />
    </div>
  );
}

export default function StatsPage() {
  useTitle("Statistics");
  const [range, setRange] = useState<Range>("30");
  const days = Number(range);
  const stats = useQuery({
    queryKey: ["stats", days],
    queryFn: () => api.getStats(days),
    refetchInterval: 15000,
  });
  const s: Stats | undefined = stats.data;

  return (
    <div className="mx-auto max-w-6xl p-6">
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-bold">Statistics</h1>
        {stats.isFetching && <Spinner className="h-4 w-4 text-zinc-500" />}
        {s && (
          <span className="text-xs text-zinc-500">updated {timeAgo(s.generated_at)}</span>
        )}
        <span className="flex-1" />
        <Tabs<Range>
          tabs={[
            { id: "7", label: "7d" },
            { id: "30", label: "30d" },
            { id: "90", label: "90d" },
            { id: "0", label: "all" },
          ]}
          active={range}
          onChange={setRange}
        />
      </div>

      {stats.isLoading ? (
        <KpiSkeleton />
      ) : stats.isError || !s ? (
        <EmptyState
          icon={<AlertTriangle className="h-10 w-10" />}
          title="Could not load statistics"
          hint="Check that the API is reachable, then try again."
        />
      ) : statsEmpty(s) ? (
        <EmptyState
          icon={<BarChart3 className="h-10 w-10" />}
          title="No data in this range"
          hint="Statistics appear once generations, jobs or LLM calls exist in the selected window."
        />
      ) : (
        <div className="flex flex-col gap-6">
          {/* KPI row */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            <StatCard
              label="Versions"
              value={s.totals.versions}
              sub={`${s.totals.projects} projects · ${s.totals.jobs} jobs`}
            />
            <StatCard
              label="Success rate"
              value={
                s.score.pass_rate != null ? `${Math.round(s.score.pass_rate * 100)}%` : "—"
              }
              sub="succeeded / terminal versions"
            />
            <StatCard
              label="Avg score"
              value={
                <span className={SCORE_TONES[scoreTone(s.score.avg)]}>
                  {s.score.avg ?? "—"}
                </span>
              }
              sub={s.score.best != null ? `best ${s.score.best}` : undefined}
            />
            <StatCard label="Avg iterations" value={s.iterations_avg ?? "—"} />
            <StatCard
              label="LLM tokens"
              value={fmtTokens(s.totals.prompt_tokens + s.totals.completion_tokens)}
              sub={`${s.totals.llm_calls} calls`}
            />
            <StatCard
              label="Tokens / success"
              value={(() => {
                const t = tokensPerSuccess(s);
                return t != null ? fmtTokens(t) : "—";
              })()}
              sub="per succeeded version"
            />
            <StatCard label="Compute time" value={fmtSecs(s.totals.compute_seconds)} />
            <StatCard
              label="Artifacts"
              value={s.totals.artifacts}
              sub={fmtBytes(s.totals.artifact_bytes)}
            />
          </div>

          {/* secondary strip */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard
              label="Mesh gens"
              value={s.mesh.generations || "—"}
              sub={(() => {
                const rate = meshCacheRate(s);
                if (!s.mesh.generations) return undefined;
                return `${s.mesh.cached} cached (${Math.round((rate ?? 0) * 100)}%) · Ø ${fmtSecs(s.mesh.avg_duration_s)}`;
              })()}
            />
            <StatCard
              label="FEM analyses"
              value={s.fem?.versions || "—"}
              sub={
                s.fem?.avg_worst_sf != null
                  ? `Ø worst SF ${s.fem.avg_worst_sf}`
                  : undefined
              }
            />
            <StatCard label="Assets" value={s.totals.assets} />
            <StatCard label="Library parts" value={s.totals.library_parts} />
          </div>

          {/* trends */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Activity (created per day)">
              <Card className="p-4">
                <StatBars
                  data={s.activity.map((d) => ({ ...d, day: d.date.slice(5) }))}
                  xKey="day"
                  bars={[
                    { key: "versions", name: "versions" },
                    { key: "jobs", name: "jobs", color: "#34d399" },
                  ]}
                />
              </Card>
            </Section>
            <Section title="Outcomes by finish day · avg score">
              <Card className="p-4">
                <StatComposed
                  data={s.activity.map((d) => ({ ...d, day: d.date.slice(5) }))}
                  xKey="day"
                  bars={[
                    { key: "succeeded", name: "succeeded", color: "#34d399" },
                    { key: "failed", name: "failed", color: "#f87171" },
                  ]}
                  line={{ key: "score_avg", name: "avg score", color: "#fbbf24" }}
                />
              </Card>
            </Section>
          </div>

          {s.totals.llm_calls > 0 && (
            <Section title="LLM tokens per day">
              <Card className="p-4">
                <StatArea
                  data={s.activity.map((d) => ({ ...d, day: d.date.slice(5) }))}
                  xKey="day"
                  areas={[{ key: "tokens", name: "tokens" }]}
                />
              </Card>
            </Section>
          )}

          {/* mode comparison */}
          {(s.mode_stats ?? []).length > 0 && (
            <Section title="Mode performance">
              <Card>
                <StatTable
                  cols={[
                    { label: "mode", render: (m) => m.mode },
                    { label: "versions", align: "right", render: (m) => m.versions },
                    {
                      label: "pass rate",
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
                      label: "Ø score",
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
                      label: "Ø iters",
                      align: "right",
                      render: (m) => m.avg_iterations ?? "—",
                    },
                    { label: "tokens", align: "right", render: (m) => fmtTokens(m.tokens) },
                  ]}
                  rows={s.mode_stats ?? []}
                  rowKey={(m) => m.mode}
                />
              </Card>
            </Section>
          )}

          {/* distributions */}
          <div className="grid gap-6 md:grid-cols-3">
            <Section title="Version status">
              <Card className="p-2">
                <StatDonut data={s.versions_by_status} colorFor={statusColor} />
              </Card>
            </Section>
            <Section title="Resolved mode">
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
            <Section title="Job status">
              <Card className="p-2">
                <StatDonut data={s.jobs_by_status} colorFor={statusColor} />
              </Card>
            </Section>
          </div>

          {/* quality */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Score distribution">
              <Card className="p-4">
                <StatBars
                  data={s.score.buckets}
                  xKey="name"
                  bars={[{ key: "count", name: "versions" }]}
                  height={180}
                  colorFor={scoreBucketColor}
                />
              </Card>
            </Section>
            <Section title="Issue sources">
              <Card>
                {(s.issue_sources ?? []).length === 0 ? (
                  <div className="p-3 text-sm text-zinc-500">no issues recorded in range</div>
                ) : (
                  <StatTable
                    cols={[
                      { label: "source", render: (i) => i.source },
                      {
                        label: "major",
                        align: "right",
                        render: (i) =>
                          i.major ? <span className="text-red-400">{i.major}</span> : "—",
                      },
                      { label: "minor", align: "right", render: (i) => i.minor || "—" },
                      { label: "total", align: "right", render: (i) => i.total },
                    ]}
                    rows={s.issue_sources ?? []}
                    rowKey={(i) => i.source}
                  />
                )}
              </Card>
            </Section>
          </div>

          {/* LLM usage tables */}
          {s.models.length > 0 && (
            <div className="grid gap-6 lg:grid-cols-2">
              <Section title="Tokens by model">
                <Card>
                  <StatTable
                    cols={[
                      {
                        label: "model",
                        render: (m) => (
                          <span className="block max-w-40 truncate" title={m.model}>
                            {m.model}
                          </span>
                        ),
                      },
                      { label: "calls", align: "right", render: (m) => m.calls },
                      {
                        label: "prompt",
                        align: "right",
                        render: (m) => fmtTokens(m.prompt_tokens),
                      },
                      {
                        label: "completion",
                        align: "right",
                        render: (m) => fmtTokens(m.completion_tokens),
                      },
                      {
                        label: "Ø latency",
                        align: "right",
                        render: (m) => fmtSecs(m.avg_latency_ms / 1000),
                      },
                    ]}
                    rows={s.models}
                    rowKey={(m) => m.model}
                  />
                </Card>
              </Section>
              <Section title="Tokens by step">
                <Card>
                  <StatTable
                    cols={[
                      { label: "step", render: (st) => stepLabel(st.step) || st.step },
                      { label: "calls", align: "right", render: (st) => st.calls },
                      {
                        label: "tokens",
                        align: "right",
                        render: (st) => fmtTokens(st.tokens),
                      },
                      {
                        label: "Ø latency",
                        align: "right",
                        render: (st) =>
                          st.avg_latency_ms != null ? fmtSecs(st.avg_latency_ms / 1000) : "—",
                      },
                      {
                        label: "wall time",
                        align: "right",
                        render: (st) => (st.wall_seconds ? fmtSecs(st.wall_seconds) : "—"),
                      },
                    ]}
                    rows={s.steps}
                    rowKey={(st) => st.step}
                  />
                </Card>
              </Section>
            </div>
          )}

          {/* performance + failures */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Job duration by type">
              <Card>
                <StatTable
                  cols={[
                    { label: "type", render: (t) => t.type },
                    { label: "count", align: "right", render: (t) => t.count },
                    {
                      label: "failed",
                      align: "right",
                      render: (t) =>
                        t.failed ? <span className="text-red-400">{t.failed}</span> : "—",
                    },
                    {
                      label: "Ø dur",
                      align: "right",
                      render: (t) => fmtSecs(t.avg_duration_s),
                    },
                    {
                      label: "p50",
                      align: "right",
                      render: (t) => fmtSecs(t.p50_duration_s),
                    },
                    {
                      label: "max",
                      align: "right",
                      render: (t) => fmtSecs(t.max_duration_s),
                    },
                  ]}
                  rows={s.job_types}
                  rowKey={(t) => t.type}
                />
              </Card>
            </Section>
            <Section title="Failures">
              <Card>
                {s.top_errors.length === 0 && s.failed_steps.length === 0 ? (
                  <div className="p-3 text-sm text-zinc-500">no failures in range</div>
                ) : (
                  <div className="divide-y divide-zinc-800/50">
                    {s.failed_steps.length > 0 && (
                      <div className="p-3">
                        <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                          last step before failure
                        </div>
                        {s.failed_steps.map((f) => (
                          <div key={f.name} className="flex justify-between text-sm">
                            <span className="text-zinc-300">{stepLabel(f.name) || f.name}</span>
                            <span className="font-mono text-zinc-500">{f.count}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {s.top_errors.length > 0 && (
                      <div className="p-3">
                        <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                          top errors
                        </div>
                        {s.top_errors.map((e) => (
                          <div key={e.name} className="flex justify-between gap-2 text-sm">
                            <span
                              className="min-w-0 flex-1 truncate font-mono text-xs text-red-300/90"
                              title={e.name}
                            >
                              {e.name}
                            </span>
                            <span className="shrink-0 font-mono text-zinc-500">{e.count}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </Card>
            </Section>
          </div>

          {/* data inventory */}
          <div className="grid gap-6 lg:grid-cols-3">
            {s.versions_by_kind.length > 0 && (
              <Section title="Version kinds">
                <Card>{namedTable(s.versions_by_kind, "kind")}</Card>
              </Section>
            )}
            {(s.assets_by_kind ?? []).length > 0 && (
              <Section title="Assets">
                <Card>{namedTable(s.assets_by_kind ?? [], "kind")}</Card>
              </Section>
            )}
            {s.artifacts_by_kind.length > 0 && (
              <Section title="Artifacts">
                <Card>{namedTable(s.artifacts_by_kind, "kind")}</Card>
              </Section>
            )}
            {s.mesh.models.length > 0 && (
              <Section title="Mesh models">
                <Card>{namedTable(s.mesh.models, "model")}</Card>
              </Section>
            )}
          </div>

          {/* projects leaderboard */}
          {s.projects.length > 0 && (
            <Section title="Projects">
              <Card>
                <StatTable
                  cols={[
                    {
                      label: "project",
                      render: (p) => (
                        <Link
                          to={`/projects/${p.id}`}
                          className="text-zinc-200 hover:text-brand-300"
                        >
                          {p.name}
                        </Link>
                      ),
                    },
                    { label: "versions", align: "right", render: (p) => p.versions },
                    {
                      label: "succeeded",
                      align: "right",
                      render: (p) => `${p.succeeded} (${Math.round((p.succeeded / p.versions) * 100)}%)`,
                    },
                    {
                      label: "Ø score",
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
                    { label: "tokens", align: "right", render: (p) => fmtTokens(p.tokens) },
                    {
                      label: "last activity",
                      align: "right",
                      render: (p) => (
                        <span className="text-zinc-500">{timeAgo(p.last_activity)}</span>
                      ),
                    },
                  ]}
                  rows={s.projects}
                  rowKey={(p) => p.id}
                />
              </Card>
            </Section>
          )}

          <div className="flex items-center gap-2 text-xs text-zinc-600">
            <RefreshCw className="h-3 w-3" />
            auto-refreshes every 15s · generated {timeAgo(s.generated_at)}
          </div>
        </div>
      )}
    </div>
  );
}
