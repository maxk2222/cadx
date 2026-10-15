import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { StatArea, StatBars, StatDonut } from "../components/charts";
import { Card, EmptyState, SectionTitle, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtSecs, fmtTokens, timeAgo } from "../lib/format";
import { stepLabel } from "../lib/steps";
import type { NamedCount, Stats } from "../lib/types";
import { useTitle } from "../lib/useTitle";

type Range = "7" | "30" | "90" | "0";

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

const TH = "px-3 py-1.5 text-left text-[11px] font-medium uppercase tracking-wider text-zinc-500";
const TD = "px-3 py-1.5 text-sm text-zinc-300";
const TD_MONO = `${TD} font-mono`;

function table(data: NamedCount[], nameLabel: string) {
  return (
    <table className="w-full">
      <thead>
        <tr className="border-b border-zinc-800">
          <th className={TH}>{nameLabel}</th>
          <th className={`${TH} text-right`}>count</th>
        </tr>
      </thead>
      <tbody>
        {data.map((r) => (
          <tr key={r.name} className="border-b border-zinc-800/50 last:border-0">
            <td className={TD}>{r.name}</td>
            <td className={`${TD_MONO} text-right`}>{r.count}</td>
          </tr>
        ))}
      </tbody>
    </table>
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

      {stats.isLoading ? null : !s || s.totals.versions === 0 ? (
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
              value={s.score.pass_rate != null ? `${Math.round(s.score.pass_rate * 100)}%` : "—"}
              sub="succeeded / terminal versions"
            />
            <StatCard
              label="Avg score"
              value={s.score.avg ?? "—"}
              sub={s.score.best != null ? `best ${s.score.best}` : undefined}
            />
            <StatCard label="Avg iterations" value={s.iterations_avg ?? "—"} />
            <StatCard
              label="LLM tokens"
              value={fmtTokens(s.totals.prompt_tokens + s.totals.completion_tokens)}
              sub={`${s.totals.llm_calls} calls`}
            />
            <StatCard label="Compute time" value={fmtSecs(s.totals.compute_seconds)} />
            <StatCard
              label="Artifacts"
              value={s.totals.artifacts}
              sub={fmtBytes(s.totals.artifact_bytes)}
            />
            <StatCard
              label="Mesh gens"
              value={s.mesh.generations || "—"}
              sub={
                s.mesh.generations
                  ? `${s.mesh.cached} cached · Ø ${fmtSecs(s.mesh.avg_duration_s)}`
                  : undefined
              }
            />
          </div>

          {/* activity */}
          <Section title="Activity">
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

          {/* tokens over time */}
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

          {/* distributions */}
          <div className="grid gap-6 md:grid-cols-3">
            <Section title="Version status">
              <Card className="p-2">
                <StatDonut data={s.versions_by_status} />
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
                <StatDonut data={s.jobs_by_status} />
              </Card>
            </Section>
          </div>

          {/* quality */}
          <Section title="Score distribution">
            <Card className="p-4">
              <StatBars
                data={s.score.buckets}
                xKey="name"
                bars={[{ key: "count", name: "versions", color: "#34d399" }]}
                height={180}
              />
            </Card>
          </Section>

          {/* LLM usage tables */}
          {s.models.length > 0 && (
            <div className="grid gap-6 lg:grid-cols-2">
              <Section title="Tokens by model">
                <Card>
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-zinc-800">
                        <th className={TH}>model</th>
                        <th className={`${TH} text-right`}>calls</th>
                        <th className={`${TH} text-right`}>prompt</th>
                        <th className={`${TH} text-right`}>completion</th>
                        <th className={`${TH} text-right`}>Ø latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s.models.map((m) => (
                        <tr key={m.model} className="border-b border-zinc-800/50 last:border-0">
                          <td className={`${TD} max-w-40 truncate`} title={m.model}>
                            {m.model}
                          </td>
                          <td className={`${TD_MONO} text-right`}>{m.calls}</td>
                          <td className={`${TD_MONO} text-right`}>{fmtTokens(m.prompt_tokens)}</td>
                          <td className={`${TD_MONO} text-right`}>
                            {fmtTokens(m.completion_tokens)}
                          </td>
                          <td className={`${TD_MONO} text-right`}>
                            {fmtSecs(m.avg_latency_ms / 1000)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              </Section>
              <Section title="Tokens by step">
                <Card>
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-zinc-800">
                        <th className={TH}>step</th>
                        <th className={`${TH} text-right`}>calls</th>
                        <th className={`${TH} text-right`}>tokens</th>
                        <th className={`${TH} text-right`}>Ø latency</th>
                        <th className={`${TH} text-right`}>wall time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s.steps.map((st) => (
                        <tr key={st.step} className="border-b border-zinc-800/50 last:border-0">
                          <td className={TD}>{stepLabel(st.step) || st.step}</td>
                          <td className={`${TD_MONO} text-right`}>{st.calls}</td>
                          <td className={`${TD_MONO} text-right`}>{fmtTokens(st.tokens)}</td>
                          <td className={`${TD_MONO} text-right`}>
                            {st.avg_latency_ms != null ? fmtSecs(st.avg_latency_ms / 1000) : "—"}
                          </td>
                          <td className={`${TD_MONO} text-right`}>
                            {st.wall_seconds ? fmtSecs(st.wall_seconds) : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              </Section>
            </div>
          )}

          {/* performance + failures */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Job duration by type">
              <Card>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      <th className={TH}>type</th>
                      <th className={`${TH} text-right`}>count</th>
                      <th className={`${TH} text-right`}>failed</th>
                      <th className={`${TH} text-right`}>Ø duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.job_types.map((t) => (
                      <tr key={t.type} className="border-b border-zinc-800/50 last:border-0">
                        <td className={TD}>{t.type}</td>
                        <td className={`${TD_MONO} text-right`}>{t.count}</td>
                        <td
                          className={`${TD_MONO} text-right ${t.failed ? "text-red-400" : ""}`}
                        >
                          {t.failed || "—"}
                        </td>
                        <td className={`${TD_MONO} text-right`}>{fmtSecs(t.avg_duration_s)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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

          {/* misc tables */}
          <div className="grid gap-6 lg:grid-cols-2">
            {s.versions_by_kind.length > 0 && (
              <Section title="Version kinds">
                <Card>{table(s.versions_by_kind, "kind")}</Card>
              </Section>
            )}
            {s.artifacts_by_kind.length > 0 && (
              <Section title="Artifacts">
                <Card>{table(s.artifacts_by_kind, "kind")}</Card>
              </Section>
            )}
          </div>

          {/* projects leaderboard */}
          {s.projects.length > 0 && (
            <Section title="Projects">
              <Card>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      <th className={TH}>project</th>
                      <th className={`${TH} text-right`}>versions</th>
                      <th className={`${TH} text-right`}>succeeded</th>
                      <th className={`${TH} text-right`}>Ø score</th>
                      <th className={`${TH} text-right`}>tokens</th>
                      <th className={`${TH} text-right`}>last activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.projects.map((p) => (
                      <tr key={p.id} className="border-b border-zinc-800/50 last:border-0">
                        <td className={TD}>
                          <Link
                            to={`/projects/${p.id}`}
                            className="text-zinc-200 hover:text-brand-300"
                          >
                            {p.name}
                          </Link>
                        </td>
                        <td className={`${TD_MONO} text-right`}>{p.versions}</td>
                        <td className={`${TD_MONO} text-right`}>{p.succeeded}</td>
                        <td className={`${TD_MONO} text-right`}>{p.avg_score ?? "—"}</td>
                        <td className={`${TD_MONO} text-right`}>{fmtTokens(p.tokens)}</td>
                        <td className={`${TD} text-right text-zinc-500`}>
                          {timeAgo(p.last_activity)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </Section>
          )}
        </div>
      )}
    </div>
  );
}
