import { useQuery } from "@tanstack/react-query";
import { ArrowRight, TriangleAlert } from "lucide-react";
import { api, errMsg } from "../lib/api";
import type {
  CompareResult,
  VersionDetail,
  VersionSummary,
} from "../lib/types";
import { Card, Chip, EmptyState, SectionTitle, Skeleton } from "./ui";

/* ---------- pure helpers (unit-tested) ---------- */

export function trunc(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, Math.max(0, n - 1))}…` : s;
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}

/** picker option label: "v3 · ab12cd34 · edit · make the wall thicker…" */
export function versionLabel(o: VersionSummary, maxLen = 48): string {
  const prompt = o.prompt.trim().replace(/\s+/g, " ") || "(no prompt)";
  return `v${o.number} · ${shortId(o.id)} · ${o.kind} · ${trunc(prompt, maxLen)}`;
}

/** default compare target: the parent when it's in the list, else the
 *  highest-numbered (latest) other version. `others` must already exclude
 *  the base version. */
export function defaultCompareTarget(
  others: VersionSummary[],
  parentId: string | null | undefined
): string | null {
  if (!others.length) return null;
  if (parentId && others.some((o) => o.id === parentId)) return parentId;
  return others.reduce((m, o) => (o.number > m.number ? o : m)).id;
}

/** mm³ → "12.3 cm³" (same convention as the analysis strip). */
export function fmtVol(mm3: number | null | undefined): string {
  if (mm3 == null || !Number.isFinite(mm3)) return "—";
  return `${(mm3 / 1000).toFixed(1)} cm³`;
}

export function fmtVec(v: number[] | null | undefined, digits = 1): string {
  return v?.length ? v.map((n) => n.toFixed(digits)).join(" × ") : "—";
}

/** signed delta: +2.5 / −3.3 / ±0 (near-zero noise collapses to ±0). */
export function fmtDelta(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (Math.abs(n) < 0.5 * 10 ** -digits) return "±0";
  return n > 0 ? `+${n.toFixed(digits)}` : `−${Math.abs(n).toFixed(digits)}`;
}

export function fmtVecDelta(v: number[] | null | undefined, digits = 1): string {
  return v?.length ? v.map((n) => fmtDelta(n, digits)).join(" × ") : "—";
}

/** tailwind text class for a signed delta: + green, − red, 0/none muted. */
export function deltaTone(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n === 0) return "text-zinc-500";
  return n > 0 ? "text-emerald-400" : "text-red-400";
}

/** params_diff cell — null/undefined means the key is absent on that side;
 *  values are JSON-encoded and truncated for the table. */
export function fmtParamVal(v: unknown, maxLen = 48): string {
  if (v === null || v === undefined) return "—";
  const s = JSON.stringify(v) ?? String(v);
  return trunc(s, maxLen);
}

/* ---------- component ---------- */

function MetricRow({
  label,
  a,
  b,
  d,
  tone,
}: {
  label: string;
  a: string;
  b: string;
  d: string;
  tone?: string;
}) {
  return (
    <tr className="border-t border-zinc-800/60 text-zinc-300">
      <td className="py-1 pr-3 text-zinc-500">{label}</td>
      <td className="py-1 pr-3 font-mono tabular-nums">{a}</td>
      <td className="py-1 pr-3 font-mono tabular-nums">{b}</td>
      <td className={`py-1 font-mono tabular-nums ${tone ?? "text-zinc-400"}`}>
        {d}
      </td>
    </tr>
  );
}

function CompareBody({
  r,
  aLabel,
  bLabel,
}: {
  r: CompareResult;
  aLabel: string;
  bLabel: string;
}) {
  const s = r.shape;
  return (
    <>
      {(!r.ok || r.error) && (
        <div className="mb-3 flex items-start gap-2 rounded-md border border-amber-800 bg-amber-950/50 p-2.5 text-xs text-amber-300">
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span className="min-w-0 break-words">
            {r.error || "comparison incomplete — partial results below"}
          </span>
        </div>
      )}

      {r.diff_summary ? (
        <p className="mb-3 whitespace-pre-wrap text-sm text-zinc-300">
          {r.diff_summary}
        </p>
      ) : (
        !s &&
        !r.params_diff.length &&
        r.ok && (
          <p className="mb-3 text-sm text-zinc-500">
            No differences detected between these versions.
          </p>
        )
      )}

      {s ? (
        <div className="mb-4">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              Geometry
            </h4>
            {s.boolean !== "ok" && (
              <Chip
                tone={s.boolean === "failed" ? "red" : "amber"}
                title={s.error ?? s.note}
              >
                <TriangleAlert className="h-3 w-3" />
                boolean diff {s.boolean}
                {s.note ? ` — ${s.note}` : ""}
              </Chip>
            )}
          </div>
          {s.boolean !== "ok" && s.error && (
            <div className="mb-1 text-[11px] text-red-400">{s.error}</div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-zinc-500">
                  <th className="pb-1 pr-3 font-medium" />
                  <th className="pb-1 pr-3 font-medium">{aLabel}</th>
                  <th className="pb-1 pr-3 font-medium">{bLabel}</th>
                  <th className="pb-1 font-medium">Δ</th>
                </tr>
              </thead>
              <tbody>
                <MetricRow
                  label="volume"
                  a={fmtVol(s.volume_a)}
                  b={fmtVol(s.volume_b)}
                  d={
                    s.volume_delta != null
                      ? `${fmtDelta(s.volume_delta / 1000)} cm³`
                      : "—"
                  }
                  tone={deltaTone(s.volume_delta)}
                />
                <MetricRow
                  label="bbox size"
                  a={fmtVec(s.bbox_a?.size)}
                  b={fmtVec(s.bbox_b?.size)}
                  d={fmtVecDelta(s.bbox_delta?.size)}
                />
                <MetricRow
                  label="solids"
                  a={String(s.solids_a)}
                  b={String(s.solids_b)}
                  d={fmtDelta(s.solids_b - s.solids_a, 0)}
                  tone={deltaTone(s.solids_b - s.solids_a)}
                />
                <MetricRow
                  label="faces"
                  a={String(s.faces_a)}
                  b={String(s.faces_b)}
                  d={fmtDelta(s.faces_b - s.faces_a, 0)}
                  tone={deltaTone(s.faces_b - s.faces_a)}
                />
                {s.boolean === "ok" && (
                  <>
                    <MetricRow
                      label="added volume (only in b)"
                      a="—"
                      b={fmtVol(s.added_volume)}
                      d={
                        s.added_volume != null
                          ? `+${(s.added_volume / 1000).toFixed(1)} cm³`
                          : "—"
                      }
                      tone="text-emerald-400"
                    />
                    <MetricRow
                      label="removed volume (only in a)"
                      a={fmtVol(s.removed_volume)}
                      b="—"
                      d={
                        s.removed_volume != null
                          ? `−${(s.removed_volume / 1000).toFixed(1)} cm³`
                          : "—"
                      }
                      tone="text-red-400"
                    />
                  </>
                )}
                {s.added_solids > 0 && (
                  <MetricRow
                    label="added solids"
                    a="—"
                    b={String(s.added_solids)}
                    d={`+${s.added_solids}`}
                    tone="text-emerald-400"
                  />
                )}
                {s.removed_solids > 0 && (
                  <MetricRow
                    label="removed solids"
                    a={String(s.removed_solids)}
                    b="—"
                    d={`−${s.removed_solids}`}
                    tone="text-red-400"
                  />
                )}
                {s.added_bbox && (
                  <MetricRow
                    label="added region bbox"
                    a="—"
                    b={fmtVec(s.added_bbox.size)}
                    d=""
                    tone="text-emerald-400"
                  />
                )}
                {s.removed_bbox && (
                  <MetricRow
                    label="removed region bbox"
                    a={fmtVec(s.removed_bbox.size)}
                    b="—"
                    d=""
                    tone="text-red-400"
                  />
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <p className="mb-3 text-xs text-zinc-500">
          No geometry available to diff (one or both versions have no solid).
        </p>
      )}

      {r.params_diff.length > 0 && (
        <div>
          <h4 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
            Parameters ({r.params_diff.length})
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full table-fixed text-xs">
              <thead>
                <tr className="text-left text-zinc-500">
                  <th className="w-1/4 pb-1 pr-3 font-medium">param</th>
                  <th className="pb-1 pr-3 font-medium">{aLabel}</th>
                  <th className="pb-1 font-medium">{bLabel}</th>
                </tr>
              </thead>
              <tbody>
                {r.params_diff.map((d) => (
                  <tr
                    key={d.key}
                    className="border-t border-zinc-800/60 text-zinc-300"
                  >
                    <td className="truncate py-1 pr-3 font-mono text-zinc-400">
                      {d.key}
                    </td>
                    <td
                      className="truncate py-1 pr-3 font-mono"
                      title={JSON.stringify(d.a) ?? ""}
                    >
                      {fmtParamVal(d.a)}
                    </td>
                    <td
                      className="truncate py-1 font-mono"
                      title={JSON.stringify(d.b) ?? ""}
                    >
                      {fmtParamVal(d.b)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

export default function ComparePanel({
  base,
  otherId,
  versions,
}: {
  base: VersionDetail;
  otherId: string | null;
  versions: VersionSummary[];
}) {
  const other = versions.find((x) => x.id === otherId) ?? null;
  const cmp = useQuery({
    queryKey: ["compare", base.id, otherId],
    queryFn: () => api.compareVersions(base.id, otherId!),
    enabled: !!otherId,
    staleTime: 60_000,
  });

  const aLabel = `v${base.number}`;
  const bLabel = other ? `v${other.number}` : otherId ? shortId(otherId) : "—";

  return (
    <Card className="mb-4 p-4">
      <SectionTitle
        right={
          <span className="flex items-center gap-1 normal-case tracking-normal">
            <Chip>{aLabel}</Chip>
            <ArrowRight className="h-3 w-3 text-zinc-600" />
            <Chip tone="blue">{bLabel}</Chip>
            {other && <Chip className="uppercase">{other.kind}</Chip>}
          </span>
        }
      >
        Compare
      </SectionTitle>

      {!otherId ? (
        <EmptyState
          title="Nothing to compare against"
          hint="Pick another version of this project above."
        />
      ) : cmp.isLoading ? (
        <div className="space-y-2" aria-busy="true">
          <Skeleton className="h-4 w-56" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-28 w-full" />
        </div>
      ) : cmp.isError ? (
        <div className="rounded-md border border-red-800 bg-red-950/50 p-2.5 text-sm text-red-300">
          {errMsg(cmp.error)}
        </div>
      ) : cmp.data ? (
        <CompareBody r={cmp.data} aLabel={aLabel} bLabel={bLabel} />
      ) : null}
    </Card>
  );
}
