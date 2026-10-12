import { Shrink } from "lucide-react";
import { Card, SectionTitle } from "./ui";

interface TopOptRun {
  it?: number;
  volfrac?: number | null;
  achieved?: number | null;
  iterations?: number;
  worst_sf?: number | null;
  min_member_mm?: number | null;
  thin_fraction?: number | null;
  chosen?: boolean;
  reasoning?: string;
  warnings?: string[];
}

interface TopOptInfo {
  volfrac_achieved?: number;
  iterations?: number;
  converged?: boolean;
  agent_runs?: TopOptRun[];
  [key: string]: unknown;
}

function sfTone(sf: number | null | undefined): string {
  if (sf == null) return "text-zinc-400";
  if (sf < 1) return "text-red-400 font-semibold";
  if (sf < 1.5) return "text-amber-400";
  return "text-emerald-400";
}

export default function TopOptPanel({ topopt }: { topopt: TopOptInfo }) {
  const runs = topopt.agent_runs ?? [];
  return (
    <Card className="p-4">
      <SectionTitle>
        <span className="inline-flex items-center gap-1">
          <Shrink className="h-3 w-3" /> Topology optimization
        </span>
      </SectionTitle>
      <div className="mb-2 text-[11px] text-zinc-500">
        {topopt.volfrac_achieved != null &&
          `volume fraction ${(topopt.volfrac_achieved * 100).toFixed(0)}%`}
        {topopt.iterations != null && ` · ${topopt.iterations} SIMP iterations`}
        {topopt.converged != null && ` · ${topopt.converged ? "converged" : "not converged"}`}
      </div>
      {runs.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-[10px] uppercase tracking-wider text-zinc-500">
              <th className="pb-1 font-medium">run</th>
              <th className="pb-1 font-medium">volfrac</th>
              <th className="pb-1 font-medium">worst SF</th>
              <th className="pb-1 font-medium">iters</th>
              <th className="pb-1 font-medium">min member</th>
              <th className="pb-1 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r, i) => (
              <tr
                key={r.it ?? i}
                className={`border-t border-zinc-800/60 ${
                  r.chosen ? "text-emerald-300" : "text-zinc-400"
                }`}
              >
                <td className="py-1 pr-2">#{r.it ?? i + 1}</td>
                <td className="py-1 pr-2">
                  {r.achieved != null
                    ? `${(r.achieved * 100).toFixed(0)}%`
                    : r.volfrac != null
                      ? `${(r.volfrac * 100).toFixed(0)}%`
                      : "—"}
                </td>
                <td className={`py-1 pr-2 ${sfTone(r.worst_sf)}`}>
                  {r.worst_sf != null ? r.worst_sf.toFixed(2) : "—"}
                </td>
                <td className="py-1 pr-2">{r.iterations ?? "—"}</td>
                <td className="py-1 pr-2">
                  {r.min_member_mm != null ? `${r.min_member_mm.toFixed(1)} mm` : "—"}
                </td>
                <td className="py-1 text-right">{r.chosen ? "✓ kept" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {runs.some((r) => r.reasoning) && (
        <div className="mt-2 text-[10px] italic text-zinc-600">
          {runs.find((r) => r.chosen)?.reasoning ?? runs[runs.length - 1]?.reasoning}
        </div>
      )}
    </Card>
  );
}
