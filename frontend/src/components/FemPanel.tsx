import { Activity } from "lucide-react";
import { useState } from "react";
import type { Artifact, FemReport } from "../lib/types";
import { Card, Lightbox, SectionTitle } from "./ui";

function sfTone(sf: number | null): string {
  if (sf == null) return "text-zinc-400";
  if (sf < 1) return "text-red-400 font-semibold";
  if (sf < 1.5) return "text-amber-400";
  return "text-emerald-400";
}

export default function FemPanel({
  fem,
  artifacts,
}: {
  fem: FemReport;
  artifacts: Artifact[];
}) {
  const [lightbox, setLightbox] = useState<string | null>(null);
  const femImgs = artifacts.filter(
    (a) => a.kind === "fem" && /\.(png|jpg)$/i.test(a.path)
  );
  const cases = fem.cases ?? [];
  const warnings = [
    ...(fem.warnings ?? []),
    ...cases.flatMap((c) => c.warnings ?? []),
  ];
  const sfs = cases
    .map((c) => c.safety_factor)
    .filter((s): s is number => s != null);
  const worst = sfs.length ? Math.min(...sfs) : null;
  const grid = (fem.grid ?? {}) as { pitch_mm?: number; elements?: number };

  return (
    <Card className="p-4">
      <SectionTitle
        right={
          worst != null && (
            <span className={`text-xs ${sfTone(worst)}`}>
              worst SF {worst.toFixed(2)}
            </span>
          )
        }
      >
        <span className="inline-flex items-center gap-1">
          <Activity className="h-3 w-3" /> Structural analysis (FEM)
        </span>
      </SectionTitle>

      <div className="mb-2 text-[11px] text-zinc-500">
        {(fem.material?.name as string) ?? "material"}
        {grid.pitch_mm != null && ` · voxel ${grid.pitch_mm} mm`}
        {grid.elements != null && ` · ${grid.elements} elements`}
      </div>

      {fem.valid === false && (
        <div className="mb-2 rounded border border-red-900/60 bg-red-950/40 px-2 py-1.5 text-[11px] font-medium text-red-400">
          degenerate solve — displacements/stress are not physically meaningful
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-[10px] uppercase tracking-wider text-zinc-500">
              <th className="pb-1 font-medium">case</th>
              <th className="pb-1 font-medium" title="solver converged">conv</th>
              <th className="pb-1 font-medium">σ max (MPa)</th>
              <th className="pb-1 font-medium">disp (mm)</th>
              <th className="pb-1 font-medium">safety</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.name} className="border-t border-zinc-800/60 text-zinc-300">
                <td className="py-1 pr-2">{c.name}</td>
                <td
                  className={`py-1 pr-2 ${c.converged ? "text-emerald-400" : "text-red-400"}`}
                  title={c.converged ? "solver converged" : "solver did not converge"}
                >
                  {c.converged ? "✓" : "✗"}
                </td>
                <td className="py-1 pr-2">{c.max_vm_mpa.toFixed(1)}</td>
                <td className="py-1 pr-2">{c.max_disp_mm.toFixed(2)}</td>
                <td className={`py-1 ${sfTone(c.safety_factor)}`}>
                  {c.safety_factor != null ? c.safety_factor.toFixed(2) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-1.5 space-y-0.5">
        {cases.map((c) => (
          <div key={c.name} className="text-[10px] text-zinc-500">
            <span className="text-zinc-400">{c.name}:</span> {c.solver} ·
            compliance {c.compliance_j.toPrecision(3)} J
            {c.residual != null && ` · residual ${c.residual.toExponential(1)}`}
            {!!c.reactions?.length &&
              ` · reactions (${c.reactions.map((n) => n.toFixed(1)).join(", ")}) N — should balance the applied loads`}
          </div>
        ))}
      </div>

      {warnings.length > 0 && (
        <div className="mt-2 space-y-0.5 text-[11px] text-amber-400/90">
          {[...new Set(warnings)].map((w, i) => (
            <div key={i}>⚠ {w}</div>
          ))}
        </div>
      )}

      {femImgs.length > 0 && (
        <div className="mt-3 grid grid-cols-3 gap-2">
          {femImgs.map((a) => (
            <button
              key={a.id}
              onClick={() => setLightbox(a.url)}
              className="overflow-hidden rounded-md border border-zinc-800 hover:border-zinc-600"
            >
              <img src={a.url} alt={a.view ?? a.path} className="w-full" />
              <div className="px-1 py-0.5 text-left text-[10px] text-zinc-500">
                {a.view ?? a.path.split("/").pop()}
              </div>
            </button>
          ))}
        </div>
      )}

      {fem.note && <div className="mt-2 text-[10px] italic text-zinc-500">{fem.note}</div>}
      <Lightbox src={lightbox} onClose={() => setLightbox(null)} />
    </Card>
  );
}
