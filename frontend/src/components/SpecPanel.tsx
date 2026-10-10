import type { PartSpec, SpecDim } from "../lib/types";
import { Card, Chip, SectionTitle } from "./ui";

const SOURCE_TONE: Record<string, "green" | "blue" | "amber"> = {
  stated: "green",
  inferred: "blue",
  assumed: "amber",
};

function DimChip({ axis, dim }: { axis: string; dim?: SpecDim }) {
  if (!dim?.value) return null;
  return (
    <Chip tone={SOURCE_TONE[dim.source ?? ""] ?? "zinc"} title={`source: ${dim.source ?? "?"}`}>
      <b>{axis}</b> {dim.value} {dim.unit ?? "mm"}
      <span className="text-[9px] opacity-70">{dim.source}</span>
    </Chip>
  );
}

export default function SpecPanel({ spec }: { spec: PartSpec }) {
  const features = spec.features ?? [];
  const dims = [
    { axis: "X", dim: spec.length_x },
    { axis: "Y", dim: spec.width_y },
    { axis: "Z", dim: spec.height_z },
  ];

  return (
    <Card className="p-4">
      <SectionTitle>Extracted spec</SectionTitle>
      {spec.summary && <p className="mb-3 text-sm text-zinc-300">{spec.summary}</p>}

      <div className="mb-3 flex flex-wrap gap-1.5">
        {dims.map((d) => (
          <DimChip key={d.axis} axis={d.axis} dim={d.dim} />
        ))}
        {spec.process && <Chip>process: {spec.process}</Chip>}
        {spec.material && <Chip>material: {spec.material}</Chip>}
        {spec.symmetry && <Chip>symmetry: {spec.symmetry}</Chip>}
        {spec.units && spec.units !== "mm" && <Chip tone="amber">units: {spec.units}</Chip>}
      </div>

      {features.length > 0 && (
        <>
          <h4 className="mb-1.5 text-xs font-semibold text-zinc-400">
            Features ({features.length})
          </h4>
          <div className="mb-3 overflow-x-auto rounded border border-zinc-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-800/60 text-zinc-400">
                <tr>
                  <th className="px-2 py-1.5 font-medium">feature</th>
                  <th className="px-2 py-1.5 font-medium">type</th>
                  <th className="px-2 py-1.5 font-medium">count</th>
                  <th className="px-2 py-1.5 font-medium">position</th>
                  <th className="px-2 py-1.5 font-medium">dims / standard</th>
                </tr>
              </thead>
              <tbody>
                {features.map((f, i) => (
                  <tr key={i} className="border-t border-zinc-800 text-zinc-300">
                    <td className="px-2 py-1.5">
                      <div className="font-medium">{f.name || "—"}</div>
                      {f.description && (
                        <div className="text-[10px] text-zinc-500">{f.description}</div>
                      )}
                    </td>
                    <td className="px-2 py-1.5">
                      <span className="rounded bg-zinc-800 px-1 font-mono text-[10px]">
                        {f.type ?? "other"}
                      </span>
                    </td>
                    <td className="px-2 py-1.5">{f.count ?? 1}×</td>
                    <td className="px-2 py-1.5 text-zinc-400">{f.position || "—"}</td>
                    <td className="px-2 py-1.5 text-zinc-400">
                      {f.standard && <div className="text-[10px]">{f.standard}</div>}
                      {f.dims &&
                        Object.entries(f.dims)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {spec.orientation_note && (
        <div className="mb-2 text-xs text-zinc-500">orientation: {spec.orientation_note}</div>
      )}
      {spec.tolerances_note && (
        <div className="mb-2 text-xs text-zinc-500">tolerances: {spec.tolerances_note}</div>
      )}

      {!!spec.must_keep?.length && (
        <div className="mb-2">
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">Must keep</h4>
          <ul className="list-inside list-disc text-xs text-zinc-400">
            {spec.must_keep.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </div>
      )}
      {!!spec.assumptions?.length && (
        <div className="mb-2">
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">Assumptions</h4>
          <ul className="list-inside list-disc text-xs text-zinc-400">
            {spec.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      )}
      {!!spec.open_questions?.length && (
        <div>
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">
            Open questions ({spec.open_questions.length})
          </h4>
          <ul className="flex flex-col gap-1 text-xs text-zinc-400">
            {spec.open_questions.map((q, i) => (
              <li key={i} className="rounded border border-zinc-800 bg-zinc-900 px-2 py-1">
                {q.blocking && <Chip tone="amber" className="mr-1">blocking</Chip>}
                {q.question}
                {q.default_assumption && (
                  <span className="text-zinc-600"> — default: {q.default_assumption}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
