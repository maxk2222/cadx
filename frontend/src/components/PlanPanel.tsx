import type { ModelingPlan } from "../lib/types";
import { Card, Chip, SectionTitle } from "./ui";

const APPROACH_TONE: Record<string, "zinc" | "purple" | "teal"> = {
  cad: "zinc",
  mesh: "purple",
  hybrid: "teal",
};

export default function PlanPanel({ plan }: { plan: ModelingPlan }) {
  return (
    <Card className="p-4">
      <SectionTitle>Modeling plan</SectionTitle>
      {plan.approach && (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <Chip tone={APPROACH_TONE[plan.approach] ?? "zinc"} className="uppercase">
            {plan.approach}
          </Chip>
          {plan.approach_reason && (
            <span className="text-xs text-zinc-500">{plan.approach_reason}</span>
          )}
        </div>
      )}
      {plan.mesh_subject && (
        <div className="mb-2 text-xs text-zinc-400">
          mesh subject: <span className="text-zinc-300">{plan.mesh_subject}</span>
        </div>
      )}
      {plan.mesh_up_hint && (
        <div className="mb-2 text-xs text-zinc-400">
          up hint: <span className="text-zinc-300">{plan.mesh_up_hint}</span>
        </div>
      )}
      {!!plan.hybrid_cad_features?.length && (
        <div className="mb-3">
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">CAD features on the mesh</h4>
          <div className="flex flex-wrap gap-1">
            {plan.hybrid_cad_features.map((f, i) => (
              <Chip key={i} tone="teal">
                {f}
              </Chip>
            ))}
          </div>
        </div>
      )}
      {!!plan.steps?.length && (
        <div className="mb-3">
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">Build steps</h4>
          <ol className="flex list-decimal flex-col gap-0.5 pl-5 text-xs text-zinc-400">
            {plan.steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </div>
      )}
      {!!plan.standard_parts?.length && (
        <div className="mb-3">
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">Standard parts</h4>
          <div className="flex flex-wrap gap-1">
            {plan.standard_parts.map((s, i) => (
              <Chip key={i}>{s}</Chip>
            ))}
          </div>
        </div>
      )}
      {!!plan.params?.length && (
        <div className="mb-3">
          <h4 className="mb-1 text-xs font-semibold text-zinc-400">Parameters</h4>
          <div className="flex flex-wrap gap-1">
            {plan.params.map((p) => (
              <Chip key={p.name} title={p.description}>
                {p.name}={String(p.value)}
                {p.unit && p.unit !== "-" ? ` ${p.unit}` : ""}
              </Chip>
            ))}
          </div>
        </div>
      )}
      {plan.notes && <div className="text-xs text-zinc-500">{plan.notes}</div>}
    </Card>
  );
}
