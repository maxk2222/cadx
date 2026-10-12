import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { GenMode, MeshHealth } from "../lib/types";

const MODES: { id: GenMode; label: string; title: string }[] = [
  { id: "auto", label: "Auto", title: "the planner decides: CAD for technical parts, mesh/hybrid for organic shapes" },
  { id: "cad", label: "CAD", title: "parametric build123d code only — precise, dimension-true parts" },
  { id: "mesh", label: "Mesh", title: "image → generated mesh — organic/sculptural shapes (needs ≥1 image + mesh server)" },
  { id: "hybrid", label: "Hybrid", title: "generated organic mesh + exact CAD features on top (needs ≥1 image + mesh server)" },
  { id: "topopt", label: "TopOpt", title: "base part + topology optimization — organic lightweight structure, FEM-verified (describe loads in the prompt)" },
];

export default function ModeSelector({
  value,
  onChange,
}: {
  value: GenMode;
  onChange: (m: GenMode) => void;
}) {
  const [health, setHealth] = useState<MeshHealth | null>(null);

  useEffect(() => {
    api.meshHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const meshOk = health?.ok ?? false;

  return (
    <div>
      <div className="flex rounded border border-zinc-700 bg-zinc-900 p-0.5">
        {MODES.map((m) => {
          const disabled = (m.id === "mesh" || m.id === "hybrid") && !meshOk;
          const active = value === m.id;
          return (
            <button
              key={m.id}
              type="button"
              title={disabled ? `${m.title} — mesh server unreachable` : m.title}
              disabled={disabled}
              onClick={() => onChange(m.id)}
              className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
                active ? "bg-brand-600 text-white" : "text-zinc-400 hover:text-zinc-200"
              } ${disabled ? "cursor-not-allowed opacity-40" : ""}`}
            >
              {m.label}
            </button>
          );
        })}
      </div>
      {(value === "mesh" || value === "hybrid") && (
        <div className="mt-1 text-[11px] text-zinc-500">
          needs at least one reference image — otherwise the run falls back to CAD.
        </div>
      )}
      {value === "auto" && !meshOk && (
        <div className="mt-1 text-[11px] text-zinc-500">
          mesh server unreachable — auto mode will only use CAD.
        </div>
      )}
      {value === "topopt" && (
        <div className="mt-1 text-[11px] text-zinc-500">
          describe loads & fixtures in the prompt (e.g. "fixed at the holes, 100 N on
          top") — the agent picks the base geometry, then optimizes it.
        </div>
      )}
    </div>
  );
}
