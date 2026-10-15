import { ArrowRight, Boxes, Check, Cpu, Image as ImageIcon, Shrink, Wand2 } from "lucide-react";
import type { GenMode, MeshHealth } from "../../lib/types";
import { Btn, Chip } from "../ui";

const MODE_CARDS: {
  id: GenMode;
  icon: typeof Cpu;
  label: string;
  title: string;
  desc: string;
}[] = [
  {
    id: "auto",
    icon: Wand2,
    label: "Auto",
    title: "the planner decides",
    desc: "Recommended default — CAD for technical parts, mesh/hybrid for organic shapes.",
  },
  {
    id: "cad",
    icon: Cpu,
    label: "CAD",
    title: "parametric build123d code",
    desc: "Precise, dimension-true parts: plates, brackets, enclosures, standard fasteners.",
  },
  {
    id: "mesh",
    icon: ImageIcon,
    label: "Mesh",
    title: "image → generated mesh",
    desc: "Organic / sculptural shapes from reference images. Needs ≥1 image + mesh server.",
  },
  {
    id: "hybrid",
    icon: Boxes,
    label: "Hybrid",
    title: "mesh base + exact CAD features",
    desc: "Generated organic body with precise bores, threads and inserts on top.",
  },
  {
    id: "topopt",
    icon: Shrink,
    label: "TopOpt",
    title: "topology optimization",
    desc: "CAD base part, then SIMP-optimized lightweight structure, FEM-verified.",
  },
];

export default function StepMode({
  value,
  onChange,
  suggested,
  hasImages,
  health,
  loading,
  onNext,
  onBack,
}: {
  value: GenMode;
  onChange: (m: GenMode) => void;
  suggested: GenMode;
  hasImages: boolean;
  /** mesh-server health from the parent query — undefined while loading */
  health: MeshHealth | undefined;
  loading?: boolean;
  onNext: () => void;
  onBack: () => void;
}) {
  const meshOk = health?.ok ?? false;

  return (
    <div>
      <h2 className="text-lg font-semibold">Generation mode</h2>
      <p className="mt-1 text-sm text-zinc-500">
        How the agent should build the part. When unsure, keep Auto or the
        recommended mode.
      </p>
      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {MODE_CARDS.map((m) => {
          // while health is still loading, don't flash-disable mesh modes
          const disabled =
            !loading && (m.id === "mesh" || m.id === "hybrid") && !meshOk;
          const active = value === m.id;
          const Icon = m.icon;
          return (
            <button
              key={m.id}
              type="button"
              disabled={disabled}
              title={disabled ? `${m.title} — mesh server unreachable` : m.title}
              onClick={() => onChange(m.id)}
              className={`relative rounded-lg border p-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                active
                  ? "border-brand-600 bg-brand-950/40"
                  : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-600"
              }`}
            >
              {active && (
                <span className="absolute right-2 top-2 flex h-4 w-4 items-center justify-center rounded-full bg-brand-500">
                  <Check className="h-3 w-3 text-white" />
                </span>
              )}
              <div className="flex items-center gap-2">
                <Icon className={`h-4 w-4 ${active ? "text-brand-300" : "text-zinc-500"}`} />
                <span className="font-medium">{m.label}</span>
                {m.id === suggested && (
                  <Chip tone="blue" className="ml-1">
                    suggested
                  </Chip>
                )}
              </div>
              <div className="mt-1 text-xs text-zinc-400">{m.desc}</div>
            </button>
          );
        })}
      </div>
      {(value === "mesh" || value === "hybrid") && !hasImages && (
        <div className="mt-2 text-[11px] text-amber-400">
          no reference images selected — the run will fall back to CAD.
        </div>
      )}
      {value === "topopt" && (
        <div className="mt-2 text-[11px] text-zinc-500">
          describe loads & fixtures in the prompt (e.g. "fixed at the holes, 100 N
          on top") — the agent picks the base geometry, then optimizes it.
        </div>
      )}
      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack}>
          Back
        </Btn>
        <Btn onClick={onNext}>
          Continue <ArrowRight className="h-4 w-4" />
        </Btn>
      </div>
    </div>
  );
}
