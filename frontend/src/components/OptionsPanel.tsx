import {
  Activity,
  ChevronDown,
  ChevronRight,
  Settings2,
  Shrink,
  Wrench,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import type { ProcessType, RunOptions } from "../lib/types";
import { Chip } from "./ui";

export const PROCESS_OPTIONS: { id: ProcessType; label: string; title: string }[] = [
  { id: "fdm", label: "FDM", title: "FDM 3D printing (filament)" },
  { id: "sla", label: "SLA", title: "SLA resin printing" },
  { id: "cnc3", label: "CNC 3-axis", title: "3-axis CNC machining" },
  { id: "cnc2d", label: "CNC 2D", title: "2D CNC cutting (sheet parts)" },
];

export const MATERIAL_OPTIONS = [
  "pla",
  "petg",
  "abs",
  "asa",
  "pa",
  "pc",
  "resin",
  "aluminum",
  "steel",
];

/** number of explicitly set options (for the collapsed badge) */
export function optionsCount(o: RunOptions): number {
  let n = 0;
  if (o.process) n++;
  if (o.material) n++;
  if (o.include_fasteners) n++;
  if (o.fem) n++;
  if (o.topopt) n++;
  if (o.max_iterations != null) n++;
  if (o.max_repairs != null) n++;
  if (o.pass_score != null) n++;
  return n;
}

/** read-only summary chips of the selected options (version header). */
export function OptionChips({ options }: { options: RunOptions | null | undefined }) {
  if (!options) return null;
  const proc = PROCESS_OPTIONS.find((p) => p.id === options.process);
  return (
    <>
      {proc && (
        <Chip tone="blue" title="manufacturing process">
          {proc.label}
        </Chip>
      )}
      {options.material && (
        <Chip tone="zinc" title="material">
          {options.material}
        </Chip>
      )}
      {options.fem && (
        <Chip tone="purple" title={options.fem_note || "FEM structural analysis"}>
          <Activity className="h-3 w-3" /> FEM
        </Chip>
      )}
      {options.topopt && (
        <Chip
          tone="teal"
          title={`topology optimization${options.topopt_volfrac ? ` → ${Math.round(options.topopt_volfrac * 100)}% volume` : ""}`}
        >
          <Shrink className="h-3 w-3" /> TopOpt
          {options.topopt_volfrac ? ` ${Math.round(options.topopt_volfrac * 100)}%` : ""}
        </Chip>
      )}
      {options.include_fasteners && (
        <Chip tone="amber" title="include standard fasteners">
          <Wrench className="h-3 w-3" /> fasteners
        </Chip>
      )}
    </>
  );
}

function Opt({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  title?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded border px-2 py-1 text-[11px] font-medium transition-colors ${
        active
          ? "border-brand-700 bg-brand-950/60 text-brand-300"
          : "border-zinc-700 bg-zinc-800/60 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
      }`}
    >
      {children}
    </button>
  );
}

function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mt-3 first:mt-0">
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

/** Collapsible run-options panel — selections are injected into the agent run. */
export default function OptionsPanel({
  value,
  onChange,
}: {
  value: RunOptions;
  onChange: (o: RunOptions) => void;
}) {
  const count = optionsCount(value);
  const [open, setOpen] = useState(count > 0);
  const set = (patch: Partial<RunOptions>) => onChange({ ...value, ...patch });
  const customMaterial =
    value.material && !MATERIAL_OPTIONS.includes(value.material)
      ? value.material
      : "";

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-zinc-400 hover:text-zinc-200"
      >
        <Settings2 className="h-3.5 w-3.5" />
        Options
        {count > 0 && (
          <span className="rounded-full bg-brand-600 px-1.5 text-[10px] text-white">
            {count}
          </span>
        )}
        <span className="flex-1" />
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
      </button>
      {open && (
        <div className="border-t border-zinc-800 px-3 pb-3 pt-2">
          <Group label="Manufacturing process">
            {PROCESS_OPTIONS.map((p) => (
              <Opt
                key={p.id}
                title={p.title}
                active={value.process === p.id}
                onClick={() =>
                  set({ process: value.process === p.id ? undefined : p.id })
                }
              >
                {p.label}
              </Opt>
            ))}
          </Group>

          <Group label="Material">
            {MATERIAL_OPTIONS.map((m) => (
              <Opt
                key={m}
                active={value.material === m}
                onClick={() =>
                  set({ material: value.material === m ? undefined : m })
                }
              >
                {m}
              </Opt>
            ))}
            <input
              className="w-32 rounded border border-zinc-700 bg-zinc-800/80 px-2 py-1 text-[11px] text-zinc-200 placeholder-zinc-600 outline-none focus:border-brand-600"
              placeholder="custom material…"
              value={customMaterial}
              onChange={(e) => set({ material: e.target.value || undefined })}
            />
          </Group>

          <Group label="Analysis & extras">
            <Opt
              title="run FEM structural analysis after generation"
              active={!!value.fem}
              onClick={() => set({ fem: !value.fem || undefined })}
            >
              <Activity className="h-3 w-3" /> FEM check
            </Opt>
            <Opt
              title="topology-optimize the part (SIMP; generates a topopt version)"
              active={!!value.topopt}
              onClick={() => set({ topopt: !value.topopt || undefined })}
            >
              <Shrink className="h-3 w-3" /> Topology optimization
            </Opt>
            <Opt
              title="include standard fasteners from bd_warehouse"
              active={!!value.include_fasteners}
              onClick={() =>
                set({ include_fasteners: !value.include_fasteners || undefined })
              }
            >
              <Wrench className="h-3 w-3" /> Standard fasteners
            </Opt>
          </Group>

          {!!value.fem && (
            <input
              className="mt-2 w-full rounded border border-zinc-700 bg-zinc-800/80 px-2 py-1.5 text-[11px] text-zinc-200 placeholder-zinc-600 outline-none focus:border-brand-600"
              placeholder="loads & fixtures (optional) — e.g. 'fixed at M4 holes, 100 N on top face'"
              value={value.fem_note ?? ""}
              onChange={(e) => set({ fem_note: e.target.value || undefined })}
            />
          )}
          {!!value.topopt && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-[11px] text-zinc-500">target volume</span>
              <input
                type="range"
                min={10}
                max={90}
                step={5}
                className="w-40 accent-brand-500"
                value={Math.round((value.topopt_volfrac ?? 0.4) * 100)}
                onChange={(e) =>
                  set({ topopt_volfrac: Number(e.target.value) / 100 })
                }
              />
              <span className="w-8 text-[11px] text-zinc-300">
                {Math.round((value.topopt_volfrac ?? 0.4) * 100)}%
              </span>
            </div>
          )}

          <Group label="Agent budget (override defaults)">
            {(
              [
                ["max_iterations", "max iterations", 1, 200],
                ["max_repairs", "max repairs", 0, 200],
                ["pass_score", "pass score", 0, 100],
              ] as const
            ).map(([key, label, min, max]) => (
              <label
                key={key}
                className="flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-800/60 px-2 py-1 text-[11px] text-zinc-400"
              >
                {label}
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={1}
                  className="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[11px] text-zinc-200 outline-none focus:border-brand-600"
                  placeholder="auto"
                  value={value[key] ?? ""}
                  onChange={(e) =>
                    set({
                      [key]:
                        e.target.value === ""
                          ? undefined
                          : Math.max(
                              min,
                              Math.min(max, Math.round(Number(e.target.value)))
                            ),
                    })
                  }
                />
              </label>
            ))}
          </Group>
        </div>
      )}
    </div>
  );
}
