import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Settings2,
  Shrink,
  Wrench,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../lib/api";
import type { MeshModelInfo, ProcessType, RunOptions } from "../lib/types";
import { Chip } from "./ui";

export const PROCESS_OPTIONS: ProcessType[] = ["fdm", "sla", "cnc3", "cnc2d"];

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
  if (o.llm_backend) n++;
  if (o.mesh_backend) n++;
  if (o.image_backend) n++;
  if (o.mesh_model) n++;
  if (o.image_model) n++;
  if (o.image_edit_model) n++;
  return n;
}

/** read-only summary chips of the selected options (version header). */
export function OptionChips({ options }: { options: RunOptions | null | undefined }) {
  const { t } = useTranslation();
  if (!options) return null;
  const proc = options.process;
  return (
    <>
      {proc && (
        <Chip tone="blue" title={t("opt.process_hint")}>
          {t(`opt.process.${proc}`)}
        </Chip>
      )}
      {options.material && (
        <Chip tone="zinc" title={t("opt.material_hint")}>
          {options.material}
        </Chip>
      )}
      {options.fem && (
        <Chip tone="purple" title={options.fem_note || t("opt.fem_hint")}>
          <Activity className="h-3 w-3" /> FEM
        </Chip>
      )}
      {options.topopt && (
        <Chip
          tone="teal"
          title={t("opt.topopt_hint") + (options.topopt_volfrac ? ` → ${Math.round(options.topopt_volfrac * 100)}% ${t("opt.volume")}` : "")}
        >
          <Shrink className="h-3 w-3" /> TopOpt
          {options.topopt_volfrac ? ` ${Math.round(options.topopt_volfrac * 100)}%` : ""}
        </Chip>
      )}
      {options.include_fasteners && (
        <Chip tone="amber" title={t("opt.fasteners_hint")}>
          <Wrench className="h-3 w-3" /> {t("opt.fasteners")}
        </Chip>
      )}
      {options.llm_backend && (
        <Chip tone="blue" title={t("opt.llm_override")}>
          llm→{options.llm_backend}
        </Chip>
      )}
      {options.mesh_backend && (
        <Chip tone="blue" title={t("opt.mesh_override")}>
          mesh→{options.mesh_backend}
        </Chip>
      )}
      {options.image_backend && (
        <Chip tone="blue" title={t("opt.img_override")}>
          img→{options.image_backend}
        </Chip>
      )}
      {options.mesh_model && (
        <Chip tone="zinc" title={t("opt.mesh_model_override")}>
          mesh: {options.mesh_model}
        </Chip>
      )}
      {options.image_model && (
        <Chip tone="zinc" title={t("opt.img_model_override")}>
          img: {options.image_model}
        </Chip>
      )}
      {options.image_edit_model && (
        <Chip tone="zinc" title={t("opt.img_edit_override")}>
          img edit: {options.image_edit_model}
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
          : "border-edge2 bg-panel2/60 text-muted hover:border-edge2 hover:text-fg"
      }`}
    >
      {children}
    </button>
  );
}

function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mt-3 first:mt-0">
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-faint">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

const selCls =
  "rounded border border-edge2 bg-panel px-1.5 py-0.5 text-[11px] text-fg outline-none focus:border-brand-600";

/** small labelled select for the backends group — first option is the
 *  "default (settings)" passthrough (empty value). */
function Sel({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: [string, string][];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1.5 rounded border border-edge2 bg-panel2/60 px-2 py-1 text-[11px] text-muted">
      {label}
      <select
        className={selCls}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}

function useDefaultOpt(t: (k: string) => string): [string, string] {
  return ["", t("opt.default_settings")];
}

function modelOpts(list: MeshModelInfo[], def: [string, string]): [string, string][] {
  return [def, ...list.map((m): [string, string] => [m.id, m.name || m.id])];
}

/** Collapsible run-options panel — selections are injected into the agent run. */
export default function OptionsPanel({
  value,
  onChange,
}: {
  value: RunOptions;
  onChange: (o: RunOptions) => void;
}) {
  const { t } = useTranslation();
  const DEFAULT_OPT = useDefaultOpt(t);
  const count = optionsCount(value);
  const [open, setOpen] = useState(count > 0);
  const set = (patch: Partial<RunOptions>) => onChange({ ...value, ...patch });

  // sidecar model lists — only fetched when the panel is open and the
  // corresponding backend override selects the sidecar
  const sidecarModels = useQuery({
    queryKey: ["meshModels"],
    queryFn: api.meshModels,
    enabled: open && value.mesh_backend === "sidecar",
    staleTime: 30_000,
    retry: false,
  });
  const imageModels = useQuery({
    queryKey: ["imageModels"],
    queryFn: api.imageModels,
    enabled: open && value.image_backend === "sidecar",
    staleTime: 30_000,
    retry: false,
  });
  const meshOpts = (sidecarModels.data ?? []).filter(
    (m) => m.kind !== "image" && m.kind !== "llm",
  );
  const customMaterial =
    value.material && !MATERIAL_OPTIONS.includes(value.material)
      ? value.material
      : "";

  return (
    <div className="rounded-lg border border-edge bg-panel/40">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-muted hover:text-fg"
      >
        <Settings2 className="h-3.5 w-3.5" />
        {t("opt.title")}
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
        <div className="border-t border-edge px-3 pb-3 pt-2">
          <Group label={t("opt.group_process")}>
            {PROCESS_OPTIONS.map((p) => (
              <Opt
                key={p}
                title={t(`opt.process_${p}_h`)}
                active={value.process === p}
                onClick={() =>
                  set({ process: value.process === p ? undefined : p })
                }
              >
                {t(`opt.process.${p}`)}
              </Opt>
            ))}
          </Group>

          <Group label={t("opt.group_material")}>
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
              className="w-32 rounded border border-edge2 bg-panel2/80 px-2 py-1 text-[11px] text-fg placeholder-faint outline-none focus:border-brand-600"
              placeholder={t("opt.custom_material")}
              value={customMaterial}
              onChange={(e) => set({ material: e.target.value || undefined })}
            />
          </Group>

          <Group label={t("opt.group_analysis")}>
            <Opt
              title={t("opt.fem_check_h")}
              active={!!value.fem}
              onClick={() => set({ fem: !value.fem || undefined })}
            >
              <Activity className="h-3 w-3" /> {t("opt.fem_check")}
            </Opt>
            <Opt
              title={t("opt.topopt_check_h")}
              active={!!value.topopt}
              onClick={() => set({ topopt: !value.topopt || undefined })}
            >
              <Shrink className="h-3 w-3" /> {t("opt.topopt_check")}
            </Opt>
            <Opt
              title={t("opt.fasteners_check_h")}
              active={!!value.include_fasteners}
              onClick={() =>
                set({ include_fasteners: !value.include_fasteners || undefined })
              }
            >
              <Wrench className="h-3 w-3" /> {t("opt.fasteners_check")}
            </Opt>
          </Group>

          {!!value.fem && (
            <input
              className="mt-2 w-full rounded border border-edge2 bg-panel2/80 px-2 py-1.5 text-[11px] text-fg placeholder-faint outline-none focus:border-brand-600"
              placeholder={t("opt.fem_note_ph")}
              value={value.fem_note ?? ""}
              onChange={(e) => set({ fem_note: e.target.value || undefined })}
            />
          )}
          {!!value.topopt && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-[11px] text-muted">{t("opt.target_volume")}</span>
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
              <span className="w-8 text-[11px] text-fg">
                {Math.round((value.topopt_volfrac ?? 0.4) * 100)}%
              </span>
            </div>
          )}

          <Group label={t("opt.group_backends")}>
            <Sel
              label="llm"
              value={value.llm_backend ?? ""}
              options={[
                DEFAULT_OPT,
                ["external", "external"],
                ["sidecar", "sidecar"],
              ]}
              onChange={(v) => set({ llm_backend: (v || undefined) as RunOptions["llm_backend"] })}
            />
            <Sel
              label="mesh"
              value={value.mesh_backend ?? ""}
              options={[
                DEFAULT_OPT,
                ["sidecar", "sidecar"],
                ["tripo", "tripo"],
              ]}
              onChange={(v) => set({ mesh_backend: (v || undefined) as RunOptions["mesh_backend"] })}
            />
            <Sel
              label="image"
              value={value.image_backend ?? ""}
              options={[
                DEFAULT_OPT,
                ["sidecar", "sidecar"],
                ["openai", "openai"],
              ]}
              onChange={(v) => set({ image_backend: (v || undefined) as RunOptions["image_backend"] })}
            />
            {value.mesh_backend === "sidecar" && (
              <Sel
                label={t("opt.mesh_model")}
                value={value.mesh_model ?? ""}
                options={modelOpts(meshOpts, DEFAULT_OPT)}
                onChange={(v) => set({ mesh_model: v || undefined })}
              />
            )}
            {value.image_backend === "sidecar" && (
              <>
                <Sel
                  label={t("opt.image_model")}
                  value={value.image_model ?? ""}
                  options={modelOpts(imageModels.data ?? [], DEFAULT_OPT)}
                  onChange={(v) => set({ image_model: v || undefined })}
                />
                <Sel
                  label={t("opt.edit_model")}
                  value={value.image_edit_model ?? ""}
                  options={modelOpts(
                    (imageModels.data ?? []).filter((m) => m.supports_edit),
                    DEFAULT_OPT,
                  )}
                  onChange={(v) => set({ image_edit_model: v || undefined })}
                />
              </>
            )}
          </Group>

          <Group label={t("opt.group_budget")}>
            {(
              [
                ["max_iterations", t("opt.max_iterations"), 1, 200],
                ["max_repairs", t("opt.max_repairs"), 0, 200],
                ["pass_score", t("opt.pass_score"), 0, 100],
              ] as const
            ).map(([key, label, min, max]) => (
              <label
                key={key}
                className="flex items-center gap-1.5 rounded border border-edge2 bg-panel2/60 px-2 py-1 text-[11px] text-muted"
              >
                {label}
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={1}
                  className="w-16 rounded border border-edge2 bg-panel px-1.5 py-0.5 text-[11px] text-fg outline-none focus:border-brand-600"
                  placeholder={t("opt.auto")}
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
