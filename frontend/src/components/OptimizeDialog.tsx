import { Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import type { LoadCase } from "../lib/types";
import { Btn, Field, Modal, inputCls } from "./ui";
import { MATERIAL_OPTIONS } from "./OptionsPanel";

const FACES = ["-x", "+x", "-y", "+y", "-z", "+z"] as const;
const FACE_LABELS: Record<string, string> = {
  "-x": "left (−X)",
  "+x": "right (+X)",
  "-y": "back (−Y)",
  "+y": "front (+Y)",
  "-z": "bottom (−Z)",
  "+z": "top (+Z)",
};
const FORCE_DIRS = ["-z", "+z", "-x", "+x", "-y", "+y"] as const;

interface CaseForm {
  name: string;
  fixed_face: string;
  load_face: string;
  force_n: number;
  force_dir: string;
}

function caseToApi(c: CaseForm): LoadCase {
  const v = [0, 0, 0];
  const axis = { x: 0, y: 1, z: 2 }[c.force_dir[1]] as number;
  v[axis] = c.force_dir.startsWith("-") ? -c.force_n : c.force_n;
  return {
    name: c.name || "case1",
    fixed: [{ kind: "face", axis: c.fixed_face }],
    loads: [{ region: { kind: "face", axis: c.load_face }, force_n: v }],
  };
}

const DEFAULT_CASE: CaseForm = {
  name: "load",
  fixed_face: "-x",
  load_face: "+x",
  force_n: 50,
  force_dir: "-z",
};

function forceToForm(force?: number[]): { force_n: number; force_dir: string } {
  if (!force?.length) return { force_n: 50, force_dir: "-z" };
  let i = 0;
  force.forEach((v, j) => {
    if (Math.abs(v) > Math.abs(force[i])) i = j;
  });
  return {
    force_n: Math.abs(force[i]) || 50,
    force_dir: `${force[i] < 0 ? "-" : "+"}${"xyz"[i]}`,
  };
}

export default function OptimizeDialog({
  open,
  onClose,
  onSubmit,
  busy,
  initial,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (body: {
    load_cases: LoadCase[];
    optimize: { target_volume_fraction: number; enabled: boolean };
    material?: string;
  }) => void;
  busy?: boolean;
  initial?: LoadCase[];
}) {
  const [cases, setCases] = useState<CaseForm[]>(
    initial?.length
      ? initial.map((lc) => ({
          name: lc.name,
          fixed_face: lc.fixed[0]?.axis || "-x",
          load_face: lc.loads[0]?.region.axis || "+x",
          ...forceToForm(lc.loads[0]?.force_n),
        }))
      : [{ ...DEFAULT_CASE }]
  );
  const [volfrac, setVolfrac] = useState(0.4);
  const [material, setMaterial] = useState("");

  const setCase = (i: number, patch: Partial<CaseForm>) =>
    setCases((cs) => cs.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  return (
    <Modal open={open} onClose={onClose} title="Topology optimization" wide>
      <div className="mb-3 text-xs text-zinc-500">
        SIMP optimization lightens the part to a target volume while keeping the load
        paths — produces organic, print-ready structures. Declares how the part is held
        and loaded.
      </div>

      {cases.map((c, i) => (
        <div
          key={i}
          className="mb-3 rounded-lg border border-zinc-800 p-3"
        >
          <div className="mb-2 flex items-center gap-2">
            <input
              className={`${inputCls} w-40`}
              value={c.name}
              onChange={(e) => setCase(i, { name: e.target.value })}
              placeholder="case name"
            />
            <span className="flex-1" />
            {cases.length > 1 && (
              <button
                className="text-zinc-500 hover:text-red-400"
                onClick={() => setCases((cs) => cs.filter((_, j) => j !== i))}
                title="remove case"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Fixed at">
              <select
                className={inputCls}
                value={c.fixed_face}
                onChange={(e) => setCase(i, { fixed_face: e.target.value })}
              >
                {FACES.map((f) => (
                  <option key={f} value={f}>
                    {FACE_LABELS[f]}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Force applied on">
              <select
                className={inputCls}
                value={c.load_face}
                onChange={(e) => setCase(i, { load_face: e.target.value })}
              >
                {FACES.map((f) => (
                  <option key={f} value={f}>
                    {FACE_LABELS[f]}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Force (N)">
              <input
                type="number"
                className={inputCls}
                min={0.1}
                step={1}
                value={c.force_n}
                onChange={(e) => setCase(i, { force_n: Number(e.target.value) || 0 })}
              />
            </Field>
            <Field label="Force direction">
              <select
                className={inputCls}
                value={c.force_dir}
                onChange={(e) => setCase(i, { force_dir: e.target.value })}
              >
                {FORCE_DIRS.map((d) => (
                  <option key={d} value={d}>
                    {d.toUpperCase()}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        </div>
      ))}

      <button
        className="mb-3 flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200"
        onClick={() => setCases((cs) => [...cs, { ...DEFAULT_CASE, name: `case${cs.length + 1}` }])}
      >
        <Plus className="h-3.5 w-3.5" /> add load case
      </button>

      <div className="grid grid-cols-2 gap-3">
        <Field label="Material" hint="leave empty to use project material">
          <select
            className={inputCls}
            value={material}
            onChange={(e) => setMaterial(e.target.value)}
          >
            <option value="">— default —</option>
            {MATERIAL_OPTIONS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
        <Field label={`Target volume: ${Math.round(volfrac * 100)}%`}>
          <input
            type="range"
            min={10}
            max={90}
            step={5}
            className="mt-2 w-full accent-brand-500"
            value={Math.round(volfrac * 100)}
            onChange={(e) => setVolfrac(Number(e.target.value) / 100)}
          />
        </Field>
      </div>

      <div className="mt-2 flex justify-end gap-2">
        <Btn variant="ghost" onClick={onClose}>
          Cancel
        </Btn>
        <Btn
          variant="success"
          busy={busy}
          disabled={!cases.length || cases.some((c) => !(c.force_n > 0))}
          onClick={() =>
            onSubmit({
              load_cases: cases.map(caseToApi),
              optimize: { enabled: true, target_volume_fraction: volfrac },
              material: material || undefined,
            })
          }
        >
          Optimize
        </Btn>
      </div>
    </Modal>
  );
}
