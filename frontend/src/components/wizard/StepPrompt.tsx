import { ArrowRight, Lightbulb } from "lucide-react";
import { Btn, inputCls } from "../ui";

const EXAMPLES = [
  "Mounting plate 80×60×8mm, four M4 clearance holes 10mm from the corners, countersunk, PLA for FDM.",
  "Desk cable clip, 40mm wide, snap-fit spring tab, PETG.",
  "Organic drawer knob, rounded pebble shape, ~45mm diameter, M4 heat-set insert on the back.",
  "L-bracket 60×40×30mm, two M5 holes per leg, carries 200N vertical — topology-optimize to 40% volume.",
];

export default function StepPrompt({
  prompt,
  onChange,
  onNext,
  busy,
  error,
}: {
  prompt: string;
  onChange: (v: string) => void;
  onNext: () => void;
  busy?: boolean;
  error?: string;
}) {
  const ok = !!prompt.trim();
  return (
    <div>
      <h2 className="text-lg font-semibold">Describe your part</h2>
      <p className="mt-1 text-sm text-zinc-500">
        Shape, dimensions (X length × Y width × Z height, mm), holes, threads,
        tolerances, material and process. The agent will measure, plan, generate
        and verify a parametric 3D model.
      </p>
      <textarea
        autoFocus
        className={`${inputCls} mt-4 h-40 resize-y`}
        placeholder={
          "e.g. Mounting plate 80×60×8mm, four M4 clearance holes at 10mm from corners, countersunk, PLA for FDM."
        }
        value={prompt}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && ok) {
            e.preventDefault();
            onNext();
          }
        }}
      />
      <div className="mt-3 flex items-start gap-2">
        <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              title="use as prompt"
              onClick={() => onChange(ex)}
              className="rounded-full border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-[11px] text-zinc-400 transition-colors hover:border-zinc-500 hover:text-zinc-200"
            >
              {ex.length > 64 ? `${ex.slice(0, 64)}…` : ex}
            </button>
          ))}
        </div>
      </div>
      {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
      <div className="mt-6 flex items-center justify-between">
        <span className="text-[11px] text-zinc-600">
          Ctrl+Enter to continue{busy ? " — creating project…" : ""}
        </span>
        <Btn onClick={onNext} disabled={!ok} busy={busy}>
          Continue <ArrowRight className="h-4 w-4" />
        </Btn>
      </div>
    </div>
  );
}
