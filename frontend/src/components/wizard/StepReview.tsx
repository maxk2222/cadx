import { Sparkles, TriangleAlert } from "lucide-react";
import type { GenMode, RunOptions } from "../../lib/types";
import OptionsPanel, { OptionChips } from "../OptionsPanel";
import { Btn, Field, inputCls } from "../ui";

export default function StepReview({
  name,
  nameAuto,
  onNameChange,
  prompt,
  assetCount,
  mode,
  suggested,
  options,
  onOptionsChange,
  busy,
  llmConfigured,
  error,
  onGenerate,
  onBack,
}: {
  name: string;
  /** still the auto-derived name (unmodified by the user) */
  nameAuto?: boolean;
  onNameChange: (v: string) => void;
  prompt: string;
  assetCount: number;
  mode: GenMode;
  suggested: GenMode;
  options: RunOptions;
  onOptionsChange: (o: RunOptions) => void;
  busy: boolean;
  /** undefined while the health check is pending */
  llmConfigured?: boolean;
  error: string;
  onGenerate: () => void;
  onBack: () => void;
}) {
  return (
    <div>
      <h2 className="text-lg font-semibold">Review &amp; generate</h2>
      <p className="mt-1 text-sm text-zinc-500">
        The project is created automatically — adjust the name, set run options,
        then start.
      </p>

      <div className="mt-4">
        <Field
          label="Project name"
          hint={nameAuto ? "auto-derived from the description — editable" : undefined}
        >
          <input
            className={inputCls}
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && prompt.trim() && !busy) onGenerate();
            }}
          />
        </Field>
      </div>

      <dl className="grid grid-cols-1 gap-x-6 gap-y-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-sm sm:grid-cols-2">
        <div className="sm:col-span-2">
          <dt className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            prompt
          </dt>
          <dd className="mt-0.5 line-clamp-3 text-zinc-300">{prompt}</dd>
        </div>
        <div>
          <dt className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            reference assets
          </dt>
          <dd className="mt-0.5 text-zinc-300">
            {assetCount === 0 ? "none" : `${assetCount} selected`}
          </dd>
        </div>
        <div>
          <dt className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            mode
          </dt>
          <dd className="mt-0.5 text-zinc-300">
            <span className="uppercase">{mode}</span>
            {suggested !== mode && (
              <span className="ml-1.5 text-[11px] text-zinc-600">
                (suggested: {suggested})
              </span>
            )}
          </dd>
        </div>
        <div className="sm:col-span-2">
          <dt className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            options
          </dt>
          <dd className="mt-1 flex flex-wrap gap-1">
            <OptionChips options={options} />
            {!options.process && !options.material && !options.fem && !options.topopt && !options.include_fasteners && (
              <span className="text-xs text-zinc-600">defaults</span>
            )}
          </dd>
        </div>
      </dl>

      <div className="mt-3">
        <OptionsPanel value={options} onChange={onOptionsChange} />
      </div>

      {llmConfigured === false && (
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-800/60 bg-amber-950/40 p-3 text-sm text-amber-300">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            No LLM provider is configured — generation will fail. Check{" "}
            <span className="font-medium">Settings → provider</span> first.
          </span>
        </div>
      )}

      {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack} disabled={busy}>
          Back
        </Btn>
        <Btn onClick={onGenerate} busy={busy} disabled={!prompt.trim()}>
          <Sparkles className="h-4 w-4" />
          Generate model{assetCount ? ` · ${assetCount} asset${assetCount === 1 ? "" : "s"}` : ""}
        </Btn>
      </div>
    </div>
  );
}
