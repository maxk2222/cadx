import { Sparkles, TriangleAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation("wizard");
  return (
    <div>
      <h2 className="text-lg font-semibold">{t("review_title")}</h2>
      <p className="mt-1 text-sm text-faint">{t("review_hint")}</p>

      <div className="mt-4">
        <Field
          label={t("project_name")}
          hint={nameAuto ? t("name_auto_hint") : undefined}
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

      <dl className="grid grid-cols-1 gap-x-6 gap-y-2 rounded-lg border border-edge bg-panel/40 p-3 text-sm sm:grid-cols-2">
        <div className="sm:col-span-2">
          <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("sum_prompt")}
          </dt>
          <dd className="mt-0.5 line-clamp-3 text-zinc-300">{prompt}</dd>
        </div>
        <div>
          <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("sum_assets")}
          </dt>
          <dd className="mt-0.5 text-zinc-300">
            {assetCount === 0
              ? t("common:state.none")
              : t("sum_assets_n", { count: assetCount })}
          </dd>
        </div>
        <div>
          <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("sum_mode")}
          </dt>
          <dd className="mt-0.5 text-zinc-300">
            <span className="uppercase">{mode}</span>
            {suggested !== mode && (
              <span className="ml-1.5 text-[11px] text-faint/80">
                ({t("sum_suggested", { mode: suggested })})
              </span>
            )}
          </dd>
        </div>
        <div className="sm:col-span-2">
          <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("sum_options")}
          </dt>
          <dd className="mt-1 flex flex-wrap gap-1">
            <OptionChips options={options} />
            {!options.process && !options.material && !options.fem && !options.topopt && !options.include_fasteners && (
              <span className="text-xs text-faint/80">{t("sum_defaults")}</span>
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
          <span>{t("no_llm_warn")}</span>
        </div>
      )}

      {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack} disabled={busy}>
          {t("common:action.back")}
        </Btn>
        <Btn onClick={onGenerate} busy={busy} disabled={!prompt.trim()}>
          <Sparkles className="h-4 w-4" />
          {t("generate")}
          {assetCount ? ` · ${t("generate_assets", { count: assetCount })}` : ""}
        </Btn>
      </div>
    </div>
  );
}
