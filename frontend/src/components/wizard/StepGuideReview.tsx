import { BookOpenCheck, TriangleAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { GuideAudience, GuideOpts } from "../../lib/types";
import { Btn, Field, inputCls } from "../ui";

const LANGS = ["de", "en", "fr", "es", "it"] as const;
const AUDIENCES: Exclude<GuideAudience, "">[] = ["beginner", "intermediate", "pro"];

function OptToggle({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded border px-2 py-1 text-[11px] font-medium transition-colors ${
        active
          ? "border-brand-700 bg-brand-950/60 text-brand-300"
          : "border-edge2 bg-panel2/60 text-muted hover:border-faint hover:text-fg"
      }`}
    >
      {children}
    </button>
  );
}

/** Guide review step: name, summary and the four run options
 *  (part generation, LLM images, output language, audience rewrite). */
export default function StepGuideReview({
  name,
  nameAuto,
  onNameChange,
  pdfName,
  cadCount,
  notes,
  opts,
  onOptsChange,
  busy,
  llmConfigured,
  error,
  onStart,
  onBack,
}: {
  name: string;
  nameAuto?: boolean;
  onNameChange: (v: string) => void;
  pdfName: string;
  cadCount: number;
  notes: string;
  opts: GuideOpts;
  onOptsChange: (o: GuideOpts) => void;
  busy: boolean;
  llmConfigured?: boolean;
  error: string;
  onStart: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("wizard");
  const set = (patch: Partial<GuideOpts>) => onOptsChange({ ...opts, ...patch });
  return (
    <div>
      <h2 className="text-lg font-semibold">{t("greview_title")}</h2>
      <p className="mt-1 text-sm text-faint">{t("greview_hint")}</p>

      <div className="mt-4">
        <Field label={t("project_name")} hint={nameAuto ? t("name_auto_hint") : undefined}>
          <input
            className={inputCls}
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !busy) onStart();
            }}
          />
        </Field>
      </div>

      <dl className="grid grid-cols-1 gap-x-6 gap-y-2 rounded-lg border border-edge bg-panel/40 p-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("greview_pdf")}
          </dt>
          <dd className="mt-0.5 truncate text-fg" title={pdfName}>
            {pdfName}
          </dd>
        </div>
        <div>
          <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("greview_parts")}
          </dt>
          <dd className="mt-0.5 text-fg">
            {cadCount === 0
              ? t("common:state.none")
              : t("sum_assets_n", { count: cadCount })}
          </dd>
        </div>
        {notes.trim() && (
          <div className="sm:col-span-2">
            <dt className="text-[11px] font-medium uppercase tracking-wider text-faint">
              {t("greview_notes")}
            </dt>
            <dd className="mt-0.5 line-clamp-3 text-fg">{notes}</dd>
          </div>
        )}
      </dl>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <OptToggle
          title={t("greview_opt_gen_h")}
          active={opts.allowGeneration}
          onClick={() => set({ allowGeneration: !opts.allowGeneration })}
        >
          {t("greview_opt_gen")}
        </OptToggle>
        <OptToggle
          title={t("greview_opt_images_h")}
          active={opts.attachImages}
          onClick={() => set({ attachImages: !opts.attachImages })}
        >
          {t("greview_opt_images")}
        </OptToggle>
        <label className="flex items-center gap-1.5 rounded border border-edge2 bg-panel2/60 px-2 py-1 text-[11px] text-muted">
          {t("greview_lang")}
          <select
            className="rounded border border-edge2 bg-panel px-1.5 py-0.5 text-[11px] text-fg outline-none focus:border-brand-600"
            value={opts.language}
            onChange={(e) => set({ language: e.target.value })}
          >
            <option value="">{t("greview_lang_auto")}</option>
            {LANGS.map((l) => (
              <option key={l} value={l}>
                {t(`version:guide_lang_${l}`)}
              </option>
            ))}
          </select>
        </label>
        <label
          className="flex items-center gap-1.5 rounded border border-edge2 bg-panel2/60 px-2 py-1 text-[11px] text-muted"
          title={t("greview_aud_h")}
        >
          {t("greview_aud")}
          <select
            className="rounded border border-edge2 bg-panel px-1.5 py-0.5 text-[11px] text-fg outline-none focus:border-brand-600"
            value={opts.audience}
            onChange={(e) => set({ audience: e.target.value as GuideAudience })}
          >
            <option value="">{t("greview_aud_orig")}</option>
            {AUDIENCES.map((a) => (
              <option key={a} value={a}>
                {t(`version:guide_audience_${a}`)}
              </option>
            ))}
          </select>
        </label>
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
        <Btn onClick={onStart} busy={busy}>
          <BookOpenCheck className="h-4 w-4" />
          {t("greview_start")}
        </Btn>
      </div>
    </div>
  );
}
