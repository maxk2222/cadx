import { ArrowRight, Lightbulb } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Btn, inputCls } from "../ui";

const EXAMPLE_KEYS = ["ex_1", "ex_2", "ex_3", "ex_4"] as const;

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
  const { t } = useTranslation("wizard");
  const ok = !!prompt.trim();
  return (
    <div>
      <h2 className="text-lg font-semibold">{t("prompt_title")}</h2>
      <p className="mt-1 text-sm text-faint">{t("prompt_hint")}</p>
      <textarea
        autoFocus
        className={`${inputCls} mt-4 h-40 resize-y`}
        placeholder={t("prompt_placeholder")}
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
        <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0 text-faint" />
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLE_KEYS.map((k) => {
            const ex = t(k);
            return (
              <button
                key={k}
                type="button"
                title={t("use_example")}
                onClick={() => onChange(ex)}
                className="rounded-full border border-edge2 bg-panel2 px-2.5 py-1 text-[11px] text-muted transition-all duration-150 hover:border-faint/50 hover:text-fg"
              >
                {ex.length > 64 ? `${ex.slice(0, 64)}…` : ex}
              </button>
            );
          })}
        </div>
      </div>
      {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
      <div className="mt-6 flex items-center justify-between">
        <span className="text-[11px] text-faint/70">
          {t("ctrl_enter_next")}
          {busy ? ` — ${t("creating_project")}` : ""}
        </span>
        <Btn onClick={onNext} disabled={!ok} busy={busy}>
          {t("continue")} <ArrowRight className="h-4 w-4" />
        </Btn>
      </div>
    </div>
  );
}
