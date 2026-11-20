import { ArrowRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Btn, inputCls } from "../ui";

/** Guide notes step — optional free text that lands in state.user_notes:
 *  tips, warnings and extra context the manual itself does not carry. */
export default function StepGuideNotes({
  prompt,
  onChange,
  onNext,
  onBack,
}: {
  prompt: string;
  onChange: (v: string) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("wizard");
  return (
    <div>
      <h2 className="text-lg font-semibold">
        {t("gnotes_title")}{" "}
        <span className="text-sm font-normal text-faint">({t("common:field.optional")})</span>
      </h2>
      <p className="mt-1 text-sm text-faint">{t("gnotes_hint")}</p>
      <textarea
        autoFocus
        className={`${inputCls} mt-4 h-32 resize-y`}
        placeholder={t("gnotes_placeholder")}
        value={prompt}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            onNext();
          }
        }}
      />
      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack}>
          {t("common:action.back")}
        </Btn>
        <div className="flex items-center gap-3">
          <button
            className="text-xs text-faint underline transition-colors hover:text-muted"
            onClick={onNext}
          >
            {t("gnotes_skip")}
          </button>
          <Btn onClick={onNext}>
            {t("continue")} <ArrowRight className="h-4 w-4" />
          </Btn>
        </div>
      </div>
    </div>
  );
}
