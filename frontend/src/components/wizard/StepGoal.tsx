import { ArrowRight, BookOpenCheck, Box } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { WizardGoal } from "../../lib/types";
import { Spinner } from "../ui";

const CARDS: { id: WizardGoal; icon: typeof Box }[] = [
  { id: "part", icon: Box },
  { id: "guide", icon: BookOpenCheck },
];

/** Step 0: what should the wizard build — a parametric part or an
 *  interactive assembly guide? Picking a card selects AND advances. */
export default function StepGoal({
  busy,
  error,
  onPick,
}: {
  /** project creation in flight — cards stay visible but inert */
  busy?: boolean;
  error?: string;
  onPick: (g: WizardGoal) => void;
}) {
  const { t } = useTranslation("wizard");
  return (
    <div>
      <h2 className="text-lg font-semibold">{t("goal_title")}</h2>
      <p className="mt-1 text-sm text-faint">{t("goal_hint")}</p>
      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {CARDS.map((c) => {
          const Icon = c.icon;
          return (
            <button
              key={c.id}
              type="button"
              disabled={busy}
              onClick={() => onPick(c.id)}
              className="group relative rounded-lg border border-edge bg-panel/60 p-4 text-left transition-all duration-150 hover:border-brand-600 hover:bg-brand-950/40 disabled:opacity-60 disabled:hover:border-edge disabled:hover:bg-panel/60"
            >
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4 text-faint transition-colors group-hover:text-brand-300" />
                <span className="font-medium">{t(`goal_${c.id}`)}</span>
                <ArrowRight className="ml-auto h-4 w-4 text-faint/40 transition-all group-hover:translate-x-0.5 group-hover:text-brand-300" />
              </div>
              <div className="mt-1 text-xs text-muted">{t(`goal_${c.id}_desc`)}</div>
            </button>
          );
        })}
      </div>
      {busy && (
        <div className="mt-3 flex items-center gap-2 text-xs text-faint">
          <Spinner className="h-3.5 w-3.5" /> {t("creating_project")}
        </div>
      )}
      {error && <div className="mt-2 text-sm text-red-400">{error}</div>}
    </div>
  );
}
