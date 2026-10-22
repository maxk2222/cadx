import { Languages } from "lucide-react";
import { useTranslation } from "react-i18next";
import { LANG_LABEL, SUPPORTED, type Lang } from "../i18n";

/** Compact language dropdown — flag-free, shows the native label. */
export default function LanguageSwitcher({ className = "" }: { className?: string }) {
  const { i18n, t } = useTranslation();
  return (
    <label
      className={`flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-zinc-200 ${className}`}
      title={t("nav:language")}
    >
      <Languages className="h-4 w-4 shrink-0" />
      <select
        aria-label={t("nav:language")}
        className="w-full cursor-pointer appearance-none bg-transparent text-xs uppercase tracking-wide outline-none md:lowercase"
        value={i18n.language}
        onChange={(e) => void i18n.changeLanguage(e.target.value as Lang)}
      >
        {SUPPORTED.map((l) => (
          <option key={l} value={l} className="bg-zinc-900 text-zinc-200">
            {LANG_LABEL[l]}
          </option>
        ))}
      </select>
    </label>
  );
}
