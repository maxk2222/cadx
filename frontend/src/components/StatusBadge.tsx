import { useTranslation } from "react-i18next";

const COLORS: Record<string, string> = {
  queued: "bg-edge2",
  running: "bg-brand-600 animate-pulse",
  waiting_for_input: "bg-amber-600",
  succeeded: "bg-emerald-600",
  needs_review: "bg-amber-700",
  failed: "bg-red-600",
  cancelled: "bg-edge2",
};

export default function StatusBadge({
  status,
  title,
}: {
  status: string;
  title?: string;
}) {
  const { t } = useTranslation();
  const key = `status.${status}`;
  const label = t(key);
  return (
    <span
      title={title}
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${COLORS[status] ?? "bg-edge2"}`}
    >
      {label === key ? status.replaceAll("_", " ") : label}
    </span>
  );
}
