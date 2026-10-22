import i18n from "../i18n";

export function locale(): string {
  return i18n.language || "en";
}

export function fmtNum(n: number): string {
  return n.toLocaleString(locale());
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return i18n.t("common:time.just_now");
  if (s < 3600)
    return i18n.t("common:time.minutes_ago", { count: Math.floor(s / 60) });
  if (s < 86400)
    return i18n.t("common:time.hours_ago", { count: Math.floor(s / 3600) });
  if (s < 86400 * 30)
    return i18n.t("common:time.days_ago", { count: Math.floor(s / 86400) });
  return new Date(t).toLocaleDateString(locale());
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString(locale());
}

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function fmtDuration(startIso: string | null | undefined, endIso?: string | null): string {
  if (!startIso) return "—";
  const start = new Date(startIso).getTime();
  const end = endIso ? new Date(endIso).getTime() : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return "—";
  const s = Math.max(0, (end - start) / 1000);
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

/** Seconds → "42s" / "3m 12s" / "1h 5m" — for step durations and ETAs. */
export function fmtSecs(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  const sec = Math.max(0, s);
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

export function fmtTokens(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 1000) return `${n}`;
  return `${(n / 1000).toFixed(1)}k`;
}

/** Compact number for chart axes — 1500 → "1.5k" (locale-aware). */
export function fmtCompact(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return new Intl.NumberFormat(locale(), {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(n);
}
