/** Pure view-helpers for the Stats page — semantic colors + derived values. */

import type { Stats } from "./types";

/** Semantic colors so e.g. "failed" is red regardless of slice order. */
export const STATUS_COLORS: Record<string, string> = {
  succeeded: "#34d399", // emerald
  needs_review: "#fbbf24", // amber
  failed: "#f87171", // red
  cancelled: "#71717a", // zinc
  running: "#7c9cf5", // brand
  queued: "#a5b4fc", // lighter brand
  waiting_for_input: "#2dd4bf", // teal
};

/** Score bucket colors: <40 red → 40-59 amber → 60-79 teal → 80-100 emerald. */
export const SCORE_BUCKET_COLORS: Record<string, string> = {
  "<40": "#f87171",
  "40-59": "#fbbf24",
  "60-79": "#2dd4bf",
  "80-100": "#34d399",
};

export function statusColor(name: string, fallback = "#7c9cf5"): string {
  return STATUS_COLORS[name] ?? fallback;
}

export function scoreBucketColor(name: string): string {
  return SCORE_BUCKET_COLORS[name] ?? "#a1a1aa";
}

/** Tone for a pass-rate progress bar: red <40%, amber <70%, emerald ≥70%. */
export function passRateTone(rate: number | null): "red" | "amber" | "green" {
  if (rate == null) return "amber";
  if (rate < 0.4) return "red";
  if (rate < 0.7) return "amber";
  return "green";
}

export function scoreTone(score: number | null): "red" | "amber" | "green" {
  if (score == null) return "amber";
  if (score < 40) return "red";
  if (score < 80) return "amber";
  return "green";
}

/** The page is empty only when nothing at all happened in range. */
export function statsEmpty(s: Stats): boolean {
  return s.totals.versions + s.totals.jobs + s.totals.llm_calls === 0;
}

/** Mean LLM tokens (prompt+completion) spent per succeeded version. */
export function tokensPerSuccess(s: Stats): number | null {
  const succeeded = s.versions_by_status.find((v) => v.name === "succeeded")?.count ?? 0;
  if (!succeeded) return null;
  return Math.round((s.totals.prompt_tokens + s.totals.completion_tokens) / succeeded);
}

/** Mesh cache hit rate 0..1, null when there were no generations. */
export function meshCacheRate(s: Stats): number | null {
  if (!s.mesh.generations) return null;
  return s.mesh.cached / s.mesh.generations;
}
