import { stepLabel } from "./steps";
import type { JobLive } from "./types";

/** One row of a pipeline display: a completed step from step_history or the
 *  currently running step (active). */
export interface StepRow {
  step: string;
  label: string;
  dt_s: number;
  active: boolean;
}

/** Merge step_history (completed steps, observed order) with the currently
 *  running live.step appended at the end, its elapsed time computed live from
 *  step_started_ts against nowMs. History only ever holds completed steps (the
 *  worker appends a step on change), so a trailing entry equal to live.step is
 *  a stale duplicate — merged, not doubled. */
export function stepRows(live: JobLive | null | undefined, nowMs: number): StepRow[] {
  if (!live) return [];
  const rows: StepRow[] = (live.step_history ?? []).map((h) => ({
    step: h.step,
    label: stepLabel(h.step),
    dt_s: h.dt_s,
    active: false,
  }));
  if (live.step) {
    const t0 = live.step_started_ts ? Date.parse(live.step_started_ts) : NaN;
    const dt = Number.isNaN(t0) ? 0 : Math.max(0, (nowMs - t0) / 1000);
    const last = rows[rows.length - 1];
    if (last && last.step === live.step) {
      last.dt_s = dt;
      last.active = true;
    } else {
      rows.push({ step: live.step, label: stepLabel(live.step), dt_s: dt, active: true });
    }
  }
  return rows;
}

/** "x,y x,y …" SVG points, min/max-normalized into width×height (y flipped so
 *  larger values sit higher). Empty input → "", a single value → one centered
 *  point. */
export function sparkPoints(values: number[], width: number, height: number): string {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const y = (v: number) => (span > 0 ? height - ((v - min) / span) * height : height / 2);
  if (values.length === 1) return `${(width / 2).toFixed(1)},${y(values[0]).toFixed(1)}`;
  return values
    .map((v, i) => `${((i / (values.length - 1)) * width).toFixed(1)},${y(v).toFixed(1)}`)
    .join(" ");
}

/** 842 → "842", 142300 → "142.3k", 2500000 → "2.5M". */
export function fmtTokens(n: number): string {
  if (n < 1000) return `${Math.round(n)}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

/** Seconds → "0:00" / "1:23" / "1:00:15" (h:mm:ss once ≥ 1h). */
export function fmtDur(s: number): string {
  const t = Math.max(0, Math.floor(s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  const p = (x: number) => String(x).padStart(2, "0");
  return h > 0 ? `${h}:${p(m)}:${p(sec)}` : `${m}:${p(sec)}`;
}

/** Linear ETA in seconds: elapsed = sum(step_history.dt_s) + running step
 *  elapsed (from step_started_ts vs Date.now()), then elapsed/progress −
 *  elapsed. null when progress ≤ 0.02 or > 0.98 or nothing has elapsed yet. */
export function etaSeconds(live: JobLive | null | undefined, progress: number): number | null {
  if (!live || progress <= 0.02 || progress > 0.98) return null;
  let elapsed = (live.step_history ?? []).reduce((a, h) => a + h.dt_s, 0);
  if (live.step && live.step_started_ts) {
    const t0 = Date.parse(live.step_started_ts);
    if (!Number.isNaN(t0)) elapsed += Math.max(0, (Date.now() - t0) / 1000);
  }
  if (elapsed <= 0) return null;
  return elapsed / progress - elapsed;
}
