/** Activity-timeline bucketing + localized tick formatting for the Stats page.
 *
 *  The backend returns a gapless list of `StatsDayBucket`s. Long ranges
 *  (90d, all-time) render unusable 1px bars, so days are folded into ISO-week
 *  or month buckets. All helpers are pure for testing.
 */

import type { StatsDayBucket } from "./types";

export type BucketMode = "day" | "week" | "month";

export interface ActivityBucket {
  /** first day of the bucket, YYYY-MM-DD */
  start: string;
  /** last day of the bucket, YYYY-MM-DD */
  end: string;
  versions: number;
  jobs: number;
  tokens: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  /** mean of the non-null daily score averages, weighted by day count=1 */
  score_avg: number | null;
}

/** ≤45 days → daily, ≤550 → ISO weeks, beyond → months. */
export function pickBucketMode(spanDays: number): BucketMode {
  if (spanDays <= 45) return "day";
  if (spanDays <= 550) return "week";
  return "month";
}

function dayOf(iso: string): Date {
  // noon avoids DST edges shifting the date when adding/subtracting days
  return new Date(`${iso}T12:00:00Z`);
}

/** Monday (ISO week start) of the week containing `d`, in UTC. */
export function isoWeekStart(d: Date): Date {
  const dow = (d.getUTCDay() + 6) % 7; // Mon=0 … Sun=6
  const out = new Date(d);
  out.setUTCDate(out.getUTCDate() - dow);
  return out;
}

export function bucketKeyFor(isoDay: string, mode: BucketMode): string {
  const d = dayOf(isoDay);
  if (mode === "day") return isoDay;
  if (mode === "month") return isoDay.slice(0, 7);
  const w = isoWeekStart(d);
  return w.toISOString().slice(0, 10);
}

/** fold day rows into buckets in input order (activity is sorted ascending). */
export function bucketActivity(
  days: StatsDayBucket[],
  mode: BucketMode
): ActivityBucket[] {
  const out = new Map<string, ActivityBucket & { scoreSum: number; scoreN: number }>();
  const order: string[] = [];
  for (const d of days) {
    const k = bucketKeyFor(d.date, mode);
    let b = out.get(k);
    if (!b) {
      b = {
        start: d.date,
        end: d.date,
        versions: 0,
        jobs: 0,
        tokens: 0,
        succeeded: 0,
        failed: 0,
        cancelled: 0,
        score_avg: null,
        scoreSum: 0,
        scoreN: 0,
      };
      out.set(k, b);
      order.push(k);
    }
    b.end = d.date;
    b.versions += d.versions;
    b.jobs += d.jobs;
    b.tokens += d.tokens;
    b.succeeded += d.succeeded;
    b.failed += d.failed;
    b.cancelled += d.cancelled ?? 0;
    if (d.score_avg != null) {
      b.scoreSum += d.score_avg;
      b.scoreN += 1;
    }
  }
  return order.map((k) => {
    const { scoreSum, scoreN, ...b } = out.get(k)!;
    return { ...b, score_avg: scoreN ? Math.round((scoreSum / scoreN) * 10) / 10 : null };
  });
}

/* ---------- localized tick/label formatting ---------- */

/** "Mar 5" / "5. März" / "5 mar" — year only when it differs from the current
 *  one, so "all time" charts spanning year boundaries stay unambiguous. */
export function fmtDayTick(isoDay: string, locale: string): string {
  const d = dayOf(isoDay);
  const opts: Intl.DateTimeFormatOptions = { day: "numeric", month: "short" };
  if (d.getUTCFullYear() !== new Date().getUTCFullYear()) opts.year = "2-digit";
  return new Intl.DateTimeFormat(locale, opts).format(d);
}

/** tick for a bucket — month name for month buckets, start-day otherwise. */
export function fmtBucketTick(b: ActivityBucket, mode: BucketMode, locale: string): string {
  if (mode === "month") {
    const d = dayOf(`${b.start}`);
    const opts: Intl.DateTimeFormatOptions = { month: "short" };
    if (d.getUTCFullYear() !== new Date().getUTCFullYear()) opts.year = "2-digit";
    return new Intl.DateTimeFormat(locale, opts).format(d);
  }
  return fmtDayTick(b.start, locale);
}

/** full-range tooltip label: "Mar 3 – Mar 9" for weeks. */
export function fmtBucketRange(b: ActivityBucket, mode: BucketMode, locale: string): string {
  if (mode === "month")
    return new Intl.DateTimeFormat(locale, { month: "long", year: "numeric" }).format(
      dayOf(b.start)
    );
  if (b.start === b.end) return fmtDayTick(b.start, locale);
  return `${fmtDayTick(b.start, locale)} – ${fmtDayTick(b.end, locale)}`;
}

/** compact axis numbers: 150000 → "150K" (locale-aware). */
export function fmtCompact(n: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(n);
}
