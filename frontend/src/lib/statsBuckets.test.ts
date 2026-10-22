import { describe, expect, it } from "vitest";
import type { StatsDayBucket } from "./types";
import {
  bucketActivity,
  bucketKeyFor,
  fmtBucketTick,
  fmtCompact,
  fmtDayTick,
  isoWeekStart,
  pickBucketMode,
} from "./statsBuckets";

const day = (date: string, over: Partial<StatsDayBucket> = {}): StatsDayBucket => ({
  date,
  versions: 0,
  jobs: 0,
  tokens: 0,
  succeeded: 0,
  failed: 0,
  cancelled: 0,
  score_avg: null,
  ...over,
});

describe("pickBucketMode", () => {
  it("day for ≤45, week ≤550, month beyond", () => {
    expect(pickBucketMode(7)).toBe("day");
    expect(pickBucketMode(45)).toBe("day");
    expect(pickBucketMode(46)).toBe("week");
    expect(pickBucketMode(90)).toBe("week");
    expect(pickBucketMode(550)).toBe("week");
    expect(pickBucketMode(551)).toBe("month");
  });
});

describe("isoWeekStart", () => {
  it("returns Monday for any weekday", () => {
    // 2025-03-05 is a Wednesday
    expect(isoWeekStart(new Date("2025-03-05T12:00:00Z")).toISOString().slice(0, 10)).toBe(
      "2025-03-03"
    );
    // Sunday belongs to the same week
    expect(isoWeekStart(new Date("2025-03-09T12:00:00Z")).toISOString().slice(0, 10)).toBe(
      "2025-03-03"
    );
    // Monday stays Monday
    expect(isoWeekStart(new Date("2025-03-03T12:00:00Z")).toISOString().slice(0, 10)).toBe(
      "2025-03-03"
    );
  });
});

describe("bucketActivity", () => {
  const rows = [
    day("2025-03-03", { versions: 2, jobs: 3, succeeded: 1, score_avg: 80 }),
    day("2025-03-04", { versions: 1, failed: 2 }),
    day("2025-03-09", { jobs: 1, cancelled: 1, score_avg: 60 }),
    day("2025-03-10", { tokens: 500 }), // next ISO week
    day("2025-04-01", { versions: 1 }), // next month
  ];

  it("keeps rows untouched in day mode", () => {
    const b = bucketActivity(rows, "day");
    expect(b).toHaveLength(5);
    expect(b[0]).toMatchObject({ start: "2025-03-03", end: "2025-03-03", versions: 2 });
  });

  it("folds days into ISO weeks", () => {
    const b = bucketActivity(rows, "week");
    expect(b).toHaveLength(3);
    expect(b[0]).toMatchObject({
      start: "2025-03-03",
      end: "2025-03-09",
      versions: 3,
      jobs: 4,
      succeeded: 1,
      failed: 2,
      cancelled: 1,
    });
    expect(b[1]).toMatchObject({ start: "2025-03-10", tokens: 500 });
  });

  it("folds days into months", () => {
    const b = bucketActivity(rows, "month");
    expect(b).toHaveLength(2);
    expect(b[0]).toMatchObject({ start: "2025-03-03", end: "2025-03-10", versions: 3 });
    expect(b[1]).toMatchObject({ start: "2025-04-01" });
  });

  it("averages score over scored days only", () => {
    const b = bucketActivity(rows, "week");
    expect(b[0].score_avg).toBe(70); // (80+60)/2, unscored day ignored
    expect(b[1].score_avg).toBeNull();
  });

  it("is empty-safe", () => {
    expect(bucketActivity([], "week")).toEqual([]);
  });
});

describe("bucketKeyFor", () => {
  it("day → identity, month → YYYY-MM, week → monday", () => {
    expect(bucketKeyFor("2025-03-05", "day")).toBe("2025-03-05");
    expect(bucketKeyFor("2025-03-05", "month")).toBe("2025-03");
    expect(bucketKeyFor("2025-03-05", "week")).toBe("2025-03-03");
  });
});

describe("tick formatting", () => {
  it("formats day ticks localized, year only when not current", () => {
    const thisYear = new Date().getUTCFullYear();
    expect(fmtDayTick(`${thisYear}-03-05`, "en")).toMatch(/Mar/);
    expect(fmtDayTick(`${thisYear}-03-05`, "de")).toMatch(/März|Mrz|Mär/i);
    // past years carry a year suffix → no ambiguity across boundaries
    expect(fmtDayTick("2020-12-30", "en")).toMatch(/Dec/);
    expect(fmtDayTick("2020-12-30", "en")).toMatch(/20/);
  });

  it("formats month buckets as month name", () => {
    const b = bucketActivity([day("2020-02-10")], "month")[0];
    expect(fmtBucketTick(b, "month", "en")).toMatch(/Feb/);
  });

  it("compacts large numbers", () => {
    expect(fmtCompact(150000, "en")).toBe("150K");
    expect(fmtCompact(0, "en")).toBe("0");
    expect(fmtCompact(1234, "en")).toMatch(/1\.2K/);
  });
});
