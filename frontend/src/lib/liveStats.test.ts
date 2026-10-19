import { describe, expect, it } from "vitest";
import { etaSeconds, fmtDur, fmtTokens, sparkPoints, stepRows } from "./liveStats";
import type { JobLive } from "./types";

const NOW = Date.parse("2024-06-01T12:00:00Z");

describe("stepRows", () => {
  it("returns [] for null/undefined live", () => {
    expect(stepRows(null, NOW)).toEqual([]);
    expect(stepRows(undefined, NOW)).toEqual([]);
  });

  it("merges history and the active step in observed order", () => {
    const live: JobLive = {
      step: "critique",
      step_started_ts: "2024-06-01T11:59:30Z", // 30s before NOW
      step_history: [
        { step: "plan", it: 0, dt_s: 5 },
        { step: "codegen", it: 1, dt_s: 83 },
      ],
    };
    expect(stepRows(live, NOW)).toEqual([
      { step: "plan", label: "plan", dt_s: 5, active: false },
      { step: "codegen", label: "codegen", dt_s: 83, active: false },
      { step: "critique", label: "critique", dt_s: 30, active: true },
    ]);
  });

  it("has no active row when live.step is unset", () => {
    const live: JobLive = {
      step_history: [{ step: "execute", it: 1, dt_s: 12.5 }],
    };
    const rows = stepRows(live, NOW);
    expect(rows).toEqual([{ step: "execute", label: "build", dt_s: 12.5, active: false }]);
    expect(rows.every((r) => !r.active)).toBe(true);
  });

  it("does not double a trailing history entry equal to live.step", () => {
    const live: JobLive = {
      step: "codegen",
      step_started_ts: "2024-06-01T12:00:00Z",
      step_history: [
        { step: "plan", it: 0, dt_s: 4 },
        { step: "codegen", it: 1, dt_s: 60 },
      ],
    };
    const rows = stepRows(live, NOW);
    expect(rows.map((r) => r.step)).toEqual(["plan", "codegen"]);
    expect(rows[1].active).toBe(true);
    expect(rows[1].dt_s).toBe(0);
  });

  it("handles a missing/invalid step_started_ts as 0 elapsed", () => {
    const live: JobLive = { step: "fem" };
    const rows = stepRows(live, NOW);
    expect(rows).toEqual([{ step: "fem", label: "fem", dt_s: 0, active: true }]);
  });
});

describe("sparkPoints", () => {
  it("returns empty string for no values", () => {
    expect(sparkPoints([], 100, 20)).toBe("");
  });

  it("returns a single centered point for one value", () => {
    expect(sparkPoints([5], 100, 20)).toBe("50.0,10.0");
  });

  it("normalizes min/max into width×height with y flipped", () => {
    // values 0,5,10 in 100×20 → min at bottom (y=20), max at top (y=0)
    expect(sparkPoints([0, 5, 10], 100, 20)).toBe("0.0,20.0 50.0,10.0 100.0,0.0");
  });

  it("flattens to mid-height when all values are equal", () => {
    expect(sparkPoints([3, 3], 100, 20)).toBe("0.0,10.0 100.0,10.0");
  });
});

describe("fmtTokens", () => {
  it("formats small counts verbatim", () => {
    expect(fmtTokens(842)).toBe("842");
    expect(fmtTokens(0)).toBe("0");
  });

  it("formats thousands as k", () => {
    expect(fmtTokens(142300)).toBe("142.3k");
    expect(fmtTokens(1000)).toBe("1.0k");
  });

  it("formats millions as M", () => {
    expect(fmtTokens(2_500_000)).toBe("2.5M");
  });
});

describe("fmtDur", () => {
  it("formats m:ss", () => {
    expect(fmtDur(83)).toBe("1:23");
    expect(fmtDur(0.4)).toBe("0:00");
    expect(fmtDur(9.9)).toBe("0:09");
  });

  it("formats h:mm:ss", () => {
    expect(fmtDur(3615)).toBe("1:00:15");
    expect(fmtDur(3600)).toBe("1:00:00");
  });
});

describe("etaSeconds", () => {
  it("returns null for null live or edge progress", () => {
    expect(etaSeconds(null, 0.5)).toBeNull();
    const live: JobLive = { step_history: [{ step: "plan", it: 0, dt_s: 10 }] };
    expect(etaSeconds(live, 0.02)).toBeNull();
    expect(etaSeconds(live, 0)).toBeNull();
    expect(etaSeconds(live, 0.99)).toBeNull();
  });

  it("returns null when nothing has elapsed", () => {
    expect(etaSeconds({ step: "plan" }, 0.5)).toBeNull();
  });

  it("estimates remaining time linearly at progress 0.5", () => {
    const live: JobLive = {
      step_history: [{ step: "plan", it: 0, dt_s: 40 }],
      step: "codegen",
      step_started_ts: new Date(Date.now() - 20_000).toISOString(),
    };
    const eta = etaSeconds(live, 0.5);
    // elapsed ≈ 60s → eta ≈ 60s (running step adds a little jitter)
    expect(eta).not.toBeNull();
    expect(eta!).toBeGreaterThan(55);
    expect(eta!).toBeLessThan(70);
  });
});
