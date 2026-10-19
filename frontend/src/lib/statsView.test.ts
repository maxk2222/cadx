import { describe, expect, it } from "vitest";
import type { Stats } from "./types";
import {
  meshCacheRate,
  passRateTone,
  scoreTone,
  statsEmpty,
  statusColor,
  tokensPerSuccess,
} from "./statsView";

const baseStats = (): Stats =>
  ({
    totals: {
      versions: 4,
      jobs: 3,
      llm_calls: 10,
      prompt_tokens: 800,
      completion_tokens: 200,
    },
    versions_by_status: [{ name: "succeeded", count: 2 }],
    mesh: { generations: 4, cached: 1, avg_duration_s: 5, models: [] },
  }) as unknown as Stats;

describe("statsView", () => {
  it("maps statuses to semantic colors", () => {
    expect(statusColor("failed")).toBe("#f87171");
    expect(statusColor("succeeded")).toBe("#34d399");
    expect(statusColor("mystery")).toBe("#7c9cf5"); // fallback
  });

  it("grades pass rates and scores", () => {
    expect(passRateTone(0.9)).toBe("green");
    expect(passRateTone(0.5)).toBe("amber");
    expect(passRateTone(0.1)).toBe("red");
    expect(passRateTone(null)).toBe("amber");
    expect(scoreTone(90)).toBe("green");
    expect(scoreTone(20)).toBe("red");
  });

  it("detects truly empty ranges", () => {
    const s = baseStats();
    expect(statsEmpty(s)).toBe(false);
    s.totals.versions = 0;
    s.totals.jobs = 0;
    s.totals.llm_calls = 0;
    expect(statsEmpty(s)).toBe(true);
    s.totals.jobs = 1; // jobs alone still count as data
    expect(statsEmpty(s)).toBe(false);
  });

  it("derives tokens-per-success and mesh cache rate", () => {
    const s = baseStats();
    expect(tokensPerSuccess(s)).toBe(500); // 1000 tokens / 2 succeeded
    expect(meshCacheRate(s)).toBe(0.25);
    s.mesh.generations = 0;
    expect(meshCacheRate(s)).toBeNull();
    s.versions_by_status = [];
    expect(tokensPerSuccess(s)).toBeNull();
  });
});
