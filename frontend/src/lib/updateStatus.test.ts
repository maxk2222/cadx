import { describe, expect, it } from "vitest";
import { behindBadge, isUpdateActive, logTail, updatePhaseLabel } from "./updateStatus";

describe("updatePhaseLabel", () => {
  it("maps known phases", () => {
    expect(updatePhaseLabel({ state: "running", phase: "deps" })).toBe("installing dependencies");
    expect(updatePhaseLabel({ state: "running", phase: "verify" })).toBe("verifying health");
  });
  it("falls back to raw phase/state", () => {
    expect(updatePhaseLabel({ state: "running", phase: "weird" })).toBe("weird");
    expect(updatePhaseLabel({ state: "idle", phase: null })).toBe("idle");
    expect(updatePhaseLabel({ state: "failed", phase: "pull" })).toBe("failed");
  });
});

describe("isUpdateActive", () => {
  it("active states", () => {
    expect(isUpdateActive({ state: "running" })).toBe(true);
    expect(isUpdateActive({ state: "checking" })).toBe(true);
    expect(isUpdateActive({ state: "rolling-back" })).toBe(true);
  });
  it("terminal/absent states", () => {
    expect(isUpdateActive({ state: "done" })).toBe(false);
    expect(isUpdateActive({ state: "failed" })).toBe(false);
    expect(isUpdateActive({ state: "idle" })).toBe(false);
    expect(isUpdateActive(null)).toBe(false);
    expect(isUpdateActive(undefined)).toBe(false);
  });
});

describe("behindBadge", () => {
  it("behind count or undefined", () => {
    expect(behindBadge(3)).toBe(3);
    expect(behindBadge(0)).toBeUndefined();
    expect(behindBadge(null)).toBeUndefined();
  });
});

describe("logTail", () => {
  it("keeps the last N lines joined", () => {
    const lines = Array.from({ length: 20 }, (_, i) => `line ${i}`);
    const out = logTail(lines, 5);
    expect(out.split("\n")).toHaveLength(5);
    expect(out).toContain("line 19");
    expect(logTail(undefined)).toBe("");
  });
});
