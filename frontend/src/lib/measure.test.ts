import { describe, expect, it } from "vitest";
import { distMm, nextMeasurePts, type Pt3 } from "./measure";

describe("distMm", () => {
  it("is 0 for identical points", () => {
    expect(distMm([1, 2, 3], [1, 2, 3])).toBe(0);
  });
  it("measures axis-aligned distances", () => {
    expect(distMm([0, 0, 0], [3, 0, 0])).toBe(3);
    expect(distMm([0, 0, 0], [0, -4, 0])).toBe(4);
  });
  it("measures a 3-4-5 triangle in 3D", () => {
    expect(distMm([0, 0, 0], [3, 4, 0])).toBeCloseTo(5);
    expect(distMm([1, 1, 1], [4, 5, 1])).toBeCloseTo(5);
  });
});

describe("nextMeasurePts", () => {
  const a: Pt3 = [0, 0, 0];
  const b: Pt3 = [1, 0, 0];
  const c: Pt3 = [2, 0, 0];
  it("collects the first two picks", () => {
    expect(nextMeasurePts([], a)).toEqual([a]);
    expect(nextMeasurePts([a], b)).toEqual([a, b]);
  });
  it("restarts after two points", () => {
    expect(nextMeasurePts([a, b], c)).toEqual([c]);
  });
});
