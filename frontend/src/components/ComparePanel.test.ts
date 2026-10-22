import { describe, expect, it } from "vitest";
import type { VersionSummary } from "../lib/types";
import {
  defaultCompareTarget,
  deltaTone,
  fmtDelta,
  fmtParamVal,
  fmtVec,
  fmtVecDelta,
  fmtVol,
  shortId,
  trunc,
  versionLabel,
} from "./ComparePanel";

const vs = (over: Partial<VersionSummary> = {}): VersionSummary => ({
  id: "id-0001",
  project_id: "p",
  parent_id: null,
  number: 1,
  kind: "generate",
  prompt: "a mounting bracket",
  name: "",
  status: "succeeded",
  score: null,
  iterations: 1,
  error: null,
  gen_mode: "auto",
  resolved_mode: "cad",
  created_at: "2024-01-01T00:00:00Z",
  thumbnail_url: null,
  ...over,
});

describe("trunc / shortId", () => {
  it("leaves short strings alone and ellipsizes long ones", () => {
    expect(trunc("hello", 10)).toBe("hello");
    const out = trunc("abcdefghij", 6);
    expect(out).toBe("abcde…");
    expect(out.length).toBe(6);
  });
  it("takes the first 8 chars of an id", () => {
    expect(shortId("0123456789abcdef")).toBe("01234567");
    expect(shortId("short")).toBe("short");
  });
});

describe("versionLabel", () => {
  it("combines index, short id, kind and truncated prompt", () => {
    const l = versionLabel(vs({ number: 3, id: "deadbeefcafe", kind: "edit" }));
    expect(l.startsWith("v3 · deadbeef · edit · ")).toBe(true);
    expect(l).toContain("a mounting bracket");
  });
  it("truncates a long prompt and collapses whitespace", () => {
    const l = versionLabel(
      vs({ prompt: `  line one\n   ${"x".repeat(80)}  ` }),
      20
    );
    expect(l).not.toContain("\n");
    expect(l.endsWith("…")).toBe(true);
  });
  it("falls back for empty prompts", () => {
    expect(versionLabel(vs({ prompt: "   " }))).toContain("(no prompt)");
  });
});

describe("defaultCompareTarget", () => {
  it("returns null with no candidates", () => {
    expect(defaultCompareTarget([], null)).toBeNull();
  });
  it("prefers the parent version when present", () => {
    const list = [
      vs({ id: "older", number: 1 }),
      vs({ id: "parent", number: 3 }),
      vs({ id: "newer", number: 5 }),
    ];
    expect(defaultCompareTarget(list, "parent")).toBe("parent");
  });
  it("falls back to the highest-numbered version", () => {
    const list = [
      vs({ id: "a", number: 2 }),
      vs({ id: "b", number: 7 }),
      vs({ id: "c", number: 4 }),
    ];
    expect(defaultCompareTarget(list, "missing")).toBe("b");
    expect(defaultCompareTarget(list, null)).toBe("b");
  });
});

describe("fmtVol", () => {
  it("converts mm³ to cm³ and handles nulls", () => {
    expect(fmtVol(null)).toBe("—");
    expect(fmtVol(undefined)).toBe("—");
    expect(fmtVol(NaN)).toBe("—");
    expect(fmtVol(12345)).toBe("12.3 cm³");
    expect(fmtVol(500)).toBe("0.5 cm³");
  });
});

describe("fmtVec / fmtVecDelta", () => {
  it("joins components with × and handles nulls", () => {
    expect(fmtVec(null)).toBe("—");
    expect(fmtVec([])).toBe("—");
    expect(fmtVec([10, 20.25, 30])).toBe("10.0 × 20.3 × 30.0");
  });
  it("signs each component in a delta vector", () => {
    expect(fmtVecDelta([0.001, -1.25, 2])).toBe("±0 × −1.3 × +2.0");
    expect(fmtVecDelta(null)).toBe("—");
  });
});

describe("fmtDelta", () => {
  it("formats sign, magnitude and digit precision", () => {
    expect(fmtDelta(null)).toBe("—");
    expect(fmtDelta(undefined)).toBe("—");
    expect(fmtDelta(2.54)).toBe("+2.5");
    expect(fmtDelta(-3.26)).toBe("−3.3");
    expect(fmtDelta(0)).toBe("±0");
    expect(fmtDelta(0.001)).toBe("±0");
    expect(fmtDelta(2, 0)).toBe("+2");
    expect(fmtDelta(-2, 0)).toBe("−2");
  });
});

describe("deltaTone", () => {
  it("maps sign to a colour class", () => {
    expect(deltaTone(1)).toBe("text-emerald-400");
    expect(deltaTone(-0.5)).toBe("text-red-400");
    expect(deltaTone(0)).toBe("text-zinc-500");
    expect(deltaTone(null)).toBe("text-zinc-500");
  });
});

describe("fmtParamVal", () => {
  it("renders — for absent sides", () => {
    expect(fmtParamVal(null)).toBe("—");
    expect(fmtParamVal(undefined)).toBe("—");
  });
  it("JSON-encodes values", () => {
    expect(fmtParamVal(2)).toBe("2");
    expect(fmtParamVal("abc")).toBe('"abc"');
    expect(fmtParamVal(true)).toBe("true");
    expect(fmtParamVal({ x: 1 })).toBe('{"x":1}');
    expect(fmtParamVal([1, 2])).toBe("[1,2]");
  });
  it("truncates long values", () => {
    const out = fmtParamVal("y".repeat(200), 48);
    expect(out.endsWith("…")).toBe(true);
    expect(out.length).toBe(48);
  });
});
