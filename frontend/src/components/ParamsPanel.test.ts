import { describe, expect, it } from "vitest";
import type { ParamDef } from "../lib/types";
import { buildOverrides, paramError } from "./ParamsPanel";

const num = (over: Partial<ParamDef> = {}): ParamDef => ({
  name: "wall",
  value: 2,
  unit: "mm",
  min: 0.5,
  max: 10,
  description: "",
  ...over,
});

describe("paramError", () => {
  it("rejects non-numeric input on a numeric param", () => {
    expect(paramError(num(), 2, "abc")).toBe("not a number");
    expect(paramError(num(), 2, "1.5")).toBeNull();
  });

  it("flags out-of-range values with the min–max bounds", () => {
    expect(paramError(num(), 2, "0.1")).toBe("outside 0.5–10");
    expect(paramError(num(), 2, "11")).toBe("outside 0.5–10");
    expect(paramError(num(), 2, "0.5")).toBeNull();
    expect(paramError(num(), 2, "10")).toBeNull();
  });

  it("uses the schema bounds when only one side is set", () => {
    expect(paramError(num({ min: null }), 2, "11")).toBe("outside −∞–10");
    expect(paramError(num({ min: null }), 2, "-5")).toBeNull();
    expect(paramError(num({ max: null }), 2, "0.1")).toBe("outside 0.5–∞");
    expect(paramError(num({ max: null }), 2, "99")).toBeNull();
  });

  it("passes empty strings and non-numeric params through", () => {
    expect(paramError(num(), 2, "")).toBeNull();
    expect(
      paramError({ ...num(), name: "s", value: "abc" }, "abc", "anything")
    ).toBeNull();
    expect(
      paramError({ ...num(), name: "b", value: true }, true, "nonsense")
    ).toBeNull();
  });
});

describe("buildOverrides", () => {
  const schema = [
    num(),
    { name: "label", value: "a", unit: "-", min: null, max: null, description: "" },
    { name: "flag", value: true, unit: "-", min: null, max: null, description: "" },
  ];

  it("sends only edited, changed values — coerced to the param type", () => {
    const out = buildOverrides(schema, null, {
      wall: "3.5",
      label: "b",
      flag: "false",
      untouched: "",
    });
    expect(out).toEqual({ wall: 3.5, label: "b", flag: false });
  });

  it("skips values equal to the current value", () => {
    expect(buildOverrides(schema, { wall: 2 }, { wall: "2" })).toEqual({});
    expect(buildOverrides(schema, null, { flag: "true" })).toEqual({});
  });

  it("never emits NaN/JSON-null for invalid numeric input", () => {
    expect(buildOverrides(schema, null, { wall: "abc" })).toEqual({});
  });
});
