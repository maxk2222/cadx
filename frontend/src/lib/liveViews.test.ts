import { describe, expect, it } from "vitest";
import {
  femViewCaption,
  isFemView,
  pickView,
  sortViews,
  viewLabel,
} from "./liveViews";

describe("pickView", () => {
  it("keeps a valid user selection", () => {
    expect(pickView(["iso", "front"], "front")).toBe("front");
  });
  it("falls back when the selection vanished", () => {
    expect(pickView(["iso", "front"], "top")).toBe("iso");
  });
  it("prefers canonical model views over fem plots", () => {
    const views = ["fem:vonmises_x", "fem:regions_x", "front"];
    expect(pickView(views, null)).toBe("front");
  });
  it("returns the first non-fem view when no preferred exists", () => {
    expect(pickView(["fem:regions_a", "measure_x"], null)).toBe("measure_x");
  });
  it("uses a fem view when only fem exists", () => {
    expect(pickView(["fem:vonmises_a"], null)).toBe("fem:vonmises_a");
  });
  it("returns null for an empty list", () => {
    expect(pickView([], null)).toBeNull();
  });
});

describe("sortViews", () => {
  it("orders model views by preference, fem last grouped", () => {
    const views = ["fem:regions_a", "top", "iso", "fem:vonmises_a", "front"];
    expect(sortViews(views)).toEqual([
      "iso",
      "front",
      "top",
      "fem:vonmises_a",
      "fem:regions_a",
    ]);
  });
});

describe("isFemView / viewLabel", () => {
  it("detects fem views", () => {
    expect(isFemView("fem:vonmises_x")).toBe(true);
    expect(isFemView("iso")).toBe(false);
  });
  it("labels fem views compactly", () => {
    expect(viewLabel("fem:vonmises_chain pull")).toBe("fem vonmises chain pull");
    expect(viewLabel("contact_sheet")).toBe("sheet");
  });
});

describe("femViewCaption", () => {
  it("parses kind + export + case", () => {
    expect(femViewCaption("vonmises_export_chain pull")).toBe(
      "von mises (export) · chain pull"
    );
    expect(femViewCaption("regions_rear brake torque")).toBe(
      "regions · rear brake torque"
    );
  });
  it("falls back to the stem", () => {
    expect(femViewCaption("grid_view")).toBe("grid view");
    expect(femViewCaption(null)).toBeNull();
  });
});
